import os
import shutil
import time
import zipfile
from typing import List
import warnings
import threading

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)

# Project modules
from src.prediction import predict_image_from_path
from src.model import load_trained_model, retrain_model
from src.preprocessing import create_data_generators

# Directories
UPLOAD_DIR = os.environ.get(
    "UPLOAD_DIR",
    r"C:\Users\fadhl\OneDrive\Desktop\ML_pipeline_summative\data\new_uploads"
)
MODELS_DIR = os.environ.get(
    "MODELS_DIR",
    r"C:\Users\fadhl\OneDrive\Desktop\ML_pipeline_summative\models"
)
TEST_DIR = os.environ.get(
    "TEST_DIR",
    r"C:\Users\fadhl\OneDrive\Desktop\ML_pipeline_summative\data\test"
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

app = FastAPI(title="Chest X-ray Pneumonia API")

# -----------------------
# Global variables
# -----------------------
SELECTED_MODEL = None
RETRAIN_STATUS = {"running": False, "progress": 0, "status": "idle"}
RETRAIN_STOP = False

# -----------------------
# Utility helpers
# -----------------------
def _save_upload_file(upload_file: UploadFile, destination: str):
    with open(destination, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    upload_file.file.close()

def _extract_zip_to_folder(zip_path: str, target_folder: str):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_folder)
    os.remove(zip_path)

def _clear_folder(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

def _get_latest_model_file():
    if not os.path.exists(MODELS_DIR):
        raise HTTPException(status_code=404, detail=f"Models directory does not exist: {MODELS_DIR}")
    all_files = os.listdir(MODELS_DIR)
    model_files = [f for f in all_files if os.path.isfile(os.path.join(MODELS_DIR, f)) and f.endswith((".keras", ".h5"))]
    if not model_files:
        raise HTTPException(status_code=404, detail=f"No .keras or .h5 model files found in {MODELS_DIR}")
    latest_model = max(model_files, key=lambda f: os.path.getmtime(os.path.join(MODELS_DIR, f)))
    return os.path.join(MODELS_DIR, latest_model)

def _get_model_to_use():
    global SELECTED_MODEL
    if SELECTED_MODEL:
        path = os.path.join(MODELS_DIR, SELECTED_MODEL)
        if os.path.exists(path):
            return path
        SELECTED_MODEL = None
    return _get_latest_model_file()

# -----------------------
# Health check
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

# -----------------------
# List available models
# -----------------------
@app.get("/models")
def list_models():
    global SELECTED_MODEL
    try:
        all_files = os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else []
        model_files = [f for f in all_files if os.path.isfile(os.path.join(MODELS_DIR, f)) and f.endswith((".keras", ".h5"))]
        latest_model_path = _get_latest_model_file() if model_files else None
        return {
            "models_directory": MODELS_DIR,
            "all_files": all_files,
            "valid_model_files": model_files,
            "selected_latest_model": latest_model_path,
            "currently_selected": SELECTED_MODEL
        }
    except Exception as e:
        return {"error": str(e), "models_directory": MODELS_DIR}

# -----------------------
# Select model
# -----------------------
@app.post("/select-model")
def select_model(model_name: str):
    global SELECTED_MODEL
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    if not model_name.endswith((".keras", ".h5")):
        raise HTTPException(status_code=400, detail="Invalid model format")
    SELECTED_MODEL = model_name
    return {"status": "success", "selected_model": SELECTED_MODEL, "message": f"Now using '{model_name}'"}

# -----------------------
# Download model
# -----------------------
@app.get("/download-model/{model_name}")
def download_model(model_name: str):
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(model_path, filename=model_name, media_type="application/octet-stream")

# -----------------------
# Predict single image
# -----------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model_file = _get_model_to_use()
    if not os.path.exists(model_file):
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_file}")
    tmp_folder = os.path.join("tmp", str(int(time.time() * 1000)))
    os.makedirs(tmp_folder, exist_ok=True)
    tmp_path = os.path.join(tmp_folder, file.filename)
    _save_upload_file(file, tmp_path)
    try:
        model = load_trained_model(model_file)
        result = predict_image_from_path(model, tmp_path)
    finally:
        shutil.rmtree(tmp_folder, ignore_errors=True)
    return JSONResponse(content=result)

# -----------------------
# Upload bulk files for retraining
# -----------------------
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    _clear_folder(UPLOAD_DIR)
    for f in files:
        dest = os.path.join(UPLOAD_DIR, f.filename)
        _save_upload_file(f, dest)
        if f.filename.lower().endswith(".zip"):
            _extract_zip_to_folder(dest, UPLOAD_DIR)
    return {"status": "ok"}

# -----------------------
# Keras callback for progress & stop
# -----------------------
from tensorflow.keras.callbacks import Callback

class ProgressCallback(Callback):
    def __init__(self, epochs):
        super().__init__()
        self.total_epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        global RETRAIN_STATUS, RETRAIN_STOP
        RETRAIN_STATUS["progress"] = int(((epoch + 1) / self.total_epochs) * 100)
        if RETRAIN_STOP:
            self.model.stop_training = True

# -----------------------
# Retraining thread
# -----------------------
def retrain_thread(epochs, batch_size, fine_tune):
    global RETRAIN_STATUS, RETRAIN_STOP
    RETRAIN_STATUS.update({"running": True, "progress": 0, "status": "running"})
    RETRAIN_STOP = False

    try:
        callback = ProgressCallback(epochs)
        model_path, history = retrain_model(
            new_data_folder=UPLOAD_DIR,
            original_train_folder=r"C:\Users\fadhl\OneDrive\Desktop\ML_pipeline_summative\data\train",
            batch_size=batch_size,
            epochs=epochs,
            output_directory=MODELS_DIR,
            fine_tune=fine_tune,
            callbacks=[callback]  # <-- pass the callback here
        )

        RETRAIN_STATUS.update({
            "running": False,
            "progress": 100,
            "status": "completed",
            "model_path": model_path,
            "summary": {k: v[-1] if isinstance(v, list) else v for k, v in history.history.items()}
        })

    except Exception as e:
        RETRAIN_STATUS.update({"running": False, "status": f"error: {str(e)}"})

# -----------------------
# Trigger retraining
# -----------------------
@app.post("/retrain")
def trigger_retrain(epochs: int = 5, batch_size: int = 32, fine_tune: bool = True):
    if not os.path.exists(UPLOAD_DIR) or not os.listdir(UPLOAD_DIR):
        raise HTTPException(status_code=400, detail="No uploaded data found. Please POST to /upload first.")
    threading.Thread(target=retrain_thread, args=(epochs, batch_size, fine_tune), daemon=True).start()
    return {"status": "started"}

# -----------------------
# Retraining progress endpoint
# -----------------------
@app.get("/retrain-progress")
def retrain_progress():
    return RETRAIN_STATUS

# -----------------------
# Stop retraining endpoint
# -----------------------
@app.post("/stop-retrain")
def stop_retrain():
    global RETRAIN_STOP
    RETRAIN_STOP = True
    return {"status": "stopping"}

# -----------------------
# Model metrics endpoint
# -----------------------
@app.get("/metrics")
def metrics(batch_size: int = 64):
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, roc_auc_score

    model_path = _get_model_to_use()
    model = load_trained_model(model_path)

    _, _, test_gen = create_data_generators(
        train_dir=r"C:\Users\fadhl\OneDrive\Desktop\ML_pipeline_summative\data\train",
        test_dir=TEST_DIR,
        img_size=(224, 224),
        batch_size=batch_size
    )

    results = model.evaluate(test_gen, verbose=1)
    loss = round(float(results[0]), 4) if isinstance(results, (list, tuple)) else round(float(results), 4)
    accuracy = round(float(results[1]), 4) if isinstance(results, (list, tuple)) and len(results) > 1 else None

    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = test_gen.classes
    precision = round(float(precision_score(y_true, y_pred)), 4)
    recall = round(float(recall_score(y_true, y_pred)), 4)
    auc = round(float(roc_auc_score(y_true, y_pred_probs)), 4)
    if accuracy is None:
        accuracy = round(float(np.mean(y_pred == y_true)), 4)

    return {"loss": loss, "accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc}

# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
