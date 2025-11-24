import os
import shutil
import time
import zipfile
from typing import List
import warnings

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

# -----------------------
# FORCE CPU and suppress TF warnings
# -----------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings('ignore', category=UserWarning)

# Project modules
from src.prediction import predict_image_from_path
from src.model import load_trained_model, retrain_model
from src.preprocessing import create_data_generators

# -----------------------
# Directories setup (root-level)
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data folders
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
UPLOAD_DIR = os.path.join(DATA_DIR, "new_uploads")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Models folder
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Chest X-ray Pneumonia API")

# Global variables
SELECTED_MODEL = None
GLOBAL_MODEL = None

# -----------------------
# Helper functions
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
    """Get the most recent model file from MODELS_DIR."""
    if not os.path.exists(MODELS_DIR):
        raise HTTPException(status_code=404, detail=f"Models directory does not exist: {MODELS_DIR}")

    all_files = os.listdir(MODELS_DIR)
    model_files = [f for f in all_files if os.path.isfile(os.path.join(MODELS_DIR, f)) and (f.endswith(".keras") or f.endswith(".h5"))]

    if not model_files:
        raise HTTPException(status_code=404, detail=f"No .keras or .h5 model files found in {MODELS_DIR}")

    model_files_with_time = [(f, os.path.getmtime(os.path.join(MODELS_DIR, f))) for f in model_files]
    model_files_with_time.sort(key=lambda x: x[1], reverse=True)

    latest_model = model_files_with_time[0][0]
    return os.path.join(MODELS_DIR, latest_model)


def _get_model_to_use():
    global SELECTED_MODEL
    if SELECTED_MODEL:
        model_path = os.path.join(MODELS_DIR, SELECTED_MODEL)
        if os.path.exists(model_path):
            return model_path
        else:
            SELECTED_MODEL = None
    return _get_latest_model_file()


# -----------------------
# Load model at startup
# -----------------------
print("Loading model at startup...")
MODEL_PATH = _get_model_to_use()
GLOBAL_MODEL = load_trained_model(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")


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
        model_files = [f for f in all_files if os.path.isfile(os.path.join(MODELS_DIR, f)) and (f.endswith(".keras") or f.endswith(".h5"))]
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
    global SELECTED_MODEL, GLOBAL_MODEL
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    if not (model_name.endswith(".keras") or model_name.endswith(".h5")):
        raise HTTPException(status_code=400, detail="Invalid model format")
    SELECTED_MODEL = model_name
    GLOBAL_MODEL = load_trained_model(model_path)
    return {"status": "success", "selected_model": SELECTED_MODEL, "message": f"Now using '{model_name}' for all operations"}


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
    tmp_folder = os.path.join(BASE_DIR, "tmp", str(int(time.time() * 1000)))
    os.makedirs(tmp_folder, exist_ok=True)
    tmp_path = os.path.join(tmp_folder, file.filename)

    await file.seek(0)
    with open(tmp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        result = predict_image_from_path(GLOBAL_MODEL, tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        shutil.rmtree(tmp_folder, ignore_errors=True)

    return JSONResponse(content=result)


# -----------------------
# Upload bulk files for retraining
# -----------------------
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    _clear_folder(UPLOAD_DIR)
    saved = []
    for upload_file in files:
        dest = os.path.join(UPLOAD_DIR, upload_file.filename)
        _save_upload_file(upload_file, dest)
        saved.append(upload_file.filename)
        if upload_file.filename.lower().endswith(".zip"):
            _extract_zip_to_folder(dest, UPLOAD_DIR)
    return {"status": "ok", "saved": saved, "upload_folder": UPLOAD_DIR}


# -----------------------
# Trigger retraining
# -----------------------
@app.post("/retrain")
def trigger_retrain(epochs: int = 5, batch_size: int = 32, fine_tune: bool = True):
    if not os.path.exists(UPLOAD_DIR) or not os.listdir(UPLOAD_DIR):
        raise HTTPException(status_code=400, detail="No uploaded data found. Please POST to /upload first.")
    model_path, history = retrain_model(new_data_folder=UPLOAD_DIR,
                                       original_train_folder=TRAIN_DIR,
                                       batch_size=batch_size,
                                       epochs=epochs,
                                       output_directory=MODELS_DIR,
                                       fine_tune=fine_tune)
    summary = {k: v[-1] if isinstance(v, list) else v for k, v in history.history.items()}
    return {"status": "completed", "model_path": model_path, "summary": summary}


# -----------------------
# Model metrics endpoint
# -----------------------
@app.get("/metrics")
def metrics(batch_size: int = 64):
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    try:
        _, _, test_gen = create_data_generators(train_dir=TRAIN_DIR,
                                               test_dir=TEST_DIR,
                                               img_size=(224, 224),
                                               batch_size=batch_size)
        results = GLOBAL_MODEL.evaluate(test_gen, verbose=1)
        loss = round(float(results[0]), 4) if isinstance(results, (list, tuple)) else round(float(results), 4)
        accuracy = round(float(results[1]), 4) if isinstance(results, (list, tuple)) and len(results) > 1 else None
        test_gen.reset()
        y_pred_probs = GLOBAL_MODEL.predict(test_gen, verbose=1)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_true = test_gen.classes
        precision = round(float(precision_score(y_true, y_pred)), 4)
        recall = round(float(recall_score(y_true, y_pred)), 4)
        auc = round(float(roc_auc_score(y_true, y_pred_probs)), 4)
        if accuracy is None:
            accuracy = round(float(np.mean(y_pred == y_true)), 4)
        return {"loss": loss, "accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {str(e)}")


# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
