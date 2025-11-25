# src/api.py
import os
import shutil
import time
import zipfile
import gc
from typing import List, Optional
import warnings
from pathlib import Path
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

# Force CPU usage and quiet TF logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

# Import tensorflow only where needed for conversion or interpreter
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input

# Project modules
from src.model import load_trained_model, retrain_model
from src.preprocessing import create_data_generators

# -----------------------
# Directories setup
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
UPLOAD_DIR = os.path.join(DATA_DIR, "new_uploads")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
os.makedirs(MODELS_DIR, exist_ok=True)

DEFAULT_INPUT_SIZE = (224, 224)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Chest X-ray Pneumonia API (TFLite-ready)")

# Global state
SELECTED_MODEL: Optional[str] = None
GLOBAL_TFLITE_INTERPRETER = None
GLOBAL_TFLITE_INPUT_DETAILS = None
GLOBAL_TFLITE_OUTPUT_DETAILS = None
GLOBAL_MODEL_PATH = None  # path to tflite file being used

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


def _list_model_files():
    if not os.path.exists(MODELS_DIR):
        return []
    return [f for f in os.listdir(MODELS_DIR) if os.path.isfile(os.path.join(MODELS_DIR, f))]


def _list_valid_models():
    files = _list_model_files()
    return [f for f in files if f.endswith(".tflite") or f.endswith(".lite") or f.endswith(".keras") or f.endswith(".h5")]


def _get_latest_keras_model_file():
    all_files = _list_model_files()
    model_files = [f for f in all_files if f.endswith(".keras") or f.endswith(".h5")]
    if not model_files:
        return None
    model_files_with_time = [(f, os.path.getmtime(os.path.join(MODELS_DIR, f))) for f in model_files]
    model_files_with_time.sort(key=lambda x: x[1], reverse=True)
    return os.path.join(MODELS_DIR, model_files_with_time[0][0])


def _get_latest_tflite_file():
    all_files = _list_model_files()
    model_files = [f for f in all_files if f.endswith(".tflite") or f.endswith(".lite")]
    if not model_files:
        return None
    model_files_with_time = [(f, os.path.getmtime(os.path.join(MODELS_DIR, f))) for f in model_files]
    model_files_with_time.sort(key=lambda x: x[1], reverse=True)
    return os.path.join(MODELS_DIR, model_files_with_time[0][0])


def _get_latest_model_file():
    tflite_file = _get_latest_tflite_file()
    if tflite_file:
        return tflite_file
    keras_file = _get_latest_keras_model_file()
    if keras_file:
        return keras_file
    raise FileNotFoundError(f"No model files (.tflite, .keras or .h5) found in {MODELS_DIR}")


def convert_keras_to_tflite(keras_model_path: str, tflite_output_path: str):
    keras_model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)
    del keras_model
    gc.collect()
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    return tflite_output_path


def load_tflite_interpreter(tflite_path: str):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def preprocess_image_for_model(image_path: str, target_size=DEFAULT_INPUT_SIZE):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x.astype("float32")


def predict_with_tflite(interpreter, input_details, output_details, image_path: str, threshold: float = 0.5):
    arr = preprocess_image_for_model(image_path, target_size=DEFAULT_INPUT_SIZE)
    input_dtype = input_details[0]["dtype"]
    if input_dtype == np.float32:
        input_data = arr.astype(np.float32)
    elif input_dtype == np.uint8:
        input_data = arr.astype(np.uint8)
    else:
        input_data = arr.astype(input_dtype)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    prob = float(output_data[0][0])
    label = "PNEUMONIA" if prob > threshold else "NORMAL"
    confidence = prob if prob > threshold else 1.0 - prob
    return {"label": label, "confidence": float(confidence), "probability": prob}


# -----------------------
# Startup: load model
# -----------------------
def _startup_load_model():
    global GLOBAL_TFLITE_INTERPRETER, GLOBAL_TFLITE_INPUT_DETAILS, GLOBAL_TFLITE_OUTPUT_DETAILS, GLOBAL_MODEL_PATH

    try:
        model_candidate = _get_latest_model_file()
    except FileNotFoundError as e:
        print("Startup: no model file found:", str(e))
        return

    model_candidate = str(model_candidate)
    print("Startup: chosen model candidate:", model_candidate)

    if model_candidate.endswith(".tflite") or model_candidate.endswith(".lite"):
        try:
            interpreter, in_det, out_det = load_tflite_interpreter(model_candidate)
            GLOBAL_TFLITE_INTERPRETER = interpreter
            GLOBAL_TFLITE_INPUT_DETAILS = in_det
            GLOBAL_TFLITE_OUTPUT_DETAILS = out_det
            GLOBAL_MODEL_PATH = model_candidate
            print("Loaded TFLite model at startup:", model_candidate)
            return
        except Exception as e:
            print("Failed to load TFLite interpreter at startup:", str(e))

    if model_candidate.endswith(".keras") or model_candidate.endswith(".h5"):
        base = os.path.splitext(os.path.basename(model_candidate))[0]
        tflite_sibling = os.path.join(MODELS_DIR, f"{base}.tflite")
        if os.path.exists(tflite_sibling):
            try:
                interpreter, in_det, out_det = load_tflite_interpreter(tflite_sibling)
                GLOBAL_TFLITE_INTERPRETER = interpreter
                GLOBAL_TFLITE_INPUT_DETAILS = in_det
                GLOBAL_TFLITE_OUTPUT_DETAILS = out_det
                GLOBAL_MODEL_PATH = tflite_sibling
                print("Found existing tflite sibling and loaded:", tflite_sibling)
                return
            except Exception as e:
                print("Failed to load existing tflite sibling:", str(e))

        try:
            print("Converting Keras model to TFLite at startup. This may be memory heavy.")
            convert_keras_to_tflite(model_candidate, tflite_sibling)
            interpreter, in_det, out_det = load_tflite_interpreter(tflite_sibling)
            GLOBAL_TFLITE_INTERPRETER = interpreter
            GLOBAL_TFLITE_INPUT_DETAILS = in_det
            GLOBAL_TFLITE_OUTPUT_DETAILS = out_det
            GLOBAL_MODEL_PATH = tflite_sibling
            print("Conversion complete and tflite loaded:", tflite_sibling)
            return
        except Exception as e:
            print("Conversion failed:", str(e))
            try:
                keras_model = tf.keras.models.load_model(model_candidate)
                def keras_predict_wrapper(image_path: str, threshold=0.5):
                    arr = preprocess_image_for_model(image_path, target_size=DEFAULT_INPUT_SIZE)
                    prob = float(keras_model.predict(arr)[0][0])
                    label = "PNEUMONIA" if prob > threshold else "NORMAL"
                    confidence = prob if prob > threshold else 1.0 - prob
                    return {"label": label, "confidence": float(confidence), "probability": prob}
                GLOBAL_TFLITE_INTERPRETER = keras_predict_wrapper
                GLOBAL_TFLITE_INPUT_DETAILS = None
                GLOBAL_TFLITE_OUTPUT_DETAILS = None
                GLOBAL_MODEL_PATH = model_candidate
                print("Keras model loaded as fallback:", model_candidate)
                return
            except Exception as e2:
                print("Failed to load keras fallback:", str(e2))

    print("Startup finished but no usable model loaded.")


_startup_load_model()


# -----------------------
# Routes
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time(), "model_in_use": GLOBAL_MODEL_PATH}


@app.get("/models")
def list_models():
    try:
        all_files = _list_model_files()
        valid = _list_valid_models()
        latest = None
        try:
            latest = _get_latest_model_file()
        except Exception:
            latest = None
        return {
            "models_directory": MODELS_DIR,
            "all_files": all_files,
            "valid_model_files": valid,
            "selected_latest_model": latest,
            "currently_selected": SELECTED_MODEL
        }
    except Exception as e:
        return {"error": str(e), "models_directory": MODELS_DIR}


@app.post("/select-model")
def select_model(model_name: str):
    global SELECTED_MODEL, GLOBAL_TFLITE_INTERPRETER, GLOBAL_TFLITE_INPUT_DETAILS, GLOBAL_TFLITE_OUTPUT_DETAILS, GLOBAL_MODEL_PATH
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    if model_name.endswith(".tflite") or model_name.endswith(".lite"):
        try:
            interpreter, in_det, out_det = load_tflite_interpreter(model_path)
            GLOBAL_TFLITE_INTERPRETER = interpreter
            GLOBAL_TFLITE_INPUT_DETAILS = in_det
            GLOBAL_TFLITE_OUTPUT_DETAILS = out_det
            GLOBAL_MODEL_PATH = model_path
            SELECTED_MODEL = model_name
            return {"status": "success", "selected_model": SELECTED_MODEL, "message": f"Loaded tflite {model_name}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load tflite interpreter: {e}")

    if model_name.endswith(".keras") or model_name.endswith(".h5"):
        base = os.path.splitext(model_name)[0]
        tflite_sibling = os.path.join(MODELS_DIR, f"{base}.tflite")
        try:
            if os.path.exists(tflite_sibling):
                interpreter, in_det, out_det = load_tflite_interpreter(tflite_sibling)
                GLOBAL_TFLITE_INTERPRETER = interpreter
                GLOBAL_TFLITE_INPUT_DETAILS = in_det
                GLOBAL_TFLITE_OUTPUT_DETAILS = out_det
                GLOBAL_MODEL_PATH = tflite_sibling
                SELECTED_MODEL = model_name
                return {"status": "success", "selected_model": SELECTED_MODEL, "message": f"Using existing tflite sibling {os.path.basename(tflite_sibling)}"}
            convert_keras_to_tflite(model_path, tflite_sibling)
            interpreter, in_det, out_det = load_tflite_interpreter(tflite_sibling)
            GLOBAL_TFLITE_INTERPRETER = interpreter
            GLOBAL_TFLITE_INPUT_DETAILS = in_det
            GLOBAL_TFLITE_OUTPUT_DETAILS = out_det
            GLOBAL_MODEL_PATH = tflite_sibling
            SELECTED_MODEL = model_name
            return {"status": "success", "selected_model": SELECTED_MODEL, "message": f"Converted {model_name} to tflite and loaded"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to convert/load model: {e}")

    raise HTTPException(status_code=400, detail="Unsupported model format")


@app.get("/download-model/{model_name}")
def download_model(model_name: str):
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(model_path, filename=model_name, media_type="application/octet-stream")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if GLOBAL_TFLITE_INTERPRETER is None:
        raise HTTPException(status_code=503, detail="No model loaded. Upload or select a model first.")

    tmp_folder = os.path.join(BASE_DIR, "tmp", str(int(time.time() * 1000)))
    os.makedirs(tmp_folder, exist_ok=True)
    tmp_path = os.path.join(tmp_folder, file.filename)

    await file.seek(0)
    with open(tmp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        if callable(GLOBAL_TFLITE_INTERPRETER) and GLOBAL_TFLITE_INPUT_DETAILS is None:
            result = GLOBAL_TFLITE_INTERPRETER(tmp_path)
        else:
            result = predict_with_tflite(GLOBAL_TFLITE_INTERPRETER, GLOBAL_TFLITE_INPUT_DETAILS, GLOBAL_TFLITE_OUTPUT_DETAILS, tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        shutil.rmtree(tmp_folder, ignore_errors=True)

    return JSONResponse(content=result)


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


@app.post("/retrain")
def trigger_retrain(epochs: int = 5, batch_size: int = 32, fine_tune: bool = True):
    if not os.path.exists(UPLOAD_DIR) or not os.listdir(UPLOAD_DIR):
        raise HTTPException(status_code=400, detail="No uploaded data found. Please POST to /upload first.")

    model_path, history = retrain_model(
        new_data_folder=UPLOAD_DIR,
        original_train_folder=TRAIN_DIR,
        batch_size=batch_size,
        epochs=epochs,
        output_directory=MODELS_DIR,
        fine_tune=fine_tune
    )
    try:
        base = os.path.splitext(os.path.basename(model_path))[0]
        tflite_path = os.path.join(MODELS_DIR, f"{base}.tflite")
        if model_path.endswith(".keras") or model_path.endswith(".h5"):
            convert_keras_to_tflite(model_path, tflite_path)
            interpreter, in_det, out_det = load_tflite_interpreter(tflite_path)
            global GLOBAL_TFLITE_INTERPRETER, GLOBAL_TFLITE_INPUT_DETAILS, GLOBAL_TFLITE_OUTPUT_DETAILS, GLOBAL_MODEL_PATH
            GLOBAL_TFLITE_INTERPRETER = interpreter
            GLOBAL_TFLITE_INPUT_DETAILS = in_det
            GLOBAL_TFLITE_OUTPUT_DETAILS = out_det
            GLOBAL_MODEL_PATH = tflite_path
    except Exception as e:
        print("Warning: conversion after retrain failed:", str(e))

    summary = {k: v[-1] if isinstance(v, list) else v for k, v in history.history.items()}
    return {"status": "completed", "model_path": model_path, "summary": summary}


@app.get("/metrics")
def metrics(batch_size: int = 64):
    from sklearn.metrics import precision_score, recall_score, roc_auc_score

    if GLOBAL_TFLITE_INTERPRETER is None:
        raise HTTPException(status_code=503, detail="No model loaded for evaluation.")

    try:
        _, _, test_gen = create_data_generators(
            train_dir=TRAIN_DIR,
            test_dir=TEST_DIR,
            img_size=DEFAULT_INPUT_SIZE,
            batch_size=batch_size
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create test data generator: {str(e)}")

    try:
        y_pred_probs = []
        y_true = []

        if callable(GLOBAL_TFLITE_INTERPRETER) and GLOBAL_TFLITE_INPUT_DETAILS is None:
            import os
            from PIL import Image
            tmp_metric_dir = os.path.join(BASE_DIR, "tmp_metric")
            os.makedirs(tmp_metric_dir, exist_ok=True)
            for i in range(len(test_gen)):
                Xbatch, ybatch = test_gen[i]
                for j in range(Xbatch.shape[0]):
                    tmp_img_path = os.path.join(tmp_metric_dir, f"{time.time_ns()}.jpg")
                    arr_uint8 = ((Xbatch[j] + 1.0) * 127.5).astype("uint8")
                    Image.fromarray(arr_uint8).save(tmp_img_path)
                    res = GLOBAL_TFLITE_INTERPRETER(tmp_img_path)
                    y_pred_probs.append(res["probability"])
                    y_true.append(int(ybatch[j]))
                    try:
                        os.remove(tmp_img_path)
                    except Exception:
                        pass
            shutil.rmtree(tmp_metric_dir, ignore_errors=True)
        else:
            for i in range(len(test_gen)):
                Xbatch, ybatch = test_gen[i]
                for j in range(Xbatch.shape[0]):
                    sample = Xbatch[j:j+1]
                    input_dtype = GLOBAL_TFLITE_INPUT_DETAILS[0]["dtype"]
                    if input_dtype == np.float32:
                        input_data = sample.astype(np.float32)
                    elif input_dtype == np.uint8:
                        input_data = sample.astype(np.uint8)
                    else:
                        input_data = sample.astype(input_dtype)
                    GLOBAL_TFLITE_INTERPRETER.set_tensor(GLOBAL_TFLITE_INPUT_DETAILS[0]["index"], input_data)
                    GLOBAL_TFLITE_INTERPRETER.invoke()
                    out = GLOBAL_TFLITE_INTERPRETER.get_tensor(GLOBAL_TFLITE_OUTPUT_DETAILS[0]["index"])
                    y_pred_probs.append(float(out[0][0]))
                    y_true.append(int(ybatch[j]))

        y_pred_probs = np.array(y_pred_probs)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        y_true = np.array(y_true)
        precision = round(float(precision_score(y_true, y_pred)), 4)
        recall = round(float(recall_score(y_true, y_pred)), 4)
        auc = round(float(roc_auc_score(y_true, y_pred_probs)), 4)
        accuracy = round(float(np.mean(y_pred == y_true)), 4)
        loss = None
        return {"loss": loss, "accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {str(e)}")


# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
