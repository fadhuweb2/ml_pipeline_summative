# Chest X-ray Pneumonia Detection API

## Project Description

This project implements an end-to-end machine learning pipeline for detecting pneumonia from chest X-ray images. It includes:

- **Data preprocessing:** Image augmentation and normalization  
- **Model creation:** Convolutional Neural Network (CNN) with optional fine-tuning  
- **Prediction:** Single image predictions via FastAPI  
- **Retraining:** Upload new data and trigger retraining    
- **Performance testing:** Flood request simulation using Locust  

The goal is to provide a user-friendly interface for both prediction and retraining while demonstrating robust performance under load.

## Video Demo

YouTube link: https://www.youtube.com/watch?v=2MDvTWdV-7M
The video demonstrates:

- Uploading an image for prediction  
- Uploading a batch of images for retraining  
- Triggering the retraining process  
- Viewing model metrics 
- Viewing data visualizations 

## Project Setup

### Prerequisites

- Python 3.10+

### Installation

Clone the repository:

```bash
git clone https://github.com/fadhuweb2/ml_pipeline_summative.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn src.api:app --reload --host localhost --port 8000

```

Start Streamlit frontend:

```bash
streamlit run ui/app.py
```

## Using the API

### Health Check

```http
GET /health
```

### Predict

Upload a single X-ray image for prediction:

```http
POST /predict
```

### Upload Data for Retraining

Upload multiple images or a ZIP file:

```http
POST /upload
```

### Trigger Retraining

```http
POST /retrain?epochs=5&batch_size=32&fine_tune=True
```

### Check Retraining Progress

```http
GET /retrain-progress
```

### Model Metrics

```http
GET /metrics
```

## Flood Request Simulation Results

Flood request simulation was performed using Locust on the `/predict` endpoint with multiple concurrent users. The key metrics from the simulation are:

| Metric                  | Value |
|-------------------------|-------|
| Total Requests          | 6     |
| Successful Requests     | 6     |
| Failed Requests         | 0     |
| Failure Rate            | 0%    |
| Average Latency         | 57978 ms |
| Median Latency          | 59000 ms |
| Max Latency             | 75595 ms |
| Requests per Second     | 0.078 |

### Observations

- The API handled concurrent requests efficiently with **no failures**  
- Average latency stayed below 60 seconds, indicating reasonable responsiveness under low to moderate load  
- Peak latency occurred at **single-request bursts**, suggesting the API is CPU or I/O-bound for large image processing  

For detailed request logs and percentiles, refer to:

- `locust_results_stats.csv`  
- `locust_results_distribution.csv`  

## Percentile Response Times (ms)

| Percentile | /predict |
|------------|----------|
| 50%        | 63000    |
| 66%        | 63000    |
| 75%        | 67000    |
| 80%        | 67000    |
| 90%        | 76000    |
| 95%        | 76000    |
| 98%        | 76000    |
| 99%        | 76000    |
| 99.9%      | 76000    |
| 99.99%     | 76000    |
| 100%       | 76000    |


