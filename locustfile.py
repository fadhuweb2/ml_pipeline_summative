from locust import HttpUser, task, between
import os

# Path to a sample image that will be sent repeatedly
SAMPLE_IMAGE_PATH = r"C:\Users\fadhl\OneDrive\Desktop\ML_pipeline_summative\data\test\PNEUMONIA\BACTERIA-40699-0002.jpeg"  

class PredictionUser(HttpUser):
    wait_time = between(0.5, 2.0)  # simulate realistic user behavior

    @task
    def predict_image(self):
        # Make sure the sample image exists
        if not os.path.exists(SAMPLE_IMAGE_PATH):
            print(f"Sample image not found at {SAMPLE_IMAGE_PATH}")
            return

        # Open the image in binary mode
        with open(SAMPLE_IMAGE_PATH, "rb") as f:
            files = {"file": (os.path.basename(SAMPLE_IMAGE_PATH), f, "image/jpeg")}
            with self.client.post("/predict", files=files, catch_response=True) as response:
                if response.status_code != 200:
                    response.failure(f"Unexpected status code: {response.status_code}")
                else:
                    try:
                        _ = response.json()
                    except Exception:
                        response.failure("Invalid JSON response")
