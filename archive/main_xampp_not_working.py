"""
# Seed quality assesment app
"""

import os
import time
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.autograd import Variable
from torchvision import models
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pymysql
from PIL import Image

# ------------------------- HELPER FUNCTIONS -------------------------
def load_model():
    # Load pre-trained MobileNetV3 large model
    model = models.mobilenet_v3_large()
    num_classes = 2
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load("mobilenet_weights_20241129_132949.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Preprocessing pipeline for images
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),         # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension
  
# ------------------------- CONFIGURATIONS -------------------------
WATCH_FOLDER = "E:\\04_SeedClassificationProject\\Soyabean_Project\\xampp\\htdocs\\mytest\\uploads"  # Change this to your folder path
MODEL_PATH = "mobilenet_weights_20241129_132949.pth"  # Path to your trained PyTorch model
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'FarmerInformation',
    'port': 3306
}
TABLE_NAME = "SeedTestResults"  # Table for storing results
TEST_ID = 1  # Replace with the appropriate TestID (CHECK LATER FOR FK CONFLICTS)
PARAMETER_NAME = "SeedQuality"  # Example parameter name
UNIT = "Probability"  # Example unit

# ------------------------- LOAD MODEL -------------------------
# Define the model (replace with your specific model if needed)
print("Loading PyTorch model...")
# Load model and predict
model = load_model()
print("Model loaded successfully.")

# ------------------------- DATABASE CONNECTION -------------------------
def insert_result(filename, value, pass_fail, notes):
    """Insert inference result into the database."""
    connection = pymysql.connect(**DB_CONFIG)
    print(f"DB connection succesfull")
    try:
        with connection.cursor() as cursor:
            sql = f"""
                INSERT INTO {TABLE_NAME} (TestID, Parameter, Value, Unit, PassFail, Notes)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (TEST_ID, PARAMETER_NAME, value, UNIT, pass_fail, notes))
        connection.commit()
        print(f"Result saved to database: {filename} -> Value: {value}, Pass/Fail: {pass_fail}")
    except Exception as e:
        print(f"Database error: {e}")
    finally:
        connection.close()

# ------------------------- FILE SYSTEM HANDLER -------------------------
class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        filepath = event.src_path
        if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"New image detected: {filepath}")
            self.process_image(filepath)

    def process_image(self, filepath):
        """Process the image through the model and save the result."""
        retries = 3  # Number of retries
        for attempt in range(retries):
            try:
                time.sleep(1)  # Wait to ensure file is accessible
                img = Image.open(filepath).convert('RGB')  # Open image
                print(f"Image opened")
                img_tensor = preprocess_image(img)
                print(f"preprocessing complete")
                # Inference
                with torch.no_grad():
                    output = model(img_tensor)
                    _, predicted = output.max(1)
                    value = predicted.item()
                print(f"Inference complete, value: {value}")
                # Pass/Fail logic
                pass_fail = "Pass" if value > 0.5 else "Fail"
                print(f"Pass/Fail : {pass_fail}")
                notes = f"Processed image: {os.path.basename(filepath)}"
                print(f"notes : {notes}")

                # Save result to database
                print(f"Inserting result for file: {filepath}")  # Debugging line
                insert_result(os.path.basename(filepath), value, pass_fail, notes)
                print(f"Successfully processed: {filepath}")
                return
            except PermissionError:
                print(f"Retrying access to {filepath} (attempt {attempt + 1})")
            except Exception as e:
                print(f"Error processing image {filepath}: {e}")
                break

# ------------------------- FOLDER WATCHER -------------------------
if __name__ == "__main__":
    print(f"Monitoring folder: {WATCH_FOLDER}")

    # Set up observer
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_FOLDER, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("Monitoring stopped.")
    observer.join()

