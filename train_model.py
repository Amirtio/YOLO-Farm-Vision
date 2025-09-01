# This script is intended to be run in a Google Colab environment
# to train a YOLO model on a custom dataset from Roboflow.

import os
from ultralytics import YOLO
from roboflow import Roboflow
from google.colab import drive, files

# --- 1. Setup Environment ---
# Mount Google Drive to save the model permanently
drive.mount('/content/drive')

# Create a directory in Google Drive to store the final model
model_save_path = '/content/drive/MyDrive/YOLO_Cow_Training'
os.makedirs(model_save_path, exist_ok=True)

# --- 2. Download Dataset from Roboflow ---
# IMPORTANT: Replace "YOUR_API_KEY" with your actual Roboflow private API key.
# It is recommended to use Colab Secrets to store your API key securely.
# For more info: https://medium.com/@igordepaula/google-colab-how-to-use-secrets-963b593a3219
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("amir-kdcbk").project("cow-and-human-htsq1")
version = project.version(2)
dataset = version.download("yolov11") # Make sure the format is correct for YOLO
data_yaml_path = f"{dataset.location}/data.yaml"

print(f"Dataset downloaded to: {dataset.location}")
print(f"Data YAML path: {data_yaml_path}")

# --- 3. Train the YOLO Model ---
# We use 'yolov11s.pt' as the base model. You can choose other versions like 'yolov8n.pt'.
# The model will be trained for 40 epochs.
# The results, including weights, will be saved in the 'runs/detect/' directory.
!yolo task=detect mode=train model=yolov11s.pt data={data_yaml_path} epochs=40 imgsz=640

# --- 4. Save the Best Model to Google Drive ---
# The training process creates a 'train' directory (e.g., train, train2, etc.).
# Find the latest training run folder and copy the 'best.pt' file.
# NOTE: You might need to change 'train' to the correct folder name (e.g., 'train2', 'train3').
latest_run_folder = 'runs/detect/train' # Adjust if necessary
best_model_path = os.path.join(latest_run_folder, 'weights/best.pt')

if os.path.exists(best_model_path):
    # Copy the best model to your mounted Google Drive
    final_destination = os.path.join(model_save_path, 'best.pt')
    !cp {best_model_path} {final_destination}
    print(f"Model successfully copied to: {final_destination}")
    
    # Optionally, download the model directly to your local machine
    print("Preparing to download the model to local machine...")
    files.download(best_model_path)
else:
    print(f"Error: Could not find the trained model at {best_model_path}. Please check the training output and folder names.")
