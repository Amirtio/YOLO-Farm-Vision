# YOLO-Farm-Vision
This project uses a YOLO model to track cows and automatically capture clear, high-quality images at the perfect moment. These images are ideal for health monitoring and herd management.
# YOLO-based Cow Detection and Tracking
> An intelligent system that automatically finds the perfect moment to capture high-quality images for health analysis.
This project utilizes a custom-trained YOLOv11 model to detect and track cows in video streams. It provides two distinct scripts for different camera angles and scenarios: one for detecting cows crossing a vertical line, and another for tracking them within a horizontal zone. The project also includes the necessary code to train the model using a custom dataset from Roboflow.

## Features

- **Custom Model Training**: Includes a script to train a YOLO model on a custom dataset from Roboflow.
- **Object Tracking**: Uses BoT-SORT and ByteTrack for robust object tracking across frames.
- **Scenario 1: Vertical Line Crossing**: Detects cows, tracks their movement, and captures an image of each cow as its right edge crosses a predefined vertical line.
- **Scenario 2: Horizontal Detection Zone**: Identifies cows within a specific horizontal zone, tracks them, and captures an image when they cross an invisible trigger line within that zone.
- **Output Generation**: Saves captured images of individual cows and generates a processed video with bounding boxes, tracking IDs, and event information.

## Project Structure

  ```
  cow-tracking-yolo/
  │
  ├── models/
  │   └── best.pt
  │
  ├── data/
  │   ├── video_angle_1.mp4
  │   └── video_angle_2.mp4
  │
  ├── output/
  │   ├── captures/
  │   └── processed_videos/
  │
  ├── train_model.py
  ├── detect_vertical_line.py
  ├── detect_horizontal_zone.py
  ├── requirements.txt
  └── README.md
  ```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/cow-tracking-yolo.git
    cd cow-tracking-yolo
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the trained model:**
    Place your trained `best.pt` file inside the `models/` directory.

## Usage

### 1. Training the Model (Optional)

The `train_model.py` script is designed to be run in a Google Colab environment to leverage free GPU resources.

1.  Upload the `train_model.py` script to your Google Colab.
2.  Follow the instructions within the script:
    - Mount your Google Drive.
    - Set up your Roboflow API key to download the dataset.
    - The script will train the model and save `best.pt` to your Google Drive.
3.  Download the `best.pt` file and place it in the `models/` directory of this project.

### 2. Running Detection Scripts

Before running, make sure to place your input videos in the `data/` directory.

#### Scenario 1: Vertical Line Crossing

This script is ideal for side-view camera angles where cows move horizontally across the frame.

-   **Configure**: Open `detect_vertical_line.py` and adjust the `video_path`, `line_x`, and other parameters as needed.
-   **Run**:
    ```bash
    python detect_vertical_line.py
    ```
-   **Output**: The processed video will be saved in `output/processed_videos/`, and captured cow images will be in `output/captures/`.

#### Scenario 2: Horizontal Detection Zone

This script is suitable for top-down or angled views where cows move vertically within a defined area.

-   **Configure**: Open `detect_horizontal_zone.py` and adjust `video_path`, `line_top_y`, `line_bottom_y`, etc.
-   **Run**:
    ```bash
    python detect_horizontal_zone.py
    ```
-   **Output**: Outputs will be generated in the same directories as the first scenario.

## Configuration

You can customize the following parameters at the top of each detection script:

-   `model_path`: Path to the trained YOLO model.
-   `video_path`: Path to the input video.
-   `output_video_path`: Path to save the processed video.
-   `capture_folder`: Directory to save captured images.
-   `line_x` / `line_top_y` / `line_bottom_y`: Coordinates for detection lines/zones.
-   `pixels_per_cm`: A calibration factor to estimate real-world distance. This needs to be adjusted based on camera perspective and distance.

---

This project is for demonstration purposes and can be extended for various real-world applications in livestock monitoring and management.
`

---
