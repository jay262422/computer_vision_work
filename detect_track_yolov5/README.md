# Person Detection and Tracking in Therapy Sessions

## Overview
This project focuses on detecting and tracking individuals (specifically children and caretakers) in video sessions. The goal is to assign unique IDs to each person, track their movements, and reassign the same ID if they re-enter the frame or become occluded and later reappear.

The final solution uses **YOLOv5** for person detection and **DeepSORT** for tracking, ensuring that multiple children and caretakers are accurately identified and tracked throughout the video.

## Model and Tracking Approach
### Detection: YOLOv5
- YOLOv5, a state-of-the-art object detection model, is employed for detecting people in the video frames. 
- The model is lightweight and fast, making it suitable for real-time applications.
- Several variations of YOLOv5 (e.g., `yolov5s`, `yolov5m`, `yolov5l`) were tested to optimize accuracy and performance.

### Tracking: DeepSORT
- DeepSORT is a robust multi-object tracking algorithm that assigns unique IDs to detected persons and tracks them across video frames.
- **DeepSORT parameters** such as `max_cosine_distance` and `nn_budget` were fine-tuned to improve re-identification, particularly for situations involving occlusions or persons leaving and re-entering the frame.
  
### Enhancements for Better Performance:
- **Confidence Thresholds**: Different confidence thresholds were tested (ranging from 0.25 to 0.75) to minimize false positives.
- **ID Handling**: The tracking system handles ID reassignment after occlusions or person re-entries to maintain consistent ID assignment.

## How to Run
### 1. Environment Setup
1. Clone the repository or extract the project files.
2. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   
## 2. Running the Code

Once the environment is set up, you can run the script on a test video by specifying the video path, output path, model, and detection confidence threshold. This allows you to perform person detection and tracking on your input video.

Use the following command in your terminal to run the code:

```bash
python detect_and_track_persons.py --video_path test_video.mp4 --output_path output_video.mp4 --model yolov5l --threshold 0.5


### 3. Parameters:

- **`--video_path`**: 
  - Description: The path to the input video file that you want to analyze.
  - Example: `--video_path ./data/test_video.mp4`
  - **Required**: Yes

- **`--output_path`**: 
  - Description: The path where the output video (with predictions and tracking overlay) will be saved.
  - Example: `--output_path ./results/output_video.mp4`
  - **Required**: Yes

- **`--model`**: 
  - Description: The YOLOv5 model variant to use for detection. You can select different models based on your desired trade-off between speed and accuracy:
    - `yolov5s`: Small and fast, but less accurate.
    - `yolov5m`: Medium-sized, balanced between speed and accuracy.
    - `yolov5l`: Large, more accurate but slower.
  - Default: `yolov5l`
  - Example: `--model yolov5s`
  - **Required**: No (default is `yolov5l`)

- **`--threshold`**: 
  - Description: The confidence threshold for person detection. Only detections with confidence scores higher than this value will be considered valid.
  - Range: The value should be between 0 and 1.
    - Lower values (e.g., 0.25) allow more detections but can introduce more false positives.
    - Higher values (e.g., 0.75) reduce the number of false positives but might miss some detections.
  - Default: `0.5`
  - Example: `--threshold 0.5`
  - **Required**: No (default is `0.5`)

### Example Command

To detect and track people in a video, you can run the following command in your terminal:

```bash
python detect_and_track_persons.py --video_path ./data/test_video.mp4 --output_path ./results/output_video.mp4 --model yolov5m --threshold 0.75
