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
