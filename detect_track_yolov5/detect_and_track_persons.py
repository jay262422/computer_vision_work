import torch
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

def detect_and_track_persons(video_path, output_path, conf_thresh,model_name):
    # Load the pre-trained YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)


    tracker = DeepSort(max_age=30, nn_budget=100,max_cosine_distance=0.3)


    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame)

        # Prepare the detections for DeepSORT tracking
        detections = []
        for *xyxy, conf, cls in results.xyxy[0]:
            if int(cls) == 0 and conf.item() > conf_thresh:  # Only consider 'person' class with confidence > conf_thresh
                bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                detections.append((bbox, conf.item(), 'person'))

        # Update tracker with new detections
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw bounding boxes, IDs, and labels on the frame
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # left, top, right, bottom coordinates
            label = f'Person {track_id}'
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(ltrb[0]), int(ltrb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Write the frame with tracking information
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()