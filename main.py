"""
Object Detection Video Processing Script

This script processes video files using different object detection models (YOLO4, YOLO5, MobileNet).
It draws bounding boxes and labels around detected objects and saves the processed video.
"""

import cv2
import argparse
from detectors import YOLO4Detector, YOLO5Detector, MobileNetDetector

def process_video(detector, video_path, output_path):
    """
    Process video using the specified object detector.

    Args:
        detector: Object detector instance (YOLO4, YOLO5, or MobileNet)
        video_path (str): Path to input video file
        output_path (str): Path to save processed video file

    Raises:
        Exception: If video file cannot be opened
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file")

    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(output_path,
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps,
                         (frame_width, frame_height))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect objects in the frame
            boxes, confidences, class_ids, classes = detector.detect(frame)

            # Apply Non-Maximum Suppression to remove overlapping boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            # Draw boxes that survived NMS
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    conf = confidences[i]
                    class_id = class_ids[i]
                    
                    # Only draw if confidence exceeds threshold
                    if conf > 0.5:
                        label = f"{classes[class_id]}: {conf:.2f}"
                        
                        # Draw bbox with shadow for better visibility
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 4)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Add black background for text
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), (0, 0, 0), -1)
                        cv2.putText(frame, label, (x, y - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("Detection", frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def main():
    """
    Main function that handles command line arguments and initializes the detection process.
    
    Supported detectors:
    - YOLO4
    - YOLO5
    - MobileNet
    """
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Object detection with different models')
    parser.add_argument('--detector', type=str, default='yolo4',
                      choices=['yolo4', 'yolo5', 'mobilenet'],
                      help='Type of detector to use (yolo4, yolo5, mobilenet)')
    parser.add_argument('--video', type=str, default='av_revolucion.mp4',
                      help='Path to video file')
    parser.add_argument('--output', type=str, default='output_detection.mp4',
                      help='Path to output file')
    
    args = parser.parse_args()

    # Select detector based on argument
    detectors = {
        'yolo4': YOLO4Detector,
        'yolo5': YOLO5Detector,
        'mobilenet': MobileNetDetector
    }
    
    detector_class = detectors.get(args.detector)
    if not detector_class:
        raise ValueError(f"Invalid detector: {args.detector}")
    
    detector = detector_class()
    process_video(detector, args.video, args.output)

if __name__ == "__main__":
    main()
