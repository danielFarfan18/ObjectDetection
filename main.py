"""
Object Detection Video Processing Script
This script processes video from files, webcams or RealSense cameras using different object detection models.
"""

import cv2
import argparse
import numpy as np
from detectors import YOLO4Detector, YOLO5Detector, MobileNetDetector

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

def setup_realsense():
    """Initialize and configure RealSense camera pipeline."""
    if not REALSENSE_AVAILABLE:
        raise ImportError("pyrealsense2 is not installed")
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align

def process_video(detector, video_source, output_path, use_realsense=False):
    """
    Process video using the specified object detector.

    Args:
        detector: Object detector instance
        video_source: Path to video file, camera index, or None for RealSense
        output_path: Path to save processed video file
        use_realsense: Boolean to indicate if RealSense camera should be used
    """
    if use_realsense:
        pipeline, align = setup_realsense()
        # Get first frame to setup video writer
        frames = pipeline.wait_for_frames()
        color_frame = align.process(frames).get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        frame_height, frame_width = frame.shape[:2]
    else:
        # Regular video/webcam setup
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise Exception("Could not open video source")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer
    out = cv2.VideoWriter(output_path,
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         30 if use_realsense else fps,
                         (frame_width, frame_height))

    try:
        while True:
            if use_realsense:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                frame = np.asanyarray(color_frame.get_data())
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            # ... existing detection and drawing code ...
            boxes, confidences, class_ids, classes = detector.detect(frame)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    conf = confidences[i]
                    class_id = class_ids[i]
                    
                    if conf > 0.5:
                        label = f"{classes[class_id]}: {conf:.2f}"
                        if use_realsense:
                            depth = depth_frame.get_distance(x + w//2, y + h//2)
                            label += f" {depth:.2f}m"
                        
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 4)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), (0, 0, 0), -1)
                        cv2.putText(frame, label, (x, y - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("Detection", frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if use_realsense:
            pipeline.stop()
        else:
            cap.release()
        out.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Object detection with different models')
    parser.add_argument('--detector', type=str, default='yolo4',
                      choices=['yolo4', 'yolo5', 'mobilenet'],
                      help='Type of detector to use')
    parser.add_argument('--source', type=str, default='av_revolucion.mp4',
                      help='Path to video file, camera index, or "realsense"')
    parser.add_argument('--output', type=str, default='output_detection.mp4',
                      help='Path to output file')
    
    args = parser.parse_args()

    # ... existing detector setup code ...
    detectors = {
        'yolo4': YOLO4Detector,
        'yolo5': YOLO5Detector,
        'mobilenet': MobileNetDetector
    }
    
    detector_class = detectors.get(args.detector)
    if not detector_class:
        raise ValueError(f"Invalid detector: {args.detector}")
    
    detector = detector_class()
    
    # Check if RealSense is requested
    use_realsense = args.source.lower() == 'realsense'
    video_source = None if use_realsense else args.source
    
    process_video(detector, video_source, args.output, use_realsense)

if __name__ == "__main__":
    main()
