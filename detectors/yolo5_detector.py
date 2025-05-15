import cv2
import numpy as np
import os

class YOLO5Detector:
    """
    A class to perform object detection using YOLOv5 model with OpenCV DNN backend.
    
    Attributes:
        model_dir (str): Directory containing the model files
        net: OpenCV DNN network
        classes (list): List of class names for detection
    """

    def __init__(self, model_dir="models"):
        """
        Initialize the YOLO5 detector.

        Args:
            model_dir (str): Path to the directory containing model files
        """
        self.model_dir = model_dir
        self.load_model()

    def load_model(self):
        """
        Load the YOLOv5 model and class names.
        
        Raises:
            FileNotFoundError: If model files are not found in the specified directory
        """
        model_path = os.path.join(self.model_dir, "yolov5n.onnx")
        classes_file = os.path.join(self.model_dir, "classes.txt")

        if not all(os.path.exists(f) for f in [model_path, classes_file]):
            raise FileNotFoundError("YOLO5 model files not found")

        # Initialize the DNN model
        self.net = cv2.dnn.readNet(model_path)

        # Set CUDA as the preferable backend and target
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Load class names from file
        with open(classes_file, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def detect(self, frame, conf_threshold=0.5):
        """
        Perform object detection on a frame.

        Args:
            frame: Input image frame
            conf_threshold (float): Confidence threshold for detections

        Returns:
            tuple: Contains boxes, confidences, class_ids, and class names
        """
        input_size = (640, 640)  # YOLOv5 standard input size
        # Prepare image for inference
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        return self._process_detections(frame, outputs, conf_threshold, input_size)

    def _process_detections(self, frame, outputs, conf_threshold, input_size):
        """
        Process the network output to extract bounding boxes and class information.

        Args:
            frame: Original input frame
            outputs: Network inference outputs
            conf_threshold (float): Confidence threshold for filtering detections
            input_size (tuple): Model input dimensions

        Returns:
            tuple: (boxes, confidences, class_ids, classes)
                - boxes: List of detection bounding boxes
                - confidences: List of confidence scores
                - class_ids: List of class IDs
                - classes: List of class names
        """
        rows = outputs.shape[1]
        frame_height, frame_width = frame.shape[:2]
        boxes = []
        confidences = []
        class_ids = []

        # Process each detection
        for i in range(rows):
            row = outputs[0][i]
            confidence = row[4]
            if confidence >= conf_threshold:
                class_scores = row[5:]
                class_id = np.argmax(class_scores)
                if class_scores[class_id] >= conf_threshold:
                    # Convert normalized coordinates to pixel coordinates
                    cx, cy, w, h = row[0:4]
                    x = int((cx - w/2) * frame_width / input_size[0])
                    y = int((cy - h/2) * frame_height / input_size[1])
                    w = int(w * frame_width / input_size[0])
                    h = int(h * frame_height / input_size[1])
                    boxes.append([x, y, w, h])
                    confidences.append(float(class_scores[class_id]))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids, self.classes
