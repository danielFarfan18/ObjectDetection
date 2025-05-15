import cv2
import numpy as np
import os

class YOLO4Detector:
    """
    A class to implement YOLO4 object detection using OpenCV's DNN module.
    
    Attributes:
        model_dir (str): Directory containing the YOLO4 model files
        net: OpenCV's DNN network object
        classes (list): List of class names that can be detected
    """

    def __init__(self, model_dir="models"):
        """
        Initialize the YOLO4 detector.

        Args:
            model_dir (str): Path to directory containing model files
        """
        self.model_dir = model_dir
        self.load_model()

    def load_model(self):
        """
        Load YOLO4 model and its configuration.
        
        Raises:
            FileNotFoundError: If model files are not found in the specified directory
        """
        model_cfg = os.path.join(self.model_dir, "yolov4-tiny.cfg")
        model_weights = os.path.join(self.model_dir, "yolov4-tiny.weights")
        classes_file = os.path.join(self.model_dir, "classes.txt")

        # Check if all required files exist
        if not all(os.path.exists(f) for f in [model_cfg, model_weights, classes_file]):
            raise FileNotFoundError("YOLO4 model files not found")

        # Initialize the network
        self.net = cv2.dnn.readNet(model_weights, model_cfg)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Load class names from file
        with open(classes_file, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def detect(self, frame, conf_threshold=0.5):
        """
        Detect objects in a given frame.

        Args:
            frame: Input image frame
            conf_threshold (float): Confidence threshold for detections

        Returns:
            tuple: Contains boxes, confidences, class_ids, and class names
        """
        # Prepare image for inference
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Get output layer names and perform forward pass
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        outs = self.net.forward(output_layers)
        return self._process_detections(frame, outs, conf_threshold)

    def _process_detections(self, frame, outs, conf_threshold):
        """
        Process network outputs to get bounding boxes and class information.

        Args:
            frame: Input image frame
            outs: Network output layers
            conf_threshold (float): Confidence threshold for detections

        Returns:
            tuple: (boxes, confidences, class_ids, classes) containing detection information
        """
        height, width = frame.shape[:2]
        boxes = []
        confidences = []
        class_ids = []

        # Process each detection
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > conf_threshold:
                    # Convert center coordinates to corner coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    # Store detection results
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids, self.classes
