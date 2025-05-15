import cv2
import os

class MobileNetDetector:
    """
    A class for object detection using MobileNet SSD model.
    
    This class implements object detection using the MobileNet SSD model
    trained on the COCO dataset. It provides methods for loading the model
    and performing object detection on frames.
    """

    def __init__(self, model_dir="models"):
        """
        Initialize the MobileNet detector.

        Args:
            model_dir (str): Directory path containing the model files
                           (default: mobilnet_model/ssd_mobilenet_v3_large_coco_2020_01_14/)
        """
        self.model_dir = model_dir
        self.load_model()

    def load_model(self):
        """
        Load the MobileNet model and its configurations.

        Loads the model weights, configuration, and class names.
        Sets up the neural network with appropriate input parameters.

        Raises:
            FileNotFoundError: If required model files are not found in the model directory
        """
        config_path = os.path.join(self.model_dir, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
        weights_path = os.path.join(self.model_dir, "frozen_inference_graph.pb")
        classes_file = os.path.join(self.model_dir, "classes.txt")

        # Check if all required files exist
        if not all(os.path.exists(f) for f in [config_path, weights_path, classes_file]):
            raise FileNotFoundError("MobileNet model files not found")

        # Initialize the detection model
        self.net = cv2.dnn_DetectionModel(weights_path, config_path)
        
        # Configure model input parameters
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        # Load class names from file
        with open(classes_file, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def detect(self, frame, conf_threshold=0.5):
        """
        Detect objects in a given frame.

        Args:
            frame: Input image frame
            conf_threshold (float): Confidence threshold for detections (default: 0.5)

        Returns:
            tuple: (boxes, scores, class_ids, classes) where:
                  - boxes: Bounding boxes coordinates
                  - scores: Confidence scores for each detection
                  - class_ids: Class IDs for each detection
                  - classes: List of class names
        """
        class_ids, scores, boxes = self.net.detect(frame, conf_threshold, 0.4)
        return boxes, scores, class_ids, self.classes
