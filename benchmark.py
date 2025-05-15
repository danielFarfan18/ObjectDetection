import json
import cv2
import numpy as np
import time
import os
from detectors import YOLO4Detector, YOLO5Detector, MobileNetDetector

class DetectorBenchmark:
    """
    A benchmark class for evaluating object detection models.
    
    This class provides functionality to measure and compare the performance
    of different object detection models using metrics such as inference time,
    precision, recall, and F1 score.

    Attributes:
        dataset_path (str): Path to the dataset directory
        annotations_file (str): Path to COCO format annotations file
        images_path (str): Path to images directory
        detectors (dict): Dictionary of detector instances
        class_mappings (dict): Mapping of detector classes to dataset classes
    """

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.annotations_file = os.path.join(dataset_path, '_annotations.coco.json')
        self.images_path = dataset_path
        self.detectors = {
            'YOLO4': YOLO4Detector(),
            'YOLO5': YOLO5Detector(),
            'MobileNet': MobileNetDetector()
        }
        # Class mapping for each detector
        self.class_mappings = {
            'MobileNetDetector': {
                3: 2,  # MobileNet class 3 -> dataset class 2
                8: 5   # MobileNet class 8 -> dataset class 5
            }
        }
        
        # Load COCO annotations
        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)

    def get_mapped_class(self, detector_name, class_id):
        """
        Maps detector class ID to dataset class ID.

        Args:
            detector_name (str): Name of the detector
            class_id (int): Original class ID from detector

        Returns:
            int: Mapped class ID for the dataset
        """
        if detector_name in self.class_mappings:
            return self.class_mappings[detector_name].get(class_id, class_id)
        return class_id

    def load_image_annotations(self, image_id):
        """
        Retrieves annotations for a specific image.

        Args:
            image_id (int): ID of the image in the dataset

        Returns:
            list: List of dictionaries containing bbox and category_id
        """
        annotations = []
        for ann in self.annotations['annotations']:
            if ann['image_id'] == image_id:
                annotations.append({
                    'bbox': ann['bbox'],  # [x, y, width, height]
                    'category_id': ann['category_id']
                })
        return annotations

    def measure_inference_time(self, detector):
        """
        Measures the inference time statistics for a detector.

        Args:
            detector: Object detector instance

        Returns:
            dict: Dictionary containing average time, FPS, and standard deviation
        """
        times = []
        for img_info in self.annotations['images']:
            img_path = os.path.join(self.images_path, img_info['file_name'])
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            start_time = time.time()
            _ = detector.detect(frame)  # Ignore the result
            inference_time = time.time() - start_time
            times.append(inference_time)
            
        return {
            'avg_time': np.mean(times),
            'fps': 1.0 / np.mean(times),
            'std_dev': np.std(times)
        }

    def measure_accuracy(self, detector):
        """
        Measures accuracy metrics using COCO annotations.

        Calculates precision, recall, and F1 score by comparing
        detector predictions with ground truth annotations.

        Args:
            detector: Object detector instance

        Returns:
            dict: Dictionary containing precision, recall, and F1 score
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for img_info in self.annotations['images']:
            img_path = os.path.join(self.images_path, img_info['file_name'])
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            # Get detections and ground truth
            boxes, scores, class_ids, _ = detector.detect(frame)
            gt_annotations = self.load_image_annotations(img_info['id'])
    
            # Prepare detected and ground truth boxes
            detected_boxes = []
            gt_boxes = []
            
            # Process detections
            for box, score, class_id in zip(boxes, scores, class_ids):
                if score > 0.5:
                    # Map class before adding
                    mapped_class = self.get_mapped_class(detector.__class__.__name__, class_id)
                    detected_boxes.append({
                        'bbox': [round(x, 2) for x in box],
                        'class_id': mapped_class,  # Use mapped class
                        'score': score
                    })
            
            # Process ground truth
            for ann in gt_annotations:
                gt_boxes.append({
                    'bbox': [round(x, 2) for x in ann['bbox']],
                    'class_id': ann['category_id']
                })
            
            # Calculate matches per image
            img_tp = 0
            img_fp = 0
            matched_gt = set()
            
            # Compare each detection
            for det in detected_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                # Find best match
                for idx, gt in enumerate(gt_boxes):
                    if idx in matched_gt:
                        continue
                    
                    if det['class_id'] == gt['class_id']:
                        iou = self.calculate_iou(det['bbox'], gt['bbox'])
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx
                
                # Evaluate match
                if best_iou > 0.5:
                    img_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    img_fp += 1
            
            # Update global counters
            total_tp += img_tp
            total_fp += img_fp
            total_fn += len(gt_boxes) - len(matched_gt)
            
        
        # Calculate final metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        }

    def calculate_iou(self, box1, box2):
        """
        Calculates Intersection over Union between two bounding boxes.

        Args:
            box1 (list): First bounding box [x, y, width, height]
            box2 (list): Second bounding box [x, y, width, height]

        Returns:
            float: IoU score between 0 and 1
        """
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        
        # Calculate intersection area
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate areas
        box1_area = box1[2] * box1[3]  # width * height
        box2_area = box2[2] * box2[3]  # width * height
        
        # Calculate union
        union = box1_area + box2_area - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou
        
    def run_benchmark(self):
        """
        Executes complete benchmark suite for all detectors.

        Measures and combines inference time and accuracy metrics
        for each configured detector.

        Returns:
            dict: Dictionary containing all benchmark results per detector
        """
        results = {}
        
        for name, detector in self.detectors.items():
            print(f"Testing {name}...")
            
            # Measure inference time
            timing = self.measure_inference_time(detector)
            
            # Measure accuracy
            accuracy = self.measure_accuracy(detector)
            
            # Combine results
            results[name] = {
                'fps': timing['fps'],
                'avg_inference_time': timing['avg_time'],
                'std_dev': timing['std_dev'],
                'precision': accuracy['precision'],
                'recall': accuracy['recall'],
                'f1': accuracy['f1']
            }
            
        return results

def main():
    """
    Main entry point for running the benchmark.
    
    Initializes the benchmark with dataset path and runs the evaluation
    for all configured detectors, displaying the results.
    """
    try:
        # Use complete path to dataset
        dataset_path = r"test"
        benchmark = DetectorBenchmark(dataset_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    results = benchmark.run_benchmark()
    
    print("\nBenchmark Results with VDD Dataset:")
    print("-" * 50)
    for model, metrics in results.items():
        print(f"\n{model}:")
        print(f"FPS: {metrics['fps']:.2f}")
        print(f"Average inference time: {metrics['avg_inference_time']*1000:.2f}ms")
        print(f"Precision: {metrics.get('precision', 0):.2%}")
        print(f"Recall: {metrics.get('recall', 0):.2%}")

if __name__ == "__main__":
    main()