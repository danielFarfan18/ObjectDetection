"""
Benchmark module for object detection models.
This module provides functionality to evaluate and compare different object detectors
using metrics like inference time, precision, recall and F1 score.
"""

import json
import cv2
import numpy as np
import time
import os
from detectors import YOLO4Detector, MobileNetDetector

class DetectorBenchmark:
    """
    A class to benchmark different object detection models.
    
    This class provides methods to evaluate object detection models by measuring
    their inference time, accuracy, and visualization capabilities.
    
    Attributes:
        dataset_path (str): Path to the dataset directory
        annotations_file (str): Path to COCO format annotations file
        images_path (str): Path to the images directory
        detectors (dict): Dictionary of detector instances
        class_mappings (dict): Mapping of detector classes to dataset classes
    """
    
    def __init__(self, dataset_path):
        """
        Initialize the benchmark with dataset path and detectors.
        
        Args:
            dataset_path (str): Path to the dataset directory
        """
        self.dataset_path = dataset_path
        self.annotations_file = os.path.join(dataset_path, '_annotations.coco.json')
        self.images_path = dataset_path
        self.detectors = {
            'YOLO4': YOLO4Detector(),
            'MobileNet': MobileNetDetector()
        }
        # Mapeo de clases para cada detector
        self.class_mappings = {
            'MobileNetDetector': {  # Cambiar 'MobileNet' a 'MobileNetDetector'
                3: 2,  # MobileNet clase 3 -> dataset clase 2
                8: 5   # MobileNet clase 8 -> dataset clase 5
            }
        }
        
        # Cargar anotaciones COCO
        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)

    def get_mapped_class(self, detector_name, class_id):
        """
        Map detector class ID to dataset class ID.
        
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
        Load ground truth annotations for a specific image.
        
        Args:
            image_id (int): ID of the image in COCO format
        
        Returns:
            list: List of dictionaries containing bounding boxes and category IDs
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
        Measure inference time statistics for a detector.
        
        Args:
            detector: Detector instance to evaluate
        
        Returns:
            dict: Dictionary containing average time, FPS and standard deviation
        """
        times = []
        for img_info in self.annotations['images']:
            img_path = os.path.join(self.images_path, img_info['file_name'])
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            start_time = time.time()
            _ = detector.detect(frame)  # Ignoramos el resultado
            inference_time = time.time() - start_time
            times.append(inference_time)
            
        return {
            'avg_time': np.mean(times),
            'fps': 1.0 / np.mean(times),
            'std_dev': np.std(times)
        }

    def measure_accuracy(self, detector):
        """
        Measure accuracy metrics for a detector using COCO annotations.
        
        Calculates precision, recall and F1 score by comparing detector predictions
        with ground truth annotations.
        
        Args:
            detector: Detector instance to evaluate
        
        Returns:
            dict: Dictionary containing precision, recall and F1 score
        """
        # Crear directorio para resultados si no existe
        results_dir = os.path.join('results', detector.__class__.__name__)
        os.makedirs(results_dir, exist_ok=True)
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for img_info in self.annotations['images']:
            img_path = os.path.join(self.images_path, img_info['file_name'])
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            # Obtener detecciones y ground truth
            boxes, scores, class_ids, _ = detector.detect(frame)
            gt_annotations = self.load_image_annotations(img_info['id'])
        
            # Preparar boxes detectados y ground truth
            detected_boxes = []
            gt_boxes = []
            
            # Procesar detecciones
            for box, score, class_id in zip(boxes, scores, class_ids):
                if score > 0.5:
                    # Mapear clase antes de añadir
                    mapped_class = self.get_mapped_class(detector.__class__.__name__, class_id)
                    detected_boxes.append({
                        'bbox': [round(x, 2) for x in box],
                        'class_id': mapped_class,  # Usar clase mapeada
                        'score': score
                    })
            
            # Procesar ground truth
            for ann in gt_annotations:
                gt_boxes.append({
                    'bbox': [round(x, 2) for x in ann['bbox']],
                    'class_id': ann['category_id']
                })
            
            # Calcular matches por imagen
            img_tp = 0
            img_fp = 0
            matched_gt = set()
            
            # Comparar cada detección
            for det in detected_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                # Buscar mejor match
                for idx, gt in enumerate(gt_boxes):
                    if idx in matched_gt:
                        continue
                    
                    if det['class_id'] == gt['class_id']:
                        iou = self.calculate_iou(det['bbox'], gt['bbox'])
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx
                
                # Evaluar match
                if best_iou > 0.7:
                    img_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    img_fp += 1
            
            # Actualizar contadores globales
            total_tp += img_tp
            total_fp += img_fp
            total_fn += len(gt_boxes) - len(matched_gt)

            img_name = os.path.splitext(img_info['file_name'])[0]
            result_path = os.path.join(results_dir, f"{img_name}_result.jpg")
            self.save_visualization(img_path, boxes, scores, class_ids, gt_annotations, result_path)

            
        
        # Calcular métricas finales
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        }

    def visualize_predictions(self, img_path, boxes, scores, class_ids, gt_annotations):
        """
        Display predictions and ground truth on an image.
        
        Args:
            img_path (str): Path to the image
            boxes (list): List of predicted bounding boxes
            scores (list): List of confidence scores
            class_ids (list): List of predicted class IDs
            gt_annotations (list): List of ground truth annotations
        """
        img = cv2.imread(img_path)
        
        # Dibujar predicciones (en rojo)
        for box, score, class_id in zip(boxes, scores, class_ids):
            if score > 0.5:
                x1, y1, w, h = [int(v) for v in box]
                cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0,0,255), 2)
                text = f"Pred: {class_id} ({score:.2f})"
                cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
        # Dibujar ground truth (en verde)
        for ann in gt_annotations:
            x1, y1, w, h = [int(v) for v in ann['bbox']]
            cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0,255,0), 2)
            text = f"GT: {ann['category_id']}"
            cv2.putText(img, text, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        # Mostrar imagen
        cv2.imshow('Predictions vs Ground Truth', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_visualization(self, img_path, boxes, scores, class_ids, gt_annotations, output_path):
        """
        Save visualization of predictions and ground truth to a file.
        
        Args:
            img_path (str): Path to source image
            boxes (list): List of predicted bounding boxes
            scores (list): List of confidence scores
            class_ids (list): List of predicted class IDs
            gt_annotations (list): List of ground truth annotations
            output_path (str): Path to save the visualization
        """
        img = cv2.imread(img_path)
        
        # Dibujar predicciones (en rojo)
        for box, score, class_id in zip(boxes, scores, class_ids):
            if score > 0.5:
                x1, y1, w, h = [int(v) for v in box]
                cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0,0,255), 2)
                text = f"Pred: {class_id} ({score:.2f})"
                cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        
        # Dibujar ground truth (en verde)
        for ann in gt_annotations:
            x1, y1, w, h = [int(v) for v in ann['bbox']]
            cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0,255,0), 2)
            text = f"GT: {ann['category_id']}"
            cv2.putText(img, text, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        # Guardar imagen
        cv2.imwrite(output_path, img)
        
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            box1 (list): First bounding box [x, y, width, height]
            box2 (list): Second bounding box [x, y, width, height]
        
        Returns:
            float: IoU score between 0 and 1
        """
        # Calcular coordenadas de intersección
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        
        # Calcular área de intersección
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calcular áreas
        box1_area = box1[2] * box1[3]  # width * height
        box2_area = box2[2] * box2[3]  # width * height
        
        # Calcular unión
        union = box1_area + box2_area - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou
        
    def run_benchmark(self):
        """
        Run complete benchmark suite on all detectors.
        
        Measures and combines all metrics including inference time,
        precision, recall and F1 score for each detector.
        
        Returns:
            dict: Dictionary containing all benchmark results per detector
        """
        results = {}
        
        for name, detector in self.detectors.items():
            print(f"Testing {name}...")
            
            # Medir tiempo de inferencia
            timing = self.measure_inference_time(detector)
            
            # Medir precisión
            accuracy = self.measure_accuracy(detector)
            
            # Combinar resultados
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
    
    Initializes benchmark with dataset path and runs evaluation
    on all configured detectors, displaying the results.
    """
    try:
        # Usa la ruta completa al dataset
        dataset_path = r"test"
        benchmark = DetectorBenchmark(dataset_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    results = benchmark.run_benchmark()
    
    print("\nBenchmark Results con Dataset VDD:")
    print("-" * 50)
    for model, metrics in results.items():
        print(f"\n{model}:")
        print(f"FPS: {metrics['fps']:.2f}")
        print(f"Average inference time: {metrics['avg_inference_time']*1000:.2f}ms")
        print(f"Precision: {metrics.get('precision', 0):.2%}")
        print(f"Recall: {metrics.get('recall', 0):.2%}")

if __name__ == "__main__":
    main()
