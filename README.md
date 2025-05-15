# Multi-Model Object Detection

This project implements real-time object detection using multiple deep learning models (YOLO4, YOLO5, and MobileNet) with OpenCV.

## Author

**Daniel Farfán**
- GitHub: [@danielFarfan18](https://github.com/danielFarfan18)
## Features

- Support for multiple object detection models:
  - YOLOv4-tiny
  - YOLOv5n
  - MobileNet SSD v3
- Real-time video processing
- Non-Maximum Suppression for better detections
- Command line interface for model selection
- Automatic model download and setup
- Visual output with bounding boxes and confidence scores

## Requirements

```bash
opencv-contrib-python==4.6.0.66
opencv-python==4.6.0.66
numpy==1.26.4
requests  # for downloading model files
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/danielFarfan18/ObjectDetection
cd ObjectDetection
```

2. Create a virtual environment (optional but recommended):
   <br>
> Using venv
```bash
  python -m venv ObjectDetector
  source ObjectDetector/bin/activate 
```

> Using conda
```bash
 conda create -n ObjectDetector python=3.10
 conda activate ObjectDetector
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── detectors/
│   ├── __init__.py
│   ├── yolo4_detector.py
│   ├── yolo5_detector.py
│   └── mobilenet_detector.py
├── models/
│   └── # Model files will be downloaded here
├── main.py
└── requirements.txt
```

## Usage

The script supports multiple video sources and detectors using command line arguments:

### Video Sources

```bash
# Process video file
python main.py --source path/to/video.mp4

# Use webcam (index 0)
python main.py --source 0

# Use RealSense camera
python main.py --source realsense
```

### Detectors

```bash
# Using YOLOv4 (default)
python main.py --detector yolo4 --source video.mp4

# Using YOLOv5
python main.py --detector yolo5 --source video.mp4

# Using MobileNet
python main.py --detector mobilenet --source video.mp4
```

### Output Options

```bash
# Save to specific output file
python main.py --detector yolo4 --source input.mp4 --output result.mp4
```

### Full Examples

```bash
# Process video file with YOLOv4
python main.py --detector yolo4 --source video.mp4 --output result.mp4

# Use webcam with MobileNet
python main.py --detector mobilenet --source 0 --output webcam_detection.mp4

# Use RealSense camera with YOLOv5 (requires RealSense SDK)
python main.py --detector yolo5 --source realsense --output depth_detection.mp4
```

Note: When using RealSense camera, make sure you have the RealSense SDK installed:

```bash
sudo apt-get install python3-pyrealsense2
```

The script will display the detection results in real-time and save the processed video to the specified output file.

### Command Line Arguments

- `--detector`: Choose the detection model (yolo4/yolo5/mobilenet)
- `--source`: Input source, could be video,webcam or realsense camera
- `--output`: Path for processed video output

## Model Files

The required model files will be automatically downloaded on first run. They will be stored in:
- YOLOv4: `models/yolov4-tiny.{cfg,weights}`
- YOLOv5: `models/yolov5n.onnx`
- MobileNet: `models/ssd_mobilenet_v3_large_coco_2020_01_14/{config,weights}`

## Controls

- Press 'q' to quit the video processing
- The processed video will be automatically saved to the specified output path

## Notes

- Make sure you have enough disk space for the model files
- CUDA acceleration is enabled by default if available (for YOLOv5)

## Citation

If you use this code in your research, please cite it as:

```bibtex
@software{daniel2025multimodel,
  author = {Farfán, Daniel},
  title = {Multi-Model Object Detection},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/danielFarfan18/ObjectDetection}
}
```
