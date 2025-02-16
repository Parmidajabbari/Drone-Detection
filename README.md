# Drone Detection

<p align="center">
  <img width=80% src="assets/images/test_image.png">
</p>

## Overview
This project focuses on **drone detection** using deep learning and computer vision techniques. The model is designed to detect drones in aerial footage and classify them accurately in real time.

## Features
- **Real-time drone detection** using a trained deep learning model.
- **Frame-by-frame analysis** for object tracking.
- **High accuracy** with minimal false positives.

## Dataset
The dataset consists of aerial images and videos containing:
- **Labeled drone instances** for supervised learning.
- **Various environmental conditions** to improve robustness.
- **Augmented data** to enhance model generalization.

## Model Architecture
The system integrates:
- **Convolutional Neural Network (CNN)** for feature extraction.
- **YOLOv5-based object detection model** for real-time detection.
- **Pre-trained weights** fine-tuned on drone-specific datasets.
- **Post-processing techniques** such as Non-Maximum Suppression (NMS) to reduce false detections.
- **Bounding box regression** to accurately locate drones in each frame.

### Detection Process
1. **Preprocessing**: Frames are extracted from the video and resized for input to the YOLOv5 model.
2. **Feature Extraction**: CNN layers extract features such as edges and shapes.
3. **Object Detection**: YOLOv5 processes each frame and predicts bounding boxes with confidence scores.
4. **Post-processing**: Non-Maximum Suppression (NMS) removes overlapping predictions.
5. **Tracking (if applicable)**: The detected drone is tracked across frames for movement analysis.

## Results
Final evaluation metrics:
- **Detection Accuracy**: 97.0% (IoU 0.50:0.95)
- **Precision**: 99.9% (IoU 0.50)
- **Recall**: 98.3% (IoU 0.50:0.95)
- **Processing Speed**: 24.9 FPS

These results highlight the model's high accuracy and efficiency in real-time drone detection. The model was trained over multiple epochs using an iterative optimization approach. Each epoch consisted of:
1. **Training Phase**: The model learned to detect drones by minimizing the classification and localization loss.
2. **Learning Rate Adjustment**: The learning rate was updated using a scheduler to optimize convergence.
3. **Evaluation Phase**: After each epoch, the model was validated on a separate dataset to assess its performance.

## Installation
To run the project locally, install the required dependencies:
```sh
pip install opencv-python numpy matplotlib torch torchvision
```

## Usage
Run the Jupyter Notebook for detection and analysis:
```sh
jupyter notebook Drone_Detection.ipynb
```

## Future Improvements
- Improve detection accuracy with **more training data**.
- Optimize performance for **embedded systems**.
- Implement **multi-object tracking** for swarm detection.

## License
This project is licensed under the MIT License.
