# Inventory Management System

## Overview
An intelligent inventory management system that leverages computer vision and deep learning for automated product tracking and management. The system uses YOLOv5 object detection to identify and track inventory items like bottles, boxes, and cups in real-time. By combining modern AI capabilities with traditional inventory management, this solution automates stock counting, monitors inventory levels, and helps prevent stockouts and overstock situations.

Key Features:
- Real-time object detection and tracking
- Automated inventory counting
- Product classification (bottles, boxes, cups)
- Database integration for inventory records
- Confidence-based detection (threshold: 0.5)

## Workflow
1. **Image Acquisition**
   - System receives input through cameras/images
   - Supports multiple image formats and real-time video feeds

2. **Object Detection**
   - YOLOv5 model processes the input
   - Detects and classifies inventory items
   - Applies confidence threshold (0.5) to ensure accuracy
   - Maps detected objects to inventory categories:
     - Class 39: Bottles
     - Class 73: Boxes
     - Class 41: Cups

3. **Data Processing**
   - Converts detection results to inventory data
   - Counts items by category
   - Validates detection confidence
   - Processes multiple items simultaneously

4. **Database Management**
   - Stores inventory counts in SQLite database
   - Updates stock levels in real-time
   - Maintains historical inventory data
   - Tracks inventory changes over time

5. **Order Management**
   - Monitors stock levels
   - Triggers alerts for low inventory
   - Supports automated order processing
   - Maintains order history

## Prerequisites
- Python >= 3.8.0
- PyTorch >= 1.8
- CUDA-enabled GPU (recommended for optimal performance)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Inventory_management
```

2. Install dependencies (if SSL issues occur, use one of these commands):
```bash
# Method 1: Standard installation
pip install -r requirements.txt

# Method 2: If SSL issues occur
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

## Required Dependencies
- YOLOv5
- PyTorch
- TorchVision
- OpenCV-Python
- NumPy
- Pandas
- Ultralytics
- SQLite3

## Features
- Object Detection using YOLOv5
- Inventory Tracking
- Database Integration with SQLite
- Real-time Processing

## Project Structure
```
Inventory_management/
├── models/
│   └── yolov5/         # YOLOv5 model files
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Troubleshooting
If you encounter SSL certificate errors during installation, try:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

## Contributing
For contributions, please create a pull request or open an issue for discussion.
