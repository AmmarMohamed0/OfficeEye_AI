# OfficeEye_AI

This project implements object tracking and calculates the time objects spend in user-defined working areas. It utilizes computer vision techniques and the YOLO object detection model to identify and track objects in a video feed. It provides insights into the time spent in specific regions of interest.

## Features

- **Object Detection:** Leverages the YOLO model to detect objects in frames.
- **Time Tracking:** Monitors and logs the time objects spend within defined areas.
- **Polygonal Area Definition:** Users can specify areas of interest using polygons.
- **Visualization:** Displays object detections, working areas, and time metrics on the video feed.

## Requirements

- Python 3.8+
- Dependencies (install using `requirements.txt`)

## Setup and Usage

### 1. Clone the Repository
```bash
git clone https://github.com/AmmarMohamed0/OfficeEye_AI.git
cd OfficeEye_AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python main.py
```

### 4. Define Working Areas
- Open the `main.py` file.
- Modify the `working_areas` variable to define the polygons for your areas of interest.
  Example:
  ```python
  working_areas = [
      [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],  # Polygon 1
      [(x5, y5), (x6, y6), (x7, y7), (x8, y8)]   # Polygon 2
  ]
  ```

### 5. Project Structure
```
.
├── main.py            # Main script for object detection and tracking
├── utilis.py          # Utility functions for area drawing and time tracking
├── requirements.txt   # List of dependencies
├── README.md          # Project documentation
└── ...                # Other files and directories
```

## Example Output
- The application displays a video feed with:
  - Detected objects highlighted.
  - Working areas outlined.
  - Time metrics shown for each area.
