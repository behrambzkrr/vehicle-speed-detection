# Vehicle Speed Detection 🚗💨

This project detects and estimates vehicle speeds in a video using **YOLOv8**, **ByteTrack**, and **perspective transform**.  
It provides real-time visualization of detected vehicles, their trajectories, and calculated speeds.

---

## ✨ Features
- 🚘 Vehicle detection with YOLOv8  
- 📍 Object tracking using ByteTrack  
- 📏 Speed estimation with perspective transformation  
- 📊 Real-time visualization with bounding boxes, labels, and traces  

---

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/behrambzkrr/vehicle-speed-detection.git
cd vehicle-speed-detection
pip install -r requirements.txt

▶️ Usage

Run the main script:

python vehicle_speed_detection.py


👉 Make sure to update the video_path in the script (vehicle_speed_detection.py) to your own video file path.

.

📦 Requirements

Python 3.8+

Ultralytics YOLOv8

Supervision

OpenCV

NumPy

You can install them manually via:

pip install ultralytics supervision opencv-python numpy

📺 Example Output


Detected vehicles with their calculated speeds (km/h) will be displayed directly on the video.
