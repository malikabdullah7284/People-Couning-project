# People Counting Project

A Machine Learning / Computer Vision project that counts people in a **webcam feed or a video file** — counting each person **only once**.

It detects people, gives each person a unique tracking ID, follows them frame to frame using appearance (Re-ID), and at the end reports the total number of unique people and the maximum number of people seen in a single frame.

## Demo

Run it on a video and it draws a green box + ID on every person, with a live counter on screen. The final result is printed in the terminal:

```
========== RESULT ==========
Total unique people (counted once each): 24
Most people in one frame               : 18
============================
```

## Features

- Works with webcam or any video file
- Each person counted only once (tracking IDs, not per-frame counting)
- Stronger detection with YOLOv8s at 1280 image size — catches small / far people
- BoT-SORT tracking with Re-ID and a 90-frame buffer, so a person hidden for a few seconds gets their same ID back
- Confirmation rule: a person is counted only after being seen in at least 5 frames, filtering out false detections
- Optional `--save` to export the counted video

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLOv8 (`yolov8s.pt`)
- BoT-SORT tracker with Re-ID

## Setup

```bash
# 1. Create and activate a virtual environment (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Webcam
python people_counter.py --source 0

# Video file
python people_counter.py --source "test.mp4"

# Video file + save the output video as output_counted.mp4
python people_counter.py --source "test.mp4" --save
```

Press **Q** to quit early. The YOLO model file is downloaded automatically on the first run.

## Limitations

- If a person leaves the frame completely and returns after a long time, the tracker may assign a new ID.
- Very crowded or low-quality videos can reduce accuracy. Using `yolov8m.pt` (edit `MODEL_NAME` in the code) improves detection at the cost of speed.

## What I Learned

- Detection alone is not enough for counting — without tracking, the same person is counted in every frame
- Model size and image size strongly affect how well small / far people are detected
- Training vs test, features, labels, classification, and overfitting — the ML basics behind this project

## Author

**Malik Muhammad Abdullah** — Computer Science student from Pakistan, learning Machine Learning and Computer Vision by building practical projects for retail, CCTV monitoring, and business automation.
