# weapon_detection

This project implements a real-time automated weapon detection system for CCTV surveillance using the YOLOv8 deep learning architecture. The system is trained on the Roboflow Universe `weapon-detection-jqd3x` dataset and optimized to detect and localize firearms and bladed weapons in live or recorded video with minimal latency.

## Project scaffold

- `requirements.txt` — Python package requirements
- `.gitignore` — standard Python and build ignores
- `dataset_prep.py` — dataset preparation and YOLOv8 data YAML generation
- `train.py` — first YOLOv8 training script for weapon detection
- `weapon_detection_proposal (1).pdf` — project proposal document

## Setup

Recommended working environment:

1. Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Dataset preparation

Prepare a dataset in YOLO format with `images/` and `labels/` directories, then run:

```powershell
python dataset_prep.py --source path\to\source --output data\weapon_dataset --grayscale --yaml data\weapon_dataset\data.yaml
```

This script will:

- copy and convert images to grayscale-compatible BGR images
- preserve label files
- split the dataset into `train`, `val`, and `test`
- generate a `data.yaml` file for YOLOv8 training

## Training

Run the first YOLOv8 training session with:

```powershell
python train.py --data data\weapon_dataset\data.yaml --model yolov8n.pt --epochs 30 --batch 16 --imgsz 640
```

Training results are saved under `runs/train/weapon_detection` by default.

## Next steps

1. Acquire and export the Roboflow dataset in YOLO format.
2. Run `dataset_prep.py` to build the training splits.
3. Train the baseline YOLOv8 model with `train.py`.
4. Add inference, optimization, and edge deployment scripts next.
