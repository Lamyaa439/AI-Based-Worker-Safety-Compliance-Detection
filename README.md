# AI-Based-Worker-Safety-Compliance-Detection

An AI-based computer vision system that monitors worker safety compliance by detecting whether workers are wearing required PPE (hardhats and safety vests) from images and video.

## Project Structure

```
AI-Based-Worker-Safety-Compliance-Detection/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 01b_data_augmentation.ipynb
│   ├── 02_model_yolov8n.ipynb
│   ├── 02b_model_yolov9t.ipynb
│   ├── 02c_model_yolo26n.ipynb
│   └── 03_evaluation_comparison.ipynb
├── results/
│   ├── overall_comparison.csv
│   └── per_class_comparison.csv
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   ├── logger.py
│   │   └── report.py
│   └── ui/
│       ├── __init__.py
│       └── interface.py
├── data/
│   └── README.md
├── models/
│   └── README.md
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python -m app.main
```

## Team

Team 7 — Thakaly
