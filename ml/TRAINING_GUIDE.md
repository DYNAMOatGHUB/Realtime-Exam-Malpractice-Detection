# ExamGuard AI — Complete Training Pipeline Guide

## Overview

This guide walks you through the full pipeline from raw images → trained YOLO model → live detection in your system.

```
Your 1000+ raw images
        ↓
  [Step 1] Upload to Roboflow
        ↓
  [Step 2] Annotate (draw bounding boxes + label each anomaly)
        ↓
  [Step 3] Export dataset in YOLOv8 format
        ↓
  [Step 4] Train in Google Colab (free GPU)
        ↓
  [Step 5] Download best.pt weights
        ↓
  [Step 6] Place in ml/weights/ → rebuild Docker
        ↓
  Live anomaly detection in ExamGuard ✅
```

---

## Step 1 — Create a Roboflow Account and Project

1. Go to **[https://roboflow.com](https://roboflow.com)** → Sign up free
2. Click **"Create New Project"**
3. Fill in:
   - **Project Name**: `exam-anomaly-detection`
   - **Annotation Group**: `exam-anomalies`
   - **Project Type**: **Object Detection**
4. Click **Create Project**

---

## Step 2 — Upload Your Images

1. In your project, click **"Upload Data"**
2. Drag and drop all your anomaly images (JPG/PNG)
3. Roboflow supports batch upload of 1000+ images
4. Wait for upload to complete

---

## Step 3 — Annotate Your Images

This is the most important step. For each image, you draw a bounding box around the person performing the anomaly and assign a class label.

### Annotation Classes to Create

Create these exact class names in your project (click **"Add Class"** for each):

| Class Name | What to Box | Example |
|---|---|---|
| `COPYING` | Student whose eyes/head is directed at neighbor's paper | Head turned sideways toward another student |
| `MOBILE_PHONE` | Visible phone in hand, on desk, or near face | Rectangle shape held in hand |
| `PASSING_NOTES` | Paper being moved between students | Paper in motion between desks |
| `UNAUTH_MATERIAL` | Book/notes/cheat sheet visible on desk | Papers with writing visible |
| `SUSPICIOUS` | Student looking around, checking surroundings | Head repeatedly turning |
| `TALKING` | Student visibly talking to neighbor | Open mouth facing another student |
| `LEAVING_SEAT` | Student partially or fully out of seat | Body raised from chair |
| `OTHER` | Any other suspicious behavior not covered above | |

### How to Draw Bounding Boxes in Roboflow

1. Click an image to open the annotation editor
2. Select **"Bounding Box"** tool (shortcut: `B`)
3. Draw a box around the **entire person** (not just the hand or face)
4. In the label dropdown, select the appropriate class
5. Press **Save** (shortcut: `S`)
6. Click **Next** to go to the next image

### Annotation Best Practices

**DO:**
- ✅ Box the entire person's upper body when anomaly involves head/eyes
- ✅ Box just the hands + object for phone/notes detections  
- ✅ Include some context around the person (don't crop too tight)
- ✅ Label every anomaly you can see in the frame
- ✅ If multiple anomalies are visible, draw multiple boxes

**DON'T:**
- ❌ Don't box students who are sitting normally (only anomalous behavior)
- ❌ Don't skip blurry or partially visible images — include them
- ❌ Don't use the `OTHER` class for everything — be specific

---

## Step 4 — Create Dataset Splits and Export

After annotating all images:

1. Click **"Generate New Version"** in your Roboflow project
2. Configure splits:
   - **Train**: 80%
   - **Valid**: 15%  
   - **Test**: 5%
3. **Preprocessing**:
   - Auto-Orient: ON
   - Resize: 640×640 (Stretch)
4. **Augmentation** (optional but recommended):
   - Flip: Horizontal
   - Rotation: -5° to +5°
   - Brightness: -15% to +15%
   - Blur: Up to 1px
5. Click **"Generate"** → wait for processing
6. Click **"Export Dataset"**
7. Format: **YOLOv8**
8. Click **"Show download code"** → copy the API key and project details

---

## Step 5 — Train in Google Colab

1. Open the training notebook:
   **[`ml/training/ExamGuard_YOLOv8_Training.ipynb`](../training/ExamGuard_YOLOv8_Training.ipynb)**

2. Upload to Google Colab:
   - Go to **[colab.research.google.com](https://colab.research.google.com)**
   - File → Upload notebook → select the `.ipynb` file

3. Enable GPU:
   - Runtime → Change runtime type → **T4 GPU** → Save

4. Run all cells in order (Shift+Enter each cell)

5. In the Roboflow download cell, paste your API key and project details

6. Training takes approximately:
   - **30–60 minutes** for 1000 images on T4 GPU
   - **80 epochs** (recommended)

### What Good Training Looks Like

After training, check the metrics in Step 7 of the notebook:

| Metric | What it means | Target |
|---|---|---|
| `mAP@50` | Main accuracy metric | > 0.60 (60%) is good |
| `Precision` | When it detects something, how often correct | > 0.70 |
| `Recall` | Of all real anomalies, how many found | > 0.60 |

If accuracy is low:
- More images = better accuracy (aim for 200+ per class)
- More careful annotations help significantly
- Try more epochs (increase to 150)

---

## Step 6 — Integrate the Trained Model

### 6a — Place the weights file

After downloading `exam_anomaly_best.pt` from Colab:

```
exam-anomaly-detection-system/
└── ml/
    └── weights/
        └── exam_anomaly_best.pt   ← place it here
```

### 6b — Update the .env file

Open `.env` and add/update:

```bash
ML_MODEL_PATH=/ml/weights/exam_anomaly_best.pt
```

### 6c — Rebuild Docker containers

```bash
docker-compose down
docker-compose up --build -d
```

The `Dockerfile.backend` already installs `ultralytics` and `opencv-python-headless`.

### 6d — Verify the model loaded

After rebuild, check the model status endpoint:

```bash
curl http://localhost:8002/api/video/model/status
```

Expected response:
```json
{
  "model_loaded": true,
  "model_path": "/ml/weights/exam_anomaly_best.pt",
  "classes": ["COPYING", "LEAVING_SEAT", "MOBILE_PHONE", ...],
  "mode": "yolo"
}
```

---

## Step 7 — Test with Your Exam Video

1. Go to `http://localhost:8000/exam-control/analyze-video/`
2. Upload a video with an anomaly
3. Click **View Live Analysis** → **Start Live Analysis**
4. You should now see:
   - ✅ Real bounding boxes on actual anomalous people
   - ✅ Correct class labels (COPYING, MOBILE_PHONE, etc.)
   - ✅ Real confidence scores from the model
   - ✅ Evidence frames captured at exact anomaly timestamps

---

## Troubleshooting

### "No trained model found — using motion-proxy detection"
→ The weights file is not in `ml/weights/`. Check the path and filename.

### Low accuracy (mAP < 0.40)
→ Need more annotated images. Aim for at least 150 images per class.
→ Check that annotations are tight and consistent.

### Model loads but detects nothing
→ Lower the confidence threshold in `video.py`: change `conf_threshold=0.45` to `0.30`
→ Check that your class names in training match `CLASS_COLORS` keys in the code

### CUDA out of memory during training
→ Reduce `BATCH_SIZE` from 16 to 8 in the Colab notebook
→ Use `yolov8n.pt` (nano) instead of `yolov8s.pt`

---

## File Reference

| File | Purpose |
|---|---|
| `ml/training/ExamGuard_YOLOv8_Training.ipynb` | Google Colab training notebook |
| `ml/models/inference_engine.py` | YOLO inference engine (loaded by FastAPI) |
| `ml/weights/exam_anomaly_best.pt` | ← Place your trained weights here |
| `backend/app/routers/video.py` | FastAPI SSE router — uses inference engine |
| `.env` → `ML_MODEL_PATH` | Path to weights file |
