import os
import cv2
import json
import csv
import base64
import torch
import numpy as np
from datetime import datetime
from flask import render_template
from urllib.parse import quote

# ================= PATHS =================
UPLOAD_DIR = "uploads"
LABEL_DIR = "outputs/labels"
VEHICLE_CSV = "outputs/vehicle_data.csv"
LPR_MODEL_PATH = "models/best.pt"

os.makedirs("outputs", exist_ok=True)

# ================= CSV INIT =================
if not os.path.exists(VEHICLE_CSV):
    with open(VEHICLE_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["lpr_no", "datetime", "driver_person_id", "vehicle_type", "tracking_id", "source"]
        )

# ================= YOLO LPR =================
from ultralytics import YOLO
lpr_model = YOLO(LPR_MODEL_PATH)

# ================= TrOCR =================
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

trocr_processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-printed"
)
trocr_model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-printed"
).to(DEVICE)

if DEVICE == "cuda":
    trocr_model = trocr_model.half()

trocr_model.eval()

def run_trocr(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pixel_values = trocr_processor(
        images=img_rgb,
        return_tensors="pt"
    ).pixel_values.to(DEVICE)

    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values)
    text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

# ================= HELPERS =================
def img_to_b64(img):
    _, buf = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

def detect_lpr(vehicle_crop):
    results = lpr_model(vehicle_crop, conf=0.25, verbose=False)

    best_box = None
    best_conf = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = lpr_model.names[cls_id]

            if cls_name.lower() == "lpr" and conf > best_conf:
                best_conf = conf
                best_box = tuple(map(int, box.xyxy[0]))

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    h, w = vehicle_crop.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    return vehicle_crop[y1:y2, x1:x2]

# ================= MAIN LOGIC =================
def vehicle_form_logic(request):
    source = request.args["source"]
    tracking_id = request.args["tracking_id"]
    frame_index = int(request.args["frame_index"])

    with open(os.path.join(LABEL_DIR, source + ".json")) as f:
        labels = json.load(f)

    frames = labels["objects"][str(tracking_id)]["frames"]
    fdata = frames[frame_index]

    path = os.path.join(UPLOAD_DIR, source)
    is_video = labels["type"] == "video"

    if is_video:
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fdata["frame"])
        _, frame = cap.read()
        cap.release()
    else:
        frame = cv2.imread(path)

    x1, y1, x2, y2 = fdata["x1"], fdata["y1"], fdata["x2"], fdata["y2"]
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    vehicle_crop = frame[y1:y2, x1:x2]
    lpr_crop = detect_lpr(vehicle_crop)

    ocr_text = ""
    if lpr_crop is not None:
        ocr_text = run_trocr(lpr_crop)

    if request.method == "POST":
        with open(VEHICLE_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                request.form.get("lpr_no", ""),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                request.form.get("driver_person_id", ""),
                labels["objects"][str(tracking_id)]["class"],
                tracking_id,
                source
            ])

    if is_video:
        fps = labels.get("fps", 25)
        start_time = fdata["frame"] / max(fps, 1)
        video_url = f"/uploads/{quote(source)}#t={start_time}"
    else:
        video_url = f"/uploads/{quote(source)}"

    return render_template(
        "vehicle.html",
        tracking_id=tracking_id,
        vehicle_img=img_to_b64(vehicle_crop),
        lpr_img=img_to_b64(lpr_crop) if lpr_crop is not None else None,
        ocr_text=ocr_text,
        is_video=is_video,
        video_url=video_url,
        datetime_now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        vehicle_type=labels["objects"][str(tracking_id)]["class"]
    )
