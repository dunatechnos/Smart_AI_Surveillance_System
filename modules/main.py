import os
import cv2
import json
import time
import threading
import csv
from ultralytics import YOLO
from datetime import datetime
import numpy as np
from modules.logger import progress, error

# ================= CONFIG =================
MODEL_PATH = "models/best.pt"
UPLOAD_DIR = "uploads"
LABEL_DIR = "outputs/labels"
LOG_CSV = "outputs/logs.csv"

ALERT_CLASSES = ["person", "car", "truck", "motorbike"]
FRAMES_PER_SECOND = 4

os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ================= MODEL =================
model = YOLO(MODEL_PATH)

# ================= GLOBAL STATE =================
GLOBAL_TRACKER_MAP = {}      # botsort_id -> custom tracking_id
SENT_ALERT_IDS = {}          # source -> set(tracking_id)
LABEL_LOCKS = {}             # source -> threading.Lock()

# ================= CSV INIT =================
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["tracking_id", "class", "source", "first_seen", "last_seen", "time", "status"]
        )

# ================= HELPERS =================
def _label_path(source):
    return os.path.join(LABEL_DIR, source + ".json")

def _new_tracking_id():
    return int(time.time() * 1000) % 100000

def _log_update(tid, cls, source, first, last):
    # generate detect time for a new tracking id
    detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []
    if os.path.exists(LOG_CSV):
        with open(LOG_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))[1:]

    # check if existing row for this tracking id exists and capture its time
    existing_time = None
    for r in rows:
        if r and r[0] == str(tid):
            # old files may or may not have a time column; if present reuse it
            existing_time = r[5] if len(r) > 5 else None
            break

    # remove any existing rows for this tracking id
    rows = [r for r in rows if not (r and r[0] == str(tid))]

    time_val = existing_time if existing_time is not None else detect_time

    rows.append([str(tid), cls, source, str(first), str(last), time_val, "active"])

    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tracking_id", "class", "source", "first_seen", "last_seen", "time", "status"])
        w.writerows(rows)

def placeholder_black_image():
    return np.zeros((120, 200, 3), dtype=np.uint8)

# ================= IMAGE PROCESS =================
def _process_image(path):
    name = os.path.basename(path)
    lp = _label_path(name)

    if os.path.exists(lp):
        os.remove(lp)   # overwrite old labels

    img = cv2.imread(path)
    res = model(img, verbose=False)[0]

    labels = {"source": name, "type": "image", "objects": {}}
    SENT_ALERT_IDS[name] = set()
    LABEL_LOCKS[name] = threading.Lock()

    for box, cls in zip(res.boxes.xyxy, res.boxes.cls):
        tid = str(_new_tracking_id())
        cls_name = model.names[int(cls)]
        x1, y1, x2, y2 = map(int, box)

        labels["objects"][tid] = {
            "class": cls_name,
            "frames": [{
                "frame": 0,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2
            }]
        }

        _log_update(tid, cls_name, name, 0, 0)

        if cls_name in ALERT_CLASSES:
            SENT_ALERT_IDS[name].add(tid)

    json.dump(labels, open(lp, "w"), indent=2)

# ================= VIDEO BACKGROUND PROCESS =================
def _background_video_process(video_path):
    source = os.path.basename(video_path)
    lp = _label_path(source)


    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    stride = max(int(fps // FRAMES_PER_SECOND), 1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    is_rtsp = video_path.startswith("rtsp://")

    if source not in LABEL_LOCKS:
        LABEL_LOCKS[source] = threading.Lock()
    if source not in SENT_ALERT_IDS:
        SENT_ALERT_IDS[source] = set()


    frame_no = 0
    last_time = time.time()
    with LABEL_LOCKS[source]:
        labels = json.load(open(lp))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % stride != 0:
            continue

        # Progress tracking
        try:
            if is_rtsp:
                now = time.time()
                if now - last_time >= 1:
                    fps_val = int(cap.get(cv2.CAP_PROP_FPS)) or 0
                    progress(source, fps_val, 0)
                    last_time = now
            else:
                progress(source, frame_no, total_frames)
        except Exception as e:
            error(f"Progress error: {e}")

        try:
            res = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]
            if res.boxes.id is None:
                continue

            updated = False

            for box, trk, cls in zip(res.boxes.xyxy, res.boxes.id, res.boxes.cls):
                trk = int(trk)

                if trk not in GLOBAL_TRACKER_MAP:
                    GLOBAL_TRACKER_MAP[trk] = _new_tracking_id()

                tid = str(GLOBAL_TRACKER_MAP[trk])
                cls_name = model.names[int(cls)]
                x1, y1, x2, y2 = map(int, box)

                with LABEL_LOCKS[source]:
                    if tid not in labels["objects"]:
                        labels["objects"][tid] = {
                            "class": cls_name,
                            "frames": []
                        }
                        _log_update(tid, cls_name, source, frame_no, frame_no)
                        updated = True

                    labels["objects"][tid]["frames"].append({
                        "frame": frame_no,
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2
                    })
                    updated = True

                    if cls_name in ALERT_CLASSES:
                        SENT_ALERT_IDS[source].add(tid)

            if updated:
                with LABEL_LOCKS[source]:
                    json.dump(labels, open(lp, "w"), indent=2)
        except Exception as e:
            error(f"Detection error: {e}")

    cap.release()

# ================= VIDEO PROCESS ENTRY =================
def _process_video(path):
    source = os.path.basename(path)
    lp = _label_path(source)

    if os.path.exists(lp):
        return

    labels = {"source": source, "type": "video", "objects": {}}
    SENT_ALERT_IDS[source] = set()
    LABEL_LOCKS[source] = threading.Lock()

    json.dump(labels, open(lp, "w"), indent=2)

    threading.Thread(
        target=_background_video_process,
        args=(path,),
        daemon=True
    ).start()

# ================= UI CALLABLE =================
def process_upload(path, source_type):
    if source_type == "image":
        _process_image(path)
    else:
        _process_video(path)

def get_current_alerts(source):
    lp = _label_path(source)
    if not os.path.exists(lp):
        return []

    with LABEL_LOCKS.get(source, threading.Lock()):
        labels = json.load(open(lp))

    alerts = []
    for tid, obj in labels["objects"].items():
        if obj["class"] in ALERT_CLASSES:
            alerts.append({
                "id": int(tid),
                "name": obj["class"]
            })
    return alerts

def get_crop(source, tracking_id, index):
    with LABEL_LOCKS[source]:
        labels = json.load(open(_label_path(source)))
        frames = labels["objects"][str(tracking_id)]["frames"]

    index = max(0, min(index, len(frames) - 1))
    f = frames[index]

    if labels["type"] == "image":
        img = cv2.imread(os.path.join(UPLOAD_DIR, source))
        h, w = img.shape[:2]
        x1, y1, x2, y2 = f["x1"], f["y1"], f["x2"], f["y2"]

        x1 = max(0, min(int(x1), w-1))
        x2 = max(0, min(int(x2), w-1))
        y1 = max(0, min(int(y1), h-1))
        y2 = max(0, min(int(y2), h-1))

        crop = img[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return placeholder_black_image(), index, 0
        return crop, index, 0

    cap = cv2.VideoCapture(os.path.join(UPLOAD_DIR, source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f["frame"])
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return placeholder_black_image(), index, max(0, len(frames) - 1)

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = f["x1"], f["y1"], f["x2"], f["y2"]

    x1 = max(0, min(int(x1), w-1))
    x2 = max(0, min(int(x2), w-1))
    y1 = max(0, min(int(y1), h-1))
    y2 = max(0, min(int(y2), h-1))

    crop = frame[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return placeholder_black_image(), index, max(0, len(frames) - 1)

    return crop, index, len(frames) - 1

def delete_tracking_id(source, tracking_id):
    lp = _label_path(source)

    with LABEL_LOCKS[source]:
        labels = json.load(open(lp))
        labels["objects"].pop(str(tracking_id), None)
        json.dump(labels, open(lp, "w"), indent=2)

    rows = []
    with open(LOG_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]

    rows = [r for r in rows if r[0] != str(tracking_id)]

    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tracking_id", "class", "source", "first_seen", "last_seen", "time", "status"])
        w.writerows(rows)
