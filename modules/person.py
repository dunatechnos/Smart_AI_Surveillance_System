import os
import cv2
import json
import csv
import pickle
import base64
import numpy as np
from datetime import datetime
from flask import render_template
from urllib.parse import quote


import logging
logging.getLogger().setLevel(logging.ERROR)
from insightface.app import FaceAnalysis

# ================= CONFIG =================
UPLOAD_DIR = "uploads"
LABEL_DIR = "outputs/labels"

FACE_PKL = "outputs/face_embeddings.pkl"
PERSON_CSV = "outputs/person_data.csv"
GUARD_CSV = "outputs/guard_entry.csv"

os.makedirs("outputs", exist_ok=True)

# ================= INSIGHTFACE =================
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ================= INIT STORAGE =================
if not os.path.exists(FACE_PKL):
    with open(FACE_PKL, "wb") as f:
        pickle.dump({}, f)

if not os.path.exists(PERSON_CSV):
    with open(PERSON_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["person_id", "name", "phone", "address", "created_on"]
        )

if not os.path.exists(GUARD_CSV):
    with open(GUARD_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["person_id", "tracking_id", "source", "in_time"]
        )

# ================= HELPERS =================
def next_person_id():
    rows = []
    with open(PERSON_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    if not rows:
        return 10000
    return max(int(r[0]) for r in rows) + 1

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def img_to_b64(img):
    _, buf = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

# ================= LOGIC FUNCTIONS =================
FACE_EMBS = []

def resolve_face_logic(request):
    data = request.json
    face_index = int(data.get("face_index", 0))

    if "FACE_EMBS" not in globals() or not FACE_EMBS or face_index >= len(FACE_EMBS):
        return {"person_id":"", "name":"", "phone":"", "address":"", "history": []}
    
    emb = FACE_EMBS[face_index]

    if not os.path.exists(FACE_PKL):
         return {"person_id":"", "name":"", "phone":"", "address":"", "history": []}

    with open(FACE_PKL, "rb") as f:
        face_db = pickle.load(f)

    found_pid = None
    for pid, e in face_db.items():
        if cosine_sim(emb, e) > 0.6:
            found_pid = pid
            break
            
    if not found_pid:
        return {"person_id":"", "name":"", "phone":"", "address":"", "history": []}

    name = phone = address = ""
    if os.path.exists(PERSON_CSV):
        with open(PERSON_CSV, newline="", encoding="utf-8") as f:
            for r in csv.reader(f):
                if r and r[0] == str(found_pid):
                    name, phone, address = r[1], r[2], r[3]
                    break

    history = []
    if os.path.exists(GUARD_CSV):
        with open(GUARD_CSV, newline="", encoding="utf-8") as gf:
            rows = list(csv.reader(gf))
            if len(rows) > 1:
                for gr in rows[1:]:
                    if gr and gr[0] == str(found_pid):
                        history.append([gr[3], gr[2], gr[1]])

    return {
        "person_id": found_pid,
        "name": name,
        "phone": phone,
        "address": address,
        "history": history
    }

def person_form_logic(request):
    source = request.args["source"]
    tracking_id = request.args["tracking_id"]
    frame_index = int(request.args["frame_index"])

    with open(os.path.join(LABEL_DIR, source + ".json"), "r") as f:
        labels = json.load(f)

    frames = labels["objects"][str(tracking_id)]["frames"]
    fdata = frames[frame_index]

    path = os.path.join(UPLOAD_DIR, source)
    is_video = labels["type"] == "video"

    if is_video:
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fdata["frame"])
        ret, frame = cap.read()
        cap.release()
    else:
        frame = cv2.imread(path)

    x1,y1,x2,y2 = fdata["x1"],fdata["y1"],fdata["x2"],fdata["y2"]
    h,w = frame.shape[:2]
    x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)

    person_crop = frame[y1:y2, x1:x2]

    faces = list(face_app.get(person_crop) or [])
    face_crops = [person_crop[int(f.bbox[1]):int(f.bbox[3]), int(f.bbox[0]):int(f.bbox[2])] for f in faces if person_crop[int(f.bbox[1]):int(f.bbox[3]), int(f.bbox[0]):int(f.bbox[2])].size>0]
    face_embeddings = [f.embedding for f in faces]
    global FACE_EMBS; FACE_EMBS = face_embeddings
    face_imgs = [img_to_b64(fc) for fc in face_crops]
    matched_id = None

    with open(FACE_PKL,"rb") as f:
        face_db = pickle.load(f)
    # try to match any detected face embeddings (pick first match)
    selected_face_index = 0
    if len(face_embeddings) == 1:
        emb = face_embeddings[0]
        for pid, e in face_db.items():
            if cosine_sim(emb, e) > 0.6:
                matched_id = pid
                break
            if matched_id is not None:
                break
    if face_embeddings and 'selected_face_index' not in locals():
        selected_face_index = 0

    name=phone=address=""
    history=[]

    if matched_id:
        with open(PERSON_CSV,newline="",encoding="utf-8") as f:
            for r in csv.reader(f):
                if r and r[0]==str(matched_id):
                    name,phone,address = r[1],r[2],r[3]
        with open(GUARD_CSV,newline="",encoding="utf-8") as f:
            for r in list(csv.reader(f))[1:]:
                if r and r[0] == str(matched_id):
                    history.append([r[3], r[2], r[1]])

    if request.method=="POST":
        face_idx = int(request.form.get("face_index", 0))
        if not face_embeddings or face_idx >= len(face_embeddings):
            emb = None
        else:
            emb = face_embeddings[face_idx]
        matched_id = None
        if emb is not None:
            for pid_db, e_db in face_db.items():
                if cosine_sim(emb, e_db) > 0.6:
                    matched_id = pid_db
                    break
        if matched_id is None and emb is not None:
            matched_id = next_person_id()
            face_db[matched_id] = emb
            with open(FACE_PKL, "wb") as f:
                pickle.dump(face_db, f)
        pid = str(matched_id)
        name=request.form["name"]
        phone=request.form["phone"]
        address=request.form["address"]
        rows=[]
        found=False
        pid = str(pid)
        with open(PERSON_CSV,newline="",encoding="utf-8") as f:
            all_rows = list(csv.reader(f))
            header = all_rows[0]
            rows = all_rows[1:]
        for r in rows:
            if r and r[0] == pid:
                r[1],r[2],r[3]=name,phone,address
                found=True
        if not found:
            rows.append([pid,name,phone,address,datetime.now().isoformat()])
        with open(PERSON_CSV,"w",newline="",encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)
            csv_writer.writerows(rows)
        with open(GUARD_CSV,"a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(
                [pid,tracking_id,source,datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            )
    if is_video:
        fps = labels.get("fps", None)
        if fps is None:
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        start_time = fdata["frame"] / max(fps, 1)
        video_url = f"/uploads/{quote(source)}#t={start_time}"
    else:
        video_url = f"/uploads/{quote(source)}"
    return render_template(
        "person.html",
        tracking_id=tracking_id,
        person_img=img_to_b64(person_crop),
        face_imgs=face_imgs,
        is_video=is_video,
        video_url=video_url,
        datetime_now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        person_id=matched_id or "",
        name=name, phone=phone, address=address,
        history=history,
        selected_face_index=selected_face_index
    )
