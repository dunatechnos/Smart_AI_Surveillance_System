import warnings
warnings.filterwarnings("ignore")

import os
os.environ["YOLO_VERBOSE"] = "False"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

import logging
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import base64
from modules.logger import success
from modules import main
from modules import person
from modules import vehicle

app = Flask(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    f = request.files["file"]
    path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(path)

    ext = f.filename.lower().split(".")[-1]
    source_type = "image" if ext in ["jpg", "jpeg", "png"] else "video"

    main.process_upload(path, source_type)
    return jsonify({"type": source_type, "source": f.filename})

@app.route("/poll_alerts")
def poll_alerts():
    return jsonify(main.get_current_alerts(request.args["source"]))

@app.route("/crop")
def crop():
    source = request.args["source"]
    tid = int(request.args["id"])
    index = int(request.args.get("index", 0))

    crop_img, idx, _ = main.get_crop(source, tid, index)
    _, buf = cv2.imencode(".jpg", crop_img)

    return jsonify({
        "index": idx,
        "image": base64.b64encode(buf).decode()
    })

@app.route("/delete_tracking_id", methods=["POST"])
def delete_tracking_id():
    data = request.json
    main.delete_tracking_id(data["source"], data["id"])
    return jsonify({"ok": True})

@app.route("/uploads/<path:p>")
def uploads(p):
    return send_from_directory(UPLOAD_DIR, p)

@app.route("/person", methods=["GET", "POST"])
def person_view():
    return person.person_form_logic(request)

@app.route("/vehicle", methods=["GET", "POST"])
def vehicle_view():
    return vehicle.vehicle_form_logic(request)

@app.route("/resolve_face", methods=["POST"])
def resolve_face():
    return person.resolve_face_logic(request)

if __name__ == "__main__":
    success("Server started: http://127.0.0.1:5000")
    app.run(port=5000, debug=False)
