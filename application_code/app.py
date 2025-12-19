# Facial Emotion Recognition Chat Application
# Using Flask, Flask-SocketIO, OpenCV, and a pre-trained FER model

# Importing necessary libraries
import base64
import hashlib
import os
import time
from PIL import Image
import cv2

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
)
from flask_socketio import SocketIO, join_room, emit
import numpy as np

from fer_model import FERModel

# Configuration
MODEL_PATH = "best_fer_cnn.keras" # Path to the pre-trained FER model
CLASS_JSON = "class_indices.json" # Path to class indices JSON

app = Flask(__name__)
app.config["SECRET_KEY"] = "super-secret-key"

# Initializing Flask-SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# FER model (loaded once)
fer = FERModel(MODEL_PATH, CLASS_JSON)

# Initializing face detector (Haar cascade)
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Making sure debug folders exist
DEBUG_DIR = "debug_faces"      # stores annotated full frames with facial boundary boxes
GRAY_DIR = "grayscale"         # stores 48x48 grayscale face crops

# Creating directories if they don't exist
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(GRAY_DIR, exist_ok=True)

# Function to extract and preprocess face region from BGR frame; takes np.ndarray (H, W, 3) input and returns 48x48 grayscale face
def extract_face_gray48_from_bgr(frame_bgr):
    # Validating input
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Empty frame passed to extract_face_gray48_from_bgr")

    annotated = frame_bgr.copy()

    # Converting to grayscale for detection
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    # Detecting faces (returns list of (x,y,w,h))
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) > 0:
        # picking largest face by area
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_gray = gray[y:y + h, x:x + w]

        # drawing face box (green)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        # Fallback: center-crop a square region from the whole frame if face not detected
        h0, w0 = gray.shape
        side = min(h0, w0)
        cx, cy = w0 // 2, h0 // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = x1 + side
        y2 = y1 + side
        face_gray = gray[y1:y2, x1:x2]

        # drawing fallback big box (blue) if no face detected
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Resizing to 48x48 for the FER model
    face_gray_resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)

    # Saving annotated frame for debugging
    timestamp = int(time.time() * 1000)
    debug_path = os.path.join(DEBUG_DIR, f"annotated_{timestamp}.jpg")

    # downscaling for convenience
    max_width = 640
    if annotated.shape[1] > max_width:
        scale = max_width / annotated.shape[1]
        annotated_small = cv2.resize(
            annotated,
            (max_width, int(annotated.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        annotated_small = annotated

    cv2.imwrite(debug_path, annotated_small)
    print(f"[DEBUG] Saved annotated frame: {debug_path}")

    return face_gray_resized


# creating a minimal web app with Flask
# a root route redirecting to /login
@app.route("/", methods=["GET"])
def root():
    return redirect(url_for("login"))

# a login route to enter username and room name
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        room = request.form.get("room", "").strip()

        if not username or not room:
            return render_template(
                "login.html",
                error="Please enter both username and channel name.",
            )

        # Saving in session
        session["username"] = username
        session["room"] = room
        return redirect(url_for("chat"))

    return render_template("login.html")

# a chat route to render the chat interface
@app.route("/chat")
def chat():
    username = session.get("username")
    room = session.get("room")
    if not username or not room:
        return redirect(url_for("login"))

    return render_template("chat.html", username=username, room=room)

# WebSocket event handlers
@socketio.on("connect")
def handle_connect():
    # Client connects; we'll ask them to emit "join" with username/room.
    print("Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")

# Handling "join" event to join a room
# Client sends:
# {#   "username": "...",
#   "room": "..."
# }
# Server responds by adding the client to the room and broadcasting a system message.
@socketio.on("join")
def handle_join(data):
    username = data.get("username")
    room = data.get("room")
    if not username or not room:
        return

    join_room(room)
    print(f"{username} joined room {room}")

    emit(
        "system_message",
        {"text": f"{username} joined the room.", "room": room},
        room=room,
    )

# Handling "send_message" event
# Client sends:
# {
#   "username": "...",
#   "room": "...",
#   "text": "...",
#   "image": "base64-encoded-image-string"
# }
# Server processes the image, predicts emotion, and broadcasts the message with emotion info.
@socketio.on("send_message")
def handle_send_message(data):
    username = data.get("username")
    room = data.get("room")
    text = data.get("text", "")
    img_b64 = data.get("image")

    if not username or not room or not img_b64:
        return

    # Decoding base64 to raw bytes, then JPEG → BGR
    try:
        img_bytes = base64.b64decode(img_b64.encode("utf-8"))
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        gray = extract_face_gray48_from_bgr(frame_bgr)

        # DEBUG: checking if we really get a new image each time
        img_hash = hashlib.md5(img_bytes).hexdigest()
        mean_val = float(gray.mean())
        std_val = float(gray.std())

        timestamp = int(time.time() * 1000)
        gray_path = os.path.join(GRAY_DIR, f"{username}_{timestamp}.png")

        # Saving the 48x48 grayscale crop in its own folder
        Image.fromarray(gray).save(gray_path)
        print(f"[DEBUG] saved grayscale patch: {gray_path}")

        print(
            f"[DEBUG] Message from {username} in room {room} "
            f"hash={img_hash} mean={mean_val:.2f} std={std_val:.2f}"
        )

    except Exception as e:
        print("Error decoding image:", e)
        return

    label, emoji, conf = fer.predict_from_gray48(gray)

    message = {
        "username": username,
        "room": room,
        "text": text,
        "label": label,
        "emoji": emoji,
        "confidence": conf,
    }

    # Broadcasting to the room: everyone in the same channel sees it
    emit("receive_message", message, room=room)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
