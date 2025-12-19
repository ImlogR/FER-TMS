// static/chat.js
// Client-side JavaScript for chat with facial expression recognition
document.addEventListener("DOMContentLoaded", () => {
  const config = window.FER_CHAT_CONFIG || {};
  const username = config.username;
  const room = config.room;

  console.log("FER_CHAT_CONFIG:", config);

  const socket = io();

  const chatLog = document.getElementById("chat-log");
  const messageInput = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const video = document.getElementById("video");
  const canvas = document.getElementById("captureCanvas");
  const ctx = canvas.getContext("2d");

  // FaceDetector (experimental API)
  let faceDetector = null;
  if ("FaceDetector" in window) {
    try {
      faceDetector = new FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
      console.log("FaceDetector API available");
    } catch (e) {
      console.warn("FaceDetector init failed, disabling:", e);
      faceDetector = null;
    }
  } else {
    console.warn("FaceDetector API not available; using full-frame fallback.");
  }

  // camera setup
  function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      console.error("navigator.mediaDevices.getUserMedia not available.");
      alert(
        "Camera access is not available.\n\n" +
        "Use a modern browser and open the app via:\n" +
        "http://localhost:5000"
      );
      return;
    }

    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then((stream) => {
        console.log("Camera stream started");
        video.srcObject = stream;
      })
      .catch((err) => {
        console.error("Error accessing camera:", err);
        alert("Could not access camera: " + err.message);
      });
  }

  startCamera();

  // SOCKET.IO events
  socket.on("connect", () => {
    console.log("Connected with id:", socket.id);
    if (username && room) {
      socket.emit("join", { username, room });
    }
  });

  socket.on("system_message", (data) => {
    const div = document.createElement("div");
    div.className = "chat-message other";
    div.textContent = "[SYSTEM] " + data.text;
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
  });

  socket.on("receive_message", (msg) => {
    const isMe = msg.username === username;
    addMessage(msg, isMe);
  });

  // UI helpers
  function addMessage(msg, isMe) {
    const div = document.createElement("div");
    div.className = "chat-message " + (isMe ? "me" : "other");

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent =
      `${msg.username} (${(msg.confidence * 100).toFixed(1)}% ` +
      `${msg.label} ${msg.emoji})`;

    const text = document.createElement("div");
    text.textContent = msg.text;

    div.appendChild(meta);
    div.appendChild(text);
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  // Capture current video frame as JPEG base64
  async function captureFrameJpegBase64() {
  if (!video || video.readyState < 2) {
    throw new Error("Camera not ready");
  }

  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = video.videoWidth || 320;
  tmpCanvas.height = video.videoHeight || 240;
  const tmpCtx = tmpCanvas.getContext("2d");

  // Drawing current video frame
  tmpCtx.drawImage(video, 0, 0, tmpCanvas.width, tmpCanvas.height);

  // Encoding as JPEG base64 (without the data URL prefix)
  const dataUrl = tmpCanvas.toDataURL("image/jpeg", 0.7); // 0.7 = quality
  const base64Data = dataUrl.split(",")[1]; // removing "data:image/jpeg;base64,"

  return base64Data;
}

  // sending message
  async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text) return;

    let img_b64;
    try {
      img_b64 = await captureFrameJpegBase64();
    } catch (err) {
      console.error("Capture failed:", err);
      alert("Failed to capture from camera. See console for details.");
      return;
    }

    socket.emit("send_message", {
      username,
      room,
      text,
      image: img_b64,
    });

    messageInput.value = "";
  }

  sendBtn.addEventListener("click", () => {
    console.log("Send button clicked");
    sendMessage();
  });

  messageInput.addEventListener("keyup", (e) => {
    if (e.key === "Enter") {
      sendMessage();
    }
  });
});
