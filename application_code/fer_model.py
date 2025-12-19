# application_code/fer_model.py
# Facial Emotion Recognition Model Wrapper

# Importing necessary libraries
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # forcing CPU to avoid CUDNN issues during prediction
import json
import numpy as np
import tensorflow as tf

IMG_HEIGHT, IMG_WIDTH = 48, 48

EMOJI_MAP = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲",
}

# FERModel class to load model and make predictions
# FERModel loads a pre-trained Keras model and provides a method to predict emotions from 48x48 grayscale face images.
class FERModel:
    def __init__(self, model_path: str, class_json_path: str):
        print("Loading FER model...")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded.")

        # Loading class indices
        with open(class_json_path, "r") as f:
            class_indices = json.load(f)
        self.idx_to_class = {v: k for k, v in class_indices.items()}
        print("Class mapping:", self.idx_to_class)
    # Predicting emotion from 48x48 grayscale face image
    # gray48: np.ndarray of shape (48, 48), dtype uint8 or float32 as input.
    # Returns: (label, emoji, confidence)
    def predict_from_gray48(self, gray48: np.ndarray):
        if gray48.shape != (IMG_HEIGHT, IMG_WIDTH):
            raise ValueError(f"Expected (48,48), got {gray48.shape}")

        arr = gray48.astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=-1)  # (48,48,1)
        arr = np.expand_dims(arr, axis=0)   # (1,48,48,1)

        preds = self.model.predict(arr, verbose=0)[0]
        print("[Debug]Raw model predictions:", preds)
        #argmax to get predicted class index
        pred_idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        label = self.idx_to_class.get(pred_idx, f"class_{pred_idx}")
        emoji = EMOJI_MAP.get(label, "🙂")
        return label, emoji, conf
