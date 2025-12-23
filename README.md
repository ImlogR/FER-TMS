# Facial Expression Recognition for Text Message Support (FER-TMS)

## Overview
This project implements a **real-time Facial Expression Recognition (FER)** system integrated into a **web-based messaging application** it is an **Emotion-Aware Messaging web-app using Facial Expression Recognition**  
The system captures a user's facial expression through their webcam, classifies the emotion using a trained **CNN model**, and transmits **only the inferred emotion (not raw images)** over a **WebSocket-based chat interface**.

The project consists of two major components:

1. **Model Development Pipeline**
   - Dataset preprocessing
   - CNN training
   - Model evaluation

2. **Real-Time FER-TMS Application**
   - Flask + Socket.IO backend
   - Client-side webcam capture
   - Emotion-aware messaging

## Folder Structure

    fer_tms/
    ├── README.md                             <- Project README
    ├── application_code/                     <- Real-time FER chat application
    │   ├── app.py                            <- Flask + Socket.IO server
    │   ├── fer_model.py                      <- FER model inference wrapper
    │   ├── best_fer_cnn.keras                <- Trained CNN model
    │   ├── class_indices.json                <- Emotion label mapping
    │   ├── static/                           <- Client-side JS & CSS
    │   ├── templates/                        <- HTML templates (login, chat)
    │   ├── debug_faces/                      <- Annotated server-side frames
    │   ├── grayscale/                        <- 48×48 grayscale face crops
    │
    └── model_development_code/               <- Offline training & evaluation
        ├── dataset/                          <- Train/test datasets
        │   ├── train/
        │   └── test/
        ├── model/                            <- Saved training outputs
        ├── model_build.ipynb                 <- CNN training notebook
        ├── model_evaluation.ipynb            <- Evaluation & metrics notebook

## Setting up the environment:
Ensure you have the following installed:

    Python version: 3.10

Then create a virtual environment using venv:

    python3 -m venv fer-tms

Activate the virtual environment:

    source fer-tms/bin/activate # for linux/mac
    fer-tms\Scripts\activate # for windows

## Installing Dependencies:
Inside the activated virtual environment run:

    pip install flask flask-socketio eventlet opencv-python numpy pillow tensorflow matplotlib scikit-learn ipykernel

## Running the Real-Time Messaging Application:
- Navigate to the application folder:
    cd application_code
- Run the Flask + Socker.IO server:
    python3 app.py
- You should see:
    Running on http://0.0.0.0:5000/
- Access the application in a browser:
    Open http://localhost:5000

Note: Be sure to open localhost:5000 not the IP, the sokets wont work without DNS.

How the app works:
1. User enters a username and a room name.
2. The browser activates the webcam.
3. Every time the user sends a message:
    - The browser captures a frame.
    - The frameis downscaled and sent to the server.
    - The server performs face detection -> 48*48 crop -> CNN inference.
    - The predicted emotion + emoji is attached to the message.
    - The message is broadcasted to all users in the same room.

Debug Outputs(server-side):
- full annotated frames with bounding boxes -> debug_faces/
- Actual 48*48 model inputs -> grayscale/

These folders are automatically populated as messagges are sent.

## Running Model Training and Evaluation:
Navigate to:

    cd model_development_code

You will see the notebooks, open them in your text editor (vs-code) for ease.
- Model building notebook:
    - model_build.ipynb:
        - loads dataset
        - Performs normalization and augmentation
        - Builds the CNN architecture
        - Trains the model
        - Saves final model as best_fer_cnn.keras
        - Saves class mapping class_indices.json
        - Plots training curves
- Model evaluation notebook:
    - model_evaluation.ipynb:
        - Loads trained model
        - Evaluates on test set
        - Computes accuracy, precision, recall, F1
        - Generates confusion matrix
        - Outputs performance tables as in report

After training, you must copy the final model files to the application folder i.e. the best_fer_cnn.heras and class_indices.json to application_code folder (These are already present in the submission if you retrain the model then copy the results as suggested).

## System Requirements:
- A machine with Python 3.10
- Webcam 
- Modern browser supporting navigator.mediaDevices
- At least 4 GB RAM
- TensorFlow CPU is sufficient

## Troubleshooting:
- Webcam not working in browser:
    - Ensure page is served via localhost, not file://
    - Allow camera permissions
- Socket.IO connection errors:
    - Clear browser cache
    - Restart eventlet if port is locked
    - use modern browser which support websockets (latest chrome, safari or brave)
- Cascaded face detection fails:
    - Poor lighting or non-frontal faces reduce accuracy
    - Annotated frames in debug_frames/ help diagnose issues
- FER predictions seem repetitive:
    - Check that new frames are being saved in grayscale/
    - verify your model is updated in application_code/

## Summary:
This project integrates machine learning, real-time communication, and computer vision into a working emotion-aware chat system.