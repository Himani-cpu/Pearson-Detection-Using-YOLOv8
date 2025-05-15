import streamlit as st
import cv2
import tempfile
import os
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import pandas as pd

# Set page config
st.set_page_config(page_title="Person Detection App", layout="wide", initial_sidebar_state="expanded")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Create directories
DETECTION_DIR = "detections"
HISTORY_FILE = "detection_history.csv"
os.makedirs(DETECTION_DIR, exist_ok=True)

# Load or create detection history CSV
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    else:
        return pd.DataFrame(columns=["Timestamp", "InputType", "PersonCount", "FilePath"])

def save_history(timestamp, input_type, count, path):
    df = load_history()
    df.loc[len(df)] = [timestamp, input_type, count, path]
    df.to_csv(HISTORY_FILE, index=False)

# Sidebar settings
st.sidebar.title("⚙️ Settings")
dark_mode = st.sidebar.toggle("🌙 Dark Mode")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
input_type = st.sidebar.radio("Select Input Type", ["Image", "Video", "Webcam"])

# Show detection history
st.sidebar.markdown("### 📜 Detection History")
history_df = load_history()
if not history_df.empty:
    st.sidebar.dataframe(history_df.tail(5), use_container_width=True)
else:
    st.sidebar.write("No detections yet.")

# Dark mode styling
if dark_mode:
    st.markdown("""
        <style>
        body { background-color: #0e1117; color: white; }
        .stApp { background-color: #0e1117; }
        </style>
    """, unsafe_allow_html=True)

st.title("🧍 Person Detection using YOLOv8")

def save_detection_image(img, suffix="image"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(DETECTION_DIR, f"detection_{suffix}_{timestamp}.jpg")
    cv2.imwrite(filename, img)
    return filename, timestamp

def process_and_display(frame, confidence_thresh):
    results = model(frame)[0]
    count = 0
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if model.names[int(cls)] == "person" and conf >= confidence_thresh:
            x1, y1, x2, y2 = map(int, box)
            count += 1
            label = f"Person {count} ({conf:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, f"Total Persons: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame, count

# Image upload
if input_type == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.resize(image, (640, 480))
        output, person_count = process_and_display(image, confidence_threshold)
        st.image(output, channels="BGR", caption="Detected Image", use_container_width=True)
        save_path, timestamp = save_detection_image(output, "image")
        save_history(timestamp, "Image", person_count, save_path)
        st.success(f"✅ Detection saved to `{save_path}`")

# Video upload
elif input_type == "Video":
    video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        last_frame = None
        person_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            output, count = process_and_display(frame, confidence_threshold)
            person_count = max(person_count, count)
            last_frame = output
            stframe.image(output, channels="BGR", use_container_width=True)
        cap.release()
        if last_frame is not None:
            save_path, timestamp = save_detection_image(last_frame, "video")
            save_history(timestamp, "Video", person_count, save_path)
            st.success(f"✅ Last frame saved to `{save_path}`")

# Webcam detection
elif input_type == "Webcam":
    run = st.checkbox("Start Webcam")
    stframe = st.empty()
    last_frame = None
    person_count = 0
    if run:
        cap = cv2.VideoCapture(0)
        while run and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            output, count = process_and_display(frame, confidence_threshold)
            person_count = max(person_count, count)
            last_frame = output
            stframe.image(output, channels="BGR", use_container_width=True)
        cap.release()
        if last_frame is not None:
            save_path, timestamp = save_detection_image(last_frame, "webcam")
            save_history(timestamp, "Webcam", person_count, save_path)
            st.success(f"✅ Last frame saved to `{save_path}`")
