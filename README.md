# Person-Detection-Using-YOLOv8
This project demonstrates real-time person detection using YOLOv8, a state-of-the-art object detection model by Ultralytics. The application can detect people in images, videos, and live webcam streams, and is built with a clean and interactive Streamlit interface.

# 🔍 Features

✅ Person detection using YOLOv8

📸 Supports image, video, and webcam input

🌗 Light/Dark mode toggle

🖼️ Displays bounding boxes with labels

📁 Easy-to-use Streamlit UI

# 🧠 Tech Stack
YOLOv8 – for real-time object detection

OpenCV – for image and video processing

Streamlit – for web-based UI

Python – backend scripting and logic

# 🛠 How to Run

### 🚀 How to Run the App Locally

Follow these steps to set up and run the person detection app:


#### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Himani-cpu/Person-Detection-Using-YOLOv8.git
cd Person-Detection-Using-YOLOv8

**2️⃣ Install Required Libraries**

bash
pip install -r requirements.txt

3️⃣ Add YOLOv8 Model Weights

Download or place your YOLOv8 model (like best.pt) in a weights/ folder.

Example (using pretrained model):

bash
mkdir weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P weights/

4️⃣ Run the Streamlit App

bash
streamlit run app.py

**🧪 Input Options Supported**

📷 Image Upload

🎞 Video Upload

🎥 Webcam Stream








