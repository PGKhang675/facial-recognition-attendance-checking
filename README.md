# 🛡️ Secure Face Attendance System
**Liveness Detection | Recognition | Emotion Analysis**

This is a real-time facial recognition application built with TensorFlow and CustomTkinter for my Applied Machine Learning subject. This system doesn't just recognize faces; it ensures the person is real (anti-spoofing) and analyzes their emotional state, making it ideal for high-security attendance or smart workplace environments.

## Project Overview
This project implements a multi-stage deep learning pipeline to provide a "triple-check" security protocol for user identification. The application is wrapped in a modern, dark-themed GUI for ease of use.

### Key Features
👤 Face Recognition: Utilizes a ResNet50 backbone to generate face embeddings, comparing them against a local database using Euclidean distance.

🛡️ Anti-Spoofing (Liveness): Powered by MobileNetV2, the system distinguishes between a real human face and a digital or printed photo/video "spoof."

🎭 Emotion Detection: Real-time sentiment analysis (Happy, Sad, Angry, etc.) using a dedicated CNN model, providing deeper insights into user interaction.

📂 User Management: Integrated tools to register new users, save embeddings to a persistent database (.pkl), and delete entries via the UI.

⚡ Threaded Performance: AI inference runs on background threads to ensure the UI remains fluid and the video feed stays lag-free.

## Getting Started
Follow these steps to set up the environment and run the application on your local machine.

### 1. Prerequisites
Ensure you have Python 3.9+ installed. It is recommended to use a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 2. Install Dependencies
Create a requirements.txt file (or run the command below) to install the necessary libraries:
```
pip install -r ./Code/requirements.txt
```
### 3. Directory Structure
The application expects your trained models and label files to be organized as follows:
```
├── Models/
│   ├── final_face_embedding_model.keras # Recognition model
│   ├── liveness_model.keras             # Anti-spoofing model
│   └── emotion_model.keras              # Emotion model
├── Code/
│   ├── app.py                           # The provided code
│   ├── emotion_labels.pkl               # Pickled list of emotion strings
│   └── employee_db.pkl                  # (Generated) User database
```

### 4. Running the App
Once your models are in place, launch the application:
```
python app.py
```
## 💡 Usage Guide
Registration: Toggle "Active Scanning" to OFF. Enter a name in the sidebar and click Save Face.

Verification: Toggle "Active Scanning" to ON. The system will highlight your face:

Green: Recognized & Real.

Red: Spoof/Fake detected.

Orange: Real person but not in the database.

Liveness Toggle: You can disable the liveness check in the sidebar for testing purposes.

