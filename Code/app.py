import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import os
import pickle
import numpy as np
import tensorflow as tf
import threading
import time

# --- CONFIGURATION ---
RECOGNITION_MODEL_PATH = '../Models/final_face_embedding_model.keras'
LIVENESS_MODEL_PATH = '../Models/liveness_model.keras'
EMOTION_MODEL_PATH = '../Models/emotion_model.keras'   
EMOTION_LABELS_PATH = 'emotion_labels.pkl' 
DB_PATH = 'employee_db.pkl'
THRESHOLD = 2.3613

class FaceAttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("Secure Attendance System (Liveness + Recognition + Emotion)")
        self.geometry("1000x750")
        ctk.set_appearance_mode("Dark")
        
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Load Models ---
        print("Loading Models... (This may take a moment)")
        
        # 1. Face Recognition (ResNet50)
        self.recog_model = tf.keras.models.load_model(RECOGNITION_MODEL_PATH)
        
        # 2. Liveness (MobileNetV2)
        self.liveness_model = tf.keras.models.load_model(LIVENESS_MODEL_PATH)
        
        # 3. Emotion (MobileNetV2)
        self.emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH)
        with open(EMOTION_LABELS_PATH, 'rb') as f:
            self.emotion_labels = pickle.load(f)
        
        print("✅ All Models Loaded!")
        
        self.db = self.load_database()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # --- NEW: Emotion Icons Map ---
        self.emotion_icons = {
            "angry": "😠",
            "disgust": "🤢",
            "fear": "😨",
            "happy": "😊",
            "neutral": "😐",
            "sad": "😢",
            "surprise": "😲"
        }
        
        # State Variables
        self.verification_active = False
        self.liveness_enabled = True
        self.current_face_frame = None
        
        # Detection Results
        self.liveness_score = 0.0
        self.is_real = False
        self.detected_name = "None"
        self.detected_dist = 0.0
        self.detected_emotion = "Neutral" # Default
        
        # Threading
        self.last_check_time = 0
        self.is_processing = False

        self.setup_ui()
        self.cap = cv2.VideoCapture(0)
        self.update_video_loop()

    def setup_ui(self):
        # Video Area
        self.video_frame = ctk.CTkFrame(self, corner_radius=10)
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="Security Control", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20, 10))

        # Status Panel
        self.status_frame = ctk.CTkFrame(self.sidebar, fg_color="#2B2B2B")
        self.status_frame.pack(padx=20, pady=10, fill="x")
        
        self.result_label = ctk.CTkLabel(self.status_frame, text="ID: --", font=("Arial", 16, "bold"))
        self.result_label.pack(pady=(10, 2))
        
        self.liveness_label = ctk.CTkLabel(self.status_frame, text="Liveness: --", font=("Arial", 14))
        self.liveness_label.pack(pady=(2, 2))

        self.emotion_label_ui = ctk.CTkLabel(self.status_frame, text="Emotion: --", font=("Arial", 14), text_color="#FFD700")
        self.emotion_label_ui.pack(pady=(2, 10))

        # Mode Switch Layout
        self.switch_container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.switch_container.pack(pady=20)
        self.reg_mode_label = ctk.CTkLabel(self.switch_container, text="Register", text_color="white")
        self.reg_mode_label.pack(side="left", padx=(0, 10))
        self.verify_switch = ctk.CTkSwitch(self.switch_container, text="Active Scanning", command=self.toggle_verification)
        self.verify_switch.pack(side="left")

        # Registration
        ctk.CTkLabel(self.sidebar, text="Register User Input", anchor="w").pack(padx=20, pady=(10, 0), anchor="w")
        self.name_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Enter Name")
        self.name_entry.pack(padx=20, pady=5, fill="x")
        self.register_btn = ctk.CTkButton(self.sidebar, text="Save Face", command=self.register_user, fg_color="green")
        self.register_btn.pack(padx=20, pady=10, fill="x")
        
        # Delete
        self.delete_btn = ctk.CTkButton(self.sidebar, text="Delete User", command=self.delete_user, fg_color="#D32F2F")
        self.delete_btn.pack(padx=20, pady=5, fill="x")

        self.log_label = ctk.CTkLabel(self.sidebar, text="", text_color="gray")
        self.log_label.pack(pady=5)

        # Liveness Toggle
        self.liveness_switch = ctk.CTkSwitch(self.sidebar, text="Liveness Check", command=self.toggle_liveness)
        self.liveness_switch.select()
        self.liveness_switch.pack(side="bottom", padx=20, pady=20, anchor="s")

    def draw_corner_rect(self, img, bbox, color=(255, 255, 255), thickness=3):
        x, y, w, h = bbox
        length = int(w * 0.2)
        cv2.line(img, (x, y), (x + length, y), color, thickness)
        cv2.line(img, (x, y), (x, y + length), color, thickness)
        cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
        cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)
        cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
        cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)
        cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
        cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)
        return img

    # --- AI Logic ---
    def predict_emotion(self, face_img):
        # Preprocess: Resize + Normalize to [0, 1]
        img = cv2.resize(face_img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0  # Manual Normalization
        img = np.expand_dims(img, axis=0)
        
        preds = self.emotion_model.predict(img, verbose=0)[0]
        idx = np.argmax(preds)
        return self.emotion_labels[idx]

    def check_liveness(self, face_img):
        # Preprocess: MobileNetV2 standard
        img = cv2.resize(face_img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype(np.float32))
        img = np.expand_dims(img, axis=0)
        return self.liveness_model.predict(img, verbose=0)[0][0]

    def get_embedding(self, face_img):
        # Preprocess: ResNet50 standard
        img = cv2.resize(face_img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.keras.applications.resnet50.preprocess_input(img.astype(np.float32))
        img = np.expand_dims(img, axis=0)
        return self.recog_model.predict(img, verbose=0)[0]

    def process_frame_task(self, face_img):
        try:
            # 1. Check Liveness
            if self.liveness_enabled:
                spoof_score = self.check_liveness(face_img)
                self.liveness_score = spoof_score
                # Note: Adjust this logic if your Real class is 1
                if spoof_score < 0.5:
                    self.is_real = False
                    self.detected_name = "SPOOF"
                    self.detected_emotion = "--" 
                    return 
                self.is_real = True
            else:
                self.is_real = True
                self.liveness_score = 0.0

            # 2. Predict Emotion (Since face is real)
            self.detected_emotion = self.predict_emotion(face_img)

            # 3. Check Identity
            if not self.db:
                self.detected_name = "Unknown"
                return

            current_emb = self.get_embedding(face_img)
            min_dist = float('inf')
            identity = "Unknown"

            for name, db_emb in self.db.items():
                dist = np.linalg.norm(current_emb - db_emb)
                if dist < min_dist:
                    min_dist = dist
                    identity = name

            self.detected_dist = min_dist
            if min_dist < THRESHOLD:
                self.detected_name = identity
            else:
                self.detected_name = "Unknown"

        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.is_processing = False

    # --- Main Loop ---
    def update_video_loop(self):
        ret, frame = self.cap.read()
        
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                (x, y, w, h) = largest_face
                self.current_face_frame = frame[y:y+h, x:x+w]

                # Visual Logic
                if not self.verification_active:
                    box_color = (255, 255, 255) # White
                else:
                    if self.liveness_enabled and not self.is_real:
                        box_color = (0, 0, 255) # Red (Spoof)
                    elif self.detected_name == "Unknown":
                        box_color = (0, 165, 255) # Orange (Unknown)
                    else:
                        box_color = (0, 255, 0) # Green (Verified)

                frame = self.draw_corner_rect(frame, (x, y, w, h), color=box_color)
                
                # Draw Emotion Text (Text Only for OpenCV)
                if self.verification_active and self.is_real and self.detected_emotion != "--":
                    cv2.putText(frame, self.detected_emotion, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

            else:
                self.current_face_frame = None
                self.detected_name = "None"
                self.detected_emotion = "--"
                self.is_real = False

            # UI Display
            cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv_img)
            img_tk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)

            # AI Trigger
            if self.verification_active and self.current_face_frame is not None:
                if (time.time() - self.last_check_time > 0.5) and not self.is_processing:
                    self.last_check_time = time.time()
                    self.is_processing = True
                    threading.Thread(target=self.process_frame_task, args=(self.current_face_frame.copy(),)).start()

                # Update Sidebar Labels
                if self.detected_name == "SPOOF":
                    self.liveness_label.configure(text=f"Liveness: FAKE ({self.liveness_score:.2f})", text_color="red")
                    self.result_label.configure(text="ID: BLOCKED", text_color="red")
                    self.emotion_label_ui.configure(text="Emotion: --", text_color="gray")
                
                elif self.detected_name == "None":
                    live_text = "Liveness: --" if self.liveness_enabled else "Liveness: OFF"
                    self.liveness_label.configure(text=live_text, text_color="gray")
                    self.result_label.configure(text="ID: --", text_color="gray")
                    self.emotion_label_ui.configure(text="Emotion: --", text_color="gray")
                
                else:
                    # Valid detection
                    if self.liveness_enabled:
                        self.liveness_label.configure(text=f"Liveness: REAL ({self.liveness_score:.2f})", text_color="green")
                    else:
                        self.liveness_label.configure(text="Liveness: OFF (Bypass)", text_color="orange")
                    
                    color = "green" if self.detected_name != "Unknown" else "orange"
                    self.result_label.configure(text=f"ID: {self.detected_name}", text_color=color)
                    
                    # --- UPDATED: Show Icon in Sidebar ---
                    # Lowercase for dict lookup, Capitalize for display
                    emotion_key = self.detected_emotion.lower()
                    icon = self.emotion_icons.get(emotion_key, "")
                    display_text = f"Emotion: {self.detected_emotion.capitalize()} {icon}"
                    self.emotion_label_ui.configure(text=display_text, text_color="#FFD700")

        self.after(10, self.update_video_loop)

    # --- Helpers ---
    def load_database(self):
        if os.path.exists(DB_PATH):
            with open(DB_PATH, 'rb') as f: return pickle.load(f)
        return {}
    def save_database(self):
        with open(DB_PATH, 'wb') as f: pickle.dump(self.db, f)
    def delete_user(self):
        dialog = ctk.CTkInputDialog(text="Enter name to delete:", title="Delete User")
        name = dialog.get_input()
        if name and name in self.db:
            del self.db[name]
            self.save_database()
            self.log_label.configure(text=f"Deleted: {name}", text_color="red")
        elif name:
            self.log_label.configure(text="User not found", text_color="red")
    def register_user(self):
        name = self.name_entry.get()
        if self.current_face_frame is not None and name:
            emb = self.get_embedding(self.current_face_frame)
            self.db[name] = emb
            self.save_database()
            self.log_label.configure(text=f"Registered: {name}", text_color="green")
            self.name_entry.delete(0, 'end')
    def toggle_verification(self):
        self.verification_active = self.verify_switch.get()
        if self.verification_active:
            self.reg_mode_label.configure(text_color="gray")
            self.verify_switch.configure(text_color="white")
        else:
            self.reg_mode_label.configure(text_color="white")
            self.verify_switch.configure(text_color="gray")
            self.result_label.configure(text="ID: --", text_color="white")
    def toggle_liveness(self):
        self.liveness_enabled = self.liveness_switch.get()
        if not self.liveness_enabled: self.liveness_label.configure(text="Liveness: OFF", text_color="gray")
    def on_closing(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = FaceAttendanceApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()