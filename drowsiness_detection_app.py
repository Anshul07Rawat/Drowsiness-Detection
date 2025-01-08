import tkinter as tk
from tkinter import messagebox, ttk
from threading import Thread
import cv2
import dlib
from scipy.spatial import distance
import pygame
import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Initialize Pygame for alarm sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("audio/alert.wav")

# Load dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
CAPTURE_FOLDER = "captures" 

# Email Configuration
EMAIL_ADDRESS = "anshulrawatansh1432@gmail.com"  
EMAIL_PASSWORD = "gwbetmceswpbpyam"  
SMTP_SERVER = "smtp.gmail.com"  
SMTP_PORT = 587

# Create captures folder if it doesn't exist
if not os.path.exists(CAPTURE_FOLDER):
    os.makedirs(CAPTURE_FOLDER)

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate Mouth Aspect Ratio (MAR)
def calculate_mar(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# Function to play alarm sound
def play_alarm():
    pygame.mixer.Sound.play(alarm_sound)

# Function to send an email
def send_email_notification(sender_name, recipient_email):
    try:
        subject = "Drowsiness Alert!"
        body = f"Alert: {sender_name} appears to be drowsy. Please check on them immediately."

        # Construct email
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

class DrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowsiness Detection App")
        self.root.geometry("600x400")
        self.root.resizable(False, False)

        # Input Fields for User Name and Recipient Name
        tk.Label(root, text="Enter Your Name:", font=("Arial", 14)).pack(pady=5)
        self.user_name_entry = tk.Entry(root, font=("Arial", 14))
        self.user_name_entry.pack(pady=5)
        

        tk.Label(root, text="Enter Recipient Email:", font=("Arial", 14)).pack(pady=5)
        self.recipient_email_entry = tk.Entry(root, font=("Arial", 14))
        self.recipient_email_entry.pack(pady=5)

        # Buttons
        self.start_button = ttk.Button(root, text="Start Detection", command=self.start_detection, width=20)
        self.start_button.pack(pady=20)
        

        self.stop_button = ttk.Button(root, text="Stop Detection", command=self.stop_detection, width=20)
        self.stop_button.pack(pady=10)
        self.stop_button.config(state=tk.DISABLED)

        self.exit_button = ttk.Button(root, text="Exit", command=root.quit, width=20)
        self.exit_button.pack(pady=10)

        self.running = False
        self.detection_thread = None

    def start_detection(self):
        user_name = self.user_name_entry.get()
        recipient_email = self.recipient_email_entry.get()

        if not user_name or not recipient_email:
            messagebox.showerror("Error", "Please enter both your name and the recipient's email.")
            return

        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.detection_thread = Thread(target=self.detect_drowsiness, args=(user_name, recipient_email))
        self.detection_thread.start()

    def stop_detection(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def detect_drowsiness(self, user_name, recipient_email):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)

                # Calculate EAR and MAR
                left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
                mouth = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]

                ear_left = calculate_ear(left_eye)
                ear_right = calculate_ear(right_eye)
                ear = (ear_left + ear_right) / 2.0
                mar = calculate_mar(mouth)

                if ear < EAR_THRESHOLD or mar > MAR_THRESHOLD:
                    cv2.putText(frame, "Drowsiness Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Save the frame as an image
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    image_path = os.path.join(CAPTURE_FOLDER, f"drowsy_{timestamp}.jpg")
                    cv2.imwrite(image_path, frame)
                    print(f"Image saved at {image_path}")

                    # Play alarm and send email notification
                    play_alarm()
                    send_email_notification(user_name, recipient_email)

                    time.sleep(2)  # Pause before the next alert to prevent spam

            cv2.imshow("Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Main Function
if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessApp(root)
    root.mainloop()
