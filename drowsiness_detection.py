# drowsiness_detection.py

import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pygame
import os
import time

# Initialize Pygame for alarm sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("audio/alert.wav")

# Load dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
PERSON_DETECTION_THRESHOLD = 1  # Threshold to play alarm for one person
CAPTURE_FOLDER = "captures"  # Folder to save captured images

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

# Class that encapsulates drowsiness detection logic
class DrowsinessDetector:
    def __init__(self, update_count_callback):
        self.detected_people_count = 0
        self.update_count_callback = update_count_callback
        self.saved_this_frame = False  # Flag to track if an image has been saved in this frame

    def save_frame(self, frame, count):
        """Save the frame as an image in the captures folder"""
        if self.saved_this_frame:
            return  # Skip saving if a photo was already saved

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(CAPTURE_FOLDER, f"drowsy_{timestamp}_{count}.jpg")
        cv2.imwrite(file_path, frame)
        print(f"Image saved as {file_path}")
        self.saved_this_frame = True  # Set flag to indicate an image has been saved

    def detect_drowsiness(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            detected_people = 0

            for face in faces:
                detected_people += 1
                landmarks = predictor(gray, face)

                # Extract eye landmarks
                left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
                ear_left = calculate_ear(left_eye)
                ear_right = calculate_ear(right_eye)
                ear = (ear_left + ear_right) / 2.0

                # Extract mouth landmarks
                mouth = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
                mar = calculate_mar(mouth)

                # Display messages and play alarm if conditions are met
                if ear < EAR_THRESHOLD:
                    cv2.putText(frame, "Drowsy: Eyes Closed!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    play_alarm()
                    self.save_frame(frame, detected_people)  # Save image when eyes are closed
                if mar > MAR_THRESHOLD:
                    cv2.putText(frame, "Drowsy: Yawning!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    play_alarm()
                    self.save_frame(frame, detected_people)  # Save image when yawning

            # Update the detected people count
            self.update_count_callback(detected_people)

            # Show the number of detected people
            cv2.putText(frame, f"People Detected: {detected_people}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.imshow("Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Reset the save flag after a small delay
            time.sleep(2)  # Wait for 2 seconds before allowing another save
            self.saved_this_frame = False

        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
