'''This script detects if a person is drowsy or not,using dlib and eye aspect ratio,mouth aspect ration and head tilt 
calculations. Uses webcam video feed as input.'''

#Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import pygame #For playing sound
import time
import dlib
import cv2
import threading

pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("audio/alert.wav")  # Replace with your sound file

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])  # Vertical distance
    B = distance.euclidean(eye[2], eye[4])  # Vertical distance
    C = distance.euclidean(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def calculate_mar(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # Vertical distance
    B = distance.euclidean(mouth[4], mouth[8])   # Vertical distance
    C = distance.euclidean(mouth[0], mouth[6])   # Horizontal distance
    mar = (A + B) / (2.0 * C)
    return mar

# Alarm function
def play_alarm():
    alarm_sound.play()

# EAR and MAR thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
CONSEC_FRAMES_EAR = 15
CONSEC_FRAMES_MAR = 15

frame_count_ear = 0
frame_count_mar = 0
alarm_on = False

# Load facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Extract eye landmarks
        left_eye = [landmarks.part(i) for i in range(36, 42)]
        right_eye = [landmarks.part(i) for i in range(42, 48)]
        left_eye = [(point.x, point.y) for point in left_eye]
        right_eye = [(point.x, point.y) for point in right_eye]

        # Extract mouth landmarks
        mouth = [landmarks.part(i) for i in range(48, 68)]
        mouth = [(point.x, point.y) for point in mouth]

        # Calculate EAR and MAR
        ear_left = calculate_ear(left_eye)
        ear_right = calculate_ear(right_eye)
        ear = (ear_left + ear_right) / 2.0
        mar = calculate_mar(mouth)

        # Visualize landmarks
        for point in left_eye + right_eye + mouth:
            cv2.circle(frame, point, 2, (255, 0, 0), -1)
        
        # Detect eye closure
        if ear < EAR_THRESHOLD:
            frame_count_ear += 1
            if frame_count_ear >= CONSEC_FRAMES_EAR and not alarm_on:
                cv2.putText(frame, "Drowsy: Eyes Closed!", (100, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                alarm_on = True
                threading.Thread(target=play_alarm, daemon=True).start()
        else:
            frame_count_ear = 0

        # Detect yawning
        if mar > MAR_THRESHOLD:
            frame_count_mar += 1
            if frame_count_mar >= CONSEC_FRAMES_MAR and not alarm_on:
                cv2.putText(frame, "Drowsy: Yawning!", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                alarm_on = True
                threading.Thread(target=play_alarm, daemon=True).start()
        else:
            frame_count_mar = 0

        # Reset alarm when no drowsiness detected
        if frame_count_ear == 0 and frame_count_mar == 0:
            alarm_on = False

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()