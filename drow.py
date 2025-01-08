import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pygame
import threading

# Initialize Pygame for sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("audio/alert.wav")  
alarm_channel = pygame.mixer.Channel(0)  # Use a dedicated channel for the alarm

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

# Function to calculate head tilt using nose and chin landmarks
def calculate_head_tilt(nose, chin):
    A = np.array(nose)
    B = np.array(chin)
    angle = np.degrees(np.arctan2(B[1] - A[1], B[0] - A[0]))
    return angle

# Alarm function
def play_alarm():
    if not alarm_channel.get_busy():  # Check if the alarm is already playing
        alarm_channel.play(alarm_sound)

# EAR, MAR, and Head Tilt thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
HEAD_TILT_THRESHOLD = 20  # Angle threshold for head tilt (in degrees)
CONSEC_FRAMES_EAR = 15
CONSEC_FRAMES_MAR = 15

frame_count_ear = 0
frame_count_mar = 0
frame_count_head = 0

alarm_on_ear = False
alarm_on_mar = False
alarm_on_head = False

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

        # Extract nose and chin landmarks for head position
        nose = (landmarks.part(30).x, landmarks.part(30).y)  # Nose tip
        chin = (landmarks.part(8).x, landmarks.part(8).y)    # Chin

        # Calculate EAR, MAR, and head tilt
        ear_left = calculate_ear(left_eye)
        ear_right = calculate_ear(right_eye)
        ear = (ear_left + ear_right) / 2.0
        mar = calculate_mar(mouth)
        head_tilt_angle = calculate_head_tilt(nose, chin)

        # Visualize landmarks
        for point in left_eye + right_eye + mouth:
            cv2.circle(frame, point, 2, (255, 0, 0), -1)
        cv2.circle(frame, nose, 2, (0, 255, 0), -1)
        cv2.circle(frame, chin, 2, (0, 255, 0), -1)

        # Detect eye closure
        if ear < EAR_THRESHOLD:
            frame_count_ear += 1
            if frame_count_ear >= CONSEC_FRAMES_EAR and not alarm_on_ear:
                cv2.putText(frame, "Drowsy: Eyes Closed!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                alarm_on_ear = True
                threading.Thread(target=play_alarm, daemon=True).start()
        else:
            frame_count_ear = 0
            alarm_on_ear = False

        # Detect yawning
        if mar > MAR_THRESHOLD:
            frame_count_mar += 1
            if frame_count_mar >= CONSEC_FRAMES_MAR and not alarm_on_mar:
                cv2.putText(frame, "Drowsy: Yawning!", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                alarm_on_mar = True
                threading.Thread(target=play_alarm, daemon=True).start()
        else:
            frame_count_mar = 0
            alarm_on_mar = False

        # Detect head tilt
        if abs(head_tilt_angle) > HEAD_TILT_THRESHOLD:
            frame_count_head += 1
            if frame_count_head >= CONSEC_FRAMES_EAR and not alarm_on_head:
                cv2.putText(frame, "Drowsy: Head Tilt!", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                alarm_on_head = True
                threading.Thread(target=play_alarm, daemon=True).start()
        else:
            frame_count_head = 0
            alarm_on_head = False

    # Reset alarm only when no conditions are active
    if not (alarm_on_ear or alarm_on_mar or alarm_on_head):
        alarm_channel.stop()

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
