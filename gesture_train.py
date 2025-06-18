import cv2
import mediapipe as mp
import numpy as np
import pickle

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

X_data = []
y_data = []

# Label for gesture - change this to collect other gestures
label = 'peace'

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    return None

# Start capturing from webcam
cap = cv2.VideoCapture(0)
print("Press 'c' to capture, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    landmarks = extract_landmarks(frame)

    if landmarks:
        mp_draw.draw_landmarks(frame, hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks[0],
                               mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f'Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Capture", frame)

    key = cv2.waitKey(1)
    if key == ord('c') and landmarks:
        X_data.append(landmarks)
        y_data.append(label)
        print(f"Captured gesture: {label}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save data
with open("gesture_data.pkl", "wb") as f:
    pickle.dump((X_data, y_data), f)

print("Data saved to gesture_data.pkl")
