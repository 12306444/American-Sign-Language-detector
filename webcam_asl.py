import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter
import time

# Load model
model = tf.keras.models.load_model("asl_model.h5")

# Class labels
class_labels = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'DEL','NOTHING','SPACE'
]

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Prediction buffer
prediction_buffer = deque(maxlen=12)
stable_label = ""

# Word formation
output_text = ""
last_added_time = time.time()
delay = 1.2  # seconds
last_letter = ""

CONFIDENCE_THRESHOLD = 0.75

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Tight crop
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size != 0:
                hand_img = cv2.resize(hand_img, (64, 64))
                hand_img = hand_img / 255.0
                hand_img = np.reshape(hand_img, (1, 64, 64, 3))

                prediction = model.predict(hand_img, verbose=0)[0]
                confidence = np.max(prediction)
                predicted_class = np.argmax(prediction)

                if confidence > CONFIDENCE_THRESHOLD:
                    prediction_buffer.append(predicted_class)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                          (0, 255, 0), 2)

    # Determine stable letter
    if len(prediction_buffer) == 12:
        most_common = Counter(prediction_buffer).most_common(1)[0][0]
        stable_label = class_labels[most_common]

    # Add letter logic
    current_time = time.time()
    if (
        stable_label not in ["NOTHING", ""] and
        stable_label != last_letter and
        current_time - last_added_time > delay
    ):
        if stable_label == "SPACE":
            output_text += " "
        elif stable_label == "DEL":
            output_text = output_text[:-1]
        else:
            output_text += stable_label

        last_added_time = current_time
        last_letter = stable_label
        prediction_buffer.clear()

    # Display letter
    cv2.putText(frame, f"Letter: {stable_label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display word
    cv2.putText(frame, f"Word: {output_text}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
