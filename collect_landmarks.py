import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

CSV_FILE = "gesture_landmarks.csv"
CLASSES = ["fist", "open_hand", "peacehand", "thumbs_up"]
SAMPLES_PER_CLASS = 200

# Create CSV header
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
        header.append("label")
        writer.writerow(header)


def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords


cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    for class_name in CLASSES:
        collected = 0
        print(f"\n=== Get ready to record: {class_name} ===")
        print("Press SPACE to start recording 200 samples.")
        print("Press Q to skip.")

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Next: {class_name} - Press SPACE",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
            cv2.imshow("Collect", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            if key == ord('q'):
                break

        while collected < SAMPLES_PER_CLASS:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                coords = extract_landmarks(hand)
                with open(CSV_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(coords + [class_name])
                collected += 1
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"{class_name}: {collected}/{SAMPLES_PER_CLASS}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
            cv2.imshow("Collect", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
print("\nDone. Saved to", CSV_FILE)