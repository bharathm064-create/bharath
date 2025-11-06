import cv2
import mediapipe as mp
import numpy as np
import threading
import platform
import os

# === SOUND ALERT FUNCTION ===
if platform.system() == "Windows":
    import winsound

    def play_alarm():
        winsound.Beep(2500, 1000)  # frequency, duration (ms)
else:
    from playsound import playsound

    def play_alarm():
        alarm_path = "alarm.mp3"
        if os.path.exists(alarm_path):
            playsound(alarm_path)
        else:
            print("[WARNING] alarm.mp3 file not found!")

# === Mediapipe FaceMesh initialization ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === EAR (Eye Aspect Ratio) calculation ===
def eye_aspect_ratio(landmarks, eye_indices):
    p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])

    # Calculate EAR using vertical and horizontal distances
    ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))
    return ear

# === Eye landmark indices for Mediapipe FaceMesh ===
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# === Thresholds and counters ===
EAR_THRESHOLD = 0.25         # EAR below this → eyes closed
CLOSED_FRAMES = 20           # Number of consecutive frames for drowsiness
blink_count = 0
closed_counter = 0
drowsy = False

# === Start webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Starting Eye Blink & Drowsiness Detection... Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read from webcam.")
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0

            # --- Drowsiness Detection ---
            if ear < EAR_THRESHOLD:
                closed_counter += 1
                if closed_counter >= CLOSED_FRAMES:
                    if not drowsy:
                        drowsy = True
                        threading.Thread(target=play_alarm, daemon=True).start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                if 1 < closed_counter < CLOSED_FRAMES:
                    blink_count += 1
                closed_counter = 0
                drowsy = False

            # --- Display info on frame ---
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {blink_count}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Eye Blink & Drowsiness Detector", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        print("[INFO] Exiting...")
        break

# === Clean up ===
cap.release()
cv2.destroyAllWindows()
