# Eye Blink & Drowsiness Detection System

This Python project uses **OpenCV** and **MediaPipe** to detect eye blinks and monitor drowsiness in real-time through your webcam.  
If prolonged eye closure is detected, the system triggers an alarm sound to alert the user.

---

## 🧠 Features

- Detects **eye blinks** and counts them in real-time.  
- Detects **drowsiness** based on the Eye Aspect Ratio (EAR).  
- Plays an **alarm sound** when eyes remain closed for a specific duration.  
- Works on **Windows**, **macOS**, and **Linux** (with slight sound differences).

---

## 🖥️ Demo

When running, the app displays your webcam feed with:
- The **EAR (Eye Aspect Ratio)** value.
- The **Blink Count**.
- A **“DROWSINESS ALERT!”** warning when eyes are closed for too long.
- An **alarm sound** for safety alert.

---

## ⚙️ Requirements

Make sure you have Python **3.8+** installed.

### Install Dependencies:
```bash
pip install opencv-python mediapipe numpy playsound
