import cv2
import time

# Face Direction & Speed Tracker
# Author: Shema Leandre

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

prev_center = None
prev_time = time.time()
direction = "Center"
speed = 0

print("[INFO] Starting Face Direction Tracker...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera not accessible.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

        current_time = time.time()
        dt = current_time - prev_time if prev_time else 0.0001

        if prev_center:
            dx = cx - prev_center[0]
            dy = cy - prev_center[1]
            if abs(dx) > abs(dy):
                direction = "Right" if dx > 0 else "Left"
            elif abs(dy) > abs(dx):
                direction = "Down" if dy > 0 else "Up"
            else:
                direction = "Center"

            distance = (dx**2 + dy**2)**0.5
            speed = distance / dt

        prev_center = (cx, cy)
        prev_time = current_time

        cv2.putText(frame, f"Dir: {direction}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Speed: {speed:.2f}px/s", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Face Direction Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
