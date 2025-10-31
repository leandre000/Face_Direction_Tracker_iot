import cv2
import time

# --- Load built-in Haar Cascade face detector ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --- Open webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot access webcam")
    exit()

# --- Get frame center ---
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
center_x, center_y = frame_width // 2, frame_height // 2

# --- Variables to track motion speed ---
prev_face_center = None
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    direction = "CENTER"
    speed = 0

    for (x, y, w, h) in faces:
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Compute center of face
        face_center = (x + w // 2, y + h // 2)
        cv2.circle(frame, face_center, 5, (0, 0, 255), -1)

        # --- Determine direction relative to frame center ---
        dx = face_center[0] - center_x
        dy = face_center[1] - center_y

        if abs(dx) > 40:
            direction = "RIGHT" if dx > 0 else "LEFT"
        elif abs(dy) > 40:
            direction = "DOWN" if dy > 0 else "UP"
        else:
            direction = "CENTER"

        # --- Estimate speed based on pixel movement per second ---
        current_time = time.time()
        if prev_face_center is not None:
            distance = ((face_center[0]-prev_face_center[0])**2 +
                        (face_center[1]-prev_face_center[1])**2) ** 0.5
            dt = current_time - prev_time
            if dt > 0:
                speed = round(distance / dt, 2)
        prev_face_center = face_center
        prev_time = current_time

        break  # track only the first face detected

    # --- Draw frame center ---
    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

    # --- Display direction and speed ---
    cv2.putText(frame, f"Dir: {direction}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Speed: {speed:.2f}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Phase 1 - Face Tracking", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()