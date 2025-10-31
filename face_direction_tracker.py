import cv2
import serial
import time

# Connect to Arduino (adjust COM port if needed)
arduino = serial.Serial('COM7', 9600, timeout=1)
time.sleep(2)  # Wait for connection to stabilize

# Load Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)
frame_center_x = cap.get(3) // 2  # Horizontal center of frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    direction = "No face detected"

    for (x, y, w, h) in faces:
        face_center_x = x + w // 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Determine direction and send command
        if face_center_x < frame_center_x - 50:
            arduino.write(b'R\n')
            direction = "Right"
        elif face_center_x > frame_center_x + 50:
            arduino.write(b'L\n')
            direction = "Left"
        else:
            arduino.write(b'S\n')
            direction = "Centered"

        break  # Only track the first face

    # Display direction on screen
    cv2.putText(frame, f"Direction: {direction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Face Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()