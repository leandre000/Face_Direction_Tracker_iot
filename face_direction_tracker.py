#!/usr/bin/env python3
"""
Face Direction Tracker with Arduino Stepper Control
------------------------------------------------
Author: Izere Shema Leandre
Description: 
    This system tracks face position in real-time using OpenCV and controls
    an Arduino-connected stepper motor based on face movement direction.
    When a face moves left or right, the stepper motor rotates accordingly.

Hardware Requirements:
    - Webcam
    - Arduino with 28BYJ-48 Stepper Motor
    - USB connection to Arduino

Dependencies:
    - OpenCV (cv2)
    - pyserial
    - time
"""

import cv2
import time
import serial

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
SERIAL_PORT = 'COM11'          # Arduino serial port (change as needed)
BAUD_RATE = 9600              # Serial communication speed
MOVEMENT_THRESHOLD = 8        # Minimum pixel movement to trigger direction change
SCALE_FACTOR = 1.3           # Face detection scaling
MIN_NEIGHBORS = 5            # Face detection minimum neighbors
MOTOR_ANGLE = 90            # Rotation angle for stepper motor

# -----------------------------------------------------------------------------
# Arduino Serial Setup
# -----------------------------------------------------------------------------
try:
    arduino = serial.Serial(port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for Arduino initialization
    print("[INFO] Connected to Arduino ✓")
except serial.SerialException as e:
    print(f"[ERROR] Failed to connect to Arduino on {SERIAL_PORT}: {e}")
    print("Please check your connection and port number.")
    exit(1)
# -----------------------------------------------------------------------------
# OpenCV Setup
# -----------------------------------------------------------------------------
# Initialize face detection cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("[ERROR] Failed to load face cascade classifier")
    arduino.close()
    exit(1)

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Failed to access webcam")
    arduino.close()
    exit(1)

# Initialize tracking variables
prev_center = None
prev_time = time.time()
direction = "CENTER"

print("[INFO] Face Left-Right Tracker Started (press 'q' to quit)")

# -----------------------------------------------------------------------------
# Main Processing Loop
# -----------------------------------------------------------------------------
while True:
    # Capture and check frame
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break
        
    # Convert to grayscale and detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS
    )
    # Find the largest face in view (assumed to be the primary subject)
    primary_face = None
    max_area = 0
    for (x, y, w, h) in faces:
        area = w * h
        if area > max_area:
            max_area = area
            primary_face = (x, y, w, h)
    
    # Process primary face if found
    if primary_face:
        # Extract face coordinates and compute center
        x, y, w, h = primary_face
        cx, cy = x + w // 2, y + h // 2
        
        # Draw face indicators
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Face boundary
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)  # Center point
        
        # Time delta for movement calculations
        current_time = time.time()
        dt = current_time - prev_time if prev_time else 1e-6
        
        # Determine direction and control motor
        if prev_center:
            # Calculate horizontal movement
            dx = cx - prev_center[0]
            
            # Check if movement exceeds threshold
            if abs(dx) > MOVEMENT_THRESHOLD:
                if dx > 0:
                    direction = "RIGHT"
                    # Send command to rotate motor right
                    arduino.write(f'rotate {MOTOR_ANGLE}\n'.encode())
                    print(f"[CMD] Sent → rotate {MOTOR_ANGLE} (RIGHT)")
                else:
                    direction = "LEFT"
                    # Send command to rotate motor left
                    arduino.write(f'rotate -{MOTOR_ANGLE}\n'.encode())
                    print(f"[CMD] Sent → rotate -{MOTOR_ANGLE} (LEFT)")
            else:
                direction = "CENTER"  # Face is relatively still
        # Update tracking variables
        prev_center = (cx, cy)
        prev_time = current_time
        
        # Display direction indicator
        text_color = (0, 255, 255) if direction in ["LEFT", "RIGHT"] else (200, 200, 200)
        text_thickness = 3 if direction in ["LEFT", "RIGHT"] else 2
        cv2.putText(frame, direction, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, text_thickness)
    else:
        # No face detected warning
        cv2.putText(frame, "NO FACE DETECTED", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow("Face-Controlled Stepper Motor", frame)
    
    # Check for quit command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Shutting down...")
        break

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
print("[INFO] Cleaning up resources...")
cap.release()
arduino.close()
cv2.destroyAllWindows()
print("[INFO] Application terminated successfully")