import cv2
import time
import numpy as np
import serial
import serial.tools.list_ports

# Constants for face detection
SCALE_FACTOR = 1.1  # Lower = more thorough (slower but more sensitive)
MIN_NEIGHBORS = 3  # Lower = more detections (may have false positives)
MIN_FACE_SIZE = (30, 30)  # Minimum face size to detect
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Movement detection thresholds
MOVEMENT_THRESHOLD = 10  # Minimum pixels of movement to trigger rotation

# Serial communication settings
BAUD_RATE = 9600

# Rotation angles
LEFT_ROTATION = -25  # Degrees to rotate when moving left
RIGHT_ROTATION = 25  # Degrees to rotate when moving right

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def find_arduino_port():
    """Auto-detect Arduino port"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if 'Arduino' in port.description or 'USB' in port.description or 'ACM' in port.device or 'ttyUSB' in port.device:
            print(f"[INFO] Found Arduino on port: {port.device}")
            return port.device
    return None

def init_serial():
    """Initialize serial connection to Arduino"""
    port = find_arduino_port()
    
    if port is None:
        print("[WARNING] Arduino not found. Running in simulation mode.")
        return None
    
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset
        print(f"[INFO] Connected to Arduino on {port}")
        return ser
    except Exception as e:
        print(f"[ERROR] Failed to connect to Arduino: {e}")
        return None

def send_rotation(ser, angle):
    """Send rotation angle to Arduino"""
    if ser is None:
        print(f"[SIMULATION] Would rotate: {angle} degrees")
        return
    
    try:
        command = f"rotate {angle}\n"
        ser.write(command.encode())
        print(f"rotate {angle}")
    except Exception as e:
        print(f"[ERROR] Failed to send command: {e}")

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot access camera.")
    exit(1)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Initialize serial connection
arduino = init_serial()

# Variables for tracking
prev_time = time.time()
prev_center = None  # Previous face center position
last_command = "Center"
last_command_time = 0
command_cooldown = 0.3  # Seconds between commands

print("[INFO] Starting Face Direction Tracker (OpenCV only)...")
print("[INFO] Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame from camera.")
        break

    # Flip frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect faces with improved parameters
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_FACE_SIZE,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    motor_command = "Center"
    rotation_angle = 0
    movement_direction = "None"
    
    if len(faces) > 0:
        # Use the largest face detected
        face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = face
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Calculate center
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        
        # Detect movement if we have a previous position
        if prev_center is not None:
            dx = cx - prev_center[0]  # Horizontal movement
            
            # Check if movement exceeds threshold
            if abs(dx) > MOVEMENT_THRESHOLD:
                if dx > 0:
                    # Moving right
                    motor_command = "Right"
                    rotation_angle = RIGHT_ROTATION
                    movement_direction = f"Right ({int(dx)}px)"
                else:
                    # Moving left
                    motor_command = "Left"
                    rotation_angle = LEFT_ROTATION
                    movement_direction = f"Left ({int(abs(dx))}px)"
                
                # Draw movement arrow
                cv2.arrowedLine(frame, prev_center, (cx, cy), (0, 255, 255), 2, tipLength=0.3)
            else:
                motor_command = "Center"
                rotation_angle = 0
                movement_direction = "Stationary"
        else:
            movement_direction = "Initializing"
        
        # Update previous center
        prev_center = (cx, cy)
        
        # Send rotation command (with cooldown)
        current_time = time.time()
        if motor_command != last_command or (current_time - last_command_time) > command_cooldown:
            if rotation_angle != 0:
                send_rotation(arduino, rotation_angle)
            last_command = motor_command
            last_command_time = current_time

        # Display information
        cv2.putText(frame, f"Motor: {motor_command} ({rotation_angle}°)", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Movement: {movement_direction}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)
    else:
        # No faces detected - reset tracking
        prev_center = None
        cv2.putText(frame, "No Face Detected", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display FPS and connection status
    fps = 1.0 / (time.time() - prev_time) if prev_time else 0
    cv2.putText(frame, f"FPS: {fps:.1f}", (FRAME_WIDTH - 120, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    status_color = (0, 255, 0) if arduino else (0, 0, 255)
    status_text = "Connected" if arduino else "Simulation"
    cv2.putText(frame, status_text, (FRAME_WIDTH - 150, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

    cv2.imshow("Face Direction Tracker (OpenCV)", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
    print("[INFO] Serial connection closed")