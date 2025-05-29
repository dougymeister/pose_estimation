import torch
from ultralytics import YOLO
import cv2
import numpy as np

def compute_angle(A, B, C):
    """Compute angle at point B formed by points A, B, C."""
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    BA = A - B
    BC = C - B
    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle_rad)

# Load YOLOv8 pose model
device = 'cpu'  # Force CPU; use 'cuda' if available
print(f"Using device: {device}")
model = YOLO('yolov8n-pose.pt').to(device)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run pose estimation
    results = model(frame, device=device)
    annotated_frame = results[0].plot()

    # Extract keypoints for first person detected
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()
        for kp in keypoints:
            # Example: Left arm angle (shoulder-elbow-wrist)
            shoulder = kp[5]
            elbow = kp[7]
            wrist = kp[9]
            angle_elbow = compute_angle(shoulder, elbow, wrist)

            # Draw lines and angle
            cv2.line(annotated_frame, tuple(shoulder.astype(int)), tuple(elbow.astype(int)), (255, 0, 0), 2)
            cv2.line(annotated_frame, tuple(wrist.astype(int)), tuple(elbow.astype(int)), (255, 0, 0), 2)
            cv2.putText(annotated_frame, f'{int(angle_elbow)} deg', tuple(elbow.astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # Example: Left leg angle (hip-knee-ankle)
            hip = kp[11]
            knee = kp[13]
            ankle = kp[15]
            angle_knee = compute_angle(hip, knee, ankle)

            cv2.line(annotated_frame, tuple(hip.astype(int)), tuple(knee.astype(int)), (0, 255, 0), 2)
            cv2.line(annotated_frame, tuple(ankle.astype(int)), tuple(knee.astype(int)), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'{int(angle_knee)} deg', tuple(knee.astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Pose Estimation with Angles', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
