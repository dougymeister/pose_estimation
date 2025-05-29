import torch
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd


def compute_angle(A, B, C):
    """Compute angle at point B formed by points A, B, C."""
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    BA = A - B
    BC = C - B
    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)

    if norm_BA == 0 or norm_BC == 0:
        return np.nan  # Undefined angle

    cosine_angle = np.dot(BA, BC) / (norm_BA * norm_BC)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)


def draw_angle_arc(image, B, angle_deg, radius=30, color=(255, 255, 255)):
    """Draw an angle arc at point B."""
    if np.isnan(angle_deg):
        return  # Skip drawing for undefined angles
    start_angle = 0
    end_angle = int(angle_deg)
    cv2.ellipse(image, tuple(B.astype(int)), (radius, radius), 0, start_angle, end_angle, color, 2)


# Load YOLOv8 pose model
device = 'cpu'  # Use 'cuda' for GPU if available
print(f"Using device: {device}")
model = YOLO('yolov8n-pose.pt').to(device)

# Load image
image_path = 'cyclist1.jpg'  # Replace with your image file path
frame = cv2.imread(image_path)
if frame is None:
    print(f"Error: Could not read image from {image_path}")
    exit()

# Run pose estimation
results = model(frame, device=device)
annotated_frame = results[0].plot()

# COCO keypoint names
keypoint_names = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
]

# Define angle triplets: (A, B, C)
angle_triplets = [
    (5, 7, 9),  # Left arm: Shoulder-Elbow-Wrist
    (6, 8, 10),  # Right arm
    (11, 13, 15),  # Left leg: Hip-Knee-Ankle
    (12, 14, 16),  # Right leg
    (5, 11, 13),  # Torso-Leg left
    (6, 12, 14)  # Torso-Leg right
]

angle_data = []

if results[0].keypoints is not None:
    keypoints = results[0].keypoints.xy.cpu().numpy()
    label_count = {}  # Track number of labels per keypoint

    for kp_index, kp in enumerate(keypoints):
        for triplet in angle_triplets:
            A = kp[triplet[0]]
            B = kp[triplet[1]]
            C = kp[triplet[2]]
            angle = compute_angle(A, B, C)

            if not np.isnan(angle):
                # Draw lines
                cv2.line(annotated_frame, tuple(A.astype(int)), tuple(B.astype(int)), (255, 0, 0), 2)
                cv2.line(annotated_frame, tuple(C.astype(int)), tuple(B.astype(int)), (255, 0, 0), 2)
                draw_angle_arc(annotated_frame, B, angle)

                # Manage label offset
                key_b = tuple(B.astype(int))
                count = label_count.get(key_b, 0)
                y_offset = count * 20  # Move text down by 20 pixels for each extra label
                label_count[key_b] = count + 1

                text_position = (key_b[0], key_b[1] + y_offset)
                cv2.putText(annotated_frame, f'{int(angle)} deg', text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                # Save data
                angle_data.append({
                    'Person': kp_index + 1,
                    'Joint': f"{keypoint_names[triplet[0]]}-{keypoint_names[triplet[1]]}-{keypoint_names[triplet[2]]}",
                    'Angle(deg)': angle,
                    'Keypoint_B_x': B[0],
                    'Keypoint_B_y': B[1]
                })
            else:
                print(f"Skipping angle for triplet {triplet}: undefined")

# Save CSV
df = pd.DataFrame(angle_data)
csv_path = 'pose_angles.csv'
df.to_csv(csv_path, index=False)
print(f"Angles saved to {csv_path}")

# Save and show image
output_image_path = 'annotated_pose_image_with_angles.jpg'
cv2.imwrite(output_image_path, annotated_frame)
print(f"Annotated image saved to: {output_image_path}")

cv2.imshow('Pose Estimation with Angles', annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()