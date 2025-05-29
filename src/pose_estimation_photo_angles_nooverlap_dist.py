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
        return np.nan
    cosine_angle = np.dot(BA, BC) / (norm_BA * norm_BC)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)

def draw_angle_arc(image, B, angle_deg, radius=30, color=(255, 255, 255)):
    """Draw an angle arc at point B."""
    if np.isnan(angle_deg):
        return
    cv2.ellipse(image, tuple(B.astype(int)), (radius, radius), 0, 0, int(angle_deg), color, 2)

def draw_label_with_background(image, text, position, font_scale=0.5, text_color=(0,255,255), bg_color=(0,0,0)):
    """Draw text with background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    x, y = position
    cv2.rectangle(image, (x, y - text_h - 4), (x + text_w, y + 4), bg_color, -1)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

# Load YOLOv8 pose model
device = 'cpu'  # Use 'cuda' for GPU
model = YOLO('yolov8n-pose.pt').to(device)

# Input: Set either image or video
input_path = 'cyclist1.jpg'  # Replace with 'your_video.mp4' for video
is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov'))

cap = None
if is_video:
    cap = cv2.VideoCapture(input_path)
else:
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Error: Could not read image from {input_path}")
        exit()

# COCO keypoint names
keypoint_names = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
]

# Define angle triplets
angle_triplets = [
    (5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16),
    (5, 11, 13), (6, 12, 14)
]

# Define distance pairs (e.g., elbow-wrist, knee-ankle)
distance_pairs = [
    (7, 9),  # Left elbow-wrist
    (13, 15) # Left knee-ankle
]

angle_data = []
frame_idx = 0

while True:
    if is_video:
        ret, frame = cap.read()
        if not ret:
            break
    elif frame_idx > 0:
        break  # Only process one image if input is image
    frame_idx += 1

    results = model(frame, device=device)
    annotated_frame = results[0].plot()

    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()
        label_count = {}

        for kp_index, kp in enumerate(keypoints):
            # Draw angles
            for triplet in angle_triplets:
                A, B, C = kp[triplet[0]], kp[triplet[1]], kp[triplet[2]]
                angle = compute_angle(A, B, C)
                if not np.isnan(angle):
                    cv2.line(annotated_frame, tuple(A.astype(int)), tuple(B.astype(int)), (255, 0, 0), 2)
                    cv2.line(annotated_frame, tuple(C.astype(int)), tuple(B.astype(int)), (255, 0, 0), 2)
                    draw_angle_arc(annotated_frame, B, angle)
                    key_b = tuple(B.astype(int))
                    count = label_count.get(key_b, 0)
                    y_offset = count * 20
                    label_count[key_b] = count + 1
                    text_position = (key_b[0], key_b[1] + y_offset)
                    draw_label_with_background(annotated_frame, f'{int(angle)} deg', text_position)
                    angle_data.append({
                        'Frame': frame_idx,
                        'Person': kp_index + 1,
                        'Joint': f"{keypoint_names[triplet[0]]}-{keypoint_names[triplet[1]]}-{keypoint_names[triplet[2]]}",
                        'Angle(deg)': angle
                    })
            # Draw distances
            for pair in distance_pairs:
                P1, P2 = kp[pair[0]], kp[pair[1]]
                distance = np.linalg.norm(P1 - P2)
                midpoint = ((P1 + P2) / 2).astype(int)
                draw_label_with_background(annotated_frame, f'{int(distance)} px', tuple(midpoint), text_color=(255,255,0))

    cv2.imshow('Pose Estimation with Angles & Distances', annotated_frame)
    if is_video:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.waitKey(0)
        break

# Save CSV
df = pd.DataFrame(angle_data)
csv_path = 'pose_angles_data.csv'
df.to_csv(csv_path, index=False)
print(f"Angles saved to {csv_path}")

# Save annotated image or video frame
output_image_path = 'annotated_output.jpg'
if not is_video:
    cv2.imwrite(output_image_path, annotated_frame)
    print(f"Annotated image saved to: {output_image_path}")

if is_video:
    cap.release()
cv2.destroyAllWindows()
