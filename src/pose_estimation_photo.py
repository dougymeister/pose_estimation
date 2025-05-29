import torch
from ultralytics import YOLO
import cv2

import torch
import numpy as np

def compute_angle(A, B, C):
    """
    Compute the angle at point B formed by points A, B, and C.

    Parameters:
    A, B, C: tuples or numpy arrays (x, y) representing 2D coordinates

    Returns:
    Angle in degrees
    """
    # Convert to numpy arrays
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    # Vectors BA and BC
    BA = A - B
    BC = C - B

    # Compute the angle using the dot product formula
    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to avoid numerical errors

    angle_deg = np.degrees(angle_rad)
    return angle_deg


print("is cuda available?:"+str(torch.cuda.is_available()))
print("torch_version="+str(torch.__version__))

# Force CPU usage
device = 'cpu'
print(f"Forcing device: {device}")

# Load the YOLOv8 pose estimation model (this automatically downloads the model if not present)
model = YOLO('yolov8n-pose.pt').to(device)  # You can replace with 'yolov8s-pose.pt', 'yolov8m-pose.pt', etc.




# Load an image
image_path = 'bball_swing1_1024x1024.jpeg'  # Replace with your image path
image = cv2.imread(image_path)

# Perform pose estimation
results = model(image)

# Visualize the results
# 'plot()' method draws keypoints and skeletons on the image
annotated_image = results[0].plot()

# Display the image using OpenCV
cv2.imshow('Pose Estimation', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# To access keypoints and bounding boxes programmatically:
for result in results:
    for keypoints in result.keypoints.xy:
        print("Keypoints:", keypoints)  # keypoints is a tensor with shape [num_keypoints, 2]
    for box in result.boxes.xyxy:
        print("Bounding Box:", box)  # box is a tensor [x1, y1, x2, y2]

    # example calculating angle between kp
    keypoints = result.keypoints.xy.cpu().numpy()[0]  # First detected person
    shoulder = keypoints[5]  # Left shoulder
    elbow = keypoints[7]     # Left elbow
    wrist = keypoints[9]     # Left wrist

    angle = compute_angle(shoulder, elbow, wrist)
    print(f"Left elbow angle: {angle:.2f} degrees")

'''
Angle Calculation Between Three Keypoints
Let’s define three keypoints as:

A (first point, e.g., shoulder)

B (vertex point, e.g., elbow)

C (third point, e.g., wrist)

We compute the angle at point B using vectors BA and BC.

Keypoint Index Reference (COCO format):
Index	Body Part
0	Nose
1	Left Eye
2	Right Eye
3	Left Ear
4	Right Ear
5	Left Shoulder
6	Right Shoulder
7	Left Elbow
8	Right Elbow
9	Left Wrist
10	Right Wrist
11	Left Hip
12	Right Hip
13	Left Knee
14	Right Knee
15	Left Ankle
16	Right Ankle

'''