import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")  # or yolov8s-pose.pt for better accuracy

# Load input image
image_path = "standing_man_legs_open.jpg"
# others:
# standing_girl_closedleg.jpg
# standing_man_closedleg_1.webp
#standing_man_slightly_closed_1.webp
#standing_man_legs_open.jpg
#
#person_standing.jpg"
img = cv2.imread(image_path)
results = model(img)[0]

# Get keypoints
keypoints = results.keypoints.xy[0].cpu().numpy()

# Define relevant points
left_ankle = keypoints[15]
right_ankle = keypoints[16]
left_hip = keypoints[11]
right_hip = keypoints[12]

# Estimate crotch as midpoint between hips
crotch_point = (left_hip + right_hip) / 2

# Function to compute height from point to line (ankle to ankle)
def point_to_line_distance(B, A, C):
    AB = B - A
    AC = C - A
    cross = np.abs(np.cross(AC, AB))
    norm = np.linalg.norm(AC)
    return cross / norm if norm != 0 else 0

# Compute pixel-based crotch-to-floor distance
crotch_to_floor_px = point_to_line_distance(crotch_point, left_ankle, right_ankle)

# Estimate person full height in pixels (top of head to midpoint between ankles)
head_top = keypoints[0]
feet_center = (left_ankle + right_ankle) / 2
full_height_px = np.linalg.norm(head_top - feet_center)

# If known actual height in cm
known_height_cm = 180  # set this based on real person
scale_cm_per_px = known_height_cm / full_height_px
crotch_to_floor_cm = crotch_to_floor_px * scale_cm_per_px

# Compute saddle heights based on known methods
lemond = crotch_to_floor_cm * 0.883
hamley = crotch_to_floor_cm * 1.09
hinault = crotch_to_floor_cm * 0.885

# Print results
print(f"\n=== Estimated Measurements ===")
print(f"Crotch-to-Floor: {crotch_to_floor_cm:.1f} cm (estimated from keypoints)")
print(f"\n=== Saddle Height Recommendations ===")
print(f"LeMond Method (0.883):     {lemond:.1f} cm")
print(f"Hamley Method (1.09):      {hamley:.1f} cm")
print(f"Hinault Method (0.885):    {hinault:.1f} cm")

# Optional: visualize
for pt in [left_ankle, right_ankle, crotch_point]:
    cv2.circle(img, tuple(pt.astype(int)), 5, (0, 255, 0), -1)
cv2.line(img, tuple(left_ankle.astype(int)), tuple(right_ankle.astype(int)), (255, 0, 0), 2)
cv2.line(img, tuple(crotch_point.astype(int)), tuple(((left_ankle + right_ankle) / 2).astype(int)), (0, 0, 255), 1)

cv2.imshow("Crotch to Floor Estimation", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
