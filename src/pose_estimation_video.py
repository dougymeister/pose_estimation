import torch
from ultralytics import YOLO
import cv2

# Load the YOLOv8 pose estimation model (auto-detects GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = YOLO('yolov8n-pose.pt').to(device)  # Replace with a different model variant if needed

# Initialize webcam (0 is default, or replace with a file path)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get webcam properties for saving output video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 FPS if FPS detection fails

# Define output video writer
output_filename = 'pose_estimation_output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Perform pose estimation (inference on specified device)
    results = model(frame, device=device)

    # Annotate the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('Real-Time Pose Estimation', annotated_frame)

    # Save the annotated frame to video
    out.write(annotated_frame)

    # Extract keypoints as PyTorch tensors
    for result in results:
        keypoints_tensor = result.keypoints.xy.to(torch.device(device))
        print("Detected keypoints (tensor):", keypoints_tensor)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved as: {output_filename}")
