from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import pose_estimation_callable1 as pec

keypoint_names = ['Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
                  'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
                  'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
                  'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle']
angle_triplets = [(5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16), (5, 11, 13), (6, 12, 14)]
distance_pairs = [(7, 9), (13, 15)]

# Define the standard COCO pose connections
COCO_CONNECTIONS = [
    (5, 7), (7, 9),  # Left Arm
    (6, 8), (8, 10),  # Right Arm
    (5, 6),  # Shoulders
    (11, 13), (13, 15),  # Left Leg
    (12, 14), (14, 16),  # Right Leg
    (11, 12),  # Hips
    (5, 11), (6, 12)  # Torso sides
]


# to avoid circular loading, we get handle to class to access helper methods.
# dont need to supply input_path
def get_annotation_mgr():  #input_path: Path):
    from annotation_manager import AnnotationManager as AnnotMgr
    mgr = AnnotMgr()
    return mgr


def generate_annotated_base(input_path: Path, output_path: Path) -> bool:
    #model = YOLO('yolov8n-pose.pt').to('cpu')
    #    def generate_all_video_annotations(input_path: Path, base_filename: str, model, output_dir: Path, fps: int = 30,  scale_percent: float = 100.0):
    #generate_all_video_annotations(input_path, model, output_path)

    try:
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"[Error] Failed to read image: {input_path}")
            return False
        cv2.imwrite(str(output_path), img)
        print(f"[Generated] Base overlay copied to {output_path}")
        return True
    except Exception as e:
        print(f"[Error] generate_annotated_base: {e}")
        return False


'''
    # Currently we write length, angle, text over transparent layer (canvas). 
    # If you want to write over original image, just delete the code below in 
     functions and write over 'image'

        # write annot length onto blank canvas. remove to write to image (and uncomment below)
        height, width = image.shape[:2]
        canvas = np.zeros((height, width, 4), dtype=np.uint8)

        # Make alpha channel visible where non-black content was drawn.....remove if want to write over output image 
        gray = cv2.cvtColor(canvas[:, :, :3], cv2.COLOR_BGR2GRAY)
        canvas[:, :, 3] = np.where(gray > 0, 255, 0).astype(np.uint8)

'''


def generate_keypoints(input_path: Path, output_path: Path, model_path: str = 'yolov8n-pose.pt') -> bool:
    print(f"generate_keypoints({input_path}, {output_path}...get_annotation_mgr().is_video(input_path.suffix.lower())={get_annotation_mgr().is_video(input_path.suffix.lower())}")
    if get_annotation_mgr().is_video(input_path):  # in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        print("[Info] Video file detected – routing to generate_video_annotations() for keypoints")
        return generate_video_annotations(input_path, output_path, kind="keypoints")

    try:
        device = 'cpu'
        model = YOLO(model_path).to(device)

        image = cv2.imread(str(input_path))
        if image is None:
            print(f"[Read Error] Unable to load image: {input_path}")
            return False

        results = model(str(input_path), device=device)
        keypoint_obj = results[0].keypoints

        if keypoint_obj is None or keypoint_obj.xy is None:
            print("[Keypoint Generation Error] No keypoints detected")
            return False

        keypoints = keypoint_obj.xy.cpu().numpy()
        height, width = image.shape[:2]
        canvas_bgr = np.zeros((height, width, 3), dtype=np.uint8)

        for kp in keypoints:
            for point in kp:
                x, y = int(point[0]), int(point[1])
                if not np.isnan(x) and not np.isnan(y):
                    cv2.circle(canvas_bgr, (x, y), 4, (0, 255, 0), -1)

        canvas = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2BGRA)
        gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
        canvas[:, :, 3] = np.where(gray > 0, 255, 0).astype(np.uint8)

        cv2.imwrite(str(output_path), canvas)
        print(f"[Generated] Keypoint overlay saved to {output_path}")
        return True

    except Exception as e:
        print(f"[Keypoint Generation Error] {e}")
        return False


def generate_video_annotations(input_path: Path, output_path: Path, kind: str = "pose_angles") -> bool:
    print(f"generate_video_annotations({input_path}, {output_path}, {kind}")
    cap=None
    writer=None
    print(f"gen_video_annot()----........pose_angles")
    if kind == "pose_angles":
        # test....
        #success, message = pec.process_pose(input_path, output_path, output_path / ".cvs", 100)
        gen_angles(input_path, output_path)
        return True

    elif kind == "pose_length":
        print(".....generate_video_annotations() - calling gen_len()")
        #success, message = pec.process_pose(input_path, output_path, output_path / ".cvs", 100)
        gen_len(input_path, output_path)
        return True

    # not using logic below for pose_length......as having issues. Also, video cant use transparent canvas, as not supported
    # so just write over frame

    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"[Video] Cannot open: {input_path}")
            return False

        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        model = YOLO('yolov8n-pose.pt').to('cpu')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, device='cpu')

            canvas = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)

            if results[0].keypoints is not None:
                keypoints = results[0].keypoints.xy.cpu().numpy()

                print(f"generate_video_annotations()..........kind={kind}")
                if kind == "pose_angles":
                    # test....
                    success, message = pec.process_pose(input_path, output_path, output_path/".cvs", 100)
                    return True
                    #draw_angles(canvas, keypoints)
                elif kind == "pose_length":
                    print(f"generate_video_annotations(kind={kind})...calling {draw_lengths}")
                    #draw_lengths_video(canvas, keypoints)
                    for kp in keypoints:
                        for i, j in distance_pairs:
                            if i < len(kp) and j < len(kp):
                                p1, p2 = kp[i], kp[j]
                                if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                                    continue
                                dist = int(np.linalg.norm(p1 - p2))
                                midpoint = ((p1 + p2) / 2).astype(int)
                                print(f"....pose_length() calling draw_label_with_back...text={dist}px")
                                # dy 06/09 - changed midpoint to midpoint[:2] to just pass first 2 fields (x,y)
                                draw_label_with_background(canvas, f"{dist}px", tuple(midpoint[:2]),
                                                              text_color=(255, 255, 0))

                    if canvas is not None:
                        writer.write(canvas)
                        #cv2.imwrite(output_path, canvas)

                elif kind == "keypoints":
                    draw_keypoints(canvas, keypoints)
                elif kind == "connections":
                    for kp in keypoints:
                        for (i, j) in COCO_CONNECTIONS:
                            if i < len(kp) and j < len(kp):
                                x1, y1 = kp[i]
                                x2, y2 = kp[j]
                                if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                                    pt1 = (int(x1), int(y1))
                                    pt2 = (int(x2), int(y2))
                                    cv2.line(canvas, pt1, pt2, (0, 255, 255, 255), 2)
                else:
                    print(f"[Video Annotation] Unknown kind: {kind}")
                    return False

                overlay_alpha(frame, canvas)

            writer.write(frame)

        cap.release()
        writer.release()
        print(f"generate_video_annotations():  [Video Annotation] Saved to {output_path}")
        return True

    except Exception as e:
        print(f"generate_video_annotations():  ***Exception*** [Video Annotation Error] {e}")
        if cap:
            cap.release()
        if writer:
            writer.release()
        return False


def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)


def is_valid_point(p):
    return not (np.isnan(p[0]) or np.isnan(p[1]))


def generate_connections(input_path: Path, output_path: Path, model_path: str = 'yolov8n-pose.pt') -> bool:
    if get_annotation_mgr().is_video(input_path):  # in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        print("[Info] Video file detected – routing to generate_video_annotations() for connections")
        return generate_video_annotations(input_path, output_path, kind="connections")

    try:
        print(f"[Keyp Connections] Generating connections using {model_path}")
        device = 'cpu'
        model = YOLO(model_path).to(device)

        results = model(str(input_path), device=device)
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"[Keyp Connections Error] Failed to read image: {input_path}")
            return False

        keypoints_obj = results[0].keypoints
        if keypoints_obj is None or keypoints_obj.xy is None:
            print("[Keyp Connections Error] No keypoints detected.")
            return False

        keypoints = keypoints_obj.xy.cpu().numpy()
        height, width = image.shape[:2]
        canvas_bgr = np.zeros((height, width, 3), dtype=np.uint8)

        for kp in keypoints:
            '''  ...reenable if you want circles and connections in 1 layer
            for point in kp:
                x, y = int(point[0]), int(point[1])
                if not np.isnan(x) and not np.isnan(y):
                    cv2.circle(canvas_bgr, (x, y), 4, (0, 255, 0), -1)
            '''
            for (i, j) in COCO_CONNECTIONS:
                if i < len(kp) and j < len(kp):
                    x1, y1 = kp[i]
                    x2, y2 = kp[j]
                    if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                        pt1 = (int(x1), int(y1))
                        pt2 = (int(x2), int(y2))
                        cv2.line(canvas_bgr, pt1, pt2, (255, 0, 0, 255), 2)

        canvas = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2BGRA)
        gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
        canvas[:, :, 3] = np.where(gray > 0, 255, 0).astype(np.uint8)

        cv2.imwrite(str(output_path), canvas)
        print(f"[Keyp Connections] Saved: {output_path}")
        return True

    except Exception as e:
        print(f"[Keyp Connections Error] {e}")
        return False


def calculate_angle(p1, p2, p3):
    try:
        a = np.array(p1) - np.array(p2)
        b = np.array(p3) - np.array(p2)
        cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    except Exception:
        return None


import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import random


def compute_angle(A, B, C):
    A, B, C = np.array(A), np.array(B), np.array(C)
    BA, BC = A - B, C - B
    norm_BA, norm_BC = np.linalg.norm(BA), np.linalg.norm(BC)
    if norm_BA == 0 or norm_BC == 0:
        return np.nan
    cosine_angle = np.dot(BA, BC) / (norm_BA * norm_BC)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))


def draw_label_with_background(image, text, position, font_scale=0.5, text_color=(0, 255, 255), bg_color=(0, 0, 0)):
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    x, y = position
    cv2.rectangle(image, (x, y - text_h - 4), (x + text_w, y + 4), bg_color, -1)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_angle_arc(image, B, angle_deg, radius=30, color=(255, 255, 255)):
    if not np.isnan(angle_deg):
        B_int = tuple(np.array(B).astype(int))
        cv2.ellipse(image, B_int, (radius, radius), 0, 0, int(angle_deg), color, 2)


def draw_pose_and_angle(image, A, B, C, angle, drawn_labels):
    if np.isnan(angle):
        return
    B = np.array(B, dtype=int)
    draw_angle_arc(image, B, angle)
    offset = (random.randint(-10, 10), random.randint(-10, 10))
    label_pos = tuple(B + offset)
    key = tuple(label_pos)
    if key not in drawn_labels:
        draw_label_with_background(image, f"{int(angle)}°", label_pos)
        drawn_labels.add(key)


def generate_pose_angles(input_path: Path, output_path: Path) -> bool:
    if get_annotation_mgr().is_video(input_path):  # in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        print("[Info] Detected video file, delegating to generate_video_annotations...")
        return generate_video_annotations(input_path, output_path, kind="pose_angles")

    try:
        device = 'cpu'
        model = YOLO("yolov8n-pose.pt").to(device)
        results = model(str(input_path), device=device)

        keypoints_obj = results[0].keypoints
        if keypoints_obj is None or keypoints_obj.xy is None:
            print("[Pose Angles] No keypoints found")
            return False

        keypoints = keypoints_obj.xy.cpu().numpy()
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"[Pose Angles] Failed to read image: {input_path}")
            return False

        label_offsets = {}
        height, width = image.shape[:2]
        canvas = np.zeros((height, width, 4), dtype=np.uint8)

        for kp in keypoints:
            for triplet in angle_triplets:
                A, B, C = kp[triplet[0]], kp[triplet[1]], kp[triplet[2]]
                if np.any(np.isnan(A)) or np.any(np.isnan(B)) or np.any(np.isnan(C)):
                    continue

                angle = compute_angle(A, B, C)
                angle_label = f"{int(round(angle))} deg" if not np.isnan(angle) else "?? deg"
                B_int = tuple(B.astype(int))
                offset_y = label_offsets.get(B_int, 0)
                label_pos = (B_int[0], B_int[1] - offset_y)
                label_offsets[B_int] = offset_y + 20

                draw_angle_arc(canvas, B, angle)
                draw_label_with_background(canvas, angle_label, label_pos)

        gray = cv2.cvtColor(canvas[:, :, :3], cv2.COLOR_BGR2GRAY)
        canvas[:, :, 3] = np.where(gray > 0, 255, 0).astype(np.uint8)
        cv2.imwrite(str(output_path), canvas)
        print(f"[Pose Angles] Saved to {output_path}")
        return True

    except Exception as e:
        print(f"[Pose Angles Error] {e}")
        return False


def draw_label_with_background(image, text, position, font_scale=0.5, text_color=(0, 255, 255), bg_color=(0, 0, 0)):
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    x, y = position
    cv2.rectangle(image, (x, y - text_h - 4), (x + text_w, y + 4), bg_color, -1)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def generate_pose_length(input_path: Path, output_path: Path) -> bool:
    if get_annotation_mgr().is_video(input_path):  # in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        print("[Info] Detected video file, delegating to generate_video_annotations...")
        return generate_video_annotations(input_path, output_path, kind="pose_length")
    try:
        model = YOLO("yolov8n-pose.pt").to("cpu")
        results = model(str(input_path), device="cpu")

        if not results or results[0].keypoints is None:
            print("[Pose Lengths] No keypoints detected")
            return False

        keypoints = results[0].keypoints.xy.cpu().numpy()
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"[Pose Lengths] Could not read image {input_path}")
            return False

        # write annot length onto blank canvas. remove to write to image (and uncomment below)
        height, width = image.shape[:2]
        canvas = np.zeros((height, width, 4), dtype=np.uint8)

        for kp in keypoints:
            for pair in distance_pairs:
                P1, P2 = kp[pair[0]], kp[pair[1]]
                distance = np.linalg.norm(P1 - P2)
                midpoint = ((P1 + P2) / 2).astype(int)
                # draw_label_with_background(image, f'{int(distance)} px', tuple(midpoint), text_color=(255, 255, 0))
                draw_label_with_background(canvas, f'{int(distance)} px', tuple(midpoint), text_color=(255, 255, 0))

        # Make alpha channel visible where non-black content was drawn
        gray = cv2.cvtColor(canvas[:, :, :3], cv2.COLOR_BGR2GRAY)
        canvas[:, :, 3] = np.where(gray > 0, 255, 0).astype(np.uint8)

        cv2.imwrite(str(output_path), canvas)  # image) #uncomment to write to image, not canvas
        print(f"[Pose Lengths] Saved to {output_path}")
        return True
    except Exception as e:
        print(f"[Pose Lengths Error] {e}")
        return False


def run_annotation_generator(base_filename: str, layer_name: str, annotation_manager, input_path: Path,
                             output_path: Path) -> bool:
    """
    Looks up and calls the appropriate generator for the given annotation layer.
    """
    layer_meta = annotation_manager.get_layer_metadata()
    layer_info = next((l for l in layer_meta if l["name"] == layer_name), None)
    if not layer_info:
        print(f"[Error] Layer '{layer_name}' not found in metadata.")
        return False

    generator_name = layer_info.get("generator")
    if not generator_name:
        print(f"[Error] No generator defined for layer '{layer_name}'")
        return False

    generator_func = globals().get(generator_name)
    if not callable(generator_func):
        print(f"[Error] Generator function '{generator_name}' not found.")
        return False

    return generator_func(input_path, output_path)


from annotation_utils import draw_keypoints, draw_angles, draw_lengths, gen_angles, gen_len


def overlay_alpha(base: np.ndarray, overlay: np.ndarray):
    """Blend a transparent RGBA overlay onto an RGB base image in-place."""
    alpha = overlay[..., 3:] / 255.0
    for c in range(3):
        base[..., c] = base[..., c] * (1 - alpha[:, :, 0]) + overlay[..., c] * alpha[:, :, 0]


def draw_connections(canvas: np.ndarray, keypoints: np.ndarray):
    for kp in keypoints:
        for i, j in COCO_CONNECTIONS:
            if i < len(kp) and j < len(kp):
                x1, y1 = kp[i]
                x2, y2 = kp[j]
                if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                    pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
                    cv2.line(canvas, pt1, pt2, (255, 255, 0), 2)
