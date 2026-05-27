import uuid
import json
import shutil
import numpy as np
import cv2
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from ultralytics import YOLO

from feedback_api import router as feedback_router

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
app.include_router(feedback_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MEDIA_DIR = BASE_DIR / "media"
STATIC_DIR = BASE_DIR / "static"

MEDIA_DIR.mkdir(exist_ok=True)

app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/images", StaticFiles(directory=STATIC_DIR / "images"), name="images")

@app.get("/")
def serve_homepage():
    return FileResponse(STATIC_DIR / "bicycleabout.html")


POSE_ANGLE_LAYERS = {
    "knee_angle": [(11, 13, 15), (12, 14, 16)],
    "hip_angle": [(5, 11, 13), (6, 12, 14)],
    #"torso_angle": [(11, 5, 6), (12, 6, 5)],
    "shoulder_angle": [(11, 5, 7), (12, 6, 8)],
    #"back_angle": [(11, 5, 6), (12, 6, 5)]  # optional if different from torso

}

'''
Measurement	Type	Keypoints	Belongs In	Rule Key Example
Leg Length	Distance	(11, 15) (12, 16)	POSE_DISTANCE_LAYERS	leg_length_left_11_15
Leg Length	Angle	(11,13,15)...	POSE_ANGLE_LAYERS	leg_length_left_11_13_15
Arm Length	Distance	(5, 9) (6, 10)	POSE_DISTANCE_LAYERS	arm_length_left_5_9
Arm Length	Angle	(5,7,9)...	POSE_ANGLE_LAYERS	arm_length_left_5_7_9
'''
POSE_DISTANCE_LAYERS = {
    "arm_reach_distance": [(5, 9), (6, 10)],
    "leg_length": [(11, 15), (12, 16)],
    "knee_length": [(13, 15), (14, 16)],
    "reach_distance": [(5, 7), (6, 8)],
    #"saddle_to_bar_distance": [(1, 0)],
}

BIKE_LANDMARK_DISTANCE_PAIRS = {
    "bike_saddle_to_bar_distance": ("saddle_center", "handlebar_center", "straight"),
    "bike_saddle_to_bar_horizontal_reach": ("saddle_center", "handlebar_center", "horizontal"),
    "bike_saddle_to_bar_vertical_drop": ("saddle_center", "handlebar_center", "vertical"),
    "bike_saddle_height": ("bottom_bracket", "saddle_center", "straight"),
    "bike_wheelbase": ("rear_axle", "front_axle", "straight"),
    "bike_bottom_bracket_to_front_axle": ("bottom_bracket", "front_axle", "straight"),
    "bike_bottom_bracket_to_rear_axle": ("bottom_bracket", "rear_axle", "straight"),
}

pose_model = YOLO("yolo11m-pose.pt")    #pose_model = YOLO("yolo26m-pose.pt").to("cpu")

'''
    "all_fit_points": [
        (5, 9), (6, 10),
        (11, 15), (12, 16),
        (13, 15), (14, 16),
        (5, 7), (6, 8),
        (1, 0)
    ]
}
'''

# Temporary in-memory storage (can be replaced with database or file)
user_profile_settings = {
    "bikeType": "Road",
    "ridingStyle": "Casual"
}


@app.post("/profile-settings")
async def update_profile_setting(request: Request):
    data = await request.json()
    setting = data.get("setting")
    value = data.get("value")
    if setting in user_profile_settings:
        user_profile_settings[setting] = value
        return JSONResponse({"status": "success", "updated": {setting: value}})
    return JSONResponse({"status": "error", "message": "Invalid setting"}, status_code=400)


@app.get("/profile-settings")
async def get_profile_settings():
    return JSONResponse(user_profile_settings)


def compute_angle_and_label(keypoints, visible, triplet):
    a, b, c = triplet
    if not all(0 <= i < len(keypoints) for i in triplet):
        print(f"[DEBUG] compute_angle_and_label()  not all(0 <= i < len(keypoints) for i in triplet  {triplet} ")
        return None
    if not all(visible[i] > 0.2 for i in triplet):
        print(
            f"[DEBUG] compute_angle_and_label() Skipping angle: low visibility {triplet} → {[visible[i][0] for i in triplet]}")
        return None

    ptA, ptB, ptC = keypoints[a], keypoints[b], keypoints[c]
    vec1 = np.array(ptA) - np.array(ptB)
    vec2 = np.array(ptC) - np.array(ptB)
    angle = np.degrees(np.arccos(
        np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6), -1.0, 1.0)
    ))

    label = f"{int(angle)}\u00B0"
    midpoint = [int((ptA[0] + ptC[0]) / 2), int((ptA[1] + ptC[1]) / 2)]

    return {
        "points": triplet,
        "angle_deg": angle,
        "label": label,
        "midpoint": midpoint,
        "line_segments": []
    }

def make_straight_reach_entry(a, b, keypoints, visible, cm_per_px):
    # Straight-line reach entry with orange solid styling
    #orange (#FFA500) and blue (#1E90FF) for the solid and dashed distance lines—they’re bright, high-contrast, and color-blind–friendly.
    # line thick=3 helps see over other stuffy
    ent = compute_distance_and_label(keypoints, visible, (a, b), cm_per_px)
    if not ent:
        return None
    ent["style"] = "solid"
    ent["color"] = "#FFA500"  # orange
    ent["thickness"] = 3

    return ent



def make_horizontal_reach_entry(a, b, keypoints, visible, cm_per_px):
    # Horizontal (horizon) reach entry with blue dashed styling
    # ensure valid straight reach before computing horizontal
    straight_ent = compute_distance_and_label(keypoints, visible, (a, b), cm_per_px)
    if not straight_ent:
        return None

    shoulder = keypoints[a]
    wrist = keypoints[b]

    # horizontal pixel distance
    horiz_px = abs(wrist[0] - shoulder[0])
    horiz_cm = horiz_px * cm_per_px if cm_per_px else None
    horiz_in = horiz_cm / 2.54 if horiz_cm is not None else None

    proj = [wrist[0], shoulder[1]]
    base_mid = [(shoulder[0] + proj[0]) / 2, shoulder[1] - 10]

    seg_base = [shoulder[0], shoulder[1], proj[0], proj[1]]
    seg_vert = [wrist[0], wrist[1], proj[0], proj[1]]

    #orange (#FFA500) and blue (#1E90FF) for the solid and dashed distance lines—they’re bright, high-contrast, and color-blind–friendly.
    # line thick=3 helps see over other stuffy
    default_label = horiz_cm is not None and f"{horiz_cm:.1f} cm" or f"{int(horiz_px)} px"
    ent = {
        "points": [a, b],
        "type": "horizontal_reach",
        "distance_px": horiz_px,
        "distance_cm": horiz_cm,
        "distance_in": horiz_in,
        "label": default_label,
        "label_px": f"{int(horiz_px)} px",
        "label_cm": horiz_cm is not None and f"{horiz_cm:.1f} cm",
        "label_in": horiz_in is not None and f"{horiz_in:.2f} in",
        "midpoint": base_mid,
        "line_segments": [seg_base, seg_vert],
        "style": "dashed",
        "color": "#1E90FF",  # blue
        "thickness": 3
    }
    return ent


# supports threshold confidence # (keypoint visibility i think)
def is_valid_point(idx, keypoints, visible, threshold=0.35):
    if not (0 <= idx < len(keypoints) and 0 <= idx < len(visible)):
        return False

    pt = keypoints[idx]
    conf = visible[idx]

    if not isinstance(conf, (int, float)) or conf < threshold:
        return False

    if pt is None or len(pt) < 2:
        return False

    x, y = float(pt[0]), float(pt[1])

    if x == 0.0 and y == 0.0:
        return False

    if np.isnan(x) or np.isnan(y):
        return False

    return True

def compute_distance_and_label(keypoints, visible, pair, cm_per_px=None):
    a, b = pair
    ptA = keypoints[a]
    ptB = keypoints[b]
    # ensure valid points
    if not (is_valid_point(a, keypoints, visible) and is_valid_point(b, keypoints, visible)):
        return None

    # straight-line distance
    dist_px = float(np.linalg.norm(np.array(ptA) - np.array(ptB)))
    dist_cm = dist_px * cm_per_px if cm_per_px else None
    dist_in = dist_cm / 2.54 if dist_cm is not None else None

    midpoint = [(ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2]

    # flat segment for straight-line
    seg_straight = [ptA[0], ptA[1], ptB[0], ptB[1]]
    default_label = dist_cm is not None and f"{dist_cm:.1f} cm" or f"{int(dist_px)} px"

    return {
        "points": pair,
        "distance_px": dist_px,
        "distance_cm": dist_cm,
        "distance_in": dist_in,
        "label": default_label,
        "label_px": f"{int(dist_px)} px",
        "label_cm": dist_cm is not None and f"{dist_cm:.1f} cm",
        "label_in": dist_in is not None and f"{dist_in:.2f} in",
        "midpoint": midpoint,
        "line_segments": [seg_straight],
    }



def generate_layer_annotations(keypoints, visible, image, layer, cm_per_px):
    labels = []
    metrics = {"distances": [], "angles": []}

    # ANGLES (unchanged)
    #  the index so you can inject “left” vs “right.”  - so i ==0 is left, etc
    if layer in POSE_ANGLE_LAYERS:
        for i, triplet in enumerate(POSE_ANGLE_LAYERS[layer]):
            angle_data = compute_angle_and_label(keypoints, visible, triplet)
            if not angle_data:
                continue

            # determine side based on index: first triplet is left, second is right
            side = "left" if i == 0 else "right"

            # build the exact key your JSON rules use
            angle_data["key"] = f"{layer}_{side}_{triplet[0]}_{triplet[1]}_{triplet[2]}"

            labels.append(angle_data)
            metrics["angles"].append(angle_data)


    # DISTANCES
    if layer in POSE_DISTANCE_LAYERS:
        for i, (a, b) in enumerate(POSE_DISTANCE_LAYERS[layer]):
            # compute left/right
            suffix = "_left" if i == 0 else "_right"
            side_key = f"{layer}{suffix}_{a}_{b}"

            # 1) straight-line reach
            straight_ent = make_straight_reach_entry(a, b, keypoints, visible, cm_per_px)
            if straight_ent:
                # tag the object you will emit
                straight_ent["key"] = side_key

                labels.append(straight_ent)
                metrics["distances"].append(straight_ent)

                # if you want per-side grouping, push a shallow copy
                copy_ent = straight_ent.copy()
                metrics.setdefault("distances_by_side", {}) \
                    .setdefault(f"{layer}{suffix}", []) \
                    .append(copy_ent)

            # 2) horizontal reach (arm only)
            if layer == "arm_reach_distance":
                horiz_ent = make_horizontal_reach_entry(a, b, keypoints, visible, cm_per_px)
                if horiz_ent:
                    horiz_ent["key"] = f"arm_reach_distance_horizon{suffix}_{a}_{b}"
                    labels.append(horiz_ent)
                    metrics["distances"].append(horiz_ent)
                    # only copy if you actually need distances_by_side for horizon:
                    # copy_h = horiz_ent.copy()
                    # metrics["distances_by_side"][f"{layer}{suffix}"].append(copy_h)

    return labels, metrics


def infer_cm_per_px_from_metrics(all_metrics):
    for layer_data in all_metrics.values():
        if not isinstance(layer_data, dict):
            continue
        for distance in layer_data.get("distances", []):
            distance_px = distance.get("distance_px")
            distance_cm = distance.get("distance_cm")
            if distance_px and distance_cm:
                return float(distance_cm) / float(distance_px)
    return None


def get_bike_landmark_point(bike_landmarks, landmark_type):
    point = bike_landmarks.get(landmark_type)
    if not isinstance(point, dict) or point.get("visible") is False:
        return None

    try:
        return [float(point["x"]), float(point["y"])]
    except (KeyError, TypeError, ValueError):
        return None


def make_bike_landmark_distance_entry(metric_key, point_a, point_b, mode, cm_per_px=None, point_labels=None):
    dx = float(point_b[0] - point_a[0])
    dy = float(point_b[1] - point_a[1])

    if mode == "horizontal":
        distance_px = abs(dx)
        line_segments = [[point_a[0], point_a[1], point_b[0], point_a[1]]]
        midpoint = [(point_a[0] + point_b[0]) / 2, point_a[1]]
        distance_type = "horizontal"
    elif mode == "vertical":
        distance_px = abs(dy)
        line_segments = [[point_b[0], point_a[1], point_b[0], point_b[1]]]
        midpoint = [point_b[0], (point_a[1] + point_b[1]) / 2]
        distance_type = "vertical"
    else:
        distance_px = float(np.linalg.norm(np.array(point_a) - np.array(point_b)))
        line_segments = [[point_a[0], point_a[1], point_b[0], point_b[1]]]
        midpoint = [(point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2]
        distance_type = "straight"

    distance_cm = distance_px * cm_per_px if cm_per_px else None
    distance_in = distance_cm / 2.54 if distance_cm is not None else None

    return {
        "key": metric_key,
        "name": metric_key,
        "layer": "bike_landmarks",
        "type": distance_type,
        "points": point_labels or [],
        "landmark_points": point_labels or [],
        "distance_px": distance_px,
        "distance_cm": distance_cm,
        "distance_in": distance_in,
        "label": distance_cm is not None and f"{distance_cm:.1f} cm" or f"{int(distance_px)} px",
        "label_px": f"{int(distance_px)} px",
        "label_cm": distance_cm is not None and f"{distance_cm:.1f} cm",
        "label_in": distance_in is not None and f"{distance_in:.2f} in",
        "midpoint": midpoint,
        "line_segments": line_segments,
        "style": "solid" if mode == "straight" else "dashed",
        "color": "#00d4ff",
        "thickness": 3,
    }


def calculate_bike_landmark_metrics(bike_landmarks, cm_per_px=None):
    metrics = {"distances": [], "angles": []}

    for metric_key, (start_type, end_type, mode) in BIKE_LANDMARK_DISTANCE_PAIRS.items():
        point_a = get_bike_landmark_point(bike_landmarks, start_type)
        point_b = get_bike_landmark_point(bike_landmarks, end_type)
        if point_a is None or point_b is None:
            continue

        metrics["distances"].append(
            make_bike_landmark_distance_entry(
                metric_key,
                point_a,
                point_b,
                mode,
                cm_per_px,
                [start_type, end_type]
            )
        )

    return metrics


def format_signed_offset_label(signed_value, unit_label):
    if signed_value is None:
        return None

    abs_value = abs(float(signed_value))
    if abs_value < 0.05:
        return f"0.0 {unit_label} aligned"

    direction = "ahead" if signed_value > 0 else "behind"
    return f"{abs_value:.1f} {unit_label} {direction}"


def compute_knee_pedal_spindle_offset(keypoints, visible, bike_landmarks, cm_per_px=None):
    pedal = get_bike_landmark_point(bike_landmarks, "visible_pedal_spindle")
    bottom_bracket = get_bike_landmark_point(bike_landmarks, "bottom_bracket")

    print(f"[ KOPS ] bottom_bracket: {bottom_bracket}")
    print(f"[ KOPS ] visible_pedal_spindle: {pedal}")

    if pedal is None or bottom_bracket is None:
        print("[ KOPS ] metric valid: False")
        return None

    dx = pedal[0] - bottom_bracket[0]
    dy = pedal[1] - bottom_bracket[1]
    if dx == 0 and dy == 0:
        print("[ KOPS ] metric valid: False")
        return None

    crank_angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    angle_from_horizontal = min(abs(crank_angle_deg), abs(abs(crank_angle_deg) - 180.0))
    print(f"[ KOPS ] crank_angle_deg: {crank_angle_deg}")

    if angle_from_horizontal > 15:
        print("[ KOPS ] metric valid: False")
        return None

    candidate_knees = []
    for knee_name, knee_idx in (("left_knee", 13), ("right_knee", 14)):
        if not is_valid_point(knee_idx, keypoints, visible):
            continue
        knee = [float(keypoints[knee_idx][0]), float(keypoints[knee_idx][1])]
        distance_to_pedal = float(np.linalg.norm(np.array(knee) - np.array(pedal)))
        candidate_knees.append((distance_to_pedal, knee_name, knee_idx, knee))

    if not candidate_knees:
        print("[ KOPS ] selected_knee: None")
        print("[ KOPS ] metric valid: False")
        return None

    _, selected_knee, selected_knee_index, knee = min(candidate_knees, key=lambda item: item[0])
    signed_offset_px = float(knee[0] - pedal[0])
    distance_px = abs(signed_offset_px)
    signed_offset_cm = signed_offset_px * cm_per_px if cm_per_px else None
    distance_cm = distance_px * cm_per_px if cm_per_px else None
    signed_offset_in = signed_offset_cm / 2.54 if signed_offset_cm is not None else None
    distance_in = distance_cm / 2.54 if distance_cm is not None else None

    if distance_px < 2:
        direction = "knee_aligned"
    elif signed_offset_px > 0:
        direction = "knee_ahead_of_pedal"
    else:
        direction = "knee_behind_pedal"

    label_px = format_signed_offset_label(signed_offset_px, "px")
    label_cm = format_signed_offset_label(signed_offset_cm, "cm")
    label_in = format_signed_offset_label(signed_offset_in, "in")

    metric = {
        "key": "knee_pedal_spindle_offset",
        "name": "knee_pedal_spindle_offset",
        "metric": "Knee to Pedal Spindle Offset",
        "layer": "knee_pedal_spindle",
        "type": "horizontal_offset",
        "selected_knee": selected_knee,
        "selected_knee_index": selected_knee_index,
        "pedal_landmark": "visible_pedal_spindle",
        "bottom_bracket_landmark": "bottom_bracket",
        "distance_px": distance_px,
        "signed_offset_px": signed_offset_px,
        "distance_cm": distance_cm,
        "signed_offset_cm": signed_offset_cm,
        "distance_in": distance_in,
        "signed_offset_in": signed_offset_in,
        "direction": direction,
        "crank_angle_deg": crank_angle_deg,
        "valid": True,
        "status_note": "Uses the selected visible knee closest to the visible pedal spindle. Best used when the crank is near horizontal.",
        "label": label_in or label_cm or label_px,
        "label_px": label_px,
        "label_cm": label_cm,
        "label_in": label_in,
        "points": ["selected_knee", "visible_pedal_spindle"],
        "landmark_points": ["visible_pedal_spindle", "bottom_bracket"],
        "line_segments": [
            [knee[0], knee[1], knee[0], pedal[1]],
            [knee[0], pedal[1], pedal[0], pedal[1]]
        ],
        "midpoint": [(knee[0] + pedal[0]) / 2, pedal[1]],
        "style": "dashed",
        "color": "#00d4ff",
        "thickness": 3
    }

    print(f"[ KOPS ] selected_knee: {selected_knee}")
    print(f"[ KOPS ] signed_offset_in: {signed_offset_in}")
    print("[ KOPS ] metric valid: True")
    return metric


def append_kops_metric(all_metrics, keypoints, visible, bike_landmarks, cm_per_px=None):
    metric = compute_knee_pedal_spindle_offset(keypoints, visible, bike_landmarks, cm_per_px)
    if metric and metric.get("valid"):
        all_metrics["knee_pedal_spindle"] = {
            "distances": [metric],
            "angles": []
        }
    return metric


def load_bike_landmarks_for_filename(filename):
    stem = Path(Path(filename).name).stem
    landmarks_path = MEDIA_DIR / f"{stem}_bike_landmarks.json"
    if not landmarks_path.exists():
        return {}, None

    with open(landmarks_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload.get("bike_landmarks", {}), payload


def append_bike_landmark_metrics(all_metrics, bike_landmarks, cm_per_px=None):
    if cm_per_px is None:
        cm_per_px = infer_cm_per_px_from_metrics(all_metrics)

    bike_metrics = calculate_bike_landmark_metrics(bike_landmarks, cm_per_px)
    if bike_metrics["distances"] or bike_metrics["angles"]:
        all_metrics["bike_landmarks"] = bike_metrics
        print("[BIKE LANDMARK] Added bike_landmarks to all_metrics")
    return bike_metrics


def resolve_media_file(filename):
    if not filename:
        return None

    relative_path = Path(str(filename).replace("\\", "/"))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return None

    return MEDIA_DIR / relative_path

# to compute custom bike wheel size ....use trained yolo wheel_model
#wheel_model_path="C:\development\Python\projects\pose_estimation\PoseEstimation\getimages\runs\wheel_train_only_v2_06259pm\weights"
wheel_model = YOLO("../getimages/runs/wheel_train_only_v2_06259pm/weights/best.pt").to("cpu")

def compute_conversion_factor(keypoints, image_path, reference_object, reference_size, reference_unit):
    # Fallback if something fails
    default_return = (None, [])

    try:
        model = wheel_model  #YOLO(WHEEL_MODEL_PATH)
        results = model(image_path)
        boxes = results[0].boxes
        labels = results[0].names
        diameters_px = []

        for box in boxes:
            cls_id = int(box.cls.item())
            label = labels[cls_id]
            xyxy = box.xyxy[0].cpu().numpy()
            w = abs(xyxy[2] - xyxy[0])
            h = abs(xyxy[3] - xyxy[1])
            diameter_px = max(w, h)
            diameters_px.append((label, diameter_px))

        if not diameters_px:
            print("[REF SCALE] No wheels detected.")
            return default_return

        # Filter by reference_object (e.g., 'front_wheel')
        ref_diameters = [d for label, d in diameters_px if label == reference_object]

        if not ref_diameters:
            print(f"[REF SCALE] No matching {reference_object} found.")
            return default_return

        avg_diameter_px = sum(ref_diameters) / len(ref_diameters)

        # Convert real-world reference size to cm
        if reference_unit == "in":
            reference_cm = reference_size * 2.54
        elif reference_unit == "cm":
            reference_cm = reference_size
        else:
            raise ValueError("Unsupported unit")

        conversion_factor = reference_cm / avg_diameter_px  # cm per pixel
        print(f"[REF SCALE] Using {reference_object}: {conversion_factor:.4f} cm/px")
        return conversion_factor, diameters_px

    except ValueError:
        print("[WARN] Invalid reference size input.")
        return default_return
    except Exception as e:
        print(f"[ERROR] Failed in compute_conversion_factor: {e}")
        return default_return


def select_best_person_index(results_obj):
    """
    Pick the most likely rider from YOLO pose detections.
    Current strategy: largest detected person bounding box.
    This avoids blindly using keypoints_raw[0] when YOLO detects multiple people.
    """
    keypoints_raw = results_obj.keypoints.xy
    boxes_raw = results_obj.boxes.xyxy if results_obj.boxes is not None else None

    num_people = len(keypoints_raw) if keypoints_raw is not None else 0
    print(f"[POSE] Detected people: {num_people}")

    if num_people == 0:
        return None, {
            "status": "failed",
            "reason": "no_people_detected",
            "detected_people": 0,
        }

    if boxes_raw is None or len(boxes_raw) == 0:
        print("[POSE] No person boxes available; defaulting to person index 0.")
        return 0, {
            "status": "warning",
            "reason": "no_boxes_available_defaulted_to_first_person",
            "detected_people": num_people,
            "selected_person_index": 0,
        }

    best_idx = 0
    best_area = -1.0

    for i in range(num_people):
        box = boxes_raw[i].cpu().numpy().tolist()
        x1, y1, x2, y2 = box
        area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        print(f"[POSE] person={i}, box={box}, area={area:.1f}")

        if area > best_area:
            best_area = area
            best_idx = i

    print(f"[POSE] Using person index: {best_idx}")

    return best_idx, {
        "status": "ok",
        "reason": "selected_largest_person_box",
        "detected_people": num_people,
        "selected_person_index": best_idx,
        "selected_person_box_area": best_area,
    }


def get_metric_angle(metrics, key):
    """Find an angle value in all_metrics by its canonical rule key."""
    for layer_data in metrics.values():
        if not isinstance(layer_data, dict):
            continue

        for angle in layer_data.get("angles", []):
            if angle.get("key") == key:
                return angle.get("angle_deg")

    return None


def get_metric_distance(metrics, key, unit="px"):
    """Find a distance value in all_metrics by its canonical rule key."""
    field = {
        "px": "distance_px",
        "cm": "distance_cm",
        "in": "distance_in",
    }.get(unit, "distance_px")

    for layer_data in metrics.values():
        if not isinstance(layer_data, dict):
            continue

        for distance in layer_data.get("distances", []):
            if distance.get("key") == key:
                return distance.get(field)

    return None


def validate_pose_geometry(all_metrics):
    """
    Reject poses that have acceptable YOLO confidence but impossible body geometry.
    This catches cases where YOLO guessed joints incorrectly.
    """
    errors = []

    def add_angle_asymmetry_check(left_key, right_key, label, max_diff):
        left = get_metric_angle(all_metrics, left_key)
        right = get_metric_angle(all_metrics, right_key)

        if left is None or right is None:
            errors.append(f"Missing {label} angle on one side.")
            return

        diff = abs(float(left) - float(right))
        if diff > max_diff:
            errors.append(
                f"{label} asymmetry too large: left={float(left):.1f}°, "
                f"right={float(right):.1f}°, diff={diff:.1f}°"
            )

    def add_distance_ratio_check(left_key, right_key, label, max_ratio):
        left = get_metric_distance(all_metrics, left_key, unit="px")
        right = get_metric_distance(all_metrics, right_key, unit="px")

        if left is None or right is None:
            errors.append(f"Missing {label} distance on one side.")
            return

        left = float(left)
        right = float(right)

        if left <= 1 or right <= 1:
            errors.append(f"{label} distance is too small or invalid.")
            return

        ratio = max(left, right) / min(left, right)
        if ratio > max_ratio:
            errors.append(
                f"{label} side-to-side ratio too large: left={left:.1f}px, "
                f"right={right:.1f}px, ratio={ratio:.2f}"
            )

    # DISABLED:
    # Side-view cycling photos often produce mirrored 0°/180° torso/back values.
    # These are not reliable rejection signals.

    # add_angle_asymmetry_check(
    #     "torso_angle_left_11_5_6",
    #     "torso_angle_right_12_6_5",
    #     "Torso angle",
    #     max_diff=45,
    # )

    # add_angle_asymmetry_check(
    #     "back_angle_left_11_5_6",
    #     "back_angle_right_12_6_5",
    #     "Back angle",
    #     max_diff=45,
    # )

    add_angle_asymmetry_check(
        "shoulder_angle_left_11_5_7",
        "shoulder_angle_right_12_6_8",
        "Shoulder angle",
        max_diff=45,
    )

    add_distance_ratio_check(
        "leg_length_left_11_15",
        "leg_length_right_12_16",
        "Leg length",
        max_ratio=1.5,
    )

    add_distance_ratio_check(
        "arm_reach_distance_left_5_9",
        "arm_reach_distance_right_6_10",
        "Arm reach",
        max_ratio=2.3,
    )

    return errors


NO_PERSON_POSE_MESSAGE = (
    "No rider/person pose was detected. Use Analyze Bike Only for bicycle-only images, "
    "or upload a side-view image with a rider for pose analysis."
)


def no_person_pose_response(filename=None, pose_quality=None):
    payload = {
        "success": False,
        "pose_detected": False,
        "reason": "no_person_pose",
        "message": NO_PERSON_POSE_MESSAGE,
        "filename": filename,
        "keypoints": [],
        "metrics": {}
    }
    if pose_quality is not None:
        payload["pose_quality"] = pose_quality
    return JSONResponse(status_code=200, content=payload)


@app.post("/annotate")
async def annotate_file(
        file: UploadFile = File(...),
        context: str = Form(...),
        layer: str = Form(...),
        reference_object: str = Form(None),
        reference_size: float = Form(None),
        reference_unit: str = Form(None),
):
    try:
        ext = Path(file.filename).suffix.lower()
        base_name = Path(file.filename).stem
        unique_id = uuid.uuid4().hex
        filename_stem = f"{base_name}_{unique_id}"
        input_path = MEDIA_DIR / f"{filename_stem}{ext}"

        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError("Failed to read uploaded image.")

        results = pose_model(image)

        keypoints_raw = results[0].keypoints.xy
        visible_raw = results[0].keypoints.conf

        if keypoints_raw is None or visible_raw is None:
            return no_person_pose_response(input_path.name)

        try:
            if len(keypoints_raw) == 0 or len(visible_raw) == 0:
                return no_person_pose_response(input_path.name)
        except TypeError:
            return no_person_pose_response(input_path.name)

        best_idx, person_selection = select_best_person_index(results[0])

        if best_idx is None:
            return no_person_pose_response(input_path.name, person_selection)

        keypoints = keypoints_raw[best_idx].cpu().numpy().tolist()
        visible = visible_raw[best_idx].cpu().numpy().tolist()

        print(f"[DEBUG] selected_person_index: {best_idx}")
        print(f"[DEBUG] keypoints[0]: {keypoints[:2]}")
        print(f"[DEBUG] visible[0]: {visible[:2]}")
        print(f"[DEBUG] len(keypoints): {len(keypoints)}, len(visible): {len(visible)}")

        try:
            if isinstance(visible, list) and len(visible) > 0 and isinstance(visible[0], list):
                visible = visible[0]
        except Exception as e:
            print(f"[DEBUG] Failed to flatten visible: {e}")

        if len(keypoints) < 17:
            print(f"[DEBUG] Not enough keypoints detected: {len(keypoints)}")
            return JSONResponse(status_code=200, content={
                "filename": input_path.name,
                "message": "Not enough keypoints detected.",
                "keypoints": keypoints,
                "metrics": {},
                "pose_quality": {
                    **person_selection,
                    "status": "failed",
                    "reason": "not_enough_keypoints",
                    "required_count": 17,
                    "detected_count": len(keypoints)
                }
            })

        # ---------------------------------------------------------
        # POSE QUALITY CHECK
        # Critical bike-fit keypoints:
        # 5,6   = shoulders
        # 11,12 = hips
        # 13,14 = knees
        # 15,16 = ankles
        # ---------------------------------------------------------
        required_points = [5, 6, 11, 12, 13, 14, 15, 16]
        confidence_threshold = 0.35   # If it rejects too many photos, try 0.25

        low_conf_points = [
            i for i in required_points
            if not is_valid_point(i, keypoints, visible, threshold=confidence_threshold)
        ]

        zero_points = [
            i for i in required_points
            if (
                i < len(keypoints)
                and (
                    keypoints[i] is None
                    or len(keypoints[i]) < 2
                    or (float(keypoints[i][0]) == 0.0 and float(keypoints[i][1]) == 0.0)
                )
            )
        ]

        if low_conf_points or zero_points:
            print(f"[POSE QUALITY] Low confidence keypoints: {low_conf_points}")
            print(f"[POSE QUALITY] Zero/invalid keypoints: {zero_points}")

            return JSONResponse(status_code=200, content={
                "filename": input_path.name,
                "message": (
                    "Pose confidence too low for reliable bike fit analysis. "
                    "Retake photo side-on with the full rider visible, camera level with the bike, "
                    "good lighting, and reduced background clutter."
                ),
                "keypoints": keypoints,
                "metrics": {},
                "pose_quality": {
                    **person_selection,
                    "status": "failed",
                    "reason": "low_confidence_keypoints",
                    "confidence_threshold": confidence_threshold,
                    "required_points": required_points,
                    "low_confidence_points": low_conf_points,
                    "zero_points": zero_points,
                    "visible_scores": {
                        str(i): visible[i] if i < len(visible) else None
                        for i in required_points
                    }
                }
            })

        print(f"/annotate: calling compute_conversion_factor(...reference_object={reference_object}, "
              f"reference_size={reference_size}, reference_unit={reference_unit})")

        cm_per_px, diameters_px = compute_conversion_factor(
            keypoints,
            input_path,
            reference_object,
            reference_size,
            reference_unit
        )

        print(f"/annotate: AFTER compute_conversion_factor(...cm_per_px={cm_per_px}, "
              f"diameters_px={diameters_px})")

        all_labels = []
        all_metrics = {}

        all_layer_names = list(POSE_ANGLE_LAYERS.keys()) + [
            k for k in POSE_DISTANCE_LAYERS.keys() if k not in POSE_ANGLE_LAYERS
        ]

        for layer_name in all_layer_names:
            labels, metrics = generate_layer_annotations(
                keypoints,
                visible,
                image,
                layer_name,
                cm_per_px
            )

            print(f"/annotate: calling generate_layer_annotations(...cm_per_px={cm_per_px}, "
                  f"labels={labels}, metrics={metrics})")

            for label in labels:
                label["layer"] = layer_name

            all_labels.extend(labels)
            all_metrics[layer_name] = metrics

            if "distances_by_side" in metrics:
                for side_key, dlist in metrics["distances_by_side"].items():
                    all_metrics.setdefault(side_key, {"distances": [], "angles": []})
                    all_metrics[side_key]["distances"].extend(dlist)

                del metrics["distances_by_side"]

        bike_landmarks, _ = load_bike_landmarks_for_filename(input_path.name)
        if bike_landmarks:
            append_bike_landmark_metrics(all_metrics, bike_landmarks, cm_per_px)
            append_kops_metric(all_metrics, keypoints, visible, bike_landmarks, cm_per_px)

        # ---------------------------------------------------------
        # GEOMETRY SANITY CHECK
        # Confidence can be high even when YOLO guesses wrong joints.
        # Reject impossible/asymmetric body geometry before feedback.
        # ---------------------------------------------------------
        geometry_errors = validate_pose_geometry(all_metrics)

        if geometry_errors:
            print(f"[POSE QUALITY] Geometry sanity failed: {geometry_errors}")

            return JSONResponse(status_code=200, content={
                "filename": input_path.name,
                "message": (
                    "Pose geometry appears unreliable for bike fit analysis. "
                    "Retake photo side-on with one rider visible, camera level with the bike, "
                    "full body in frame, and clearer joint visibility."
                ),
                "keypoints": keypoints,
                "metrics": {},
                "pose_quality": {
                    **person_selection,
                    "status": "failed",
                    "reason": "geometry_sanity_failed",
                    "geometry_errors": geometry_errors,
                    "confidence_threshold": confidence_threshold,
                    "required_points": required_points,
                    "visible_scores": {
                        str(i): visible[i] if i < len(visible) else None
                        for i in required_points
                    }
                }
            })

        json_output_path = MEDIA_DIR / f"{filename_stem}_layers.json"

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump({
                "keypoints": keypoints,
                "visible": visible,
                "labels": all_labels,
                "metrics": all_metrics,
                "cm_per_px": cm_per_px,
                "pose_quality": {
                    **person_selection,
                    "status": "ok",
                    "confidence_threshold": confidence_threshold,
                    "required_points": required_points
                }
            }, f, indent=2)

        return {
            "filename": f"{filename_stem}{ext}",
            "message": "Annotation complete",
            "keypoints": keypoints,
            "metrics": all_metrics,
            "pose_quality": {
                **person_selection,
                "status": "ok",
                "confidence_threshold": confidence_threshold,
                "required_points": required_points
            }
        }

    except Exception as e:
        print("[SERVER ERROR] /annotate exception:", str(e))
        return JSONResponse(status_code=500, content={
            "error": "Annotation failed.",
            "details": str(e)
        })


@app.post("/annotate_b4chg05232026")
async def annotate_file(
        file: UploadFile = File(...),
        context: str = Form(...),
        layer: str = Form(...),
        reference_object: str = Form(None),
        reference_size: float = Form(None),
        reference_unit: str = Form(None),
):
    try:
        ext = Path(file.filename).suffix.lower()
        base_name = Path(file.filename).stem
        unique_id = uuid.uuid4().hex
        filename_stem = f"{base_name}_{unique_id}"
        input_path = MEDIA_DIR / f"{filename_stem}{ext}"

        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError("Failed to read uploaded image.")

        from ultralytics import YOLO
        pose_model = YOLO("yolo26m-pose.pt")  # 05222026 - m was n originally
        results = pose_model(image)
        # Extract and flatten keypoints and visibility
        keypoints_raw = results[0].keypoints.xy
        visible_raw = results[0].keypoints.conf

        if keypoints_raw is None or visible_raw is None:
            raise ValueError("No keypoints found in pose result.")

        keypoints = keypoints_raw[0].cpu().numpy().tolist()  # (17, 2)
        visible = visible_raw[0].cpu().numpy().tolist()  # (17,)

        print(f"[DEBUG] keypoints[0]: {keypoints[:2]}")
        print(f"[DEBUG] visible[0]: {visible[:2]}")
        print(f"[DEBUG] len(keypoints): {len(keypoints)}, len(visible): {len(visible)}")

        try:
            if isinstance(visible, list) and isinstance(visible[0], list):
                visible = visible[0]
        except Exception as e:
            print(f"[DEBUG] Failed to flatten keypoints: {e}")

        if len(keypoints) < 17:
            print(f"[DEBUG] Not enough keypoints detected: {len(keypoints)}")
            return JSONResponse(status_code=200, content={
                "filename": input_path.name,
                "message": "Not enough keypoints detected.",
                "keypoints": keypoints,
                "metrics": {}
            })


        print(f"/annotate: calling compute_conversion_factor(...reference_object={reference_object}, "
              f"reference_size={reference_size}, reference_unit={reference_unit})")
        # calc for converting px to in/cm
        cm_per_px, diameters_px = compute_conversion_factor(
            keypoints,
            input_path,
            reference_object,
            reference_size,
            reference_unit
        )
        print(f"/annotate: AFTER compute_conversion_factor(...cm_per_px={cm_per_px}, "
              f"diameters_px={diameters_px})")

        all_labels = []
        all_metrics = {}

        # Collect all unique layer names from both ANGLE and DISTANCE layer sets
        all_layer_names = list(POSE_ANGLE_LAYERS.keys()) + [
            k for k in POSE_DISTANCE_LAYERS.keys() if k not in POSE_ANGLE_LAYERS
        ]

        for layer_name in all_layer_names:

            # generate_layer_annotations()
            labels, metrics = generate_layer_annotations(
                keypoints, visible, image, layer_name,
                #                reference_object, reference_size, reference_unit
                cm_per_px)  #new
            print(f"/annotate: calling generate_layer_annotations(...cm_per_px={cm_per_px}, "
                  f"labels={labels}, metrics={metrics})")


            for label in labels:
                label["layer"] = layer_name
            all_labels.extend(labels)

            # Merge base metrics
            all_metrics[layer_name] = metrics

            # Handle side-specific metrics if present
            if "distances_by_side" in metrics:
                for side_key, dlist in metrics["distances_by_side"].items():
                    all_metrics.setdefault(side_key, {"distances": [], "angles": []})
                    all_metrics[side_key]["distances"].extend(dlist)
                del metrics["distances_by_side"]  # clean up to avoid duplication

        json_output_path = MEDIA_DIR / f"{filename_stem}_layers.json"
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump({
                "keypoints": keypoints,
                "labels": all_labels,
                "metrics": all_metrics
            }, f, indent=2)

        return {
            "filename": f"{filename_stem}{ext}",
            "message": "Annotation complete",
            "keypoints": keypoints,
            "metrics": all_metrics
        }


    except Exception as e:
        print("[SERVER ERROR] /annotate exception:", str(e))
        return JSONResponse(status_code=500, content={"error": "Annotation failed.", "details": str(e)})


async def annotate_file_was_good0701(
        file: UploadFile = File(...),
        context: str = Form(...),
        layer: str = Form(...),
        reference_object: str = Form(None),
        reference_size: float = Form(None),
        reference_unit: str = Form(None),
):
    try:
        ext = Path(file.filename).suffix.lower()
        base_name = Path(file.filename).stem
        unique_id = uuid.uuid4().hex
        filename_stem = f"{base_name}_{unique_id}"
        input_path = MEDIA_DIR / f"{filename_stem}{ext}"

        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError("Failed to read uploaded image.")

        from ultralytics import YOLO
        #pose_model = YOLO("yolov8n-pose.pt") #was
        pose_model = YOLO("yolo26m-pose.pt")
        results = pose_model(image)
        # Extract and flatten keypoints and visibility
        keypoints_raw = results[0].keypoints.xy
        visible_raw = results[0].keypoints.conf

        if keypoints_raw is None or visible_raw is None:
            raise ValueError("No keypoints found in pose result.")

        keypoints = keypoints_raw[0].cpu().numpy().tolist()  # (17, 2)
        visible = visible_raw[0].cpu().numpy().tolist()  # (17,)

        print(f"[DEBUG] keypoints[0]: {keypoints[:2]}")
        print(f"[DEBUG] visible[0]: {visible[:2]}")
        print(f"[DEBUG] len(keypoints): {len(keypoints)}, len(visible): {len(visible)}")

        try:
            if isinstance(visible, list) and isinstance(visible[0], list):
                visible = visible[0]
        except Exception as e:
            print(f"[DEBUG] Failed to flatten keypoints: {e}")

        if len(keypoints) < 17:
            print(f"[DEBUG] Not enough keypoints detected: {len(keypoints)}")
            return JSONResponse(status_code=200, content={
                "filename": input_path.name,
                "message": "Not enough keypoints detected.",
                "keypoints": keypoints,
                "metrics": {}
            })

        all_labels = []
        all_metrics = {}

        all_layer_names = set(POSE_ANGLE_LAYERS.keys()).union(POSE_DISTANCE_LAYERS.keys())

        for layer_name in all_layer_names:  #POSE_ANGLE_LAYERS:
            labels, metrics = generate_layer_annotations(
                keypoints, visible, image, layer_name,
                reference_object, reference_size, reference_unit
            )
            for label in labels:
                label["layer"] = layer_name
            all_labels.extend(labels)

            # Merge base metrics
            all_metrics[layer_name] = metrics

            # Handle side-specific metrics if present
            if "distances_by_side" in metrics:
                for side_key, dlist in metrics["distances_by_side"].items():
                    all_metrics.setdefault(side_key, {"distances": [], "angles": []})
                    all_metrics[side_key]["distances"].extend(dlist)
                del metrics["distances_by_side"]  # clean up to avoid duplication

        json_output_path = MEDIA_DIR / f"{filename_stem}_layers.json"
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump({
                "keypoints": keypoints,
                "labels": all_labels,
                "metrics": all_metrics
            }, f, indent=2)

        return {
            "filename": f"{filename_stem}{ext}",
            "message": "Annotation complete",
            "keypoints": keypoints,
            "metrics": all_metrics
        }

    except Exception as e:
        print("[SERVER ERROR] /annotate exception:", str(e))
        return JSONResponse(status_code=500, content={"error": "Annotation failed.", "details": str(e)})


@app.get("/analyze")
async def analyze_overlay(layer: str = Query(...), filename: str = Query(...)):
    stem = Path(filename).stem
    json_path = MEDIA_DIR / f"{stem}_layers.json"
    image_path = MEDIA_DIR / f"{filename}"

    if not json_path.exists() or not image_path.exists():
        return JSONResponse(status_code=404, content={"error": "Required data not found."})

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_labels = [lbl for lbl in data.get("labels", []) if lbl.get("layer") == layer]
    metrics_for_layer = data.get("metrics", {}).get(layer, {})

    return {
        "blob_url": filename,
        "labels": filtered_labels,
        "keypoints": data.get("keypoints", []),
        "metrics": {
            "distances": metrics_for_layer.get("distances", []),
            "angles": metrics_for_layer.get("angles", [])
        }
    }


@app.get("/bike-landmarks")
async def get_bike_landmarks(image: str = Query(...)):
    safe_filename = Path(image).name
    stem = Path(safe_filename).stem
    exact_path = MEDIA_DIR / f"{stem}_bike_landmarks.json"
    dataset_labels_dir = MEDIA_DIR / "bike_landmark_annotations" / "labels"

    print(f"[BIKE LANDMARK] Exact lookup path: {exact_path}")

    processed_candidates = []
    dataset_candidates = []
    landmarks_path = None
    matched_by = None

    if exact_path.exists():
        landmarks_path = exact_path
        matched_by = "exact"
    else:
        processed_candidates = list(MEDIA_DIR.glob(f"{stem}_*_bike_landmarks.json"))
        if processed_candidates:
            landmarks_path = max(processed_candidates, key=lambda p: p.stat().st_mtime)
            matched_by = "processed_uuid_fallback"
        elif dataset_labels_dir.exists():
            dataset_candidates = list(dataset_labels_dir.glob(f"{stem}_*_bike_landmarks.json"))
            if dataset_candidates:
                landmarks_path = max(dataset_candidates, key=lambda p: p.stat().st_mtime)
                matched_by = "landmark_dataset_fallback"

    print(f"[BIKE LANDMARK] Processed fallback candidates: {[p.name for p in processed_candidates]}")
    print(f"[BIKE LANDMARK] Dataset fallback candidates: {[p.name for p in dataset_candidates]}")

    if not landmarks_path:
        return {
            "found": False,
            "source": None,
            "matched_by": None,
            "bike_landmarks": {}
        }

    print(f"[BIKE LANDMARK] Loaded landmarks from: {landmarks_path}")

    with open(landmarks_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return {
        "found": True,
        "source": landmarks_path.name,
        "matched_by": matched_by,
        "bike_landmarks": payload.get("bike_landmarks", {}),
        "saved_image": payload.get("saved_image"),
        "original_filename": payload.get("original_filename")
    }


@app.post("/bike-landmarks")
async def save_bike_landmarks(request: Request):
    data = await request.json()

    image_filename = data.get("image_filename") or data.get("filename")
    if not image_filename:
        return JSONResponse(status_code=400, content={"error": "Missing image filename."})

    safe_filename = Path(image_filename).name
    stem = Path(safe_filename).stem
    bike_landmarks = data.get("bike_landmarks") or {}

    payload = {
        "filename": safe_filename,
        "image_filename": safe_filename,
        "image_width": data.get("image_width"),
        "image_height": data.get("image_height"),
        "bike_landmarks": bike_landmarks
    }

    output_path = MEDIA_DIR / f"{stem}_bike_landmarks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    response_payload = {
        "message": "Bike landmarks saved.",
        "filename": safe_filename,
        "path": output_path.name
    }

    layers_path = MEDIA_DIR / f"{stem}_layers.json"
    if layers_path.exists():
        with open(layers_path, "r", encoding="utf-8") as f:
            layers_data = json.load(f)

        all_metrics = layers_data.setdefault("metrics", {})
        cm_per_px = layers_data.get("cm_per_px") or infer_cm_per_px_from_metrics(all_metrics)
        bike_metrics = append_bike_landmark_metrics(all_metrics, bike_landmarks, cm_per_px)
        print(f"[BIKE LANDMARK] Computed bike landmark metrics: {bike_metrics}")
        keypoints = layers_data.get("keypoints") or []
        visible = layers_data.get("visible") or []
        if keypoints and visible:
            append_kops_metric(all_metrics, keypoints, visible, bike_landmarks, cm_per_px)

        with open(layers_path, "w", encoding="utf-8") as f:
            json.dump(layers_data, f, indent=2)

        response_payload["metrics"] = all_metrics
        response_payload["bike_landmark_metrics"] = bike_metrics

    return response_payload


@app.post("/bike-landmark-annotation")
async def save_bike_landmark_annotation(
        file: UploadFile = File(...),
        original_filename: str = Form(...),
        image_width: str = Form(None),
        image_height: str = Form(None),
        bike_landmarks: str = Form(...),
):
    safe_original_filename = Path(original_filename or file.filename).name
    original_stem = Path(safe_original_filename).stem
    ext = Path(file.filename or safe_original_filename).suffix.lower() or ".jpg"
    unique_stem = f"{original_stem}_{uuid.uuid4().hex}"

    annotation_root = MEDIA_DIR / "bike_landmark_annotations"
    images_dir = annotation_root / "images"
    labels_dir = annotation_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    saved_image_name = f"{unique_stem}{ext}"
    saved_image_path = images_dir / saved_image_name
    with open(saved_image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        parsed_landmarks = json.loads(bike_landmarks)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid bike_landmarks JSON."})

    def parse_dimension(value):
        try:
            return int(float(value)) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None

    saved_image_relative = f"bike_landmark_annotations/images/{saved_image_name}"
    landmarks_file_name = f"{unique_stem}_bike_landmarks.json"
    landmarks_path = labels_dir / landmarks_file_name

    payload = {
        "image": saved_image_name,
        "original_filename": safe_original_filename,
        "saved_image": saved_image_relative,
        "image_width": parse_dimension(image_width),
        "image_height": parse_dimension(image_height),
        "bike_landmarks": parsed_landmarks
    }

    with open(landmarks_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "status": "success",
        "saved_image": saved_image_relative,
        "landmarks_file": f"bike_landmark_annotations/labels/{landmarks_file_name}",
        "filename": saved_image_relative,
        "stem": unique_stem
    }


@app.post("/recalculate-bike-metrics")
async def recalculate_bike_metrics(request: Request):
    data = await request.json()
    image_filename = data.get("image_filename") or data.get("filename")
    if not image_filename:
        return JSONResponse(status_code=400, content={"error": "Missing image filename."})

    safe_filename = Path(image_filename).name
    stem = Path(safe_filename).stem
    layers_path = MEDIA_DIR / f"{stem}_layers.json"
    landmarks_path = MEDIA_DIR / f"{stem}_bike_landmarks.json"

    if not layers_path.exists():
        return JSONResponse(status_code=404, content={"error": "Layers file not found."})
    if not landmarks_path.exists():
        return JSONResponse(status_code=404, content={"error": "Bike landmarks file not found."})

    with open(layers_path, "r", encoding="utf-8") as f:
        layers_data = json.load(f)

    with open(landmarks_path, "r", encoding="utf-8") as f:
        landmarks_data = json.load(f)

    all_metrics = layers_data.setdefault("metrics", {})
    cm_per_px = data.get("cm_per_px") or layers_data.get("cm_per_px") or infer_cm_per_px_from_metrics(all_metrics)
    bike_landmarks = landmarks_data.get("bike_landmarks", {})
    bike_metrics = append_bike_landmark_metrics(all_metrics, bike_landmarks, cm_per_px)
    print(f"[BIKE LANDMARK] Computed bike landmark metrics: {bike_metrics}")
    keypoints = layers_data.get("keypoints") or []
    visible = layers_data.get("visible") or []
    if keypoints and visible:
        append_kops_metric(all_metrics, keypoints, visible, bike_landmarks, cm_per_px)

    with open(layers_path, "w", encoding="utf-8") as f:
        json.dump(layers_data, f, indent=2)

    return {
        "message": "Bike landmark metrics recalculated.",
        "filename": safe_filename,
        "metrics": all_metrics,
        "bike_landmark_metrics": bike_metrics
    }


@app.post("/analyze-bike-geometry")
async def analyze_bike_geometry(request: Request):
    print("[BIKE GEOMETRY] Starting bike-only analysis")

    data = await request.json()
    image_filename = data.get("image") or data.get("filename") or data.get("image_filename")
    if not image_filename:
        return JSONResponse(status_code=400, content={"error": "Missing image filename."})

    image_path = resolve_media_file(image_filename)
    safe_filename = Path(str(image_filename).replace("\\", "/")).as_posix()
    stem = Path(safe_filename).stem

    bike_landmarks = data.get("bike_landmarks") or {}
    if not bike_landmarks:
        bike_landmarks, _ = load_bike_landmarks_for_filename(safe_filename)

    print(f"[BIKE GEOMETRY] Loaded landmarks: {bike_landmarks}")

    if not bike_landmarks:
        return {
            "message": "No bike landmarks found. Add or load bike landmarks first.",
            "metrics": {}
        }

    reference_object = data.get("reference_object")
    reference_size = data.get("reference_size")
    reference_unit = data.get("reference_unit")
    cm_per_px = None

    if image_path and image_path.exists() and reference_object and reference_size and reference_unit:
        try:
            cm_per_px, _ = compute_conversion_factor(
                [],
                image_path,
                reference_object,
                float(reference_size),
                reference_unit
            )
        except Exception as e:
            print(f"[BIKE GEOMETRY] Scale calculation skipped: {e}")

    print(f"[BIKE GEOMETRY] cm_per_px: {cm_per_px}")

    bike_metrics = calculate_bike_landmark_metrics(bike_landmarks, cm_per_px)
    print(f"[BIKE GEOMETRY] Computed metrics: {bike_metrics}")

    metrics = {}
    if bike_metrics["distances"] or bike_metrics["angles"]:
        metrics["bike_landmarks"] = bike_metrics

    layers_path = MEDIA_DIR / f"{stem}_layers.json"
    if layers_path.exists():
        with open(layers_path, "r", encoding="utf-8") as f:
            layers_data = json.load(f)
    else:
        layers_data = {
            "keypoints": [],
            "labels": [],
            "metrics": {}
        }

    layers_data.setdefault("keypoints", [])
    layers_data.setdefault("labels", [])
    layers_data.setdefault("metrics", {})
    layers_data["metrics"]["bike_landmarks"] = bike_metrics

    with open(layers_path, "w", encoding="utf-8") as f:
        json.dump(layers_data, f, indent=2)

    return {
        "message": "Bike geometry analysis complete",
        "pose_detected": False,
        "metrics": metrics,
        "filename": safe_filename
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mainbike:app",
                host="127.0.0.1",
                #host="0.0.0.0" ,
                port=8000, timeout_keep_alive=300,
                workers=1)
