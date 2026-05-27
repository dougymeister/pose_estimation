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
            raise ValueError("No keypoints found in pose result.")

        best_idx, person_selection = select_best_person_index(results[0])

        if best_idx is None:
            return JSONResponse(status_code=200, content={
                "filename": input_path.name,
                "message": (
                    "No rider/person was detected. Retake the photo side-on with the full rider visible."
                ),
                "keypoints": [],
                "metrics": {},
                "pose_quality": person_selection
            })

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
                "labels": all_labels,
                "metrics": all_metrics,
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

    return {
        "message": "Bike landmarks saved.",
        "filename": safe_filename,
        "path": output_path.name
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mainbike:app",
                host="127.0.0.1",
                #host="0.0.0.0" ,
                port=8000, timeout_keep_alive=300,
                workers=1)
