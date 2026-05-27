from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from mainbike_interactive import generate_label_json

# feedback api .py stuff
from fastapi import FastAPI
from feedback_api import router as feedback_router

# suggested layout
# your_project/
# ├── mainbike.py               # Main app
# ├── feedback_api.py           # Feedback route
# ├── fit_rules.json            # Rule datastore
# ├── utils/                    # Optional: for helpers like angle/distance calculators
# ├── static/                   # Images, fonts
# ├── templates/                # HTML if using Jinja2


app = FastAPI()
app.include_router(feedback_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Annotation style configuration
ANNOTATION_STYLE = {
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "font_scale": 0.6,
    "font_color": (255,255,255), #(0, 0, 255),  # Red text
    "font_thickness": 1,
    "keypoint_color": (0, 255, 0),  # Green keypoints
    "keypoint_radius": 5,
    "connection_color": (255, 0, 0),  # Blue lines
    "connection_thickness": 2
}

REFERENCE_KEYPOINTS = {
    "wheel_diameter": (15, 16),     # left and right ankle
    "crank_length": (11, 15),       # left hip to left ankle
    "top_tube": (5, 6),             # left shoulder to right shoulder
}
'''
REFERENCE_KEYPOINTS = {
    "wheel_diameter": (15, 16),             # left ankle to right ankle (approx. bottom wheel width)
    "crank_length_left": (11, 15),          # left hip to left ankle
    "crank_length_right": (12, 16),         # right hip to right ankle
    "top_tube": (5, 6),                     # left shoulder to right shoulder
}
'''

MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="static/images"), name="images") #access images using website/images...

model = YOLO("yolov8n-pose.pt")
model.to("cpu")
wheel_model_path="C:\development\Python\projects\pose_estimation\PoseEstimation\getimages\runs\wheel_train_only_v2_06259pm\weights"
wheel_model = YOLO("../getimages/runs/wheel_train_only_v2_06259pm/weights/best.pt").to("cpu")


def_label_width=42
def_label_width_deg = 50


# Maps each layer to a pose triplet and segment for midpoint
#CONSTANTS
KEYPOINT_DICT = {
    0: "nose",
    1: "neck",
    2: "right_shoulder",
    3: "right_elbow",
    4: "right_wrist",
    5: "left_shoulder",
    6: "left_elbow",
    7: "left_wrist",
    8: "right_hip",
    9: "right_knee",
    10: "right_ankle",
    11: "left_hip",
    12: "left_knee",
    13: "left_ankle",
    14: "right_eye",
    15: "left_eye",
    16: "right_ear",
    17: "left_ear"
}

POSE_ANGLE_LAYERS = {
    "knee_angle": [(11, 13, 15), (12, 14, 16)],
    "hip_angle": [(5, 11, 13), (6, 12, 14)],
    "shoulder_angle": [(1, 5, 7), (1, 6, 8)],
    "torso_angle": [(0, 1, 5), (0, 1, 6)],
    "arm_angle": [(5, 7, 9), (6, 8, 10)],
    "back_angle": [(11, 23, 24)],
    "all_fit_points": [
        (11, 13, 15), (12, 14, 16),
        (5, 11, 13), (6, 12, 14),
        (11, 5, 7), (12, 6, 8),
        (11, 5, 6), (12, 6, 5),
        (11, 5, 6), (12, 6, 5)
    ]
}


POSE_ANGLE_SEGMENTS = {
    (11, 13, 15): (13, 15),   # Left knee
    (12, 14, 16): (14, 16),   # Right knee
    (5, 11, 13): (11, 13),    # Left hip
    (6, 12, 14): (12, 14),    # Right hip
    (5, 7, 9): (7, 9),        # Left arm
    (6, 8, 10): (8, 10),      # Right arm
    (1, 5, 7): (5, 7),        # Shoulder angle L
    (1, 6, 8): (6, 8),        # Shoulder angle R
    (0, 1, 5): (1, 5),        # Torso
    (0, 1, 6): (1, 6)         # Torso
}

POSE_CONNECTIONS = [
    (5, 7), (7, 9),     # Left arm: shoulder–elbow–wrist
    (6, 8), (8, 10),    # Right arm
    (5, 6),             # Shoulders
    (5, 11), (6, 12),   # Torso sides
    (11, 12),           # Hips
    (11, 13), (13, 15), # Left leg
    (12, 14), (14, 16)  # Right leg
]

ANNOTATION_STYLE = {
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "font_scale": 0.5,
    "font_thickness": 1,
    "font_color": (255, 255, 255),
    "keypoint_color": (0, 255, 0),
    "keypoint_radius": 4,
    "connection_color": (255, 0, 0),
    "connection_thickness": 2
}
###############END CONSTSNT




POSE_ANGLE_PAIRS = [
    # --- Lower Body ---
    (11, 13, 15),  # Left Knee: hip–knee–ankle
    (12, 14, 16),  # Right Knee
    (5, 11, 13),   # Left Hip: shoulder–hip–knee
    (6, 12, 14),   # Right Hip
    (11, 23, 13),  # Alt Left Hip: (for 25-keypoint models)
    (12, 24, 14),  # Alt Right Hip

    # --- Upper Body: Arm angles ---
    (5, 7, 9),     # Left Arm: shoulder–elbow–wrist
    (6, 8, 10),    # Right Arm

    # --- Additional arm/shoulder joints (optional granularity) ---
    (1, 5, 7),     # Neck–Left shoulder–Elbow
    (1, 6, 8),     # Neck–Right shoulder–Elbow

    # --- Optional Torso angles ---
    (0, 1, 5),     # Nose–Neck–Left Shoulder (optional)
    (0, 1, 6),     # Nose–Neck–Right Shoulder

    # --- Custom: Spinal or alignment angles (if supported) ---
    (11, 23, 24), #etc. if using full body models
]



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

# ---- helper for side label ----
def get_side_label(a, b, c=None):
    # Any one of the indices can indicate a side
    left_ids = {5, 7, 9, 11, 13, 15, 23}
    right_ids = {6, 8, 10, 12, 14, 16, 24}
    ids = {a, b} if c is None else {a, b, c}

    if ids & left_ids:
        return "Left: "
    elif ids & right_ids:
        return "Right: "
    return ""


def compute_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return round(np.degrees(angle), 1)


def filter_keypoints_and_connections2(keypoints, layer):
    # Return all indices if not filtering based on confidence
    #return set(range(len(keypoints)))
    visible = set(i for i, pt in enumerate(keypoints) if pt[0] > 0 and pt[1] > 0)
    print(f"[DEBUG] filter_keypoints_and_connections2() Visible keypoints: {visible}")
    return visible
    '''
    If you later use a model that provides keypoints.conf (like results[0].keypoints.conf from YOLOv8), you can use that to filter based on confidence like so:
    
    def filter_keypoints_and_connections(keypoints, layer, conf=None, threshold=0.3):
    if conf is not None:
        return set(i for i, c in enumerate(conf) if c > threshold)
    return set(range(len(keypoints)))
    
    And in /annotate, you'd call:

    visible = filter_keypoints_and_connections(keypoints, layer, keypoint_confidence)
    '''

    '''
    if layer == "leg":
        return {11, 12, 13, 14, 15, 16}  # hips, knees, ankles
    elif layer in ["arm", "arm_angle", "arm_dist", "reach"]:
        return {5, 6, 7, 8, 9, 10}  # shoulders, elbows, wrists
    elif layer in ["all", "all_fit_points"]:
        return set(range(len(keypoints)))
    else:
        return set(range(len(keypoints)))  # fallback: show all keypoints for unknown layers
    '''


def filter_keypoints_and_connections(keypoints, layer):
    if layer in POSE_ANGLE_LAYERS:
        triplets = POSE_ANGLE_LAYERS[layer]
        visible = {i for triplet in triplets for i in triplet}
    else:
        visible = set()

    if layer in POSE_DISTANCE_LAYERS:
        pairs = POSE_DISTANCE_LAYERS[layer]
        visible.update(i for pair in pairs for i in pair)

    # Fallback: if no angle or distance matches, show everything
    if not visible:
        visible = set(range(len(keypoints)))

    return visible

#This is valid in Python, but OpenCV's cv2.putText() doesn't support Unicode, so the degree symbol ° often won’t render correctly in images
# only solution is to convert image to unicode supported image (PIL), then save and convert back for cv2
# OR display html &deg symbol on web page...which wont work as we bake it into the actual image.
enc_degree_symbol="\u00B0"

def draw_distance_line(image, pt1, pt2, label_pos=None, label=None, color=(255, 140, 0)):
    midpoint = label_pos or ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
    cv2.line(image, pt1, pt2, color, ANNOTATION_STYLE["connection_thickness"])

    distance = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))
    # dy debug - to not display label px text .......should just do this if we are sure conversion factor is used/drawn in calling loop
    if 1==1:
        return distance

    label = label or f"{distance}px"

    # Default label position: midpoint
    if label_pos is None:
        label_pos = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)

    # Text size and box
    font = ANNOTATION_STYLE["font"]
    font_scale = ANNOTATION_STYLE["font_scale"]
    thickness = ANNOTATION_STYLE["font_thickness"]
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

    box_coords = (
        (label_pos[0], label_pos[1] - text_height - 4),
        (label_pos[0] + text_width + 6, label_pos[1] + 4)
    )

    # Background box
    overlay = image.copy()
    cv2.rectangle(overlay, box_coords[0], box_coords[1], (0, 0, 0), -1)
    ###### INTERESTING - for "All" layers, changing alpha/beta higher to 1.0 makes bgnd image clear
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    # Text
    label_width_x_offset = def_label_width_deg if "deg" in label else def_label_width #  x offset if deg in label

    cv2.rectangle(image, (label_pos[0], label_pos[1] - 14 - 4), (label_pos[0] + label_width_x_offset, label_pos[1] + 4), (0, 0, 0), -1)

    # USING annotated = draw_unicode_text(...)  to handle unicode degree symbol for angles....if we dont use here, dont bother
    cv2.putText(image, label, (label_pos[0] + 3, label_pos[1]), font, font_scale,
                color, thickness, cv2.LINE_AA)

    return distance


def draw_unicode_text(img_cv, text, position, font_path="arial.ttf", font_size=16, color=(255, 255, 255)):
    img_pil = Image.fromarray(img_cv)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

'''
YOLOv8 Pose Keypoints Mapping
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

 Useful Bicycle Fitting Angles and Distances
✅ Common Angles (3 Keypoints)
Metric Name	    Keypoints (A-B-C)	                                    Description
Knee Angle	    Hip (11 or 12) - Knee (13 or 14) - Ankle (15 or 16)	    Measures knee extension at bottom of pedal stroke
Hip Angle	    Shoulder (5 or 6) - Hip (11 or 12) - Knee (13 or 14)	Hip flexion while riding
Shoulder Angle	Ear (3 or 4) - Shoulder (5 or 6) - Elbow (7 or 8)	    Posture and handlebar reach
Elbow Angle	    Shoulder - Elbow - Wrist	                            Arm extension on handlebar
Torso Angle	    Shoulder - Hip - Horizontal axis	                    Back posture / aero position
Back Angle	    Neck - Shoulder - Hip	                                Spine alignment
Ankle Angle	    Knee - Ankle - Toe (custom, if added)	                Foot alignment during pedal stroke

**Use side-specific points (e.g. Left Hip, Left Knee, Left Ankle) depending on the camera side.

Common Distances (2 Keypoints)
Metric Name	            Keypoints (A-B)	                    Description
Leg Length (Virtual)	Hip - Ankle	                        Checks extension fit
Arm Reach Distance	    Shoulder - Wrist	                Handlebar reach measurement
Saddle to Bar Distance	Hip - Wrist (or Shoulder - Bar)	    Indicates stretch between seat and handlebar
Torso Length	        Shoulder - Hip	                    Core length and posture
Crank Length Proxy	    Knee - Ankle	                    Motion range at pedal rotation
Shoulder Width	        Left Shoulder - Right Shoulder	    Frame fit / handlebar width
Hip Width	            Left Hip - Right Hip	            Saddle width estimation

Notes for Implementation
You’ll need to choose left or right side depending on your camera angle (e.g., use left-side points for side-profile bike fitting).
For pixel distances, you can calibrate using a known reference (e.g., wheel diameter or crank length).
'''

POSE_ANGLE_LAYERS = {
    "knee_angle": [(11, 13, 15), (12, 14, 16)],
    "hip_angle": [(5, 11, 13), (6, 12, 14)],
    "shoulder_angle": [(11, 5, 7), (12, 6, 8)],
    "torso_angle": [(11, 5, 6), (12, 6, 5)],
    "back_angle": [(11, 5, 6), (12, 6, 5)]
}

# need to modify if length/connections needs 3 pts vs 2, to calculate length (see leg_length vs)
# Step 1: Declare individual layers first
POSE_DISTANCE_LAYERS = {
    "arm_reach_distance": [(5, 9), (6, 10)],  # 5=left_shoulder, 9=left_wrist; 6=right_shoulder, 10=right_wris
    "leg_length": [(11, 13, 15), (12, 14, 16)],
    "saddle_to_bar_distance": [(11, 9), (12, 10)],
}


# Step 2: Add all_fit_points using existing keys
POSE_DISTANCE_LAYERS["all_fit_points"] = (
    POSE_DISTANCE_LAYERS["arm_reach_distance"] +
    POSE_DISTANCE_LAYERS["leg_length"] +
    POSE_DISTANCE_LAYERS["saddle_to_bar_distance"]
)


#annotate logic will now treat all_fit_points as a valid layer using this
POSE_ANGLE_LAYERS["all_fit_points"] = (
    POSE_ANGLE_LAYERS.get("knee_angle", []) +
    POSE_ANGLE_LAYERS.get("hip_angle", []) +
    POSE_ANGLE_LAYERS.get("shoulder_angle", []) +
    POSE_ANGLE_LAYERS.get("torso_angle", []) +
    POSE_ANGLE_LAYERS.get("back_angle", [])
)



def detect_and_scale_from_wheels(image_path, known_diameter_in=None, target_label="rear_wheel"):
    img = cv2.imread(str(image_path))
    if img is None:
        print("[ERROR] Could not load image for wheel detection.")
        return None, {}, None

    results = wheel_model(img)[0]
    boxes = results.boxes
    names = results.names

    diameters = {}
    scale = None
    rear_wheel_px = None

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        label = names.get(cls_id, f"class_{cls_id}")
        xyxy = box.xyxy[0].cpu().numpy()
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        d = max(w, h)
        diameters[label] = d
        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 255), 2)
        cv2.putText(img, f"{label}: {int(d)}px", (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if target_label in diameters and known_diameter_in:
        scale = known_diameter_in / diameters[target_label]

    debug_path = Path(image_path).with_name("debug_wheel_overlay.jpg")
    cv2.imwrite(str(debug_path), img)

    return scale, diameters, rear_wheel_px

def convert_cm_to_in(val_cm: float) -> float:
    return round(val_cm / 2.54, 2)

def adjust_distance_metrics(distances_list: list, reference_unit: str) -> list:
    adjusted = []
    for item in distances_list:
        item_copy = item.copy()
        if reference_unit == "in" and "distance_cm" in item:
            item_copy["distance_in"] = round(item["distance_cm"] / 2.54, 2)
        adjusted.append(item_copy)
    return adjusted



def adjust_metric_units(metrics: dict, reference_unit: str) -> dict:
    """
    Converts metrics from cm to in if needed and renames keys.
    """
    adjusted = {}
    for key, val in metrics.items():
        if key.endswith("_cm_left") or key.endswith("_cm_right"):
            base_key = key.replace("_cm", "")  # e.g. "horizontal_reach_cm_left" -> "horizontal_reach_left"
            if reference_unit == "in":
                adjusted[base_key] = convert_cm_to_in(val)
            else:
                adjusted[base_key] = round(val, 2)
        else:
            adjusted[key] = round(val, 2) if isinstance(val, (int, float)) else val
    return adjusted

# Example usage in annotation pipeline
# image_path = "media/sample.jpg"
# scale_in, diameters_px = detect_and_scale_from_wheels(image_path)
# if scale_in:
#     print(f"[INFO] Scale = {scale_in:.4f} inches/px based on rear wheel diameter")
#     print(f"Front wheel = {diameters_px.get('front_wheel', 'N/A')} px")

def compute_conversion_factor(keypoints, image_path, reference_object, reference_size, reference_unit):
    # Fallback if something fails
    default_return = (None, [])

    try:
        model = wheel_model #YOLO(WHEEL_MODEL_PATH)
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


def get_midpoint(p1, p2):
    return (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))


def get_angle_midpoint(p1, p2, p3):
    """Compute the midpoint between p1 and p3 (arms of angle), slightly offset upward."""
    if p1 is None or p3 is None:
        return (0, 0)
    return (
        int((p1[0] + p3[0]) / 2),
        int((p1[1] + p3[1]) / 2) - 10
    )

def get_angle_midpoint_segment(a_pt, b_pt, c_pt, layer_name):
    """
    Returns the (start, end) points of the leader line segment.
    Midpoint is calculated visually between the outer points (a, c).
    """
    # Midpoint for label
    midpoint = ((a_pt[0] + c_pt[0]) // 2, (a_pt[1] + c_pt[1]) // 2 - 10)

    # Project a segment starting from b (joint), pointing toward midpoint
    dx = midpoint[0] - b_pt[0]
    dy = midpoint[1] - b_pt[1]
    scale = 1.0  # you can tweak this if needed
    seg_start = (b_pt[0], b_pt[1])
    seg_end = (int(b_pt[0] + dx * scale), int(b_pt[1] + dy * scale))

    return seg_start, seg_end


LEFT_BODY_PARTS = [5, 7, 9, 11, 13, 15, 23]
RIGHT_BODY_PARTS = [6, 8, 10, 12, 14, 16, 24]

def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=5):
    """Draw a dotted line between pt1 and pt2 on img."""
    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
    if dist == 0:
        return
    points = np.linspace(0, 1, int(dist // gap))
    for t in points[::2]:  # Skip every other to simulate gaps
        start = (int(pt1[0] * (1 - t) + pt2[0] * t), int(pt1[1] * (1 - t) + pt2[1] * t))
        end = (int(pt1[0] * (1 - (t + 1/len(points))) + pt2[0] * (t + 1/len(points))),
               int(pt1[1] * (1 - (t + 1/len(points))) + pt2[1] * (t + 1/len(points))))
        cv2.line(img, start, end, color, thickness)

def is_valid_point(pt):
    return (
        pt is not None
        and isinstance(pt, (list, tuple, np.ndarray))
        and len(pt) == 2
        and all(np.issubdtype(type(x), np.number) and not np.isnan(x) for x in pt)
    )

def create_line_segment(start, end, color="black", style="solid", thickness=2):
    # Convert BGR or RGB tuple to rgba() string if needed
    if isinstance(color, (list, tuple)) and len(color) in [3, 4]:
        r, g, b = color[:3]
        a = color[3] / 255 if len(color) == 4 else 1.0
        color = f"rgba({r},{g},{b},{a:.2f})"
    elif isinstance(color, str):
        pass  # already a valid CSS color string
    else:
        color = "black"

    return {
        "start": [int(start[0]), int(start[1])],
        "end": [int(end[0]), int(end[1])],
        "color": color,
        "style": style,
        "thickness": int(thickness)
    }

##########NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
def compute_distance_and_label(keypoints, visible, pair, ref_obj=None, ref_size=None, ref_unit="cm"):
    a, b = pair
    if not all(0 <= i < len(keypoints) for i in pair):
        return None
    if not all(visible[i][0] > 0.5 for i in pair):
        return None

    ptA, ptB = keypoints[a], keypoints[b]
    dist_px = np.linalg.norm(np.array(ptA) - np.array(ptB))
    dist_cm = None
    dist_in = None

    if ref_size and ref_size > 0:
        scale_factor = None
        if ref_unit == "cm":
            scale_factor = ref_size / ref_size  # identity, already cm
        elif ref_unit == "in":
            scale_factor = ref_size / ref_size * 2.54  # convert to cm
        if scale_factor:
            dist_cm = dist_px / scale_factor
            dist_in = dist_cm / 2.54

    label = f"{int(dist_px)} px"
    midpoint = [int((ptA[0] + ptB[0]) / 2), int((ptA[1] + ptB[1]) / 2)]

    return {
        "points": pair,
        "distance_px": dist_px,
        "distance_cm": dist_cm,
        "distance_in": dist_in,
        "label": label,
        "midpoint": midpoint,
        "line_segments": [[ptA[0], ptA[1], ptB[0], ptB[1]]]
    }

##########NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
def compute_angle_and_label(keypoints, visible, triplet):
    a, b, c = triplet
    if not all(0 <= i < len(keypoints) for i in triplet):
        return None
    if not all(visible[b][0] > 0.5 for b in triplet):
        return None

    ptA, ptB, ptC = keypoints[a], keypoints[b], keypoints[c]
    vec1 = np.array(ptA) - np.array(ptB)
    vec2 = np.array(ptC) - np.array(ptB)
    angle = np.degrees(np.arccos(
        np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6), -1.0, 1.0)
    ))

    label = f"{int(angle)}°"
    midpoint = [int((ptA[0] + ptC[0]) / 2), int((ptA[1] + ptC[1]) / 2)]

    return {
        "points": triplet,
        "angle_deg": angle,
        "label": label,
        "midpoint": midpoint,
        "line_segments": []
    }


####NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
def generate_layer_annotations(
    keypoints,
    visible,
    image,
    layer,
    reference_object=None,
    reference_size=None,
    reference_unit="cm"
):
    from copy import deepcopy
    annotated = image.copy()
    labels = []
    metrics = {"distances": [], "angles": []}

    if layer in POSE_ANGLE_LAYERS:
        angle_defs = POSE_ANGLE_LAYERS[layer]
        for triplet in angle_defs:
            angle_data = compute_angle_and_label(keypoints, visible, triplet)
            if angle_data:
                labels.append(angle_data)
                metrics["angles"].append(angle_data)

    if layer in POSE_DISTANCE_LAYERS:
        distance_defs = POSE_DISTANCE_LAYERS[layer]
        for pair in distance_defs:
            dist_data = compute_distance_and_label(keypoints, visible, pair, reference_object, reference_size, reference_unit)
            if dist_data:
                labels.append(dist_data)
                metrics["distances"].append(dist_data)

    return labels, metrics


def compute_distance(p1, p2):
    """Compute Euclidean distance between two points."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

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

        print(f"**[SERVER DEBUG] Received layer: {layer} reference_object={reference_object} reference_size={reference_size}  reference_unit={reference_unit}")
        ext = Path(file.filename).suffix.lower()
        unique_id = uuid.uuid4().hex[:6]
        base_name = Path(file.filename).stem
        filename_stem = f"{base_name}_{unique_id}"
        input_path = MEDIA_DIR / f"{filename_stem}{ext}"
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        is_video = ext in [".mp4", ".webm", ".mov", ".avi"]
        unique_id = uuid.uuid4().hex[:6]
        output_path = MEDIA_DIR / f"{base_name}_annot_{layer}_{unique_id}{ext}"

        img = cv2.imread(str(input_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        results = model.predict(source=img, device='cpu', save=False, verbose=False)

        ########## NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
        keypoints = results[0].keypoints.xy.cpu().numpy().tolist()
        visible = results[0].keypoints.conf.cpu().numpy().tolist()

        all_labels = []
        all_metrics = {}

        for layer_name in POSE_ANGLE_LAYERS.keys():
            labels, metrics = generate_layer_annotations(
                keypoints=keypoints,
                visible=visible,
                image=img,
                layer=layer_name,
                reference_object=reference_object,
                reference_size=reference_size,
                reference_unit=reference_unit,
            )
            for label in labels:
                label["layer"] = layer_name
            all_labels.extend(labels)
            all_metrics[layer_name] = metrics

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
            "metrics": all_metrics,
        }

    except Exception as e:
        print("[SERVER ERROR] /annotate Exception:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

    ######### ENDDDD NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW

        annotated = img.copy()
        distances_px = []
        distances_and_angles = []
        cm = None
        line_segments = []  # Add this before loop

        for kp_set in results[0].keypoints.xy:
            print(f"[DEBUG] Loaded POSE_ANGLE_LAYERS['all_fit_points']: {POSE_ANGLE_LAYERS['all_fit_points']}")

            keypoints = kp_set.cpu().numpy().astype(int)
            visible = filter_keypoints_and_connections(keypoints, layer)

            conversion_factor, diameters_px  = compute_conversion_factor(
                keypoints, input_path, reference_object, reference_size, reference_unit
            )
            print(f"[DEBUG] /annotate: Wheel diameters found: {diameters_px}")

            for idx, pt in enumerate(keypoints):
                if idx in visible:
                    cv2.circle(annotated, tuple(pt), ANNOTATION_STYLE["keypoint_radius"],
                               ANNOTATION_STYLE["keypoint_color"], -1)

            # realize is yolo detects certain points are not visible (there but hidden) this logic may not draw line or kp,
            # but distance is still calculated
            for a, b in POSE_CONNECTIONS:
                if layer == "arm_reach_distance" and {a, b} == {5, 6}:
                    continue  # Skip shoulder-to-shoulder line for clarity in arm reach layer
                if a in visible and b in visible and a < len(keypoints) and b < len(keypoints):
                    print(f"[DEBUG] Drawing line from {a} to {b}")
                    cv2.line(annotated, tuple(keypoints[a]), tuple(keypoints[b]),
                             ANNOTATION_STYLE["connection_color"], ANNOTATION_STYLE["connection_thickness"])
                else:
                    print(f"[SKIP] Line from {a} to {b}: missing or not visible")

            if layer == "arm_reach_distance":

                LEFT_TRIANGLE_COLOR = (255, 165, 0)  # Orange
                RIGHT_TRIANGLE_COLOR = (0, 128, 255)  # Blue
                for side_label, shoulder_idx, elbow_idx, wrist_idx in [("Left", 5, 7, 9), ("Right", 6, 8, 10)]:
                    if all(i in visible and i < len(keypoints) for i in [shoulder_idx, wrist_idx]):
                        shoulder = keypoints[shoulder_idx]
                        wrist = keypoints[wrist_idx]

                        if is_valid_point(shoulder) and is_valid_point(wrist):

                            # Use fallback elbow: midpoint between shoulder and wrist
                            elbow = ((shoulder[0] + wrist[0]) // 2, (shoulder[1] + wrist[1]) // 2)

                            if is_valid_point(elbow):
                                # dy 6/27 - test....see if would draw arm length on arm_reach. Only draws straight line
                                #cv2.line(annotated, tuple(shoulder), tuple(elbow), (255, 165, 0), 2)
                                #cv2.line(annotated, tuple(elbow), tuple(wrist), (255, 165, 0), 2)


                                # Proceed with triangle drawing and JSON labeling
                                # Flat horizontal base
                                projected_wrist = (wrist[0], shoulder[1])
                                horiz_base = abs(wrist[0] - shoulder[0])
                                projected_wrist = (wrist[0], shoulder[1])  # Flat base
                                base_midpoint = ((shoulder[0] + projected_wrist[0]) // 2, shoulder[1])

                                # Draw triangle (even without elbow) - alternates color from settings above
                                color = LEFT_TRIANGLE_COLOR if "Left" in side_label else RIGHT_TRIANGLE_COLOR
                                color = rgba = f"rgba({color[0]},{color[1]},{color[2]},1.0)"
                                line_segments.append(create_line_segment(shoulder, elbow, color=color, style="solid", thickness=2))
                                line_segments.append(create_line_segment(elbow, wrist, color=color, style="solid", thickness=2))
                                line_segments.append(create_line_segment(shoulder, projected_wrist, color=color, style="solid", thickness=2))
                                line_segments.append(create_line_segment(wrist, projected_wrist, color=color, style="solid", thickness=2))

                                print(f"debug - line_segments = {line_segments}")
                                #cv2.line(annotated, tuple(shoulder), tuple(elbow), color, 2)
                                #cv2.line(annotated, tuple(elbow), tuple(wrist), color, 2)
                                #draw_dotted_line(annotated, tuple(shoulder), projected_wrist, color, 2)
                                #draw_dotted_line(annotated, tuple(wrist), projected_wrist, color, 2)

                                label_text = f"{horiz_base}px"
                                if conversion_factor:
                                    cm = horiz_base * conversion_factor
                                    if reference_unit == "in":
                                        cm /= 2.54
                                        label_text = f"{side_label} Horizontal Reach: {cm:.2f} in"
                                    else:
                                        label_text = f"{side_label} Horizontal Reach: {cm:.2f} cm"

                                # adjust left and right starting point of text and leader line
                                if "Left" in side_label:
                                    label_midpoint = (base_midpoint[0] - 30, base_midpoint[1] - 10)  # shift left & up
                                else:
                                    label_midpoint = (base_midpoint[0] + 30, base_midpoint[1] - 10)  # shift right & up

                                distances_and_angles.append({
                                    "points": [int(shoulder_idx), int(elbow_idx), int(wrist_idx)],
                                    "type": "horizontal_reach",
                                    "distance_px": int(horiz_base),
                                    "distance_cm": round(float(cm), 2) if cm is not None else None,
                                    "label": label_text,
                                    "midpoint": [int(base_midpoint[0]), int(base_midpoint[1])],
                                    "line_segments": line_segments if not is_video else []
                                })

                                print(f"[DEBUG] {side_label} horizontal reach: {label_text} at {base_midpoint}")
                            else:
                                print(f"[DEBUG] reached else of: if all(is_valid_point(pt) for pt in [shoulder, elbow, wrist])")
                                print( f"[DEBUG] is_valid_point() failed for  elbow={elbow}")
                        else:
                            print(f"[SKIP] Triangle skipped due to invalid point: shoulder={shoulder}, wrist={wrist}")
                            print( f"[DEBUG] is_valid_point() failed for shoulder={shoulder}, wrist={wrist}")
                    else:
                        # 🔁 Fallback: draw shoulder-to-wrist if triangle cannot be drawn
                        if shoulder_idx in visible and wrist_idx in visible:
                            print(f"[FALLBACK] Drawing line from {shoulder_idx} to {wrist_idx}")
                            cv2.line(annotated, tuple(keypoints[shoulder_idx]), tuple(keypoints[wrist_idx]),
                                     ANNOTATION_STYLE["connection_color"], ANNOTATION_STYLE["connection_thickness"])



            # Handle angles (3 keypoints)
            if layer in POSE_ANGLE_LAYERS:
                for a, b, c in POSE_ANGLE_LAYERS[layer]:
                    if all(i in visible and i < len(keypoints) for i in [a, b, c]):
                        angle = compute_angle(keypoints[a], keypoints[b], keypoints[c])
                        seg_start, seg_end = get_angle_midpoint_segment(keypoints[a], keypoints[b], keypoints[c], layer)

                        midpoint = get_midpoint(seg_start, seg_end)
                        side_prefix = get_side_label(a, b, c)
                        prefix = "Left:" if a in LEFT_BODY_PARTS else "Right:" if a in RIGHT_BODY_PARTS else ""
                        label = f"{prefix} {round(angle, 1)}°"

                        if is_video:
                            cv2.rectangle(annotated, (midpoint[0], midpoint[1] - 18), (midpoint[0] + 70, midpoint[1]),
                                          (0, 0, 0), -1)
                            cv2.putText(annotated, label, midpoint, ANNOTATION_STYLE["font"],
                                        ANNOTATION_STYLE["font_scale"], ANNOTATION_STYLE["font_color"],
                                        ANNOTATION_STYLE["font_thickness"])

                        distances_and_angles.append({
                            "points": [a, b, c],
                            "angle_deg": round(angle, 1),
                            "label": label,
                            "midpoint": list(midpoint),
                            "line_segments": line_segments  # ensure it's [] when unused
                        })
                        print(f"[DEBUG] Drawing angle {angle} at {midpoint} for points {a}-{b}-{c}")

            # Handle distances (2 or 3 keypoints)
            if layer in POSE_DISTANCE_LAYERS or layer == "all_fit_points":
                print(f"[DEBUG] keypoints shape: {keypoints.shape}, length: {len(keypoints)}")
                max_index = keypoints.shape[0]

                for i, entry in enumerate(POSE_DISTANCE_LAYERS[layer]):
                    if len(entry) == 2:
                        a, b = entry
                        print(f"[DEBUG] Checking points {a}-{b} for layer '{layer}'")
                        if a < max_index and b < max_index and a in visible and b in visible:
                            pt1, pt2 = tuple(keypoints[a]), tuple(keypoints[b])
                            dist = compute_distance(pt1, pt2)
                            midpoint = get_midpoint(pt1, pt2)

                            label_text = f"{int(dist)} px"
                            distance_cm = None
                            if conversion_factor:
                                distance_cm = dist * conversion_factor
                                label_text = f"{distance_cm:.1f} cm"
                                if reference_unit == "in":
                                    distance_cm = distance_cm / 2.54
                                    label_text = f"{distance_cm:.1f} in"
                                prefix = "Left:" if a in LEFT_BODY_PARTS else "Right:" if a in RIGHT_BODY_PARTS else ""
                                label_text = f"{prefix} {label_text}"

                            if is_video:
                                cv2.putText(annotated, label_text, (midpoint[0], midpoint[1] + 14),
                                            ANNOTATION_STYLE["font"], ANNOTATION_STYLE["font_scale"],
                                            ANNOTATION_STYLE["font_color"], ANNOTATION_STYLE["font_thickness"])

                            distances_and_angles.append({
                                "points": [a, b],
                                "distance_px": int(dist),
                                "distance_cm": round(distance_cm, 2) if distance_cm else None,
                                "label": label_text,
                                "midpoint": list(midpoint),
                                "line_segments": line_segments  # ensure it's [] when unused
                            })
                            print(f"...adding distance to distances_and_angles ... {distances_and_angles[-1]}")

                    elif len(entry) == 3:
                        a, b, c = entry
                        print(f"[DEBUG] Checking points {a}-{b}-{c} for layer '{layer}'")
                        if all(k < max_index and k in visible for k in [a, b, c]):
                            pt1, pt2, pt3 = tuple(keypoints[a]), tuple(keypoints[b]), tuple(keypoints[c])
                            d1 = compute_distance(pt1, pt2)
                            d2 = compute_distance(pt2, pt3)
                            total_dist = d1 + d2

                            midpoint = get_midpoint(pt1, pt3)
                            label_text = f"{int(total_dist)} px"
                            distance_cm = None
                            if conversion_factor:
                                distance_cm = total_dist * conversion_factor
                                label_text = f"{distance_cm:.1f} cm"
                                if reference_unit == "in":
                                    distance_cm = distance_cm / 2.54
                                    label_text = f"{distance_cm:.1f} in"
                                prefix = "Left:" if a in LEFT_BODY_PARTS else "Right:" if a in RIGHT_BODY_PARTS else ""
                                label_text = f"{prefix} {label_text}"

                            if is_video:
                                cv2.putText(annotated, label_text, (midpoint[0], midpoint[1] + 14),
                                            ANNOTATION_STYLE["font"], ANNOTATION_STYLE["font_scale"],
                                            ANNOTATION_STYLE["font_color"], ANNOTATION_STYLE["font_thickness"])

                            distances_and_angles.append({
                                "points": [a, b, c],
                                "distance_px": int(total_dist),
                                "distance_cm": round(distance_cm, 2) if distance_cm else None,
                                "label": label_text,
                                "midpoint": list(midpoint),
                                "line_segments": line_segments  # ensure it's [] when unused
                            })
                            print(f"...adding 3-pt distance to distances_and_angles ... {distances_and_angles[-1]}")


        cv2.imwrite(str(output_path), annotated)

        print(f"[ANNOTATE] Pose layer applied: {layer}, Output path: {output_path}")

        # Example final section of your /annotate route
        distances_adjusted = adjust_distance_metrics(distances_and_angles, reference_unit)

        return JSONResponse({
            "media_type": "image/png",
            "blob_url": output_path.name,
            "metrics": {
                "distances": distances_adjusted
            },
            "labels": distances_and_angles,
            "keypoints": keypoints.tolist()
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze")
async def analyze_overlay(layer: str = Query(...), filename: str = Query(...)):
    stem = Path(filename).stem  # e.g. 'cyclist1_abc123'
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

def detect_and_draw_wheel(image_path, debug_output_path="wheel_debug.jpg", model_path="yolov8n.pt"):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WHEEL DETECT] Failed to read image: {image_path}")
        return None

    # Load YOLOv8
    model = YOLO(model_path)

    # Detect objects
    results = model.predict(image, save=False, device="cpu")[0]
    bike_boxes = []
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        print(f"[DEBUG] Detected label: {box.cls}")
        if int(cls) == 1:  # 'bicycle' class in COCO
            x1, y1, x2, y2 = map(int, box)
            bike_boxes.append((x1, y1, x2, y2))

    if not bike_boxes:
        print("[WHEEL DETECT] No bicycle detected.")
        return None

    # Use the largest detected bike box
    x1, y1, x2, y2 = max(bike_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    roi = image[y1:y2, x1:x2].copy()

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=90, #30,
        minRadius=30,
        maxRadius=200
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # Draw circle
            cv2.circle(roi, center, radius, (0, 255, 0), 2)
            cv2.putText(roi, f"{radius*2}px", (center[0]-10, center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 1)

        # Save debug image
        debug_path = Path(debug_output_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), roi)
        print(f"[WHEEL DETECT] Debug image saved: {debug_output_path}")

        return int(circles[0, 0, 2] * 2)  # return diameter in px
    else:
        print("[WHEEL DETECT] No wheels detected by HoughCircles.")
        return None


@app.get("/media/{filename}")
async def get_file(filename: str):
    file_path = MEDIA_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mainbike:app",
                host="127.0.0.1",
                #host="0.0.0.0" ,
                port=8000, timeout_keep_alive=300,
                workers=2)
