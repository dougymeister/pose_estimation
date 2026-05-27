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

app = FastAPI()

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


MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="static/images"), name="images") #access images using website/images...

model = YOLO("yolov8n-pose.pt")
model.to("cpu")

POSE_CONNECTIONS = [
    (5, 7), (7, 9),  # Left Arm
    (6, 8), (8, 10),  # Right Arm
    (5, 6),          # Shoulders
    (11, 13), (13, 15),  # Left Leg
    (12, 14), (14, 16),  # Right Leg
    (11, 12)         # Hips
]


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


def compute_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return round(np.degrees(angle), 1)


def filter_keypoints_and_connections(keypoints, layer):
    if layer == "leg":
        return {11, 12, 13, 14, 15, 16}  # hips, knees, ankles
    elif layer in ["arm", "arm_angle", "arm_dist", "reach"]:
        return {5, 6, 7, 8, 9, 10}  # shoulders, elbows, wrists
    elif layer == "all":
        return set(range(len(keypoints)))
    else:
        return set()  # fallback for unknown layer

#This is valid in Python, but OpenCV's cv2.putText() doesn't support Unicode, so the degree symbol ° often won’t render correctly in images
# only solution is to convert image to unicode supported image (PIL), then save and convert back for cv2
# OR display html &deg symbol on web page...which wont work as we bake it into the actual image.
enc_degree_symbol="\u00B0"

def draw_distance_line(image, pt1, pt2, label_pos=None, label=None, color=(255, 140, 0)):
    midpoint = label_pos or ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
    cv2.line(image, pt1, pt2, color, ANNOTATION_STYLE["connection_thickness"])

    distance = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))
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
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    # Text
    cv2.putText(image, label, (label_pos[0] + 3, label_pos[1]), font, font_scale,
                color, thickness, cv2.LINE_AA)

    return distance



@app.post("/annotate")
async def annotate_file(
    file: UploadFile = File(...),
    context: str = Form(...),
    layer: str = Form(...)
):
    try:
        ext = Path(file.filename).suffix.lower()
        base_name = Path(file.filename).stem
        input_path = MEDIA_DIR / f"{base_name}_{uuid.uuid4().hex}{ext}"
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        is_video = ext in [".mp4", ".webm", ".mov", ".avi"]
        unique_id = uuid.uuid4().hex[:6]
        output_path = MEDIA_DIR / f"{base_name}_annot_{layer}_{unique_id}{ext}"

        distances_px = []

        if is_video:
            cap = cv2.VideoCapture(str(input_path))
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(source=frame, device='cpu', save=False, verbose=False)
                annotated = frame.copy()

                for kp_set in results[0].keypoints.xy:
                    keypoints = kp_set.cpu().numpy().astype(int)
                    visible = filter_keypoints_and_connections(keypoints, layer)

                    for idx, pt in enumerate(keypoints):
                        if idx in visible:
                            cv2.circle(annotated, tuple(pt), ANNOTATION_STYLE["keypoint_radius"],
                                       ANNOTATION_STYLE["keypoint_color"], -1)

                    for a, b in POSE_CONNECTIONS:
                        if a in visible and b in visible:
                            cv2.line(annotated, tuple(keypoints[a]), tuple(keypoints[b]),
                                     ANNOTATION_STYLE["connection_color"], ANNOTATION_STYLE["connection_thickness"])

                    if layer in ["leg","arm_angle", "reach", "all"]:
                        for a, b, c in POSE_ANGLE_PAIRS:
                            if all(i in visible for i in [a, b, c]):
                                angle = compute_angle(keypoints[a], keypoints[b], keypoints[c])
                                midpoint = ((keypoints[a][0] + keypoints[c][0]) // 2,
                                            (keypoints[a][1] + keypoints[c][1]) // 2 - 10)
                                cv2.putText(annotated, f"{angle} deg", midpoint, ANNOTATION_STYLE["font"],
                                            ANNOTATION_STYLE["font_scale"], ANNOTATION_STYLE["font_color"],
                                            ANNOTATION_STYLE["font_thickness"])

                    if layer in ["arm_dist", "reach", "all"]:
                        arm_pairs = [(5, 7), (7, 9), (6, 8), (8, 10)]
                        for i, (a, b) in enumerate(arm_pairs):
                            if a in visible and b in visible:
                                pt1, pt2 = tuple(keypoints[a]), tuple(keypoints[b])
                                label_pos = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2 + (i + 1) * 25)
                                distance = draw_distance_line(annotated, pt1, pt2, label_pos=label_pos, color=(255, 140, 0))
                                distances_px.append({"points": [a, b], "distance_px": distance})

                if out is None:
                    height, width = frame.shape[:2]
                    out = cv2.VideoWriter(str(output_path), fourcc, 20.0, (width, height))
                out.write(annotated)

            cap.release()
            out.release()
            return FileResponse(output_path, media_type="video/mp4")

        # IMAGE BRANCH
        img = cv2.imread(str(input_path))
        print(f"arm_angle ...reached IMAGE BRANCH...layer={layer}")
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        results = model.predict(source=img, device='cpu', save=False, verbose=False)
        annotated = img.copy()
        text_overlay = img.copy()
        knee_angles = []

        for kp_set in results[0].keypoints.xy:
            keypoints = kp_set.cpu().numpy().astype(int)
            visible = filter_keypoints_and_connections(keypoints, layer)

            for idx, pt in enumerate(keypoints):
                if idx in visible:
                    cv2.circle(annotated, tuple(pt), ANNOTATION_STYLE["keypoint_radius"],
                               ANNOTATION_STYLE["keypoint_color"], -1)
                    cv2.circle(text_overlay, tuple(pt), ANNOTATION_STYLE["keypoint_radius"],
                               ANNOTATION_STYLE["keypoint_color"], -1)

            for a, b in POSE_CONNECTIONS:
                if a in visible and b in visible:
                    cv2.line(annotated, tuple(keypoints[a]), tuple(keypoints[b]),
                             ANNOTATION_STYLE["connection_color"], ANNOTATION_STYLE["connection_thickness"])
                    cv2.line(text_overlay, tuple(keypoints[a]), tuple(keypoints[b]),
                             ANNOTATION_STYLE["connection_color"], ANNOTATION_STYLE["connection_thickness"])

            if layer in ["leg", "arm_angle", "reach", "all"]:
                print(f"arm_angle ...reached [arm_angle,reach, all]")
                for a, b, c in POSE_ANGLE_PAIRS:
                    print(f"arm_angle text   in loop....[a,b,c]...is i in visible set?={all(i in visible for i in [a, b, c])}")
                    if all(i in visible for i in [a, b, c]):
                        angle = compute_angle(keypoints[a], keypoints[b], keypoints[c])
                        midpoint = ((keypoints[a][0] + keypoints[c][0]) // 2,
                                    (keypoints[a][1] + keypoints[c][1]) // 2 - 10)
                        print(f"arm_angle text   in loop...angle={angle}  midpoint={midpoint}")
                        cv2.putText(annotated, f"{angle} deg", midpoint, ANNOTATION_STYLE["font"],
                                    ANNOTATION_STYLE["font_scale"], ANNOTATION_STYLE["font_color"],
                                    ANNOTATION_STYLE["font_thickness"])
                        cv2.putText(text_overlay, f"{angle} deg", midpoint, ANNOTATION_STYLE["font"],
                                    ANNOTATION_STYLE["font_scale"], ANNOTATION_STYLE["font_color"],
                                    ANNOTATION_STYLE["font_thickness"])
                        fcolor=ANNOTATION_STYLE["font_color"]
                        print(f"arm_angle text  annotated= {angle} deg,  text_overlay, f{angle} deg), fontcolor={fcolor}")
            if layer in ["arm_dist", "reach", "all"]:
                arm_pairs = [(5, 7), (7, 9), (6, 8), (8, 10)]
                for i, (a, b) in enumerate(arm_pairs):
                    if a in visible and b in visible:
                        pt1, pt2 = tuple(keypoints[a]), tuple(keypoints[b])
                        label_pos = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2 + (i + 1) * 25)
                        distance = draw_distance_line(annotated, pt1, pt2, label_pos=label_pos, color=(255, 140, 0))
                        distances_px.append({"points": [a, b], "distance_px": distance})
                        draw_distance_line(text_overlay, pt1, pt2, label_pos=label_pos, color=(255, 140, 0))

        cv2.imwrite(str(output_path), annotated)
        text_output_path = MEDIA_DIR / f"{base_name}_annot_{layer}_{unique_id}_text.png"
        cv2.imwrite(str(text_output_path), text_overlay)

        return JSONResponse({
            "media_type": "image/png",
            "blob_url": output_path.name,
            "text_layer_url": text_output_path.name,
            "metrics": {
                "knee_angle": max(knee_angles) if knee_angles else None,
                "distances": distances_px,
                "recommended": "65°135 deg"
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# 06/17 12pm ###################################################
# original....prior to conditional layer request logic
#################################################################
@app.post("/annotate_orig")
async def annotate_file(
        file: UploadFile = File(...),
        context: str = Form(...),
        layer: str = Form(...)
):
    try:
        print("recieved annotate request")
        ext = Path(file.filename).suffix.lower()
        base_name = Path(file.filename).stem
        input_path = MEDIA_DIR / f"{base_name}_{uuid.uuid4().hex}{ext}"
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        is_video = ext in [".mp4", ".webm", ".mov", ".avi"]
        output_path = MEDIA_DIR / f"{base_name}_annot_{layer}{ext}"

        ''' 
        To handle requests for specific layer...need logic similar to checking layer:

        layer = form.layer  # could be: 'all', 'leg', 'arm', 'reach'
        if layer == "leg":
            # draw only leg keypoints and angle
        elif layer == "arm":
            # draw only arm keypoints and arm angle
        elif layer == "reach":
            # draw arm keypoints, angles, connections/lengths
        elif layer == "all":
            # full overlay
        '''

        print(f"recieved annotate request  video?={is_video}")
        if is_video:
            cap = cv2.VideoCapture(str(input_path))
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(source=frame, device='cpu', save=False, verbose=False)
                annotated = frame.copy()

                for kp_set in results[0].keypoints.xy:
                    keypoints = kp_set.cpu().numpy().astype(int)

                    for pt in keypoints:
                        cv2.circle(annotated, tuple(pt), ANNOTATION_STYLE["keypoint_radius"],
                                   ANNOTATION_STYLE["keypoint_color"], -1)

                    for a, b in POSE_CONNECTIONS:
                        if a < len(keypoints) and b < len(keypoints):
                            cv2.line(annotated, tuple(keypoints[a]), tuple(keypoints[b]),
                                     ANNOTATION_STYLE["connection_color"], ANNOTATION_STYLE["connection_thickness"])

                    for a, b, c in POSE_ANGLE_PAIRS:
                        if all(i < len(keypoints) for i in [a, b, c]):
                            angle = compute_angle(keypoints[a], keypoints[b], keypoints[c])
                            midpoint = tuple(((np.array(keypoints[a]) + np.array(keypoints[c])) // 2).astype(int))
                            cv2.putText(annotated, f"{angle} deg", midpoint,
                                        ANNOTATION_STYLE["font"],
                                        ANNOTATION_STYLE["font_scale"],
                                        ANNOTATION_STYLE["font_color"],
                                        ANNOTATION_STYLE["font_thickness"])

                if out is None:
                    height, width = frame.shape[:2]
                    out = cv2.VideoWriter(str(output_path), fourcc, 20.0, (width, height))
                out.write(annotated)

            cap.release()
            out.release()
            return FileResponse(output_path, media_type="video/mp4")

        else:
            img = cv2.imread(str(input_path))
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image file")

            results = model.predict(source=img, device='cpu', save=False, verbose=False)
            annotated = img.copy()
            text_overlay = img.copy()
            knee_angles = []

            for kp_set in results[0].keypoints.xy:
                keypoints = kp_set.cpu().numpy().astype(int)

                for pt in keypoints:
                    cv2.circle(annotated, tuple(pt), ANNOTATION_STYLE["keypoint_radius"],
                               ANNOTATION_STYLE["keypoint_color"], -1)
                    cv2.circle(text_overlay, tuple(pt), ANNOTATION_STYLE["keypoint_radius"],
                               ANNOTATION_STYLE["keypoint_color"], -1)

                for a, b in POSE_CONNECTIONS:
                    if a < len(keypoints) and b < len(keypoints):
                        cv2.line(annotated, tuple(keypoints[a]), tuple(keypoints[b]),
                                 ANNOTATION_STYLE["connection_color"], ANNOTATION_STYLE["connection_thickness"])
                        cv2.line(text_overlay, tuple(keypoints[a]), tuple(keypoints[b]),
                                 ANNOTATION_STYLE["connection_color"], ANNOTATION_STYLE["connection_thickness"])

                for a, b, c in POSE_ANGLE_PAIRS:
                    if all(i < len(keypoints) for i in [a, b, c]):
                        angle = compute_angle(keypoints[a], keypoints[b], keypoints[c])
                        knee_angles.append(angle)
                        midpoint = tuple(((np.array(keypoints[a]) + np.array(keypoints[c])) // 2).astype(int))
                        cv2.putText(text_overlay, f"{angle} deg", midpoint,
                                    ANNOTATION_STYLE["font"],
                                    ANNOTATION_STYLE["font_scale"],
                                    ANNOTATION_STYLE["font_color"],
                                    ANNOTATION_STYLE["font_thickness"])

            cv2.imwrite(str(output_path), annotated)
            text_output_path = MEDIA_DIR / f"{base_name}_annot_{layer}_text.png"
            cv2.imwrite(str(text_output_path), text_overlay)

            return JSONResponse({
                "media_type": "image/png",
                "blob_url": output_path.name,
                "text_layer_url": text_output_path.name,
                "metrics": {
                    "knee_angle": max(knee_angles) if knee_angles else None,
                    "recommended": "65°135 deg"
                }
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
