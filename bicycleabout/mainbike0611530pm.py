from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

model = YOLO("yolov8n-pose.pt")
model.to("cpu")  # Force model to CPU

# COCO keypoint connections
POSE_CONNECTIONS = [
    (5, 7), (7, 9),  # Left Arm
    (6, 8), (8, 10),  # Right Arm
    (5, 6),          # Shoulders
    (11, 13), (13, 15),  # Left Leg
    (12, 14), (14, 16),  # Right Leg
    (11, 12)         # Hips
]

POSE_ANGLE_PAIRS = [
    (11, 13, 15),  # Left Knee
    (12, 14, 16),  # Right Knee
    (5, 11, 13),   # Left Hip
    (6, 12, 14)    # Right Hip
]


def compute_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return round(np.degrees(angle), 1)


@app.get("/")
async def serve_home():
    return FileResponse("static/index.html")


@app.post("/annotate")
async def annotate_file(
    file: UploadFile = File(...),
    context: str = Form(...),
    layer: str = Form(...)
):
    try:
        ext = Path(file.filename).suffix
        base_name = Path(file.filename).stem
        input_path = MEDIA_DIR / f"{base_name}_{uuid.uuid4().hex}{ext}"
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        output_path = MEDIA_DIR / f"{base_name}_annot_{layer}.png"
        img = cv2.imread(str(input_path))

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        results = model.predict(source=img, device='cpu', save=False, verbose=False)
        annotated = img.copy()
        text_overlay = img.copy()
        knee_angles = []

        for kp_set in results[0].keypoints.xy:
            keypoints = kp_set.cpu().numpy().astype(int)

            # Draw keypoints and lines
            for i, pt in enumerate(keypoints):
                cv2.circle(annotated, tuple(pt), 5, (0, 255, 0), -1)
                cv2.circle(text_overlay, tuple(pt), 5, (0, 255, 0), -1)

            for a, b in POSE_CONNECTIONS:
                if a < len(keypoints) and b < len(keypoints):
                    cv2.line(annotated, tuple(keypoints[a]), tuple(keypoints[b]), (255, 0, 0), 2)
                    cv2.line(text_overlay, tuple(keypoints[a]), tuple(keypoints[b]), (255, 0, 0), 2)

            for a, b, c in POSE_ANGLE_PAIRS:
                if all(i < len(keypoints) for i in [a, b, c]):
                    angle = compute_angle(keypoints[a], keypoints[b], keypoints[c])
                    knee_angles.append(angle)
                    midpoint = tuple(((np.array(keypoints[a]) + np.array(keypoints[c])) // 2).astype(int))
                    cv2.putText(text_overlay, f"{angle}°", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Save annotated layer
        cv2.imwrite(str(output_path), annotated)

        # Save text layer
        text_output_path = MEDIA_DIR / f"{base_name}_annot_{layer}_text.png"
        cv2.imwrite(str(text_output_path), text_overlay)

        return JSONResponse({
            "media_type": "image/png",
            "blob_url": output_path.name,
            "text_layer_url": text_output_path.name,
            "metrics": {
                "knee_angle": max(knee_angles) if knee_angles else None,
                "recommended": "65°–135°"
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


@app.post("/annotate-frame")
async def annotate_frame(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        results = model(img, device="cpu")  # Force CPU
        annotated_img = results[0].plot()

        # Encode to JPEG
        success, jpeg = cv2.imencode(".jpg", annotated_img)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode image")

        return Response(content=jpeg.tobytes(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mainbike:app", host="127.0.0.1", port=8000, reload=True)
