from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from fastapi.routing import APIRoute
import os
from pathlib import Path
import shutil

import uvicorn, os, tempfile
import logging

from pose_estimation_callable1 import process_pose, generate_annotation_files

from annotation_manager import AnnotationManager
from annotation_generators import run_annotation_generator


# import pose_estimation_callable1

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Available routes:")
    for route in app.routes:
        if isinstance(route, APIRoute):
            print(f"{route.path} [{','.join(route.methods)}]")
    yield  # Startup done — let the app run


app = FastAPI()  # lifespan=lifespan)

UPLOAD_DIR = Path("uploaded")
ANNOTATION_DIR = Path("annotations")
STATIC_DIR = Path("static")
POSE_OUTPUT_PATH = "media"

UPLOAD_DIR.mkdir(exist_ok=True)
ANNOTATION_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
Path(POSE_OUTPUT_PATH).mkdir(exist_ok=True)

#POSE_OUTPUT_PATH = "static/pose_images/outputs"
#POSE_MEDIA_REF = "media"  # URL to request media files
CHUNK_SIZE = 1024 * 1024


# new python module to support annotation layers, naming conventions, layer names, gen annot filenames, helper funcs
#POSE_OUTPUT_PATH = Path("media")
annotation_manager = AnnotationManager(Path(POSE_OUTPUT_PATH))


#Dynamic vs. Static Serving: Use route handlers for dynamic file serving when you need to process or validate requests.
# Use StaticFiles for serving static content without additional logic.
app.mount("/media", StaticFiles(directory=POSE_OUTPUT_PATH), name="media")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

#TODO: dont expose during PROD.....use ENV var to check
#if os.getenv("ENV") != "production":
'''
@app.get("/routes")
async def list_routes():
    return [
        {"path": r.path, "methods": list(r.methods)}
        for r in app.routes
        if isinstance(r, APIRoute)
    ]
'''


@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"--> {request.method} {request.url}")
    response = await call_next(request)
    return response

@app.get("/annotation/layers")
async def get_annotation_layer_metadata():
    return annotation_manager.get_layer_metadata()


########################################################################
# file uploaded to server to process by calling method duplicate of @app.post("/generate/annotation/{layer}")
# but we cant use @app.post Form() input, so creating similar method to process.
#
#   async def upload_base(file: UploadFile = File(...)):
#       file: The name of the parameter that will hold the uploaded file.
#       UploadFile: A FastAPI type for handling uploaded files.
#       = File(...): Tells FastAPI this parameter should come from an uploaded file (in a multipart/form-data POST request).
#       The ... means it's required.
#   uploaded file saved to upload, and copy in media to make available for query
#   we will write processed output to media also just to see if easier
#
@app.post("/upload/base")
async def upload_base(file: UploadFile = File(...)):
    print(f"upload_base(): upload for file {file.filename}")
    base_path = UPLOAD_DIR / file.filename
    with open(base_path, "wb") as f:  # writes base image to
        shutil.copyfileobj(file.file, f)

    # Also copy to media dir for client access
    shutil.copy(base_path, Path(POSE_OUTPUT_PATH) / file.filename)

    # now call generate_annotation_core() to create our default base annotation layer
    # we will decide (probably from config file) what that consists of.
    # TODO: layers (base_default, etc) need to be in dict - { layer_name, method, layers to use list }
    #input_media_copy = POSE_OUTPUT_PATH + "/" + file.filename
    # TODO: hardcoded scale....would need to pass from form
    scale_percent = 100.0
    await generate_annotation_core("annotated_base", file.filename, scale_percent)
    return {"message": "File uploaded successfully."}


# same as web call below, but we need to call locally and web call has Form()
async def generate_annotation_core(layer: str, base_filename: str, scale_percent: float = 100.0):
    print(f"generate_annotation_core(): Generating {layer} for {base_filename}  scale_percent={scale_percent}")


    input_path = Path(POSE_OUTPUT_PATH) / base_filename

    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Base file not found")

    # Determine file extension
    suffix = input_path.suffix.lower()

    # dy 6/5 - new method using AnnotationManager class, replace code below it
    output_path = annotation_manager.get_annotation_path(base_filename, layer)
    # Set output path depending on file type
    # output get ext/stem removed
    #if suffix in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
    #    output_path = input_path.with_name(f"{input_path.stem}_annot_{layer}.png")
    #elif suffix in [".mp4", ".avi", ".mov", ".mkv"]:
    #    output_path = input_path.with_name(f"{input_path.stem}_annot_{layer}.mp4")
    #else:
    #    raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
    # Debug info
    print(f"generate_annotation_core(): Input path: {input_path}")
    print(f"generate_annotation_core(): Output path: {output_path}")
    print(
        f"generate_annotation_core(): File type: {'video' if suffix in ['.mp4', '.avi', '.mov', '.mkv'] else 'image'}")
    #output_path = Path(POSE_OUTPUT_PATH) / f"{base_filename}_annot_{layer}.png"

    # Do your processing here
    # providing input_path=src to process,  output_path=folder/name of annot layer to create.
    # input_path is src image (base), output_path is dest (media)
    # scale_percent would need to get value from Form (we dont have for upload/..), default to 100.
    #   TODO: Client would need to pass in upload_base then to generate_annotation_core
    success = False
    #scale_percent = 100.0  # see TODO above
    output_csv = output_path.with_suffix(".csv")  # change ext for csv

    # removing any existing annot layers (other than core) so we start fresh each time, otherwise client grabs any
    # existing on button click
    input_path_stem= Path(input_path).stem
    print(f"input_path stem..........{input_path_stem}....output_path={output_path}")
    for layer in annotation_manager.get_layer_names():
        if not layer == "annotated_base":
            old_layer_path_to_remove = Path(POSE_OUTPUT_PATH) / Path(f"{input_path_stem}_annot_{layer}{input_path.suffix}")

            print(f"generate_annotation_core(): cleanup on old layers, calling annotation_manager.safe_delete() on {old_layer_path_to_remove}")
            print(f"...does file exist {old_layer_path_to_remove.exists()}")
            annotation_manager.safe_delete(old_layer_path_to_remove)
            # example: f"{base_filename}_annot_keypoints.png"), keypoints_layer

    logger.info(
        f"generate_annotation_core(): calling process_pose(input_path={input_path}, output_path={output_path}, scale_percent={scale_percent})")

    # return status, and optional message ("", None). Message should be populated if error to provide back to client
    input_path_str = str(input_path)  #process_pose expects string paths
    output_path_str = str(output_path)
    output_cs_str = str(output_csv)



    success, message = process_pose(input_path, output_path, output_csv, scale_percent)

    logger.info(f"generate_annotation_core(): returns status {success} {message}  layer: {layer}")
    response = {"message": f"status {success} {message}  layer: {layer}"}
    return response


# Replace the annotation generation block in generate_annotation()
@app.post("/generate/annotation/{layer}")
async def generate_annotation(layer: str, base_filename: str = Form(...)): #, skip_frames: int = Form(0) ):
    print(f"generate_annotation(): Generating {layer} for {base_filename}")
    input_path = Path(POSE_OUTPUT_PATH) / base_filename

    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Base file not found")

    # Use AnnotationManager to determine file output path
    output_path = annotation_manager.get_annotation_path(base_filename, layer)

    print(f"generate_annotation(): CALLING annotation_manager.generate_layer({Path(base_filename)}, {layer} ") #, skip_frames={skip_frames}")
    # Call centralized generator dispatcher
    success = annotation_manager.generate_layer(base_filename, layer) #, skip_frames=skip_frames) # amgr creates  output_path)
    print(f"generate_annotation(): annotation_manager.generate_layer  returns {success}")

    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to generate layer {layer}")

    return {"message": f"Layer {layer} generated successfully."}

########################################################################
#   receive a file to create annotation layer from.
#   inputs:  layer name to create
#          source file
#
#   calls process() to create annotated layer
#   returns path to annotated file for client to load/display
#
# possible layer requests ['annotated_base', 'keypoints', 'connections', 'pose_angles', 'pose_length'];
@app.post("/generate/annotation2/{layer}")
async def generate_annotation2(layer: str, base_filename: str = Form(...)):
    print(f"generate_annotation(): Generating {layer} for {base_filename}")
    input_path = Path(POSE_OUTPUT_PATH) / base_filename

    # dy 6/5 - new class method to handle file names, replaces code below it
    output_path = annotation_manager.get_annotation_path(base_filename, layer)
    # dy 6/5 Strip extension from filename for output file
    #base_filename_no_stem = Path(base_filename).stem
    #output_path = Path(POSE_OUTPUT_PATH) / f"{base_filename_no_stem}_annot_{layer}.png"

    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Base file not found")

    # providing input_path=src to process,  output_path=folder/name of annot layer to create.
    #process(input_path, output_path, scale_percent: float = Form(100.0))
    # process(request: Request, file: UploadFile, background_tasks: BackgroundTasks, scale_percent: float = Form(100.0)):
    success = False
    scale_percent = 100.0  #float = Form(100.0)
    output_csv = output_path.with_suffix(".csv")  # change ext for csv
    input_path_str = str(input_path)  # process_pose expects string paths
    output_path_str = str(output_path)
    output_cs_str = str(output_csv)

    success, message, list_file_generated = generate_annotation_files(input_path, output_path, output_csv,
                                                                      scale_percent)
    #success, message = process_pose(input_path, output_path, output_csv, scale_percent)

    print(
        f"generate_annotation(): ############## returns status {success} {message}  layer: {layer}  list_file_generated={list_file_generated}")

    # dy 6/5 - new class method to handle file names, replaces code below it
    suffix = annotation_manager.get_output_suffix(base_filename)
    # testing for video...... return files list
    #suffix = input_path.suffix.lower()

    # Get the folder from output_path
    input_path_folder = input_path.parent   #Path object
    output_dir_folder = output_path.parent
    out_filename = output_path.name
    if list_file_generated and get_file_type(suffix) == "video":
        for vid_created_file in list_file_generated:
            print(f"    generate_annotation(): copying annot video layers.... shutil.copy({vid_created_file}, {output_dir_folder/ out_filename}")
            shutil.copy(vid_created_file, output_dir_folder/ out_filename)

    response = {"message": f"status {success} {message}  layer: {layer}"}
    return response

    # Placeholder annotation: just copy input for now - will be created layer
    #  process will write to output_path directly when integrated
    #    shutil.copy(input_path, output_path)
    #return {"message": f"{layer} generated."}


###test
@app.get("/annotation/{filename}/{layer}")
async def get_annotation_layer(filename: str, layer: str):
    logger.info(f"get_annotation_layer() - filename={filename}, layer={layer}")

    # Ensure clean filename
    filename = Path(filename).name
    logger.info(f"Sanitized filename = {filename}")

    #dy added 6/5...troubleshoot
    # Strip extension from filename.........filename=base file, we want to send back the layer file. Need to remove filename.ext to match
    #filename = Path(filename).stem

    # dy later 6/5 - new class will manage filename creation, access. Then can delete line above (stem) as class handles it
    file_path = annotation_manager.get_annotation_path(filename, layer)
    #file_path = Path(POSE_OUTPUT_PATH) / f"{filename}_annot_{layer}.png"

    logger.info(f"get_annotation_layer(): file_path.resolve(): {file_path.resolve()}")
    if not file_path.exists():
        logger.warning("get_annotation_layer(): File does not exist on disk")
        raise HTTPException(status_code=404, detail="Annotation not found")


    logger.info(f"get_annotation_layer(): returning file_path {file_path}")
    return FileResponse(file_path)


   # logger.info(f"get_annotation_layer(): file_path.resolve(): {file_path.resolve()}")
    #if not file_path.exists():
     #   logger.warning("get_annotation_layer(): File does not exist on disk")
      #  raise HTTPException(status_code=404, detail="Annotation not found")




@app.head("/annotation/{filename}/{layer}")
async def head_annotation_layer(filename: str, layer: str):
    logger.info(f"app.head()   get_annotation_layer() - filename={filename}, layer={layer}")
    filename = Path(filename).name

    # dy later 6/5 - new class will manage filename creation, access.
    file_path = annotation_manager.get_annotation_path(filename, layer)
    #file_path = Path(POSE_OUTPUT_PATH) / f"{filename}_annot_{layer}.png"
    if not file_path.exists():
        raise HTTPException(status_code=404)
    return Response(status_code=200)


def get_file_type(suffix: str) -> str:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    suffix = suffix.lower()

    if suffix in image_exts:
        return "image"
    elif suffix in video_exts:
        return "video"
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/process/")
async def process(request: Request, file: UploadFile, background_tasks: BackgroundTasks,
                  scale_percent: float = Form(100.0)):
    # file: UploadFile, scale_percent: float = Form(100.0)):

    logger.debug("process()....entered, uploaded file: " + file.filename)
    logger.debug("process() root_path" + request.scope.get("root_path"))

    base_dir = Path(__file__).parent
    file_path_in = base_dir / 'static' / 'pose_images' / 'inputs' / file.filename  # 'cyclist1.jpg'
    file_path_out = base_dir / 'static' / 'pose_images' / 'outputs' / file.filename  # 'cyclist1.jpg'

    input_path = str(file_path_in)
    output_path = str(file_path_out)
    logger.debug(f"post.... input_path {input_path}   output_path={output_path}")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    logger.info(
        f"post...calling process_pose(input_path={input_path}, output_path={output_path}, scale_percent={scale_percent}")
    output_csv = output_path + ".csv"
    process_pose(input_path, output_path, output_csv, scale_percent)
    logger.debug(f"post..returned from process_pose()")

    # Queue generate_annotation_files() to run in the background
    background_tasks.add_task(generate_annotation_files, input_path, output_path, output_csv, scale_percent)

    # returning relative path that the client JS will use to build full url to serve video
    base_name, original_ext = os.path.splitext(input_path)
    media_path = f"{POSE_OUTPUT_PATH}/{file.filename}_all" + original_ext  # {POSE_OUTPUT_PATH}/{file.filename}"
    logger.info(f"process():  POST..returning output_url:{media_path}")
    return {"output_url": media_path}  # output_url}
    # sample returned:   "/static/pose_images/outputs/cyclist1.jpg


if __name__ == "__main__":
    uvicorn.run("fastapi-aws-deployment.ec2.app.main:app", host="127.0.0.1", port=8009, timeout_keep_alive=300,
                workers=4, log_level="debug")  # reload=True,
