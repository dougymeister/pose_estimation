from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
import uvicorn, os, tempfile
from fastapi.staticfiles import StaticFiles
from pose_estimation_callable1 import process_pose
from pathlib import Path

#import pose_estimation_callable1

app = FastAPI()

#app.mount("/", StaticFiles(directory="static",html = True), name="static")

@app.post("/process/")
async def process(file: UploadFile, scale_percent: float = Form(100.0)):
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, file.filename)
    output_path = os.path.join(temp_dir, f"annotated_{file.filename}")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    process_pose(input_path, output_path, scale_percent)
    return {"output_url": f"/download/{os.path.basename(output_path)}"}


@app.get("/download/{filename}")
async def download(filename: str):
    return FileResponse(os.path.join(tempfile.gettempdir(), filename))

@app.get("/")
async def read_index():
    index_file = Path(__file__).parent / "index.html"
    return FileResponse(index_file)


#def read_root():
#    return {"Hello": "World"}

if __name__ == "__main__":
    uvicorn.run("fastapi-aws-deployment.ec2.app.main:app", host="127.0.0.1", port=8000, reload=True)

# to RUN uvicorn in pycharm
#  uvicorn fastapi-aws-deployment.ec2.app.main:app --reload --host 127.0.0.1 --port 8000
# ** seems to look for libs in  File "C:\Users\dougy\anaconda3\Lib\site-packages\
