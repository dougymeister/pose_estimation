from pathlib import Path
from typing import List, Dict
from typing import Literal
import cv2
from PIL import Image
from annotation_generators import run_annotation_generator  # Ensure this import is present at the top
import gc
import time


class AnnotationManager:
    def __init__(self,
                 pose_output_path: Path = None):  # added None so can create class with no args -use as helper to methods
        self.pose_output_path = pose_output_path

        self._layers = [
            {"name": "annotated_base", "label": "Merged Layer", "format": "inherit",
             "description": "Base with light overlay", "generator": "generate_annotated_base"},
            {"name": "keypoints", "label": "Keypoints", "format": "inherit", "description": "Detected keypoints",
             "generator": "generate_keypoints"},
            {"name": "connections", "label": "Connections", "format": "inherit", "description": "Keypoint connections",
             "generator": "generate_connections"},
            {"name": "pose_angles", "label": "Angle  Text", "format": "inherit", "description": "Pose joint angles",
             "generator": "generate_pose_angles"},
            {"name": "pose_length", "label": "Length Text", "format": "inherit", "description": "Pose bone lengths",
             "generator": "generate_pose_length"}
        ]

        self._video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        self._image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

        image_formats_with_alpha = {'.png', '.tiff', '.webp'}
        video_formats_with_alpha = {'.mov', '.webm'}  # Alpha support depends on codec

    #helper to fetch label by name
    def get_label_for_layer(self, name: str) -> str:
        for layer in self._layers:
            if layer["name"] == name:
                return layer.get("label", name)
        return name

    def get_method_for_layer(self, name: str) -> str:
        for layer in self._layers:
            if layer["name"] == name:
                return layer.get("generator", "")
        return ""

    # use to create layer without referencing annotation_generators directly
    # usage:
    #    success = annotation_manager.generate_layer("example.jpg", "keypoints")
    #
    def generate_layer(self, base_filename: str, layer: str) -> bool:
        """
        Generate the specified annotation layer for the given base file.
        Returns True if generation was successful.
        """
        """
        Dynamically invokes the method associated with the layer to generate its output.
        Returns True if successful, False otherwise.
        """
        method_name = self.get_method_for_layer(layer)
        if not method_name:
            print(f"[Layer Generation Error] No method specified for layer '{layer}'")
            return False

        func_return = False
        input_path = Path(self.pose_output_path) / base_filename
        output_path = self.get_annotation_path(base_filename, layer)
        try:
            #            if self.is_video(str(input_path)):
            #                func_return = run_annotation_generator(base_filename, layer, self, input_path, output_path,skip_frames=skip_frames)
            #            else:
            func_return = run_annotation_generator(base_filename, layer, self, input_path, output_path)
        except Exception as e:
            print(f"[Layer Generation Error] Failed to execute '{method_name}': {e}")
            return False

        return func_return

    def get_layer_metadata(self) -> List[Dict]:
        return self._layers

    def get_layer_names(self) -> List[str]:
        return [layer["name"] for layer in self._layers]

    def get_base_stem(self, base_filename: str) -> str:
        return Path(base_filename).stem

    def is_video(self, base_filename: str) -> bool:
        print(
            f"is_video(): {base_filename}....Path(base_filename).suffix.lower() in self._video_exts={Path(base_filename).suffix.lower() in self._video_exts}")
        return Path(base_filename).suffix.lower() in self._video_exts

    def is_image(self, base_filename: str) -> bool:
        return Path(base_filename).suffix.lower() in self._image_exts

    ''' USAGE EXAMPLES
    media_type = annotation_manager.get_file_type("runner.mov")  # returns "video"
    media_type = annotation_manager.get_file_type("pose.jpg")    # returns "image"
    media_type = annotation_manager.get_file_type("data.xyz")    # returns "unknown"
    '''

    def get_file_type(self, base_filename: str) -> str:
        suffix = Path(base_filename).suffix.lower()
        if suffix in self._image_exts:
            return "image"
        elif suffix in self._video_exts:
            return "video"
        else:
            return "unknown"
        #   To raise exception...    raise ValueError(f"Unsupported file type: {suffix}")

    def get_annotation_filename(self, base_filename: str, layer: str) -> str:
        base_stem = self.get_base_stem(base_filename)
        return f"{base_stem}_annot_{layer}"

    def get_annotation_layer_from_filename(self, filename: str) -> str:
        if "_annot_" in filename:
            return filename.split("_annot_")[-1].split(".")[0]
        return ""

    def get_annotation_path(self, base_filename: str, layer: str) -> Path:
        suffix = Path(base_filename).suffix
        filename = self.get_annotation_filename(base_filename, layer) + suffix
        return self.pose_output_path / filename

    def get_output_suffix(self, base_filename: str) -> str:
        return Path(base_filename).suffix

    @staticmethod
    def supports_alpha(filepath: str | Path) -> Literal['image', 'video', 'none']:
        """
        Check if the file extension supports alpha transparency.
        Returns 'image', 'video', or 'none'.
        """
        ext = Path(filepath).suffix.lower()
        if ext in AnnotationManager.image_formats_with_alpha:
            return 'image'
        elif ext in AnnotationManager.video_formats_with_alpha:
            return 'video'
        return 'none'

    @staticmethod
    def convert_image_format(input_path: str | Path, output_path: str | Path) -> bool:
        """
        Convert image from one format to another (e.g., JPG to PNG).
        Returns True if successful.
        """
        try:
            with Image.open(input_path) as img:
                img.save(output_path)
            print(f"[Image Converted] {input_path} → {output_path}")
            return True
        except Exception as e:
            print(f"[Image Conversion Error] {e}")
            return False

    @staticmethod
    def convert_video_format(
            input_path: str | Path,
            output_path: str | Path,
            codec: str = 'mp4v'
    ) -> bool:
        """
        Convert video from one format to another using OpenCV (e.g., AVI to MP4).
        Returns True if successful.
        """
        try:
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                print(f"[Video Open Error] Could not open {input_path}")
                return False

            fourcc = cv2.VideoWriter_fourcc(*codec)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frame_count += 1

            cap.release()
            out.release()
            print(f"[Video Converted] {input_path} → {output_path}  Frames: {frame_count}")
            return True
        except Exception as e:
            print(f"[Video Conversion Error] {e}")
            return False

    # usage example
    #manager = AnnotationManager(pose_output_path=Path("media"))
    #success = manager.generate_layer("pose_angles", Path("media/input.mp4"), Path("media/output_pose_angles.png"))

    def _copy_file(self, src: Path, dst: Path) -> bool:
        """Simple file copy for stub methods (acts as a placeholder for actual logic)."""
        try:
            with open(src, 'rb') as f_in, open(dst, 'wb') as f_out:
                f_out.write(f_in.read())
            return True
        except Exception as e:
            print(f"[File Copy Error] {e}")
            return False

    @staticmethod
    def safe_delete(path: Path) -> bool:
        """
        Attempts to safely delete a file at the given path.
        Ensures resources are cleaned up and handles common exceptions.

        Args:
            path (Path): The file path to delete.

        Returns:
            bool: True if successfully deleted or does not exist, False if an error occurred.
        """
        print(f"**REMOVING**   safe_delete(path: Path)= {path}")
        try:
            if path.exists():
                gc.collect()
                time.sleep(0.1)  # Give time for OS to release file locks
                path.unlink()
                print(f"[SafeDelete] Deleted: {path}")
            return True
        except Exception as e:
            print(f"[SafeDelete Error] Could not delete {path}: {e}")
            return False
