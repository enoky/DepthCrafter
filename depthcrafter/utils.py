import numpy as np
import cv2
import matplotlib.cm as cm
import torch

def read_video_frames(video_path, process_length, target_fps, max_res):
    # a simple function to read video frames
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_aspect_ratio = original_width / original_height

    # Resize the video while maintaining the original aspect ratio if height or width exceeds max_res
    if max(original_height, original_width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = int(original_height * scale)
        width = int(original_width * scale)
    else:
        height = original_height
        width = original_width

    # Ensure the dimensions are divisible by 64 (for model requirements)
    height_64 = round(height / 64) * 64
    width_64 = round(width / 64) * 64

    if target_fps < 0:
        target_fps = original_fps

    stride = max(round(original_fps / target_fps), 1)

    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (process_length > 0 and frame_count >= process_length):
            break
        if frame_count % stride == 0:
            # Resize frame to be divisible by 64 for processing
            frame = cv2.resize(frame, (width_64, height_64))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frames.append(frame.astype("float32") / 255.0)
        frame_count += 1
    cap.release()

    frames = np.array(frames)
    return frames, target_fps, original_aspect_ratio, original_height, original_width, width_64, height_64



def save_video(
    video_frames,
    output_video_path,
    fps: int = 15,
    original_aspect_ratio: float = None,
    original_height: int = None,
    original_width: int = None,
) -> str:
    # The frames are processed at dimensions that are multiples of 64
    height_64, width_64 = video_frames[0].shape[:2]
    is_color = video_frames[0].ndim == 3
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Resize the saved frames to exactly half of the original resolution
    if original_aspect_ratio and original_height and original_width:
        target_height = original_height // 2  # exact half of the original height
        target_width = original_width // 2    # exact half of the original width
    else:
        target_height = height_64 // 2
        target_width = width_64 // 2

    # Video writer will use the exact half dimensions
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps, (target_width, target_height), isColor=is_color
    )

    for frame in video_frames:
        frame = (frame * 255).astype(np.uint8)
        if is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Resize frame to half of the original resolution
        frame_resized = cv2.resize(frame, (target_width, target_height))
        video_writer.write(frame_resized)

    video_writer.release()
    return output_video_path



class ColorMapper:
    # a color mapper to map depth values to a certain colormap
    def __init__(self, colormap: str = "inferno"):
        self.colormap = torch.tensor(cm.get_cmap(colormap).colors)

    def apply(self, image: torch.Tensor, v_min=None, v_max=None):
        # assert len(image.shape) == 2
        if v_min is None:
            v_min = image.min()
        if v_max is None:
            v_max = image.max()
        image = (image - v_min) / (v_max - v_min)
        image = (image * 255).long()
        image = self.colormap[image]
        return image


def vis_sequence_depth(depths: np.ndarray, v_min=None, v_max=None):
    visualizer = ColorMapper()
    if v_min is None:
        v_min = depths.min()
    if v_max is None:
        v_max = depths.max()
    res = visualizer.apply(torch.tensor(depths), v_min=v_min, v_max=v_max).numpy()
    return res
