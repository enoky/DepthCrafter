import gc
import os
import numpy as np
import torch
import argparse
from diffusers.training_utils import set_seed

from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from depthcrafter.utils import save_video, read_video_frames  # Removed vis_sequence_depth import


class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload: str = "model",
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            subfolder="unet",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # Load weights of other components from the provided checkpoint
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # For saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
        if cpu_offload is not None:
            if cpu_offload == "sequential":
                # This will be slower but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown CPU offload option: {cpu_offload}")
        else:
            self.pipe.to("cuda")
        # Enable attention slicing and xformers memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
            print("Xformers is not enabled")
        self.pipe.enable_attention_slicing()

    def infer(
            self,
            video: str,
            num_denoising_steps: int,
            guidance_scale: float,
            save_folder: str = "./demo_output",
            window_size: int = 110,
            process_length: int = 195,
            overlap: int = 25,
            max_res: int = 1024,
            target_fps: int = 15,
            seed: int = 42,
            track_time: bool = True,
            save_npz: bool = False,
    ):
        set_seed(seed)

        frames, target_fps, original_aspect_ratio, original_height, original_width, width_64, height_64 = read_video_frames(
            video, process_length, target_fps, max_res
        )
        print(f"==> Video name: {video}, frames shape: {frames.shape}")

        # Inference the depth map using the DepthCrafter pipeline
        with torch.inference_mode():
            res = self.pipe(
                frames,
                height=height_64,  # processed height (multiple of 64)
                width=width_64,  # processed width (multiple of 64)
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=window_size,
                overlap=overlap,
                track_time=track_time,
            ).frames[0]
        # Convert the three-channel output to a single-channel depth map
        res = res.sum(-1) / res.shape[-1]
        # Normalize the depth map to [0, 1] across the whole video
        res = (res - res.min()) / (res.max() - res.min())
        # Save the depth map with the target FPS
        save_path = os.path.join(
            save_folder, os.path.splitext(os.path.basename(video))[0]
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_npz:
            np.savez_compressed(save_path + ".npz", depth=res)
        save_video(
            res,
            save_path + "_depth.mp4",
            fps=target_fps,
            original_aspect_ratio=original_aspect_ratio,
            original_height=original_height,
            original_width=original_width
        )
        return [save_path + "_depth.mp4"]

    def run(
        self,
        input_video,
        num_denoising_steps,
        guidance_scale,
        max_res=1024,
        process_length=195,
    ):
        res_path = self.infer(
            input_video,
            num_denoising_steps,
            guidance_scale,
            max_res=max_res,
            process_length=process_length,
        )
        # Clear the cache for the next video
        gc.collect()
        torch.cuda.empty_cache()
        return res_path


if __name__ == "__main__":
    # Running configs
    # The most important arguments for memory saving are `cpu_offload`, `enable_xformers`, `max_res`, and `window_size`
    # The most important arguments for trade-off between quality and speed are
    # `num_inference_steps`, `guidance_scale`, and `max_res`
    parser = argparse.ArgumentParser(description="DepthCrafter")
    parser.add_argument(
        "--directory", type=str, default="./examples/scenes", help="Path to the directory with video files"
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default="./demo_output",
        help="Folder to save the output",
    )
    parser.add_argument(
        "--unet-path",
        type=str,
        default="model",
        help="Path to the UNet model",
    )
    parser.add_argument(
        "--pre-train-path",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid-xt",
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--process-length", type=int, default=0, help="Number of frames to process"
    )
    parser.add_argument(
        "--cpu-offload",
        type=str,
        default="model",
        choices=["model", "sequential", None],
        help="CPU offload option",
    )
    parser.add_argument(
        "--target-fps", type=int, default=24000/1001, help="Target FPS for the output video"
    )  # -1 for original fps
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-inference-steps", type=int, default=6, help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=1.2, help="Guidance scale"
    )
    parser.add_argument("--window-size", type=int, default=110, help="Window size")
    parser.add_argument("--overlap", type=int, default=25, help="Overlap size")
    parser.add_argument("--max-res", type=int, default=640, help="Maximum resolution")
    parser.add_argument("--save_npz", type=bool, default=False, help="Save npz file")
    parser.add_argument("--track_time", type=bool, default=False, help="Track time")

    args = parser.parse_args()

    depthcrafter_demo = DepthCrafterDemo(
        unet_path=args.unet_path,
        pre_train_path=args.pre_train_path,
        cpu_offload=args.cpu_offload,
    )

    # Get all video files from the directory
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]  # Add other video extensions if needed
    video_files = [
        os.path.join(args.directory, file)
        for file in os.listdir(args.directory)
        if any(file.endswith(ext) for ext in video_extensions)
    ]

    # Process each video file in the directory
    for video in video_files:
        depthcrafter_demo.infer(
            video,
            args.num_inference_steps,
            args.guidance_scale,
            save_folder=args.save_folder,
            window_size=args.window_size,
            process_length=args.process_length,
            overlap=args.overlap,
            max_res=args.max_res,
            target_fps=args.target_fps,
            seed=args.seed,
            track_time=args.track_time,
            save_npz=args.save_npz,
        )
        # Clear the cache for the next video
        gc.collect()
        torch.cuda.empty_cache()
