import threading
import gc
import os
import glob
import shutil
import json
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from diffusers.training_utils import set_seed
from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from depthcrafter.utils import save_video, read_video_frames

class DepthCrafterDemo:
    """
    Class to handle the DepthCrafter inference.
    """
    def __init__(self, unet_path: str, pre_train_path: str, cpu_offload: str = "model"):
        """
        Initializes the DepthCrafter pipeline.

        Args:
            unet_path (str): Path to the UNet model.
            pre_train_path (str): Path to the pre-trained model.
            cpu_offload (str, optional): CPU offload strategy ('model', 'sequential'). Defaults to "model".
        """
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        if cpu_offload == "sequential":
            self.pipe.enable_sequential_cpu_offload()
        elif cpu_offload == "model":
            self.pipe.enable_model_cpu_offload()
        else:
            raise ValueError(f"Unknown CPU offload option: {cpu_offload}")
        self.pipe.enable_attention_slicing()

    def infer(self, video, num_denoising_steps, guidance_scale, save_folder, window_size, process_length, overlap, max_res, seed):
        """
        Performs depth inference on a video.

        Args:
            video (str): Path to the input video.
            num_denoising_steps (int): Number of denoising steps.
            guidance_scale (float): Guidance scale for inference.
            save_folder (str): Folder to save the depth map video.
            window_size (int): Window size for temporal processing.
            process_length (int): Number of frames to process.
            overlap (int): Overlap between windows.
            max_res (int): Maximum resolution of the video.
            seed (int): Random seed for reproducibility.

        Returns:
            str: The save path of the depth map video
        """
        set_seed(seed)
        frames, target_fps = read_video_frames(video, process_length, -1, max_res, "open")
        with torch.inference_mode():
            res = self.pipe(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=window_size,
                overlap=overlap,
            ).frames[0]
        res = res.sum(-1) / res.shape[-1]
        res = (res - res.min()) / (res.max() - res.min())
        save_path = os.path.join(save_folder, os.path.splitext(os.path.basename(video))[0])
        os.makedirs(save_folder, exist_ok=True)
        save_video(res, save_path + "_depth.mp4", fps=target_fps)
        return save_path + "_depth.mp4"

    def run(self, video, **kwargs):
        """
        Runs the depth inference and handles cleanup.

        Args:
            video (str): Path to the input video.
            **kwargs: Additional parameters for inference.
        """
        self.infer(video, **kwargs)
        gc.collect()
        torch.cuda.empty_cache()


class DepthCrafterGUI:
    """
    GUI class for the DepthCrafter application.
    """
    CONFIG_FILENAME = "config.json"

    def __init__(self, root):
        """
        Initializes the GUI.

        Args:
            root (tk.Tk): The main Tkinter window.
        """
        self.root = root
        self.root.title("DepthCrafter GUI")

        # Default values before loading config
        self.input_dir = tk.StringVar(value="./input_clips")
        self.output_dir = tk.StringVar(value="./output_depthmaps")
        self.guidance_scale = tk.DoubleVar(value=1.0)
        self.inference_steps = tk.IntVar(value=5)
        self.window_size = tk.IntVar(value=110)
        self.max_res = tk.IntVar(value=960)
        self.overlap = tk.IntVar(value=25)
        self.seed = tk.IntVar(value=42)
        self.cpu_offload = tk.StringVar(value="model")

        # Attempt to load config from file
        self.load_config()

        self.processing_thread = None
        self.create_widgets()

        # Ensure settings are saved on exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        """Creates and arranges all the GUI widgets."""
        # Input/Output Folders
        frame = tk.LabelFrame(self.root, text="Directories")
        frame.pack(fill="x", padx=10, pady=5)
        tk.Label(frame, text="Input Folder:").grid(row=0, column=0, sticky="e")
        tk.Entry(frame, textvariable=self.input_dir, width=50).grid(row=0, column=1)
        tk.Button(frame, text="Browse", command=self.browse_input).grid(row=0, column=2)
        tk.Label(frame, text="Output Folder:").grid(row=1, column=0, sticky="e")
        tk.Entry(frame, textvariable=self.output_dir, width=50).grid(row=1, column=1)
        tk.Button(frame, text="Browse", command=self.browse_output).grid(row=1, column=2)

        # Parameters
        param_frame = tk.LabelFrame(self.root, text="Parameters")
        param_frame.pack(fill="x", padx=10, pady=5)
        self.add_param(param_frame, "Guidance Scale", self.guidance_scale, 0)
        self.add_param(param_frame, "Inference Steps", self.inference_steps, 1)
        self.add_param(param_frame, "Window Size", self.window_size, 2)
        self.add_param(param_frame, "Max Resolution", self.max_res, 3)
        self.add_param(param_frame, "Overlap", self.overlap, 4)
        self.add_param(param_frame, "Seed", self.seed, 5)

        tk.Label(param_frame, text="CPU Offload Mode:").grid(row=6, column=0, sticky="e")
        cpu_offload_box = ttk.Combobox(
            param_frame, textvariable=self.cpu_offload, values=["model", "sequential"]
        )
        cpu_offload_box.grid(row=6, column=1, padx=5)

        # Controls
        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(pady=10)
        tk.Button(ctrl_frame, text="Start", command=self.start_thread).pack(side="left", padx=5)
        tk.Button(ctrl_frame, text="Exit", command=self.on_close).pack(side="right", padx=5)

        # Logs
        log_frame = tk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.log = tk.Text(log_frame, state="disabled", height=10)
        self.log.pack(fill="both", expand=True)

    def add_param(self, parent, label, var, row):
        """Helper function to create a parameter entry field."""
        tk.Label(parent, text=label + ":").grid(row=row, column=0, sticky="e")
        tk.Entry(parent, textvariable=var).grid(row=row, column=1, padx=5, pady=2)

    def browse_input(self):
        """Opens a file dialog to select the input folder and uses os.path.normpath to fix path formatting"""
        folder = filedialog.askdirectory(initialdir=os.path.normpath(self.input_dir.get()))
        if folder:
           self.input_dir.set(os.path.normpath(folder))

    def browse_output(self):
        """Opens a file dialog to select the output folder and uses os.path.normpath to fix path formatting"""
        folder = filedialog.askdirectory(initialdir=os.path.normpath(self.output_dir.get()))
        if folder:
           self.output_dir.set(os.path.normpath(folder))

    def log_message(self, message):
        """Logs a message to the GUI log."""
        self.log.config(state="normal")
        self.log.insert("end", message + "\n")
        self.log.config(state="disabled")
        self.log.see("end")

    def start_thread(self):
        """Starts a new thread for processing."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self.start_processing, daemon=True)
            self.processing_thread.start()

    def start_processing(self):
        """
        Main processing logic.
        """
        try:
            self.log_message("Starting processing...")
            demo = DepthCrafterDemo(
                unet_path="tencent/DepthCrafter",
                pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
                cpu_offload=self.cpu_offload.get(),
            )
            for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
                videos = glob.glob(os.path.join(self.input_dir.get(), ext))
                finished_folder = os.path.join(self.input_dir.get(), "finished")
                # Ensure the 'finished' folder exists
                os.makedirs(finished_folder, exist_ok=True)
                for video in videos:
                    self.log_message(f"Processing: {video}")
                    demo.run(
                        video,
                        num_denoising_steps=self.inference_steps.get(),
                        guidance_scale=self.guidance_scale.get(),
                        save_folder=self.output_dir.get(),
                        window_size=self.window_size.get(),
                        process_length=-1,
                        overlap=self.overlap.get(),
                        max_res=self.max_res.get(),
                        seed=self.seed.get(),
                    )
                    shutil.move(video, finished_folder)
            self.log_message("Processing complete!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_close(self):
        """
        Saves the config and closes the main window.
        """
        # Save configuration before closing
        self.save_config()
        self.root.destroy()

    def save_config(self):
        """
        Saves the current settings to the configuration file.
        """
        config = {
            "input_dir": self.input_dir.get(),
            "output_dir": self.output_dir.get(),
            "guidance_scale": self.guidance_scale.get(),
            "inference_steps": self.inference_steps.get(),
            "window_size": self.window_size.get(),
            "max_res": self.max_res.get(),
            "overlap": self.overlap.get(),
            "seed": self.seed.get(),
            "cpu_offload": self.cpu_offload.get(),
        }
        with open(self.CONFIG_FILENAME, "w") as f:
            json.dump(config, f, indent=4)

    def load_config(self):
        """
         Loads settings from the configuration file.
        """
        if os.path.exists(self.CONFIG_FILENAME):
            try:
                with open(self.CONFIG_FILENAME, "r") as f:
                    config = json.load(f)
                # Use os.path.normpath to ensure path correctness
                self.input_dir.set(os.path.normpath(config.get("input_dir", "./input_clips")))
                self.output_dir.set(os.path.normpath(config.get("output_dir", "./output_depthmaps")))
                self.guidance_scale.set(config.get("guidance_scale", 1.0))
                self.inference_steps.set(config.get("inference_steps", 5))
                self.window_size.set(config.get("window_size", 110))
                self.max_res.set(config.get("max_res", 960))
                self.overlap.set(config.get("overlap", 25))
                self.seed.set(config.get("seed", 42))
                self.cpu_offload.set(config.get("cpu_offload", "model"))
            except Exception as e:
                # If there's an error reading config, just use defaults
                print(f"Warning: Could not load config: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DepthCrafterGUI(root)
    root.mainloop()
