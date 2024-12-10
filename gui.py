import threading
import gc
import os
import glob
import shutil
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from diffusers.training_utils import set_seed
from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from depthcrafter.utils import save_video, read_video_frames


class DepthCrafterDemo:
    def __init__(self, unet_path: str, pre_train_path: str, cpu_offload: str = "model"):
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
        self.infer(video, **kwargs)
        gc.collect()
        torch.cuda.empty_cache()


class DepthCrafterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DepthCrafter GUI")
        self.input_dir = tk.StringVar(value="./input_clips")
        self.output_dir = tk.StringVar(value="./output_depthmaps")
        self.guidance_scale = tk.DoubleVar(value=1.0)
        self.inference_steps = tk.IntVar(value=5)
        self.window_size = tk.IntVar(value=110)
        self.max_res = tk.IntVar(value=960)
        self.overlap = tk.IntVar(value=25)
        self.seed = tk.IntVar(value=42)
        self.cpu_offload = tk.StringVar(value="model")
        self.processing_thread = None
        self.create_widgets()

    def create_widgets(self):
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
        ttk.Combobox(
            param_frame, textvariable=self.cpu_offload, values=["model", "sequential"]
        ).grid(row=6, column=1, padx=5)

        # Controls
        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(pady=10)
        tk.Button(ctrl_frame, text="Start", command=self.start_thread).pack(side="left", padx=5)
        tk.Button(ctrl_frame, text="Exit", command=self.root.destroy).pack(side="right", padx=5)

        # Logs
        log_frame = tk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.log = tk.Text(log_frame, state="disabled", height=10)
        self.log.pack(fill="both", expand=True)

    def add_param(self, parent, label, var, row):
        tk.Label(parent, text=label + ":").grid(row=row, column=0, sticky="e")
        tk.Entry(parent, textvariable=var).grid(row=row, column=1, padx=5, pady=2)

    def browse_input(self):
        folder = filedialog.askdirectory(initialdir=".")
        if folder:
            self.input_dir.set(folder)

    def browse_output(self):
        folder = filedialog.askdirectory(initialdir=".")
        if folder:
            self.output_dir.set(folder)

    def log_message(self, message):
        self.log.config(state="normal")
        self.log.insert("end", message + "\n")
        self.log.config(state="disabled")
        self.log.see("end")

    def start_thread(self):
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self.start_processing, daemon=True)
            self.processing_thread.start()

    def start_processing(self):
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



if __name__ == "__main__":
    root = tk.Tk()
    app = DepthCrafterGUI(root)
    root.mainloop()
