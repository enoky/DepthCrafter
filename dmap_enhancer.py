import os
import cv2
import glob
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import queue

def variable_gamma_correction(disparity_map, gamma_min=0.8, gamma_max=2.0, alpha=1.0):
    """Applies a variable gamma correction based on pixel intensity."""
    img_norm = disparity_map.astype(np.float32) / 255.0
    gamma_map = gamma_min + (gamma_max - gamma_min) * (img_norm**alpha)
    enhanced = np.power(img_norm, 1.0 / gamma_map)
    return (enhanced * 255).astype(np.uint8)


def process_videos(input_folder, output_folder, gamma_min=0.8, gamma_max=2.0, alpha=1.0, output_format="mp4", log_queue=None, blur_enabled=False, blur_sigma=1.0):
    """
    Processes all MP4 disparity videos in the input folder.
    Applies variable gamma correction and optional Gaussian blur.
    Writes out either:
    - Enhanced MP4 files, or
    - A single continuous PNG sequence of 16-bit PNGs.
    """
    def log(msg):
        if log_queue is not None:
            log_queue.put(msg)

    os.makedirs(output_folder, exist_ok=True)
    video_paths = sorted(glob.glob(os.path.join(input_folder, "*.mp4")))

    if not video_paths:
        log("No MP4 files found in the input directory.")
        return

    frame_counter = 0  # Used only if outputting PNG sequence

    for video_path in video_paths:
        log(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log(f"Warning: Could not open {video_path}. Skipping.")
            continue

        # If outputting MP4:
        if output_format == "mp4":
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = os.path.join(output_folder, f"{base_name}_enhanced.mp4")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height), isColor=False)
        else:
            # For PNG sequence, we do not open a VideoWriter
            out = None

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if len(frame.shape) == 3 and frame.shape[2] == 3:
                disparity_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                disparity_frame = frame

            enhanced_frame = variable_gamma_correction(
                disparity_frame,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                alpha=alpha
            )

            # Apply optional Gaussian blur
            if blur_enabled and blur_sigma > 0:
                enhanced_frame = cv2.GaussianBlur(enhanced_frame, (0, 0), blur_sigma)

            if output_format == "mp4":
                # Write frame to MP4 file
                out.write(enhanced_frame)
            else:
                # Write frame as 16-bit PNG
                # Convert 8-bit image to 16-bit by scaling:
                # 0 -> 0, 255 -> 65535
                enhanced_16 = (enhanced_frame.astype(np.uint16) * 257)
                png_name = f"{frame_counter:08d}.png"
                cv2.imwrite(os.path.join(output_folder, png_name), enhanced_16)
                frame_counter += 1

            frame_count += 1

        cap.release()
        if out is not None:
            out.release()

        if output_format == "mp4":
            log(f"Output written to: {output_filename} (Total frames: {frame_count})")
        else:
            log(f"Processed {frame_count} frames from {video_path}.")

    log("Processing complete.")


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Disparity Video Enhancer")
        self.pack(fill="both", expand=True)

        self.input_folder = tk.StringVar(value="./enhancer_input")
        self.output_folder = tk.StringVar(value="./enhancer_output")
        self.gamma_min = tk.DoubleVar(value=0.8)
        self.gamma_max = tk.DoubleVar(value=2.0)
        self.alpha = tk.DoubleVar(value=1.0)
        self.output_format = tk.StringVar(value="mp4")

        self.blur_enabled = tk.BooleanVar(value=True)
        self.blur_sigma = tk.DoubleVar(value=1.2)

        self.log_queue = queue.Queue()
        self.processing_thread = None

        self.create_widgets()
        self.check_log_queue()

    def create_widgets(self):
        # Input folder
        input_frame = tk.Frame(self)
        input_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(input_frame, text="Input Folder:").pack(side="left")
        tk.Entry(input_frame, textvariable=self.input_folder, width=40).pack(side="left", padx=5)
        tk.Button(input_frame, text="Browse...", command=self.browse_input).pack(side="left")

        # Output folder
        output_frame = tk.Frame(self)
        output_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(output_frame, text="Output Folder:").pack(side="left")
        tk.Entry(output_frame, textvariable=self.output_folder, width=40).pack(side="left", padx=5)
        tk.Button(output_frame, text="Browse...", command=self.browse_output).pack(side="left")

        # Gamma parameters
        params_frame = tk.Frame(self)
        params_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(params_frame, text="Gamma Min:").grid(row=0, column=0, sticky="e", padx=5)
        tk.Entry(params_frame, textvariable=self.gamma_min, width=10).grid(row=0, column=1, padx=5)
        tk.Label(params_frame, text="Gamma Max:").grid(row=0, column=2, sticky="e", padx=5)
        tk.Entry(params_frame, textvariable=self.gamma_max, width=10).grid(row=0, column=3, padx=5)
        tk.Label(params_frame, text="Alpha:").grid(row=0, column=4, sticky="e", padx=5)
        tk.Entry(params_frame, textvariable=self.alpha, width=10).grid(row=0, column=5, padx=5)

        # Output format
        format_frame = tk.Frame(self)
        format_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(format_frame, text="Output Format:").pack(side="left")
        tk.Radiobutton(format_frame, text="MP4", variable=self.output_format, value="mp4").pack(side="left", padx=5)
        tk.Radiobutton(format_frame, text="PNG Sequence", variable=self.output_format, value="png").pack(side="left", padx=5)

        # Blur settings
        blur_frame = tk.Frame(self)
        blur_frame.pack(fill="x", padx=10, pady=5)
        tk.Checkbutton(blur_frame, text="Enable Gaussian Blur", variable=self.blur_enabled).pack(side="left", padx=5)
        tk.Label(blur_frame, text="Blur Sigma:").pack(side="left", padx=5)
        tk.Entry(blur_frame, textvariable=self.blur_sigma, width=10).pack(side="left", padx=5)

        # Process button
        button_frame = tk.Frame(self)
        button_frame.pack(fill="x", padx=10, pady=5)
        tk.Button(button_frame, text="Process", command=self.start_processing).pack()

        # Log output
        log_frame = tk.Frame(self)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.log_text = tk.Text(log_frame, wrap="word", height=10)
        self.log_text.pack(fill="both", expand=True)
        self.log("Ready.")

    def browse_input(self):
        folder = filedialog.askdirectory(initialdir=self.input_folder.get(), title="Select Input Folder")
        if folder:
            self.input_folder.set(folder)

    def browse_output(self):
        folder = filedialog.askdirectory(initialdir=self.output_folder.get(), title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)

    def log(self, message):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.update_idletasks()

    def start_processing(self):
        input_dir = self.input_folder.get()
        output_dir = self.output_folder.get()
        gamma_min_val = self.gamma_min.get()
        gamma_max_val = self.gamma_max.get()
        alpha_val = self.alpha.get()
        out_format = self.output_format.get()
        blur_en = self.blur_enabled.get()
        blur_sigma_val = self.blur_sigma.get()

        if not os.path.isdir(input_dir):
            messagebox.showerror("Error", "Invalid input folder.")
            return

        if not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Could not create output folder:\n{e}")
                return

        self.log("Starting processing...")

        # Start the worker thread for processing
        self.processing_thread = threading.Thread(
            target=process_videos,
            args=(input_dir, output_dir, gamma_min_val, gamma_max_val, alpha_val, out_format, self.log_queue, blur_en, blur_sigma_val),
            daemon=True
        )
        self.processing_thread.start()

    def check_log_queue(self):
        """Check the log queue and update the log text widget periodically."""
        while True:
            try:
                msg = self.log_queue.get_nowait()
                self.log(msg)
            except queue.Empty:
                break
        self.after(100, self.check_log_queue)  # check again after 100 ms

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
