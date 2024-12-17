import os
import cv2
import glob
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import queue

def convert_videos_to_png_sequence(input_folder, output_folder, log_queue=None):
    """
    Converts all common video formats in the input folder into a PNG sequence in the output folder.
    Frames are saved as 8-digit zero-padded filenames starting with 00000000.png.
    """
    def log(msg):
        if log_queue is not None:
            log_queue.put(msg)

    os.makedirs(output_folder, exist_ok=True)

    # List of common video file extensions
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.flv", "*.wmv", "*.mpeg", "*.mpg"]
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not video_paths:
        log("No video files found in the input directory.")
        return

    video_paths = sorted(video_paths)
    frame_counter = 0

    for video_path in video_paths:
        log(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log(f"Warning: Could not open {video_path}. Skipping.")
            continue

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to PNG without altering it.
            png_name = f"{frame_counter:08d}.png"
            cv2.imwrite(os.path.join(output_folder, png_name), frame)
            frame_counter += 1
            frame_count += 1

        cap.release()
        log(f"Processed {frame_count} frames from {video_path}.")

    log("Conversion complete.")

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Video to PNG Sequence Converter")
        self.pack(fill="both", expand=True)

        self.input_folder = tk.StringVar(value="./video_input")
        self.output_folder = tk.StringVar(value="./png_output")

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

        # Convert button
        button_frame = tk.Frame(self)
        button_frame.pack(fill="x", padx=10, pady=5)
        tk.Button(button_frame, text="Convert", command=self.start_conversion).pack()

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

    def start_conversion(self):
        input_dir = self.input_folder.get()
        output_dir = self.output_folder.get()

        if not os.path.isdir(input_dir):
            messagebox.showerror("Error", "Invalid input folder.")
            return

        if not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Could not create output folder:\n{e}")
                return

        self.log("Starting conversion...")

        # Start the worker thread for processing
        self.processing_thread = threading.Thread(
            target=convert_videos_to_png_sequence,
            args=(input_dir, output_dir, self.log_queue),
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
