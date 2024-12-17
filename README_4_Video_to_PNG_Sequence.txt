Video to PNG Sequence GUI
=========================

Overview
--------
The Video to PNG Sequence GUI is a simple tool for converting videos into 
a sequence of PNG image frames. Each video frame is saved as an individual 
PNG file, starting with the filename `00000000.png` and incrementing for 
each subsequent frame.

This tool provides a user-friendly graphical interface to select input and 
output folders and process multiple videos in a single operation.

Features
--------
1. **Multi-Format Video to PNG Conversion**:
   - Converts videos in common formats (e.g., MP4, AVI, MOV, MKV, FLV, WMV, 
     MPEG) into a PNG sequence.
   - Automatically scans the input folder for supported video files.

2. **Sequential Frame Naming**:
   - Frame filenames are zero-padded to 8 digits (e.g., `00000000.png`, 
     `00000001.png`, etc.), ensuring compatibility with `iw3`.

3. **Simple and Intuitive GUI**:
   - Easily select input and output folders with a file browser.
   - Start the conversion process with a single button click.
   - Monitor progress and completion status in the real-time log window.

Notes
-----
- The input folder can contain videos in various formats; unsupported formats 
  will be skipped.
- The output folder will store all PNG files from the processed videos.
- Ensure sufficient disk space is available for the converted image sequence.
- The conversion process does not alter video frames, preserving original 
  quality.