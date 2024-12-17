MP4 to PNG Sequence GUI
=======================

Overview
--------
The MP4 to PNG Sequence GUI is a simple tool for converting MP4 videos into 
a sequence of PNG image frames. Each video frame is saved as an individual 
PNG file, starting with the filename `00000000.png` and incrementing for 
each subsequent frame.

This tool provides a user-friendly graphical interface to select input and 
output folders and process multiple videos in a single operation.

Features
--------
1. **MP4 to PNG Conversion**:
   - Converts all MP4 files in the selected input folder into a PNG sequence.
   - Each frame is saved as a separate PNG file.

2. **Sequential Frame Naming**:
   - Frame filenames are zero-padded to 8 digits (e.g., `00000000.png`, 
     `00000001.png`, etc.), ensuring compatibility with `iw3`.

3. **Simple and Intuitive GUI**:
   - Easily select input and output folders with a file browser.
   - Start the conversion process with a single button click.
   - Monitor progress and completion status in the real-time log window.

Notes
-----
- Input folder should contain MP4 files; unsupported formats will be skipped.
- The output folder will store all PNG files from the processed videos.
- Ensure sufficient disk space is available for the converted image sequence.