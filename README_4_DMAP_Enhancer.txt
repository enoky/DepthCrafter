Depthmap Enhancer GUI
=====================

Overview
--------
The Depthmap Enhancer GUI is a user-friendly application designed to enhance 
disparity videos (depth maps). It applies advanced gamma correction to improve 
visual clarity and optionally smooths the output using Gaussian blur. The tool 
supports saving enhanced outputs as MP4 videos or a sequence of PNG frames.

Features
--------
1. **Variable Gamma Correction**:
   - Dynamically adjusts gamma values for each pixel based on intensity.
   - Enhances contrast and detail in disparity maps for better visualization.
   - Parameters:
     - `Gamma Min`: Sets the minimum gamma value (default: 0.8).
     - `Gamma Max`: Sets the maximum gamma value (default: 2.0).
     - `Alpha`: Controls the intensity weighting (default: 1.0).

2. **Optional Gaussian Blur**:
   - Smooths enhanced frames by applying a Gaussian blur.
   - Configurable blur strength using the `Blur Sigma` value (default: 1.2).

3. **Flexible Output Formats**:
   - **MP4 Video**: Processes videos and saves the enhanced output in MP4 format.
   - **PNG Sequence**: Extracts and saves each enhanced frame as an individual PNG image.

4. **Simple and Intuitive GUI**:
   - Select input and output folders using the built-in file browser.
   - Configure enhancement parameters directly in the GUI.
   - View progress and log messages in real-time.

Notes
-----
- Input folder should contain MP4 videos; unsupported files will be skipped.
- Output folder will store the processed files with appropriate naming.

