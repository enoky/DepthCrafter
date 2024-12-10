==================================================
DepthCrafter GUI - Parameter Guide
==================================================

DESCRIPTION
-----------
DepthCrafter processes videos into temporally consistent depth maps using advanced video diffusion models. Below is an explanation of the configurable parameters.

==================================================
INPUT/OUTPUT PARAMETERS
==================================================
1. Input Folder
   - Directory containing the video files to process.
   - Default: ./input_clips

2. Output Folder
   - Directory to save the processed depth videos.
   - Default: ./depthmap_output

==================================================
PROCESSING PARAMETERS
==================================================
3. Guidance Scale
   - Adjusts the balance between model creativity and fidelity to the input.
   - Default: 1.0

4. Inference Steps
   - Number of steps for depth generation. Higher values improve quality but take longer.
   - Default: 5

5. Window Size
   - Number of frames processed per batch. Larger values may require more memory.
   - Default: 110

6. Maximum Resolution
   - Maximum width of the video frames.
   - Default: 960

7. Overlap
   - Number of overlapping frames between processing batches to ensure smooth transitions.
   - Default: 25

8. Seed
   - Sets the random seed for reproducibility of results.
   - Default: 42

==================================================
RESOURCE MANAGEMENT
==================================================
9. CPU Offload Mode
   - Optimizes memory usage during processing.
   - Options:
       - model: Faster
       - sequential: Slower
   - Default: model