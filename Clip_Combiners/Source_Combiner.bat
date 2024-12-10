@echo off
setlocal enabledelayedexpansion

REM Define the output video name
set output_video=combined_video.mp4

REM Delete existing file_list.txt if it exists
if exist file_list.txt del file_list.txt

REM Loop through all MP4 files in the current directory
echo Generating file_list.txt...
for %%f in (*.mp4) do (
    echo file '%%f' >> file_list.txt
)

REM Check if file_list.txt was successfully created
if not exist file_list.txt (
    echo No MP4 files found in the directory. Exiting.
    pause
    exit /b
)

REM Display the generated file list for verification
echo file_list.txt has been created with the following content:
type file_list.txt

REM Perform concatenation using FFmpeg
echo Starting concatenation with FFmpeg...
ffmpeg -f concat -safe 0 -i file_list.txt -c copy "%output_video%"

REM Check if concatenation was successful
if exist "%output_video%" (
    echo Concatenation completed successfully! Output file: %output_video%
) else (
    echo An error occurred during concatenation.
)

pause
