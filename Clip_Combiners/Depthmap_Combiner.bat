@echo off
setlocal enabledelayedexpansion

REM Define temporary and output file names
set temp_folder=temp_videos
set output_video=combined_video.mp4

REM Create a temporary folder to store scaled videos
if not exist %temp_folder% mkdir %temp_folder%

REM Loop through all MP4 files in the current directory
echo Scaling videos...
set counter=0
for %%f in (*.mp4) do (
    set /a counter+=1
    ffmpeg -i "%%f" -vf scale=960:400 -c:v h264_nvenc -cq 16 -preset medium "%temp_folder%\scaled_%%~nf.mp4"
)

REM Check if scaling was successful
if %counter%==0 (
    echo No MP4 files found in the directory. Exiting.
    pause
    exit /b
)

REM Generate file_list.txt for concatenation
echo Generating file_list.txt...
if exist file_list.txt del file_list.txt
for %%f in (%temp_folder%\*.mp4) do (
    echo file '%%f' >> file_list.txt
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

REM Clean up temporary files
echo Cleaning up temporary files...
rmdir /s /q %temp_folder%
del file_list.txt

pause
