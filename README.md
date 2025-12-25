# Ezgrid

A desktop app for exporting video grids from multiple clips, with proxy support, auto tile sizing, and advanced export options.

## Requirements
- Python 3.8+
- PySide6
- FFmpeg (not included, see below)

## Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Download FFmpeg and add it to your PATH, or place `ffmpeg.exe` and `ffprobe.exe` in the same folder as your videos.

## Usage
- Run the app:
  ```
  python main.py
  ```
- Import your video clips, adjust grid and export settings, and export your grid video.

## Not Included
- The `ffmpeg/` folder and FFmpeg binaries are NOT included in this repository. Download them from [ffmpeg.org](https://ffmpeg.org/download.html).

