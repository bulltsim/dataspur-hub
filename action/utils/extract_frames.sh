#!/usr/bin/env bash
# Simple utility to extract frames from a video file using ffmpeg.
# Usage: ./extract_frames.sh <video_path> <output_dir> [fps]

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <video_path> <output_dir> [fps]"
  exit 1
fi

video_path="$1"
output_dir="$2"
fps="${3:-30}"

mkdir -p "$output_dir"
# Extract frames at the specified frame rate (default 30fps)
ffmpeg -i "$video_path" -vf fps=$fps "$output_dir/frame_%05d.jpg"
