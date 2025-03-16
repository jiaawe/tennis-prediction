import os
import cv2
import glob
from tqdm import tqdm
import subprocess
import json
from pathlib import Path

def extract_frames(video_path, output_dir):
    """Extract frames from a video file."""
    # Create output directory
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(output_dir, video_name)
    
    if os.path.exists(frames_dir):
        print(f"Skipping frame extraction for {video_name} - exists at {frames_dir}")
        return frames_dir
    
    os.makedirs(frames_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Extract frames with progress bar
    with tqdm(total=frame_count, desc=f"Extracting frames from {video_name}") as pbar:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save frame
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    return frames_dir

def run_inference(frames_dir, config_file, checkpoint_path, text_prompt):
    """Run inference on extracted frames."""
    video_name = os.path.basename(frames_dir)
    bbox_path = os.path.join("data/bbox", f"{video_name}.json")
    
    # Check if JSON already exists
    if os.path.exists(bbox_path):
        print(f"Skipping inference for {video_name} - JSON exists at {bbox_path}")
        return None
    
    # Create inference output directory
    inference_dir = os.path.join("data/inference", video_name)
    os.makedirs(inference_dir, exist_ok=True)
    
    # Build and run inference command
    cmd = [
        "python", "Open-GroundingDino/inference_on_a_folder.py",
        "-c", config_file,
        "-p", checkpoint_path,
        "-i", frames_dir,
        "-t", text_prompt,
        "-o", inference_dir
    ]
    
    print(f"\nRunning inference on {video_name}")
    subprocess.run(cmd)

def process_all_videos(args):
    """Process all videos in the rallies directory."""
    # Get all video files
    video_files = glob.glob(os.path.join(args.video_dir, "*.mp4"))
    
    # Create necessary directories
    os.makedirs("data/frames", exist_ok=True)
    os.makedirs("data/inference", exist_ok=True)
    os.makedirs("data/bbox", exist_ok=True)
    
    print(f"Found {len(video_files)} videos to process")
    
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\nProcessing video: {video_name}")
        
        # Check if JSON already exists
        bbox_path = os.path.join("data/bbox", f"{video_name}.json")
        if os.path.exists(bbox_path):
            print(f"Skipping {video_name} - JSON exists at {bbox_path}")
            continue
        
        # Extract frames
        frames_dir = extract_frames(video_path, "data/frames")
        
        # Run inference
        run_inference(
            frames_dir,
            args.config_file,
            args.checkpoint_path,
            args.text_prompt
        )
        
        print(f"Completed processing {video_name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("Video Processing Pipeline")
    parser.add_argument("--video_dir", type=str, default="data/rallies",
                      help="Directory containing MP4 files")
    parser.add_argument("--config_file", type=str, default="Open-GroundingDino/tools/GroundingDINO_SwinT_OGC.py",
                      help="Path to config file")
    parser.add_argument("--checkpoint_path", type=str, default="Open-GroundingDino/logs/checkpoint0014.pth",
                      help="Path to checkpoint file")
    parser.add_argument("--text_prompt", type=str, default="tennis player",
                      help="Text prompt for detection")
    
    args = parser.parse_args()
    
    # Process all videos
    process_all_videos(args)