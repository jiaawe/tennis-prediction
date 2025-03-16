import os
import glob
import cv2
from tqdm import tqdm
import subprocess
from pathlib import Path

def extract_frames(video_path, output_dir):
    """Extract frames from a video file."""
    # Create output directory
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(output_dir, video_name)
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
    # Create inference output directory
    inference_dir = frames_dir.replace('frames', 'inference')
    os.makedirs(inference_dir, exist_ok=True)
    
    # Build inference command
    cmd = [
        "python", "inference_on_a_folder.py",
        "-c", config_file,
        "-p", checkpoint_path,
        "-i", frames_dir,
        "-t", text_prompt,
        "-o", inference_dir
    ]
    
    # Run inference
    print(f"\nRunning inference on {os.path.basename(frames_dir)}")
    subprocess.run(cmd)
    return inference_dir

def create_video_from_frames(frames_dir, output_path, fps=30):
    """Create video from frames."""
    frames = sorted(glob.glob(os.path.join(frames_dir, "*_pred.jpg")))
    if not frames:
        print(f"No processed frames found in {frames_dir}")
        return None
        
    # Read first frame to get dimensions
    frame = cv2.imread(frames[0])
    height, width = frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    with tqdm(total=len(frames), desc="Creating video") as pbar:
        for frame_path in frames:
            frame = cv2.imread(frame_path)
            out.write(frame)
            pbar.update(1)
            
    out.release()
    return output_path

def process_videos(video_dir, frames_base_dir, config_file, checkpoint_path, text_prompt):
    """Process all videos in directory."""
    # Get all video files
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\nProcessing video: {video_name}")
        
        # Extract frames
        frames_dir = extract_frames(video_path, frames_base_dir)
        
        # Run inference
        inference_dir = run_inference(
            frames_dir,
            config_file,
            checkpoint_path,
            text_prompt
        )
        
        # Create output video
        processed_dir = "data/processed"
        os.makedirs(processed_dir, exist_ok=True)
        output_video_path = os.path.join(processed_dir, f"{video_name}_processed.mp4")
        
        # Get FPS from original video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        # Create final video
        final_video = create_video_from_frames(inference_dir, output_video_path, fps)
        
        print(f"Completed processing {video_name}")
        print(f"Frames saved to: {frames_dir}")
        print(f"Inference results saved to: {inference_dir}")
        if final_video:
            print(f"Final video saved to: {final_video}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("Video Processing and Inference")
    parser.add_argument("--video_dir", type=str, default="data/rallies",
                      help="Directory containing MP4 files")
    parser.add_argument("--frames_dir", type=str, default="data/frames",
                      help="Base directory for extracted frames")
    parser.add_argument("--config_file", type=str, default="tools/GroundingDINO_SwinT_OGC.py",
                      help="Path to config file")
    parser.add_argument("--checkpoint_path", type=str, default="logs/checkpoint0014.pth",
                      help="Path to checkpoint file")
    parser.add_argument("--text_prompt", type=str, default="tennis player",
                      help="Text prompt for detection")
    
    args = parser.parse_args()
    
    # Process all videos
    process_videos(
        args.video_dir,
        args.frames_dir,
        args.config_file,
        args.checkpoint_path,
        args.text_prompt
    )