import cv2
from pathlib import Path
import os
from tennis_analyzer import TennisPlayerAnalyzer

def get_video_fps(video_path):
    """Get FPS from video file"""
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps

def process_all_rallies(base_dir="data"):
    base_dir = Path(base_dir)
    bbox_dir = base_dir / "bbox"
    frames_dir = base_dir / "frames"
    rallies_dir = base_dir / "rallies"
    
    # Get all JSON files
    json_files = list(bbox_dir.glob("*_boxes.json"))
    
    for json_file in json_files:
        # Extract rally ID from JSON filename
        rally_id = json_file.stem.replace("_boxes", "")
        print(f"\nProcessing rally: {rally_id}")
        
        # Check if frames exist
        rally_frames_dir = frames_dir / rally_id
        if not rally_frames_dir.exists():
            print(f"No frames found for rally {rally_id}, skipping...")
            continue
        
        # Get FPS from original video
        video_path = rallies_dir / f"{rally_id}.mp4"
        if not video_path.exists():
            print(f"Video file not found for rally {rally_id}, using default 30 fps")
            fps = 30
        else:
            fps = get_video_fps(video_path)
        
        # Setup output directories
        output_dir = base_dir / "processed_frames" / rally_id
        video_output = base_dir / "processed_videos" / f"{rally_id}.mp4"
        video_output.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Initialize and run analyzer
            analyzer = TennisPlayerAnalyzer(rally_frames_dir, json_file)
            print(f"Processing frames...")
            processed_frames = analyzer.process_frames(output_dir, rally_id)
            
            print(f"Creating video with {fps} fps...")
            analyzer.create_video(processed_frames, video_output, fps)
            
        except Exception as e:
            print(f"Error processing rally {rally_id}: {str(e)}")
            continue

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch Process Tennis Rallies')
    parser.add_argument('--base_dir', type=str, default="data",
                      help='Base directory containing data')
    
    args = parser.parse_args()
    process_all_rallies(args.base_dir)