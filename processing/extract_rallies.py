import os
import json
import argparse
from moviepy.editor import VideoFileClip

def extract_rallies(video_files):
    # Create the output directory if it doesn't exist
    output_dir = os.path.join('data', 'rallies')
    os.makedirs(output_dir, exist_ok=True)

    for video_file in video_files:
        video_path = os.path.join('data', 'videos', video_file)
        base_name = os.path.splitext(video_file)[0]
        
        # Handle different JSON file naming conventions
        json_file = f"{base_name}_transformed.json"
        json_path = os.path.join('data', 'transformed', json_file)
        
        if not os.path.exists(json_path):
            json_file = f"{base_name}_*transformed.json"
            json_path = os.path.join('data', 'transformed', json_file)
        
        if not os.path.exists(json_path):
            print(f"JSON file not found for {video_file}")
            continue
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        fps = data['fps']
        rallies = data['rallies']
        
        with VideoFileClip(video_path) as video:
            total_frames = int(video.duration * fps)
            
            for rally in rallies:
                rally_id = rally['video_id']
                start_frame = int(rally_id.split('_')[-2])
                end_frame = int(rally_id.split('_')[-1])
                
                # Add 30 frames to the start and end, respecting video boundaries
                extended_start_frame = max(0, start_frame - 30)
                extended_end_frame = min(total_frames, end_frame + 30)
                
                start_time = extended_start_frame / fps
                end_time = extended_end_frame / fps
                
                output_path = os.path.join(output_dir, f"{rally_id}.mp4")
                
                rally_clip = video.subclip(start_time, end_time)
                rally_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
                
                print(f"Extracted rally: {rally_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract rallies from tennis videos.")
    parser.add_argument("videos", nargs="+", help="List of video files to process")
    args = parser.parse_args()
    
    extract_rallies(args.videos)