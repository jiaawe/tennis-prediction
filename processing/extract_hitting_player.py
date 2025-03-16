import os
import json
import cv2
import numpy as np
from tqdm import tqdm

class HittingPlayerVisualizer:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.width = 1920
        self.height = 1080
        
    def _get_poses_with_bbox(self, frame_data, max_poses=4):
        """Extract top N poses with their bounding boxes."""
        if not frame_data:
            return (np.zeros((max_poses, 17, 2)), 
                   np.zeros((max_poses, 17)), 
                   np.zeros((max_poses, 4)))
            
        # Sort poses by bbox confidence
        sorted_poses = sorted(frame_data, key=lambda x: x.get('bbox_confidence', 0), reverse=True)
        
        # Initialize arrays
        poses = np.zeros((max_poses, 17, 2))
        confidences = np.zeros((max_poses, 17))
        bboxes = np.zeros((max_poses, 4))
        
        # Fill in available poses
        for i in range(min(len(sorted_poses), max_poses)):
            pose = sorted_poses[i]
            keypoints = np.array(pose['keypoints']).reshape(-1, 2)
            confidence = np.array(pose['keypoint_confidence'])
            bbox = np.array(pose['bbox'])
            
            poses[i] = keypoints
            confidences[i] = confidence
            bboxes[i] = bbox
            
        return poses, confidences, bboxes
    
    def _get_hitting_player_label(self, bboxes, player_position):
        """Determine which pose corresponds to the hitting player."""
        if len(bboxes) == 0:
            return 0
            
        # Convert bboxes to center coordinates
        centers = np.zeros((len(bboxes), 2))
        for i, bbox in enumerate(bboxes):
            if np.all(bbox == 0):  # Skip empty bboxes
                continue
            centers[i] = [(bbox[0] + bbox[2])/2/self.width, 
                         (bbox[1] + bbox[3])/2/self.height]
        
        # Calculate distances to player position
        distances = np.sqrt(np.sum((centers - player_position)**2, axis=1))
        
        # Return index of closest bbox (if any valid bbox exists)
        valid_boxes = ~np.all(bboxes == 0, axis=1)
        if np.any(valid_boxes):
            distances[~valid_boxes] = float('inf')
            return np.argmin(distances)
        return 0

    def process_video(self, video_id):
        # Create output directory for this video
        video_output_dir = os.path.join(self.output_dir, video_id)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Find the matching pose file for this video ID
        pose_file = None
        for filename in os.listdir(os.path.join(self.data_dir, 'pose')):
            if filename.endswith('_pose.json') and video_id in filename:
                pose_file = os.path.join(self.data_dir, 'pose', filename)
                break
        
        # If no direct match, try with just the YouTube ID part
        if pose_file is None:
            youtube_id = video_id.split('_')[0]
            for filename in os.listdir(os.path.join(self.data_dir, 'pose')):
                if filename.endswith('_pose.json') and filename.startswith(youtube_id):
                    pose_file = os.path.join(self.data_dir, 'pose', filename)
                    break
        
        if pose_file is None:
            print(f"No matching pose file found for video {video_id}")
            return
            
        print(f"Using pose file: {os.path.basename(pose_file)} for video {video_id}")
        with open(pose_file) as f:
            pose_data = json.load(f)
        
        # Extract the YouTube ID for transform files
        youtube_id = video_id.split('_')[0]
        
        # Load rally data - look for any transform file that could match this video
        transform_files = [f for f in os.listdir(os.path.join(self.data_dir, 'transformed')) 
                          if f.startswith(youtube_id) and f.endswith('.json')]
        
        if not transform_files:
            print(f"No transform files found for video {video_id}")
            return
            
        # Find specific event frames to process
        event_frames = {}
        for transform_file in transform_files:
            with open(os.path.join(self.data_dir, 'transformed', transform_file)) as f:
                rally_data = json.load(f)
                
                for rally in rally_data['rallies']:
                    rally_video_id = rally.get('video_id', '')
                    
                    # Only process events from the current video folder
                    if video_id in rally_video_id or rally_video_id in video_id:
                        for event in rally['events']:
                            frame = event['frame']
                            # Get relative player position
                            relative_width = event.get('relative_player_width', 0.0)
                            relative_height = event.get('relative_player_height', 0.0)
                            player_position = np.array([relative_width, relative_height])
                            
                            frame_key = f"frame_{frame:06d}"
                            if frame_key in pose_data:
                                event_frames[frame_key] = {
                                    'player_position': player_position,
                                    'event': event['event']
                                }
        
        # Only process the event frames (not all frames)
        print(f"Found {len(event_frames)} event frames to process for video {video_id}")
        
        # Process each frame that has pose data
        frames_dir = os.path.join(self.data_dir, 'frames', video_id)
        if not os.path.exists(frames_dir):
            print(f"Frames directory not found: {frames_dir}")
            return
            
        print(f"Processing video {video_id}...")
        for frame_key, event_info in tqdm(event_frames.items()):
            player_position = event_info['player_position']
            event_type = event_info['event']
            
            # Load the original frame
            frame_path = os.path.join(frames_dir, f"{frame_key}.jpg")
            if not os.path.exists(frame_path):
                continue
                
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
                
            # Get poses and bboxes
            if frame_key in pose_data:
                poses, confidences, bboxes = self._get_poses_with_bbox(pose_data[frame_key])
                
                # Find the hitting player
                hitting_player_idx = self._get_hitting_player_label(bboxes, player_position)
                
                # Get the hitting player's bbox
                hitting_bbox = bboxes[hitting_player_idx]
                
                # Convert normalized bbox coordinates to pixel coordinates
                if not np.all(hitting_bbox == 0):  # If valid bbox
                    x1, y1, x2, y2 = hitting_bbox
                    x1, x2 = int(x1), int(x2)
                    y1, y2 = int(y1), int(y2)
                    
                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Extract player ID and shot type from event
                    event_parts = event_type.split('_')
                    player_id = event_parts[0] if len(event_parts) > 0 else ""
                    shot_type = event_parts[3] if len(event_parts) > 3 else ""
                    
                    # Add player label with more information
                    label = f"{player_id} {shot_type}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Save the annotated frame
            output_path = os.path.join(video_output_dir, f"{frame_key}.jpg")
            cv2.imwrite(output_path, frame)

def main():
    # Define data directory and output directory
    data_dir = 'data'
    output_dir = os.path.join(data_dir, 'hitting_player')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizer
    visualizer = HittingPlayerVisualizer(data_dir, output_dir)
    
    # Get all frame folders and extract the video IDs
    frames_dir = os.path.join(data_dir, 'frames')
    video_ids = [d for d in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, d))]
    
    print(f"Found {len(video_ids)} video folders to process")
    
    # Process each video
    for video_id in video_ids:
        visualizer.process_video(video_id)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()