import cv2
import json
import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO

class TennisPlayerAnalyzer:
    def __init__(self, frames_dir, bbox_file, yolo_model_path="models/yolo11x-pose.pt"):
        self.frames_dir = Path(frames_dir)
        self.bbox_file = Path(bbox_file)
        self.predictions = self.load_predictions()
        self.model = YOLO(yolo_model_path)
        self.scale_factor = 1.8
        
    def load_predictions(self):
        with open(self.bbox_file, 'r') as f:
            return json.load(f)
    
    def expand_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        
        new_w = min(w * self.scale_factor, x2)
        new_h = min(h * self.scale_factor, y2)
        
        x1_new = max(0, int(x_center - new_w/2))
        y1_new = max(0, int(y_center - new_h/2))
        x2_new = int(x_center + new_w/2)
        y2_new = int(y_center + new_h/2)
        
        return [x1_new, y1_new, x2_new, y2_new]
    
    def process_pose_keypoints(self, pose_results):
        try:
            keypoints = np.zeros((17, 2), dtype=np.float32)
            confidence = np.zeros(17, dtype=np.float32)
            
            if len(pose_results) > 0:
                kpts = pose_results[0].keypoints  # Take first detection only
                if len(kpts) > 0:
                    keypoints = kpts.xy.cpu().numpy()[0][:17, :2]
                    confidence = kpts.conf.cpu().numpy()[0][:17] if kpts.conf is not None else np.ones(17)
            
            return keypoints, confidence
        except Exception as e:
            print(f"Error processing pose keypoints: {str(e)}")
            return np.zeros((17, 2), dtype=np.float32), np.zeros(17, dtype=np.float32)

    def draw_bbox(self, frame, bbox, confidence):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f"Player ({confidence:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def draw_pose(self, frame, keypoints):
        if len(keypoints) == 0:
            return

        left_indices = {5, 7, 9, 11, 13, 15}
        right_indices = {6, 8, 10, 12, 14, 16}
        center_indices = {0, 1, 2, 3, 4}

        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Face
            (5, 6), (5, 11), (6, 12), (11, 12),  # Body
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16)  # Right leg
        ]

        BLUE = (255, 0, 0)
        RED = (0, 0, 255)
        GREEN = (0, 255, 0)

        for p1, p2 in skeleton:
            if (p1 < len(keypoints) and p2 < len(keypoints) and 
                keypoints[p1][0] > 0 and keypoints[p1][1] > 0 and 
                keypoints[p2][0] > 0 and keypoints[p2][1] > 0):
                
                pt1 = tuple(map(int, keypoints[p1]))
                pt2 = tuple(map(int, keypoints[p2]))

                if p1 in left_indices and p2 in left_indices:
                    color = BLUE
                elif p1 in right_indices and p2 in right_indices:
                    color = RED
                else:
                    color = GREEN

                cv2.line(frame, pt1, pt2, color, 2)

        for i, (x, y) in enumerate(keypoints):
            if x > 0 and y > 0:
                if i in left_indices:
                    color = BLUE
                elif i in right_indices:
                    color = RED
                else:
                    color = GREEN

                cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    def process_frames(self, output_dir, rally_id):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_files = sorted(self.frames_dir.glob("frame_*.jpg"))
        if not frame_files:
            raise FileNotFoundError(f"No frames found in {self.frames_dir}")
        
        processed_frames = []
        all_poses = {}
        
        for frame_path in frame_files:
            frame_number = frame_path.stem.split('_')[1]
            frame = cv2.imread(str(frame_path))
            
            if frame is None:
                continue
                
            frame_with_poses = frame.copy()
            frame_poses = []
            
            # Get predictions for this frame
            frame_predictions = self.predictions.get(f"frame_{frame_number}", [])
            
            for pred in frame_predictions:
                try:
                    bbox = pred['bbox']
                    confidence = pred['confidence']
                    
                    # Get expanded bbox for pose detection
                    expanded_bbox = self.expand_bbox(bbox)
                    x1, y1, x2, y2 = expanded_bbox
                    
                    # Extract player crop
                    player_crop = frame[y1:y2, x1:x2]
                    if player_crop.size > 0:  # Check if crop is valid
                        # Get pose keypoints and confidence
                        pose_results = self.model(player_crop)
                        crop_keypoints, keypoint_conf = self.process_pose_keypoints(pose_results)
                        
                        # Only store if we got valid keypoints
                        if np.any(crop_keypoints):
                            # Map keypoints back to original frame coordinates
                            valid_mask = crop_keypoints[:, 0] != 0
                            crop_keypoints[valid_mask, 0] += x1
                            crop_keypoints[valid_mask, 1] += y1
                            
                            # Store pose information
                            pose_info = {
                                'bbox': bbox,
                                'bbox_confidence': float(confidence),
                                'keypoints': crop_keypoints.tolist(),
                                'keypoint_confidence': keypoint_conf.tolist()
                            }
                            frame_poses.append(pose_info)
                            
                            # Draw pose and bbox
                            self.draw_pose(frame_with_poses, crop_keypoints)
                            self.draw_bbox(frame_with_poses, bbox, confidence)
                
                except Exception as e:
                    print(f"Error processing bbox in frame {frame_number}: {str(e)}")
                    continue
            
            # Store poses for this frame
            if frame_poses:  # Only store if we have valid poses
                all_poses[f"frame_{frame_number}"] = frame_poses
            
            # Save processed frame
            output_path = output_dir / f"processed_{frame_number}.jpg"
            cv2.imwrite(str(output_path), frame_with_poses)
            processed_frames.append(output_path)
        
        # Save complete poses file in data/pose directory
        pose_dir = Path("data") / "pose"
        pose_dir.mkdir(exist_ok=True)
        pose_file = pose_dir / f"{rally_id}_pose.json"
        
        with open(pose_file, 'w') as f:
            json.dump(all_poses, f, indent=2)
        
        return processed_frames

    def create_video(self, processed_frames, output_path, fps=30):
        if not processed_frames:
            print("No processed frames available to create video")
            return
        
        first_frame = cv2.imread(str(processed_frames[0]))
        height, width = first_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame_path in processed_frames:
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                out.write(frame)
        
        out.release()
        print(f"Video saved to {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Tennis Player Pose Analysis')
    parser.add_argument('--rally', type=str, required=True,
                      help='Rally name (e.g., rally_name)')
    parser.add_argument('--fps', type=int, default=30,
                      help='FPS for output video')
    parser.add_argument('--base_dir', type=str, default="data",
                      help='Base directory containing data')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    frames_dir = base_dir / "frames" / args.rally
    bbox_file = base_dir / "bbox" / f"{args.rally}_boxes.json"
    output_dir = base_dir / "processed_frames" / args.rally
    video_output = base_dir / "processed_videos" / f"{args.rally}.mp4"
    
    video_output.parent.mkdir(parents=True, exist_ok=True)
    
    analyzer = TennisPlayerAnalyzer(frames_dir, bbox_file)
    processed_frames = analyzer.process_frames(output_dir, args.rally)
    analyzer.create_video(processed_frames, video_output, args.fps)

if __name__ == "__main__":
    main()