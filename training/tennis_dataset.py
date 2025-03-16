import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class TennisDataset(Dataset):
    def __init__(self, data_dir, sequence_length=30, video_ids=None, train_label='shot_type', max_poses=4):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.max_poses = max_poses
        self.video_ids = video_ids
        self.width = 1920
        self.height = 1080
        
        self.samples = []
        self.serve_types = set()
        self.label_map = {}
        self._load_data(train_label=train_label)
        
        if not self.label_map:
            self.label_map = {serve_type: idx for idx, serve_type in enumerate(sorted(self.serve_types))}
        print(f"Label mapping: {self.label_map}")
        
        self._get_class_distribution()
        
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
    
    def _get_bbox(self, frame_data, normalize=True):
        if not frame_data:
            return np.zeros((0, 4))
        all_bboxes = []
        for item in frame_data:
            bbox = np.array(item['bbox'])
            
            if normalize:
                x_min, y_min, x_max, y_max = bbox
                
                # Normalize to [0, 1] range
                x_min = x_min / self.width
                y_min = y_min / self.height
                x_max = x_max / self.width
                y_max = y_max / self.height
                
                bbox = np.array([x_min, y_min, x_max, y_max])
                
            all_bboxes.append(bbox)
        
        return np.array(all_bboxes)
    
    def _get_hitting_player_label(self, bboxes, player_position):
        if len(bboxes) == 0:
            return -1
        
        player_x, player_y = player_position
        
        bbox_centers = np.zeros((len(bboxes), 2))
        for i, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            bbox_centers[i] = [center_x, center_y]
        
        distances = np.sqrt(np.sum((bbox_centers - player_position) ** 2, axis=1))
        closest_bbox_idx = np.argmin(distances)
        
        return closest_bbox_idx
    
    def _get_hitting_players(self, bboxes, player_position):
        if len(bboxes) == 0:
            return -1, -1
        
        player_x, player_y = player_position
        
        # Calculate center points of all bounding boxes
        bbox_centers = np.zeros((len(bboxes), 2))
        for i, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            bbox_centers[i] = [center_x, center_y]
        
        distances = np.sqrt(np.sum((bbox_centers - player_position) ** 2, axis=1))
        hitting_player_idx = np.argmin(distances)
        
        hitting_player_center = bbox_centers[hitting_player_idx]
        
        # Find the closest player on the width/x-axis, excluding the hitting player
        width_distances = []
        for i, center in enumerate(bbox_centers):
            if i != hitting_player_idx:
                width_distance = abs(center[0] - hitting_player_center[0])
                width_distances.append((i, width_distance))
        
        if not width_distances:
            return hitting_player_idx, -1
        
        width_distances.sort(key=lambda x: x[1])
        hitting_partner_idx = width_distances[0][0]
        
        return hitting_player_idx, hitting_partner_idx
        
    def _draw_bbox_on_image(self, raw_image_path, output_image_path, bbox, color=(0, 255, 0), thickness=2):
        # Read the image
        image = cv2.imread(raw_image_path)
        if image is None:
            print(f"Failed to read image: {raw_image_path}")
            return 
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_image_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Convert normalized coordinates back to pixel coordinates 
        x_min, y_min, x_max, y_max = bbox
        x_min = int(x_min * self.width)
        y_min = int(y_min * self.height)
        x_max = int(x_max * self.width)
        y_max = int(y_max * self.height)
        
        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # Save the output image
        cv2.imwrite(output_image_path, image)
        
        return image
    
    def _extract_player_bbox(self, raw_image_path, output_image_path, bbox, target_size=(224, 224)):
        # Read the image
        image = cv2.imread(raw_image_path)
        if image is None:
            print(f"Failed to read image: {raw_image_path}")
            return None
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_image_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Convert normalized coordinates to pixel coordinates
        x_min, y_min, x_max, y_max = bbox
        x_min = int(x_min * self.width)
        y_min = int(y_min * self.height)
        x_max = int(x_max * self.width)
        y_max = int(y_max * self.height)
        
        # Calculate center of the bounding box
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        # Calculate original width and height
        original_width = x_max - x_min
        original_height = y_max - y_min
        
        # Scale the bbox by 2x while keeping the center fixed
        scaled_width = original_width * 2
        scaled_height = original_height * 2
        
        # Calculate new bbox coordinates
        new_x_min = max(0, center_x - scaled_width // 2)
        new_y_min = max(0, center_y - scaled_height // 2)
        new_x_max = min(self.width, center_x + scaled_width // 2)
        new_y_max = min(self.height, center_y + scaled_height // 2)
        
        # Crop the player from the image using the scaled bbox
        player_image = image[new_y_min:new_y_max, new_x_min:new_x_max]
        
        # If the crop is empty, handle the error
        if player_image.size == 0:
            print(f"Empty bounding box for image: {raw_image_path}")
            return None
        
        # Resize to standardized dimensions for CNN
        standardized_image = cv2.resize(player_image, target_size)
        
        # Save the standardized player image
        cv2.imwrite(output_image_path, standardized_image)
        
        return standardized_image

    
    def _load_data(self, train_label, n=10):
        print(f'Training for label: {train_label}')
        transform_dir = os.path.join(self.data_dir, 'transformed')
        
        for file in os.listdir(transform_dir):
            if not file.endswith('.json'):
                continue
                
            video_id = '_'.join(file.split('_')[:-1])
            if self.video_ids and video_id not in self.video_ids:
                continue
            
            offset_videos = ["Granollers_Zeballos vs Arevalo_Rojer  _ Toronto 2023 Doubles Semi-Finals", 
                             "Nick Kyrgios_Thanasi Kokkinakis vs Jack Sock_John Isner _ Indian Wells 2022 Doubles Highlights",
                             "Rajeev Ram_Joe Salisbury vs Tim Puetz_Michael Venus _ Cincinnati 2022 Doubles Final",
                             "Salisbury_Ram vs Krawietz_Puetz  _ Toronto 2023 Doubles Semi-Finals",
                            #  "VUPKfQgXy8g",
                            #  "EMBw_kXc574",
                            #  "eGFJAG-2jM8"
                             ]
            
            if any(video in video_id for video in offset_videos):
                offset = 30 # 30 frames before the event (which is specified during processing)
            else:
                offset = 0
            print(f"Using offset: {offset} for video: {video_id}")
            
            with open(os.path.join(transform_dir, file)) as f:
                rally_data = json.load(f)
                    
                for rally in rally_data['rallies']:
                    
                    # load pose and bbox file
                    pose_file = os.path.join(self.data_dir, 'pose', f"{rally['video_id']}_pose.json")
                    bbox_file = os.path.join(self.data_dir, 'bbox', f"{rally['video_id']}_boxes.json")
                    
                    
                    start_frame = rally['video_id'].split('_')[-2]
                    
                    if not os.path.exists(pose_file):
                        print(f"Pose file not found: {pose_file}")
                        continue
                        
                    if not os.path.exists(bbox_file):
                        print(f"Bbox file not found: {bbox_file}")
                        continue
                        
                    with open(pose_file) as f:
                        pose_data = json.load(f)
                    
                    with open(bbox_file) as f:
                        bbox_data = json.load(f)
                        
                    for event in rally['events']:
                        frame = event['frame']
                        frame_key = f"frame_{(frame + offset):06d}" # add offset
                        frame_key_n = f"frame_{(frame + offset + n):06d}" # add offset
                        
                        if frame_key not in pose_data:
                            continue
                        
                        if frame_key_n not in pose_data:
                            print(f"Frame {frame_key_n} not found in pose data")
                            continue
                            
                        # Get poses, confidences, and bboxes
                        poses, confidences, _ = self._get_poses_with_bbox(pose_data[frame_key], self.max_poses)
                        
                        # Get bbox
                        bboxes = self._get_bbox(bbox_data[frame_key])
                        bboxes_n = self._get_bbox(bbox_data[frame_key_n])
                        
                        # Get relative player position
                        relative_width = event.get('relative_player_width', 0.0)
                        relative_height = event.get('relative_player_height', 0.0)
                        player_position = np.array([relative_width, relative_height])
                        
                        # Get hitting player label
                        # hitting_player = self._get_hitting_player_label(bboxes, player_position)
                        hitting_player, hitting_partner = self._get_hitting_players(bboxes, player_position)
                        hitting_player_n = self._get_hitting_player_label(bboxes_n, player_position)
                        
                        if hitting_partner == -1 or hitting_player == -1:
                            print(f'no hitting player or partner found in {frame_key}')
                            continue
                        
                        if hitting_player_n == -1:
                            print(f'no hitting player found in {frame_key_n}')
                            continue
                        
                        # Draw bbox on image
                        # hitting_player_bbox = bboxes[hitting_player] if hitting_player >= 0 else np.zeros(4)
                        # raw_image_path = os.path.join(self.data_dir, 'frames', rally['video_id'], f"{frame_key}.jpg")
                        # output_image_path = os.path.join(self.data_dir, 'hitting_player', rally['video_id'], f"frame_{(frame):06d}.jpg")
                        # self._draw_bbox_on_image(raw_image_path, output_image_path, hitting_player_bbox)
                        # self._extract_player_bbox(raw_image_path, output_image_path, hitting_player_bbox)
                        
                        hitting_player_bbox = bboxes[hitting_player] if hitting_player >= 0 else np.zeros(4)
                        hitting_partner_bbox = bboxes[hitting_partner] if hitting_partner >= 0 else np.zeros(4)
                        raw_image_path = os.path.join(self.data_dir, 'frames', rally['video_id'], f"{frame_key}.jpg")
                        output_image_path = os.path.join(self.data_dir, 'hitting_player', rally['video_id'], f"frame_{(frame):06d}.jpg")
                        output_image_path_partner = os.path.join(self.data_dir, 'hitting_partner', rally['video_id'], f"frame_{(frame):06d}.jpg")
                        # self._extract_player_bbox(raw_image_path, output_image_path_partner, hitting_partner_bbox)
                        
                        hitting_player_n_bbox = bboxes_n[hitting_player_n] if hitting_player_n >= 0 else np.zeros(4)
                        raw_image_path_n = os.path.join(self.data_dir, 'frames', rally['video_id'], f"{frame_key_n}.jpg")
                        output_image_path_n = os.path.join(self.data_dir, 'hitting_player_n', rally['video_id'], f"frame_{(frame + n):06d}.jpg")
                        # self._extract_player_bbox(raw_image_path_n, output_image_path_n, hitting_player_n_bbox)
                        
                        # Get event type
                        event_type = event['event']
                        serve_parts = event_type.split('_')

                        if train_label == 'side': # training for side (forehand/backhand)
                            is_serve = serve_parts[4]
                            
                            if is_serve == 'serve' or is_serve == 'second-serve':
                                continue
                            
                            serve_type = serve_parts[3]

                        elif train_label == 'shot_type': # training for serve type
                            serve_type = serve_parts[4]
                            
                            if serve_type == 'serve' or serve_type == 'second-serve':
                                continue
                                
                            if serve_type == 'return':
                                serve_type = 'swing'
                            # if serve_type == 'second-serve': serve_type = 'serve'
                            # if serve_type == 'return': serve_type = 'swing'
                            # if serve_type != 'serve': serve_type = 'non-serve'

                        elif train_label == 'shot_direction':
                            serve_type = serve_parts[5]
                            if serve_type == 'cc' or serve_type == 'io':
                                serve_type = 'cross'
                            elif serve_type == 'dl' or serve_type == 'ii':
                                serve_type = 'straight'
                            else:
                                serve_type = 'cross'
                        
                        elif train_label == 'serve_direction':
                            serve_type = serve_parts[5]
                            if serve_type != 't' and serve_type != 'w' and serve_type != 'b':
                                continue

                        elif train_label == 'formation':
                            is_serve = serve_parts[4]
                            if is_serve != 'serve' and is_serve != 'second-serve':
                                continue
                            serve_type = serve_parts[6]

                        elif train_label == 'outcome':
                            serve_type = serve_parts[7]
                            if serve_type == 'in': continue
                            
                        elif train_label == 'is_serve':
                            serve_type = serve_parts[4]
                            if serve_type == 'second-serve': serve_type = 'serve'
                            if serve_type != 'serve': serve_type = 'non-serve'

                        else:
                            raise(ValueError('Incorrect Train Label'))
                        
                        self.serve_types.add(serve_type)
                        self.samples.append({
                            'poses': poses,
                            'confidences': confidences,
                            'bboxes': bboxes,
                            'player_position': player_position,
                            'hitting_player': hitting_player,
                            'hitting_partner': hitting_partner,
                            # 'hitting_player_pose': poses[hitting_player],
                            'hitting_player_bbox': bboxes[hitting_player],
                            'hitting_partner_bbox': bboxes[hitting_partner],
                            'video_id': video_id,
                            'frame':  f"frame_{(frame):06d}",
                            'serve_type': serve_type,
                            'side': serve_parts[3],
                            'image_path': output_image_path,
                            'image_path_partner': output_image_path_partner,
                            'image_path_n': output_image_path_n,
                            'hitting_player_n_bbox': hitting_player_n_bbox,
                            'hitting_player_n': hitting_player_n
                        })
                        
    def _get_class_distribution(self):
        # Initialize a dictionary to store class counts
        class_counts = {}
        
        # Iterate through the dataset
        for i in range(len(self.samples)):
            label = self.samples[i]['serve_type']
            label_name = self.label_map[label]  # Convert to class name if you have a label map
            
            # Update counts
            if label_name in class_counts:
                class_counts[label_name] += 1
            else:
                class_counts[label_name] = 1
        
        # Calculate percentages
        total = len(self.samples)
        class_distribution = {
            k: {
                'count': v,
                'percentage': (v / total) * 100
            } for k, v in class_counts.items()
        }
        
        for class_name, stats in class_distribution.items():
            print(f"{class_name}: {stats['count']} samples ({stats['percentage']:.2f}%)")
        
        return class_distribution
    
    def _normalize_poses(self, poses):
        """Normalize pose coordinates to [0, 1]"""
        normalized = poses.copy()
        normalized[..., 0] /= self.width
        normalized[..., 1] /= self.height
        return normalized
    
    def _normalize_bboxes(self, bboxes):
        """Normalize bbox coordinates to [0, 1]"""
        normalized = bboxes.copy()
        normalized[:, [0, 2]] /= self.width
        normalized[:, [1, 3]] /= self.height
        return normalized
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Normalize coordinates
        normalized_poses = self._normalize_poses(sample['poses'])
        normalized_bboxes = self._normalize_bboxes(sample['bboxes'])
        
        # Prepare pose features with confidence
        poses_with_conf = np.concatenate([
            normalized_poses,
            sample['confidences'][..., np.newaxis]
        ], axis=-1)
        
        return {
            # 'poses': torch.tensor(poses_with_conf, dtype=torch.float32),
            # 'bboxes': torch.tensor(normalized_bboxes, dtype=torch.float32),
            # 'player_position': torch.tensor(sample['player_position'], dtype=torch.float32),
            'hitting_player': torch.tensor(sample['hitting_player'], dtype=torch.long),
            'hitting_partner': torch.tensor(sample['hitting_partner'], dtype=torch.long),
            'hitting_player_n': torch.tensor(sample['hitting_player_n'], dtype=torch.long),
            'serve_type': torch.tensor(self.label_map[sample['serve_type']], dtype=torch.long),
            # 'side': sample['side'],
            'image_path': sample['image_path'],
            'image_path_partner': sample['image_path_partner'],
            'image_path_n': sample['image_path_n'],
            # 'hitting_player_pose': torch.tensor(sample['hitting_player_pose'], dtype=torch.float32)
        }
    
    def __len__(self):
        return len(self.samples)
    
    