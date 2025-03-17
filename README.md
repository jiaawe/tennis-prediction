# Tennis Prediction

A deep learning system for analyzing tennis matches and predicting various aspects of tennis gameplay from video data.

## Overview

This project uses computer vision and deep learning to analyze tennis doubles matches. It processes videos to:

1. Extract frames from tennis rally videos
2. Detect tennis players using GroundingDINO
3. Analyze player poses using YOLO-Pose
4. Create datasets with player positions, movements, and shot information
5. Train deep learning models to predict various aspects of tennis play:
   - Shot type (serve, return, volley, etc.)
   - Shot direction (cross-court, down-the-line)
   - Player side (forehand/backhand)
   - Serve direction (wide, T, body)
   - Formation
   - Shot outcome

## Directory Structure

```
tennis-prediction/
├── data/                     # Data directory (created during processing)
│   ├── bbox/                 # Bounding box information
│   ├── frames/               # Extracted video frames
│   ├── hitting_partner/      # Partner images
│   ├── hitting_player/       # Hitting player images
│   ├── hitting_player_n/     # Player images n frames later
│   ├── inference/            # Detection visualization
│   ├── pose/                 # Pose data
│   ├── processed_frames/     # Frames with annotations
│   ├── processed_videos/     # Videos with annotations
│   ├── rallies/              # Extracted rally clips
│   ├── transformed/          # Transformed data
│   └── videos/               # Original videos
├── models/                   # Pre-trained models
├── processing/               # Processing scripts
│   ├── batch_process.py
│   ├── extract_hitting_player.py
│   ├── extract_rallies.py
│   ├── inference_on_a_folder.py
│   ├── process_all_videos.py
│   ├── process_videos.py
│   └── tennis_analyzer.py
├── Open-GroundingDino/       # Submodule for GroundingDINO
└── training/                 # Training scripts
    ├── dataset_functions.py
    ├── tennis_dataset.py
    ├── train_double_frame_cnn.py
    ├── train_double_pose_gcn.py
    ├── train_single_frame_cnn.py
    └── train_single_pose_gcn.py
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA (for GPU acceleration)
- OpenCV
- Other dependencies (see requirements.txt)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tennis-prediction.git
   cd tennis-prediction
   ```

2. Set up GroundingDINO:
   ```bash
   git clone https://github.com/IDEA-Research/GroundingDINO.git Open-GroundingDINO
   cd Open-GroundingDINO
   pip install -e .
   ```

3. Download pre-trained models:
   ```bash
   # For YOLO-Pose
   mkdir -p models
   wget -O models/yolo11x-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt
   ```

4. Install other requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Data Processing Pipeline

The processing pipeline consists of several steps:

### 1. Split Videos into Rallies

Extract individual rally clips from full match videos:

```bash
python processing/extract_rallies.py data/videos/your_match_video.mp4
```

This script:
- Reads the original video files
- Processes the corresponding metadata JSON files
- Extracts rally segments with a 30-frame buffer at start and end
- Saves rally clips to `data/rallies/`

### 2. Extract Frames

Extract frames from each rally video:

```bash
python processing/process_all_videos.py --video_dir data/rallies
```

This script:
- Extracts frames from each rally video
- Saves frames to `data/frames/{video_id}/`

### 3. Player Detection

Detect tennis players in each frame using GroundingDINO:

```bash
python Open-GroundingDINO/inference_on_a_folder.py -c Open-GroundingDINO/tools/GroundingDINO_SwinT_OGC.py -p Open-GroundingDINO/logs/checkpoint0014.pth -i data/frames/{video_id} -t "tennis player" -o data/inference/{video_id}
```

This script:
- Uses a text prompt ("tennis player") to detect players
- Saves bounding box coordinates to `data/bbox/{video_id}_boxes.json`
- Saves visualization images to `data/inference/{video_id}/`

### 4. Pose Estimation

Extract player poses using YOLO-Pose:

```bash
python processing/batch_process.py --base_dir data
```

This script:
- Uses the tennis_analyzer.py module to process all videos
- Detects 17 keypoints for each player
- Saves pose data to `data/pose/{video_id}_pose.json`
- Creates processed videos with pose overlays in `data/processed_videos/`

## Model Architectures

The project implements multiple model architectures to analyze different aspects of tennis gameplay:

### 1. CNN Models

#### Single Frame CNN
- Based on a pre-trained ResNet50 backbone
- Takes a single frame of the hitting player as input
- Outputs predictions for:
  - is_serve (serve vs. non-serve)
  - shot_type (volley, groundstroke, smash, etc.)
  - side (forehand/backhand)

#### Dual Frame CNN
- Uses two ResNet50 backbones in parallel
- Processes both the hitting player and a second image
- Two variants:
  - Player-Partner model: Uses images of both players
  - Temporal model: Uses current frame and a future frame (n frames later)
- Used for predicting:
  - formation (using partner image)
  - shot_direction (cross-court, down-the-line)
  - serve_direction (wide, T, body)
  - outcome (using frame n=10 frames later)

### 2. Graph Convolutional Networks (GCN) for Pose Analysis

GCNs are particularly well-suited for pose analysis as they can directly model the skeletal structure of the human body as a graph.

#### GCN Layer
The fundamental building block is the `GCNLayer` which implements graph convolution operations:
```python
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # x shape: (batch_size, num_nodes, in_features)
        # adj shape: (num_nodes, num_nodes) - adjacency matrix
        support = torch.matmul(adj, x)  # Propagate features
        output = self.linear(support)  # Transform features
        return output
```

#### Single Pose GCN
The `PoseGCN` model processes a single player's pose:
- Treats the 17 body keypoints as nodes in a graph
- Uses an adjacency matrix that represents the human skeleton structure
- Takes input features of shape (batch_size, 17, 3) where each keypoint has x, y coordinates and a confidence value
- Processes through multiple GCN layers and outputs classification predictions

#### Dual Pose GCN
The `DualPoseGCN` model extends the single-pose approach to analyze two poses:
- Processes two separate poses (hitting player and partner, or current and future frames)
- Uses two parallel GCN backbones with shared architecture but separate weights
- Combines features from both poses for prediction
- Can be used for:
  - Formation analysis (using player and partner)
  - Shot direction prediction (using current and future frames)
  - Serve direction prediction
  - Outcome prediction

#### Skeleton Structure
The models use a predefined human skeleton structure to create the adjacency matrix:
```
skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face connections
    (5, 6), (5, 11), (6, 12), (11, 12),  # Body connections
    (5, 7), (7, 9),  # Left arm connections
    (6, 8), (8, 10),  # Right arm connections
    (11, 13), (13, 15),  # Left leg connections
    (12, 14), (14, 16)  # Right leg connections
]
```

### Advantages of GCN Models
- **Structure-aware**: GCNs explicitly model the skeletal structure of players
- **Efficient**: Requires fewer parameters than CNNs to model human poses
- **Pose-focused**: Focuses specifically on body position and movement rather than appearance
- **Rotation-invariant**: Better generalization to different camera angles
- **Complementary**: Provides a different modeling approach that complements the CNN models

## Training Models

The project includes four main training scripts:

### 1. Single Frame CNN

Trains a CNN model using single frames to predict player attributes:

```bash
python training/train_single_frame_cnn.py
```

### 2. Dual Frame CNN

Trains a CNN model using two frames to predict match dynamics:

```bash
python training/train_double_frame_cnn.py
```

### 3. Single Pose GCN

Trains a GCN using only the hitting player's pose:

```bash
python training/train_single_pose_gcn.py
```

Used for predicting:
- Shot type (swing, volley, smash, etc.)
- Side (forehand/backhand)
- Serve prediction (is_serve)

### 4. Dual Pose GCN

Trains a GCN using two poses:

```bash
python training/train_double_pose_gcn.py
```

Used for predicting:
- Formation (using partner pose)
- Shot direction (cross-court, down-the-line)
- Serve direction (wide, T, body)
- Shot outcome (using future frame)

## Dataset Structure

The `TennisDataset` class in `training/tennis_dataset.py` handles:
- Loading pose and bounding box data
- Extracting hitting player and partner data
- Creating normalized input features
- Mapping events to appropriate labels for training
- Data augmentation and preprocessing

Each training script includes:
- Hyperparameter settings
- Model initialization
- Training and validation loops
- Metrics calculation (accuracy, precision, recall, F1-score)
- Confusion matrix visualization
- Model checkpoint saving
- Training history plots

## Results and Evaluation

Model evaluation includes:
- Accuracy metrics
- Confusion matrices
- Precision, recall, and F1-score
- AUC for each class

Results are saved to the `experiments` directory with the following structure:
```
experiments/
├── cnn_run_{timestamp}_{label}/     # For CNN models
│   ├── best_model.pth               # Saved model weights
│   ├── confusion_matrix.png         # Confusion matrix visualization
│   ├── evaluation_metrics.txt       # Detailed metrics report
│   ├── hyperparameters.json         # Training configuration
│   └── training_history.png         # Loss and accuracy plots
└── gcn_run_{timestamp}_{label}/     # For GCN models
    ├── [similar structure]
```

## Notes

- The scripts assume a specific directory structure, which is created during processing.
- Some scripts are designed to be run within the Open-GroundingDINO directory.
- The system is designed for doubles tennis matches but can be adapted for singles.
- Different models (CNN vs GCN) have different strengths - CNNs excel at visual patterns while GCNs are better for structural pose information.

## Acknowledgments

- This project uses [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for player detection
- Pose estimation is performed using [Ultralytics YOLO-Pose](https://github.com/ultralytics/ultralytics)
- The GCN implementation is inspired by spatial graph convolutional networks for pose-based action recognition