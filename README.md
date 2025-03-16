# Tennis Prediction

A deep learning system for analyzing tennis matches and predicting various aspects of tennis gameplay from video data.

## Overview

This project uses computer vision and deep learning to analyze tennis doubles matches. It processes videos to:

1. Extract frames from tennis rally videos
2. Detect tennis players using GroundingDINO
3. Analyze player poses using YOLO-Pose
4. Create datasets with player positions, movements, and shot information
5. Train CNN models to predict various aspects of tennis play:
   - Shot type (serve, return, etc.)
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
    ├── tennis_dataset.py
    ├── train_double_frame_cnn.py
    └── train_single_frame_cnn.py
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

### 2. Process the Rallies

For each rally, extract frames, detect players, and analyze poses:

```bash
# Process all rally videos, put process_all_videos.py, process_videos.py, inference_on_a_folder.py files inside Open-GroundingDINO folder
python Open-GroundingDINO/process_all_videos.py --video_dir data/rallies --config_file Open-GroundingDINO/tools/GroundingDINO_SwinT_OGC.py --checkpoint_path Open-GroundingDINO/logs/checkpoint0014.pth --text_prompt "tennis player"

# Run pose detection on all detected players
python processing/batch_process.py --base_dir data
```

## Training Models

The project includes two main training scripts:

### 1. Single Frame CNN

Trains a CNN model using single frames to predict:
- is_serve
- shot_type
- side (forehand/backhand)

```bash
python training/train_single_frame_cnn.py
```

### 2. Dual Frame CNN

Trains a CNN model using multiple frames to predict:
- formation (using partner image)
- shot_direction (cross-court, down-the-line)
- serve_direction (wide, T, body)
- outcome (using frame n=10 frames later)

```bash
python training/train_double_frame_cnn.py
```

## Model Architecture

The project uses two main model architectures:

1. **Single Frame CNN**: Based on ResNet50, processes a single image of the hitting player.

2. **Dual Frame CNN**: Uses two ResNet50 backbones, one for the hitting player and one for either:
   - The hitting partner (for formation, serve_direction)
   - A future frame (for outcome, shot_direction)

Both models are pre-trained on ImageNet and fine-tuned on tennis data.

## Results and Evaluation

Model evaluation includes:
- Accuracy metrics
- Confusion matrices
- Precision, recall, and F1-score
- AUC for each class

Results are saved to the `experiments` directory.

## Notes

- The scripts assume a specific directory structure, which is created during processing.
- Some scripts are designed to be run within the Open-GroundingDINO directory.
- The system is designed for doubles tennis matches but can be adapted for singles.

## Acknowledgments

- This project uses [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for player detection
- Pose estimation is performed using [Ultralytics YOLO-Pose](https://github.com/ultralytics/ultralytics)