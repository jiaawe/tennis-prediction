import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import json
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
import seaborn as sns
from collections import Counter

from tennis_dataset import TennisDataset
from dataset_functions import custom_collate_fn, calculate_class_weights, plot_confusion_matrix, print_metrics_report

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

# GCN model for pose data
class PoseGCN(nn.Module):
    def __init__(self, num_classes, num_keypoints=17, hidden_dim=64):
        super(PoseGCN, self).__init__()
        
        # Input features: x, y, confidence for each keypoint
        self.input_dim = 3
        
        # Define the skeleton adjacency matrix
        self.register_buffer('adj', self._get_adjacency_matrix(num_keypoints))
        
        # GCN layers
        self.gcn1 = GCNLayer(self.input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_keypoints, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def _get_adjacency_matrix(self, num_keypoints):
        # Define the human pose skeleton connections
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Face
            (5, 6), (5, 11), (6, 12), (11, 12),  # Body
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16)  # Right leg
        ]
        
        # Create adjacency matrix
        adj = torch.zeros(num_keypoints, num_keypoints)
        
        # Fill adjacency matrix based on skeleton
        for i, j in skeleton:
            adj[i, j] = 1
            adj[j, i] = 1  # Undirected graph
            
        # Add self-loops
        adj = adj + torch.eye(num_keypoints)
        
        # Normalize adjacency matrix
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        
        return adj
    
    def forward(self, x):
        # x shape: (batch_size, num_keypoints, 3) - 3 = (x, y, confidence)
        
        # Apply GCN layers
        x = F.relu(self.gcn1(x, self.adj))
        x = F.relu(self.gcn2(x, self.adj))
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get labels
        labels = batch['serve_type']
        
        # Handle labels - may be a list if sizes vary
        if isinstance(labels, list):
            labels = torch.stack(labels).to(device)
        else:
            labels = labels.to(device)
        
        # Get poses of hitting player
        hitting_player_indices = batch['hitting_player']
        
        # Get pose data
        poses = batch['poses']  # Shape: (batch_size, max_poses, 17, 3)
        
        # Extract the pose of the hitting player from each sample
        batch_size = poses.size(0)
        player_poses = []
        
        for i in range(batch_size):
            player_idx = hitting_player_indices[i].item()
            if player_idx >= 0 and player_idx < poses.size(1):
                player_pose = poses[i, player_idx]  # Shape: (17, 3)
            else:
                # If invalid index, use zeros
                player_pose = torch.zeros(17, 3)
            player_poses.append(player_pose)
        
        # Stack into batch tensor
        player_poses = torch.stack(player_poses).to(device)  # Shape: (batch_size, 17, 3)
            
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(player_poses)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    epoch_acc = 100 * correct / total if total > 0 else 0
    
    return epoch_loss, epoch_acc

# Evaluation function
def evaluate(model, dataloader, criterion, device, label_map=None, compute_metrics=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # For computing metrics
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get labels
            labels = batch['serve_type']
            
            # Handle labels - may be a list if sizes vary
            if isinstance(labels, list):
                labels = torch.stack(labels).to(device)
            else:
                labels = labels.to(device)
            
            # Get poses of hitting player
            hitting_player_indices = batch['hitting_player']
            
            # Get pose data
            poses = batch['poses']  # Shape: (batch_size, max_poses, 17, 3)
            
            # Extract the pose of the hitting player from each sample
            batch_size = poses.size(0)
            player_poses = []
            
            for i in range(batch_size):
                player_idx = hitting_player_indices[i].item()
                if player_idx >= 0 and player_idx < poses.size(1):
                    player_pose = poses[i, player_idx]  # Shape: (17, 3)
                else:
                    # If invalid index, use zeros
                    player_pose = torch.zeros(17, 3)
                player_poses.append(player_pose)
            
            # Stack into batch tensor
            player_poses = torch.stack(player_poses).to(device)  # Shape: (batch_size, 17, 3)
            
            # Forward pass
            outputs = model(player_poses)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for metrics calculation
            if compute_metrics:
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    epoch_acc = 100 * correct / total if total > 0 else 0
    
    # Compute detailed metrics if requested
    metrics = {}
    if compute_metrics and all_labels:
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        metrics['confusion_matrix'] = cm
        
        # Precision, recall, F1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        metrics['support'] = support
        
        # AUC if applicable (for each class in a one-vs-rest fashion)
        try:
            if len(np.unique(all_labels)) > 1:
                auc = []
                for i in range(len(np.unique(all_labels))):
                    # One-vs-rest approach for multiclass
                    y_true_binary = (all_labels == i).astype(int)
                    y_score = all_probabilities[:, i]
                    if len(np.unique(y_true_binary)) > 1:  # Need both classes present
                        auc.append(roc_auc_score(y_true_binary, y_score))
                    else:
                        auc.append(float('nan'))
                metrics['auc'] = auc
        except Exception as e:
            print(f"Error computing AUC: {e}")
            metrics['auc'] = [float('nan')] * len(np.unique(all_labels))
        
        # Map numerical labels to class names if provided
        if label_map:
            # Convert the label_map from {class_name: index} to {index: class_name}
            inverse_map = {v: k for k, v in label_map.items()}
            metrics['class_names'] = [inverse_map.get(i, f"Unknown-{i}") for i in range(len(label_map))]
        else:
            metrics['class_names'] = [str(i) for i in range(len(np.unique(all_labels)))]
    
    return epoch_loss, epoch_acc, metrics

if __name__ == '__main__':
    # Seed
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define videos list and split into train, validation, and test sets
    all_train_videos = [
        'an7MXASRyI0',
        'eGFJAG-2jM8',
        'EMBw_kXc574',
        'Granollers_Zeballos vs Arevalo_Rojer  _ Toronto 2023 Doubles Semi-Finals',
        'Nick Kyrgios_Thanasi Kokkinakis vs Jack Sock_John Isner _ Indian Wells 2022 Doubles Highlights',
        'Rajeev Ram_Joe Salisbury vs Tim Puetz_Michael Venus _ Cincinnati 2022 Doubles Final'
    ]
    
    # Split all_train_videos into train and validation
    train_videos = all_train_videos[:-2]  # Use first 4 videos for training
    val_videos = all_train_videos[-2:]    # Use last 2 videos for validation
    
    test_videos = [
        'Salisbury_Ram vs Krawietz_Puetz  _ Toronto 2023 Doubles Semi-Finals',
        'VUPKfQgXy8g'
    ]
    
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    MAX_POSES = 4  # Keep this for the original dataset
    label = 'shot_direction_all'  # Can be changed to any of the available labels
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('experiments', f'gcn_run_{timestamp}_{label}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = TennisDataset(
        data_dir='data',
        video_ids=train_videos,
        max_poses=MAX_POSES,
        train_label=label
    )
    
    # Create validation dataset
    val_dataset = TennisDataset(
        data_dir='data',
        video_ids=val_videos,
        max_poses=MAX_POSES,
        train_label=label
    )
    
    test_dataset = TennisDataset(
        data_dir='data',
        video_ids=test_videos,
        max_poses=MAX_POSES,
        train_label=label
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    
    # Create validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Get number of classes from the dataset's label_map
    num_classes = len(train_dataset.label_map)
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {train_dataset.label_map}")
    
    # Calculate class weights for the loss function
    class_weights, class_counts = calculate_class_weights(train_dataset)
    
    # Save hyperparameters
    hyperparams = {
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'max_poses': MAX_POSES,
        'model': 'PoseGCN',
        'hidden_dim': 64,
        'class_balanced': True,
        "class_mappings": train_dataset.label_map,
        'class_counts': class_counts
    }
    with open(os.path.join(save_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, indent=4)
    
    # Initialize model, criterion with class weights, and optimizer
    model = PoseGCN(num_classes=num_classes, num_keypoints=17, hidden_dim=64).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Train model
    print("Starting training...")
    best_val_acc = 0
    
    for epoch in range(NUM_EPOCHS):
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device)
        
        # Validation (now using val_loader instead of test_loader) 
        val_loss, val_acc, _ = evaluate(
            model, val_loader, criterion, device)
    
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch statistics
        print(f'\nEpoch [{epoch+1}/{NUM_EPOCHS}]:')
        print(f'  Training Loss: {train_loss:.4f}')
        print(f'  Training Accuracy: {train_acc:.2f}%')
        print(f'  Validation Loss: {val_loss:.4f}')
        print(f'  Validation Accuracy: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
            }, os.path.join(save_dir, 'best_model.pth'))
    
    # Save training history and plots
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()
    
    # Load the best model for final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate with detailed metrics on the test set
    print("Computing final evaluation metrics...")
    _, final_acc, metrics = evaluate(
        model, 
        test_loader, 
        criterion, 
        device, 
        label_map=train_dataset.label_map,
        compute_metrics=True
    )
    
    # Plot and save confusion matrix
    if 'confusion_matrix' in metrics:
        plot_confusion_matrix(
            metrics['confusion_matrix'], 
            metrics['class_names'], 
            os.path.join(save_dir, 'confusion_matrix.png')
        )
    
    # Print and save detailed metrics report
    print_metrics_report(metrics, os.path.join(save_dir, 'evaluation_metrics.txt'))
    
    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model and all evaluation results saved to: {save_dir}")