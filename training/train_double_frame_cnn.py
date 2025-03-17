import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datetime import datetime
import json
import random
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
import seaborn as sns
from collections import Counter
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, efficientnet_b0, efficientnet_v2_m, EfficientNet_V2_M_Weights

from tennis_dataset import TennisDataset
from dataset_functions import custom_collate_fn, evaluate, train_epoch, calculate_class_weights, plot_confusion_matrix, print_metrics_report

class DualImageTennisCNN(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DualImageTennisCNN, self).__init__()
        # Create two separate backbones for player and partner
        self.player_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.partner_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        
        self.player_features = nn.Sequential(*list(self.player_backbone.children())[:-1])
        self.partner_features = nn.Sequential(*list(self.partner_backbone.children())[:-1])
        
        # Get feature dimensions (2048 for ResNet50)
        self.feature_dim = 2048
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 512),  # Combine features from both players
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, player_img, partner_img):
        # Extract features from both images
        player_features = self.player_features(player_img)
        partner_features = self.partner_features(partner_img)
        
        # Flatten feature maps
        player_features = torch.flatten(player_features, 1)
        partner_features = torch.flatten(partner_features, 1)
        
        # Concatenate features from both images
        combined_features = torch.cat((player_features, partner_features), dim=1)
        
        # Pass through classifier
        output = self.classifier(combined_features)
        
        return output

# Image transformations
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


if __name__ == '__main__':
    # Seed
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define train and test video IDs
    train_videos = [
        'an7MXASRyI0',
        'eGFJAG-2jM8',
        'EMBw_kXc574',
        'Granollers_Zeballos vs Arevalo_Rojer  _ Toronto 2023 Doubles Semi-Finals',
        'Nick Kyrgios_Thanasi Kokkinakis vs Jack Sock_John Isner _ Indian Wells 2022 Doubles Highlights',
        'Rajeev Ram_Joe Salisbury vs Tim Puetz_Michael Venus _ Cincinnati 2022 Doubles Final'
    ]
    
    test_videos = [
        'Salisbury_Ram vs Krawietz_Puetz  _ Toronto 2023 Doubles Semi-Finals',
        'VUPKfQgXy8g'
    ]
    
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0001
    MAX_POSES = 4  # Keep this for the original dataset
    label = 'serve_direction'
    
    if label == 'outcome' or label == 'shot_direction':
        print(f'using n frames later')
        second_path = 'image_path_n'
    else:
        print(f'using partner image')
        second_path = 'image_path_partner'
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('experiments', f'dual_cnn_run_{timestamp}_{label}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Get image transformations
    train_transform, test_transform = get_transforms()
    
    # Create datasets using the existing TennisDataset class
    print("Creating datasets...")
    train_dataset = TennisDataset(
        data_dir='data',
        video_ids=train_videos,
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
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
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
        'model': 'DualImageResNet50',
        'class_balanced': True,
        "class_mappings": train_dataset.label_map,
        "class counts": class_counts
    }
    with open(os.path.join(save_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, indent=4)
    
    # Initialize model, criterion with class weights, and optimizer
    model = DualImageTennisCNN(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Use different learning rates for different parts of the model
    optimizer = torch.optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': LEARNING_RATE},
        {'params': model.player_features.parameters(), 'lr': LEARNING_RATE/10},
        {'params': model.partner_features.parameters(), 'lr': LEARNING_RATE/10}
    ], weight_decay=1e-4)
    
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
            model, train_loader, criterion, optimizer, device, train_transform, second_path=second_path)
        
        # Validation
        if (epoch + 1) % 5 == 0:
            val_loss, val_acc, _ = evaluate(
                model, test_loader, criterion, device, test_transform, second_path=second_path)
        
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
    
    # Evaluate with detailed metrics
    print("Computing final evaluation metrics...")
    _, final_acc, metrics = evaluate(
        model, 
        test_loader, 
        criterion, 
        device, 
        test_transform, 
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