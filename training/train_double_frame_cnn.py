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

# Calculate class weights for balanced training
def calculate_class_weights(dataset):
    # Count samples in each class
    class_counts = Counter()
    for i in range(len(dataset)):
        label = dataset[i]['serve_type'].item()
        class_counts[label] += 1
    
    # Calculate weights: 1 / (frequency)
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    
    # Normalize weights so they sum to n_classes
    n_classes = len(class_counts)
    weight_sum = sum(class_weights.values())
    class_weights = {cls: weight * n_classes / weight_sum for cls, weight in class_weights.items()}
    
    # Convert to tensor format for the loss function
    weights = torch.zeros(n_classes)
    for cls, weight in class_weights.items():
        weights[cls] = weight
    
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {weights}")
    
    return weights, class_counts

# Updated training function to load both player and partner images
def train_epoch(model, dataloader, criterion, optimizer, device, transform, second_path='image_path_partner'):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get labels and image paths
        labels = batch['serve_type'].to(device)
        player_image_paths = batch['image_path']
        partner_image_paths = batch[second_path]
        
        # Load and transform images for both players
        player_images = []
        partner_images = []
        
        # Process player images
        for path in player_image_paths:
            try:
                img = Image.open(path).convert('RGB')
                img = transform(img)
                player_images.append(img)
            except Exception as e:
                print(f"Error loading player image {path}: {e}")
                # Create a blank image if loading fails
                img = torch.zeros(3, 224, 224)
                player_images.append(img)
        
        # Process partner images
        for path in partner_image_paths:
            try:
                img = Image.open(path).convert('RGB')
                img = transform(img)
                partner_images.append(img)
            except Exception as e:
                print(f"Error loading partner image {path}: {e}")
                # Create a blank image if loading fails
                img = torch.zeros(3, 224, 224)
                partner_images.append(img)
        
        # Stack images into batch tensors
        if player_images and partner_images:
            player_images = torch.stack(player_images).to(device)
            partner_images = torch.stack(partner_images).to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with both images
            outputs = model(player_images, partner_images)
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

# Updated evaluation function for dual images
def evaluate(model, dataloader, criterion, device, transform, label_map=None, compute_metrics=False, second_path='image_path_partner'):
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
            # Get labels and image paths
            labels = batch['serve_type'].to(device)
            player_image_paths = batch['image_path']
            partner_image_paths = batch[second_path]
            
            # Load and transform images for both players
            player_images = []
            partner_images = []
            
            # Process player images
            for path in player_image_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img = transform(img)
                    player_images.append(img)
                except Exception as e:
                    print(f"Error loading player image {path}: {e}")
                    img = torch.zeros(3, 224, 224)
                    player_images.append(img)
            
            # Process partner images
            for path in partner_image_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img = transform(img)
                    partner_images.append(img)
                except Exception as e:
                    print(f"Error loading partner image {path}: {e}")
                    img = torch.zeros(3, 224, 224)
                    partner_images.append(img)
            
            # Stack images into batch tensors
            if player_images and partner_images:
                player_images = torch.stack(player_images).to(device)
                partner_images = torch.stack(partner_images).to(device)
                
                # Forward pass with both images
                outputs = model(player_images, partner_images)
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

# Plot confusion matrix
def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Print detailed metrics report
def print_metrics_report(metrics, save_path=None):
    report_str = "\n===== EVALUATION METRICS =====\n"
    
    # Mapping numerical indices to class names
    class_names = metrics['class_names']
    
    # Print per-class metrics
    report_str += "\nPer-class metrics:\n"
    report_str += "Class\t\tPrecision\tRecall\t\tF1\t\tSupport\t\tAUC\n"
    
    for i, class_name in enumerate(class_names):
        # For cleaner output formatting
        class_name_padded = class_name[:10] + (class_name[10:] and '..')
        
        # Get metrics
        precision = metrics['precision'][i] if i < len(metrics['precision']) else float('nan')
        recall = metrics['recall'][i] if i < len(metrics['recall']) else float('nan')
        f1 = metrics['f1'][i] if i < len(metrics['f1']) else float('nan')
        support = metrics['support'][i] if i < len(metrics['support']) else 0
        auc = metrics['auc'][i] if 'auc' in metrics and i < len(metrics['auc']) else float('nan')
        
        # Format the output
        report_str += f"{class_name_padded:<12}\t{precision:.4f}\t\t{recall:.4f}\t\t{f1:.4f}\t\t{support}\t\t{auc:.4f}\n"
    
    # Print macro and weighted averages
    report_str += "\nMacro average:\n"
    report_str += f"Precision: {np.nanmean(metrics['precision']):.4f}\n"
    report_str += f"Recall: {np.nanmean(metrics['recall']):.4f}\n"
    report_str += f"F1: {np.nanmean(metrics['f1']):.4f}\n"
    if 'auc' in metrics:
        report_str += f"AUC: {np.nanmean(metrics['auc']):.4f}\n"
    
    # Print weighted averages
    weights = metrics['support'] / np.sum(metrics['support'])
    report_str += "\nWeighted average:\n"
    report_str += f"Precision: {np.sum(metrics['precision'] * weights):.4f}\n"
    report_str += f"Recall: {np.sum(metrics['recall'] * weights):.4f}\n"
    report_str += f"F1: {np.sum(metrics['f1'] * weights):.4f}\n"
    
    print(report_str)
    
    # Save to file if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_str)

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