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

def custom_collate_fn(batch):
    """
    Without custom collate, there will be an error when using DataLoader to load the data.
    """
    # Initialize an empty dictionary to store the batched data
    batch_dict = {}
    
    # Get all keys from the first sample
    keys = batch[0].keys()
    
    for key in keys:
        # Handle image paths differently - don't stack these
        if key in ['image_path', 'image_path_partner', 'image_path_n', 'side', 'video_id', 'frame']:
            batch_dict[key] = [item[key] for item in batch]
            continue
        
        # Try to stack tensors if they have the same size
        try:
            batch_dict[key] = torch.stack([item[key] for item in batch])
        except:
            # If stacking fails, store as a list
            batch_dict[key] = [item[key] for item in batch]
    
    return batch_dict


# Updated training function to load images during training
def train_epoch(model, dataloader, criterion, optimizer, device, transform):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get labels and image paths
        labels = batch['serve_type']
        
        # Handle labels - may be a list if sizes vary
        if isinstance(labels, list):
            labels = torch.stack(labels).to(device)
        else:
            labels = labels.to(device)
            
        image_paths = batch['image_path']
        
        # Load and transform images
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                img = transform(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # Create a blank image if loading fails
                img = torch.zeros(3, 224, 224)
                images.append(img)
        
        # Stack images into a batch tensor
        if images:
            images = torch.stack(images).to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
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

# Updated evaluation function with detailed metrics collection
def evaluate(model, dataloader, criterion, device, transform, label_map=None, compute_metrics=False):
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
            labels = batch['serve_type']
            
            # Handle labels - may be a list if sizes vary
            if isinstance(labels, list):
                labels = torch.stack(labels).to(device)
            else:
                labels = labels.to(device)
                
            image_paths = batch['image_path']
            
            # Load and transform images
            images = []
            for path in image_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img = transform(img)
                    images.append(img)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
                    # Create a blank image if loading fails
                    img = torch.zeros(3, 224, 224)
                    images.append(img)
            
            # Stack images into a batch tensor
            if images:
                images = torch.stack(images).to(device)
                
                # Forward pass
                outputs = model(images)
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