import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import argparse
import os
import numpy as np
from tqdm import tqdm


def create_model(num_classes):
    """Create MobileNetV3 model"""
    model = models.mobilenet_v3_small(pretrained=True)
    # 替换最后一层
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def get_data_loaders(data_dir, batch_size=32, train_ratio=0.8, val_ratio=0.2):
    """Create data loaders with automatic train/val split"""
    # Data preprocessing
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create full dataset first
    full_dataset = datasets.ImageFolder(data_dir, transform=None)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    
    print(f"Total samples: {total_size}")
    print(f"Train samples: {train_size} ({train_ratio:.1%})")
    print(f"Validation samples: {val_size} ({val_ratio:.1%})")
    
    # Split indices
    torch.manual_seed(42)  # For reproducible splits
    train_indices, val_indices = random_split(
        range(total_size), [train_size, val_size]
    )
    
    # Create separate datasets with different transforms
    train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, len(full_dataset.classes)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, num_classes):
    """Validate model and return detailed metrics"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Get predictions and scores
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Collect predictions, labels, and scores for detailed metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probabilities.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    # Calculate detailed metrics
    detailed_metrics = calculate_metrics(all_labels, all_predictions, all_scores, num_classes)
    
    return epoch_loss, epoch_acc, detailed_metrics


def calculate_metrics(y_true, y_pred, y_scores, num_classes):
    """Calculate detailed classification metrics"""
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Calculate per-class metrics
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    ap_per_class = np.zeros(num_classes)
    
    for i in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        
        # Precision and Recall
        precision_per_class[i] = tp / (tp + fp + 1e-8)
        recall_per_class[i] = tp / (tp + fn + 1e-8)
        
        # Average Precision (AP)
        if np.sum(y_true == i) > 0:  # Only calculate if class exists in ground truth
            # Create binary labels for this class
            binary_true = (y_true == i).astype(int)
            # Use the confidence scores for this class
            class_scores = y_scores[:, i]
            ap_per_class[i] = calculate_ap(binary_true, class_scores)
        else:
            ap_per_class[i] = 0.0
    
    # Calculate weighted averages
    class_counts = np.bincount(y_true, minlength=num_classes)
    total_samples = len(y_true)
    
    # Weighted precision and recall
    precision = np.sum(precision_per_class * class_counts) / total_samples
    recall = np.sum(recall_per_class * class_counts) / total_samples
    
    # mAP calculations
    map50 = np.mean(ap_per_class)  # For classification, this is essentially macro-averaged AP
    map50_95 = map50  # For classification, we don't have IoU thresholds, so it's the same
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'map50': map50 * 100,
        'map50_95': map50_95 * 100,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'ap_per_class': ap_per_class
    }


def calculate_ap(y_true, y_scores):
    """Calculate Average Precision for binary classification"""
    # Sort by scores in descending order
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Calculate precision and recall at each threshold
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    
    # Avoid division by zero
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (np.sum(y_true) + 1e-8)
    
    # Calculate AP using the trapezoidal rule
    # Add endpoints for proper integration
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])
    
    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    
    # Calculate AP
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    return ap


def main():
    parser = argparse.ArgumentParser(description='Train image classifier')
    parser.add_argument('--data-dir', type=str, required=True, help='Dataset directory path')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training data ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation data ratio (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model-path', type=str, default='best.pth', help='Model save path')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint for resuming training')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio - 1.0) > 1e-6:
        raise ValueError(f"Train ratio ({args.train_ratio}) + Val ratio ({args.val_ratio}) must equal 1.0")
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory does not exist: {args.data_dir}")
    
    # Auto select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, num_classes = get_data_loaders(
        args.data_dir, args.batch_size, args.train_ratio, args.val_ratio
    )
    print(f"Number of classes: {num_classes}")
    
    # Create model
    model = create_model(num_classes)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Initialize training variables
    best_val_acc = 0.0
    start_epoch = 0
    
    # Resume training if checkpoint is provided
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available (for backward compatibility)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("Warning: Optimizer state not found in checkpoint, using fresh optimizer")
        
        # Load scheduler state if available (for backward compatibility)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("Warning: Scheduler state not found in checkpoint, using fresh scheduler")
        
        # Load training progress
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        start_epoch = checkpoint.get('epoch', 0)
        
        # Verify num_classes matches
        checkpoint_classes = checkpoint.get('num_classes', num_classes)
        if checkpoint_classes != num_classes:
            print(f"Warning: Number of classes mismatch! Checkpoint: {checkpoint_classes}, Current: {num_classes}")
            print("This may cause issues if the model architecture doesn't match the data")
        
        print(f"Resumed from epoch {start_epoch}, best validation accuracy: {best_val_acc:.2f}%")
    elif args.resume:
        print(f"Warning: Resume checkpoint not found at {args.resume}, starting from scratch")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc, val_metrics = validate(model, val_loader, criterion, device, num_classes)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Val Precision: {val_metrics['precision']:.2f}%, Val Recall: {val_metrics['recall']:.2f}%")
        print(f"Val mAP@50: {val_metrics['map50']:.2f}%, Val mAP@50:95: {val_metrics['map50_95']:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'num_classes': num_classes,
                'best_val_acc': best_val_acc
            }, args.model_path)
            print(f"Best model saved, validation accuracy: {best_val_acc:.2f}%")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = args.model_path.replace('.pth', f'_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'num_classes': num_classes,
                'best_val_acc': best_val_acc
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()