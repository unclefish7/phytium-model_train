import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_model(num_classes):
    """Create MobileNetV3 model"""
    model = models.mobilenet_v3_small(pretrained=True)
    # 替换最后一层
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def get_test_loader(test_dir, batch_size=32):
    """Create test data loader"""
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return test_loader, test_dataset.classes


def load_model(model_path, device):
    """Load trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint['num_classes']
    best_val_acc = checkpoint.get('best_val_acc', 0)
    
    model = create_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully, classes: {num_classes}, best validation accuracy: {best_val_acc:.2f}%")
    return model, num_classes


def predict(model, test_loader, device):
    """Make predictions and collect confidence scores"""
    all_predictions = []
    all_labels = []
    all_scores = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Get predictions and confidence scores
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_scores)


def calculate_detailed_metrics(y_true, y_pred, y_scores, num_classes):
    """Calculate detailed classification metrics including mAP"""
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Calculate per-class metrics
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    f1_per_class = np.zeros(num_classes)
    ap_per_class = np.zeros(num_classes)
    
    for i in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        
        # Precision, Recall, F1
        precision_per_class[i] = tp / (tp + fp + 1e-8)
        recall_per_class[i] = tp / (tp + fn + 1e-8)
        f1_per_class[i] = 2 * precision_per_class[i] * recall_per_class[i] / (precision_per_class[i] + recall_per_class[i] + 1e-8)
        
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
    
    # Weighted precision, recall, F1
    precision = np.sum(precision_per_class * class_counts) / total_samples
    recall = np.sum(recall_per_class * class_counts) / total_samples
    f1_score = np.sum(f1_per_class * class_counts) / total_samples
    
    # mAP calculations
    map50 = np.mean(ap_per_class)  # Macro-averaged AP
    map50_95 = map50  # For classification, same as mAP@50
    
    # Calculate confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'map50': map50,
        'map50_95': map50_95,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'ap_per_class': ap_per_class,
        'confusion_matrix': cm
    }


def calculate_ap(y_true, y_scores):
    """Calculate Average Precision for binary classification"""
    if np.sum(y_true) == 0:  # No positive samples
        return 0.0
    
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


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar(im)
    
    # Set labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add values to each cell
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to: {save_path}")


def print_detailed_results(metrics, class_names):
    """Print detailed evaluation results"""
    print("\n" + "="*60)
    print("Model Evaluation Results")
    print("="*60)
    
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted Precision: {metrics['precision']:.4f}")
    print(f"Weighted Recall: {metrics['recall']:.4f}")
    print(f"Weighted F1-Score: {metrics['f1_score']:.4f}")
    print(f"mAP@50: {metrics['map50']:.4f}")
    print(f"mAP@50:95: {metrics['map50_95']:.4f}")
    
    print("\nDetailed metrics by class:")
    print("-" * 80)
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AP':<10}")
    print("-" * 80)
    
    for i, class_name in enumerate(class_names):
        precision = metrics['precision_per_class'][i]
        recall = metrics['recall_per_class'][i]
        f1 = metrics['f1_per_class'][i]
        ap = metrics['ap_per_class'][i]
        print(f"{class_name:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {ap:<10.4f}")


def main():
    parser = argparse.ArgumentParser(description='Test image classifier')
    parser.add_argument('--test-dir', type=str, required=True, help='Test data path')
    parser.add_argument('--model-path', type=str, default='best.pth', help='Model file path')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--save-cm', type=str, default='confusion_matrix.png', help='Confusion matrix save path')
    
    args = parser.parse_args()
    
    # Check test data path
    if not os.path.exists(args.test_dir):
        raise ValueError(f"Test data path does not exist: {args.test_dir}")
    
    # Auto select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, num_classes = load_model(args.model_path, device)
    
    # Create test data loader
    test_loader, class_names = get_test_loader(args.test_dir, args.batch_size)
    print(f"Test data classes: {class_names}")
    print(f"Number of test samples: {len(test_loader.dataset)}")
    
    # Make predictions
    predictions, true_labels, confidence_scores = predict(model, test_loader, device)
    
    # Calculate evaluation metrics
    metrics = calculate_detailed_metrics(true_labels, predictions, confidence_scores, num_classes)
    
    # Print detailed results
    print_detailed_results(metrics, class_names)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, args.save_cm)
    
    print("\nTesting completed!")


if __name__ == '__main__':
    main()