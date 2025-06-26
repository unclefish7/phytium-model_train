import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
import cv2
import time


def create_model(num_classes):
    """Create MobileNetV3 model"""
    model = models.mobilenet_v3_small(pretrained=True)
    # 替换最后一层
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def load_model(model_path, device):
    """Load trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint['num_classes']
    best_val_acc = checkpoint.get('best_val_acc', 0)
    
    model = create_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    print(f"Number of classes: {num_classes}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, num_classes


def get_image_transform():
    """Get image preprocessing transform"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def classify_image(model, image_path, transform, device, class_names=None):
    """Classify a single image"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Record inference time
        start_time = time.time()
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Calculate inference time
        inference_time = time.time() - start_time
            
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        
        # Get top-5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, min(5, probabilities.size(1)))
        top5_predictions = []
        for i in range(top5_prob.size(1)):
            class_idx = top5_indices[0][i].item()
            prob = top5_prob[0][i].item()
            class_name = class_names[class_idx] if class_names else f"Class_{class_idx}"
            top5_predictions.append((class_name, prob))
        
        return {
            'predicted_class': predicted_class,
            'predicted_class_name': class_names[predicted_class] if class_names else f"Class_{predicted_class}",
            'confidence': confidence_score,
            'top5_predictions': top5_predictions,
            'inference_time': inference_time
        }
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def classify_images_in_directory(model, image_dir, transform, device, class_names=None):
    """Classify all images in a directory"""
    # Supported image extensions
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    results = []
    image_files = []
    
    # Collect all image files
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images to classify...")
    
    # Classify each image
    for i, image_path in enumerate(image_files):
        result = classify_image(model, image_path, transform, device, class_names)
        if result:
            result['image_path'] = image_path
            result['image_name'] = os.path.basename(image_path)
            results.append(result)
            
            # Print progress with inference time
            print(f"Processing ({i+1}/{len(image_files)}): {os.path.basename(image_path)} - "
                  f"Inference time: {result['inference_time']:.4f}s")
    
    return results


def visualize_predictions(results, output_dir, class_names=None, max_images_per_class=10):
    """Visualize classification results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by predicted class
    class_results = defaultdict(list)
    for result in results:
        class_results[result['predicted_class_name']].append(result)
    
    # Create visualization for each class
    for class_name, class_imgs in class_results.items():
        # Sort by confidence (highest first)
        class_imgs.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Limit number of images per class
        class_imgs = class_imgs[:max_images_per_class]
        
        if not class_imgs:
            continue
            
        # Calculate grid size
        n_images = len(class_imgs)
        n_cols = min(5, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten() if n_images > 1 else [axes]
        
        fig.suptitle(f'Class: {class_name} (Top {n_images} predictions)', fontsize=16)
        
        for i, result in enumerate(class_imgs):
            try:
                # Load and display image
                img = Image.open(result['image_path']).convert('RGB')
                
                if i < len(axes):
                    axes[i].imshow(img)
                    axes[i].set_title(f'{result["image_name"]}\nConf: {result["confidence"]:.3f}', 
                                    fontsize=10)
                    axes[i].axis('off')
            except Exception as e:
                print(f"Error loading image {result['image_path']}: {str(e)}")
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        # Save figure
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'class_{class_name}_predictions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization for class {class_name}: {output_path}")


def create_summary_visualization(results, output_dir):
    """Create summary visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Class distribution
    class_counts = defaultdict(int)
    confidences_by_class = defaultdict(list)
    
    for result in results:
        class_name = result['predicted_class_name']
        class_counts[class_name] += 1
        confidences_by_class[class_name].append(result['confidence'])
    
    # Plot class distribution
    plt.figure(figsize=(12, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(classes, counts)
    plt.title('Predicted Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom')
    
    # Plot confidence distribution
    plt.subplot(1, 2, 2)
    all_confidences = [result['confidence'] for result in results]
    plt.hist(all_confidences, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Confidence Score Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Images')
    plt.axvline(np.mean(all_confidences), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_confidences):.3f}')
    plt.legend()
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'classification_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary visualization: {summary_path}")
    
    # Create confidence statistics per class
    plt.figure(figsize=(12, 6))
    class_names = list(confidences_by_class.keys())
    mean_confidences = [np.mean(confidences_by_class[cls]) for cls in class_names]
    std_confidences = [np.std(confidences_by_class[cls]) for cls in class_names]
    
    plt.errorbar(range(len(class_names)), mean_confidences, yerr=std_confidences, 
                fmt='o', capsize=5, capthick=2)
    plt.title('Mean Confidence by Class')
    plt.xlabel('Class')
    plt.ylabel('Mean Confidence')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    confidence_path = os.path.join(output_dir, 'confidence_by_class.png')
    plt.savefig(confidence_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confidence analysis: {confidence_path}")


def save_results_json(results, output_path):
    """Save classification results to JSON file"""
    # Convert results to JSON-serializable format
    json_results = []
    for result in results:
        json_result = {
            'image_path': result['image_path'],
            'image_name': result['image_name'],
            'predicted_class': result['predicted_class'],
            'predicted_class_name': result['predicted_class_name'],
            'confidence': float(result['confidence']),
            'inference_time': float(result['inference_time']),
            'top5_predictions': [(name, float(prob)) for name, prob in result['top5_predictions']]
        }
        json_results.append(json_result)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")


def print_classification_summary(results):
    """Print classification summary statistics"""
    if not results:
        print("No classification results to summarize.")
        return
    
    print("\n" + "="*60)
    print("Classification Summary")
    print("="*60)
    
    # Overall statistics
    total_images = len(results)
    mean_confidence = np.mean([r['confidence'] for r in results])
    
    print(f"Total images processed: {total_images}")
    print(f"Average confidence: {mean_confidence:.3f}")
    
    # Class distribution
    class_counts = defaultdict(int)
    class_confidences = defaultdict(list)
    
    for result in results:
        class_name = result['predicted_class_name']
        class_counts[class_name] += 1
        class_confidences[class_name].append(result['confidence'])
    
    print("\nClass distribution:")
    print("-" * 50)
    for class_name, count in sorted(class_counts.items()):
        mean_conf = np.mean(class_confidences[class_name])
        percentage = (count / total_images) * 100
        print(f"{class_name:<20}: {count:>4} images ({percentage:5.1f}%) - Avg conf: {mean_conf:.3f}")
    
    # High/Low confidence predictions
    high_conf_threshold = 0.9
    low_conf_threshold = 0.5
    
    high_conf_count = sum(1 for r in results if r['confidence'] >= high_conf_threshold)
    low_conf_count = sum(1 for r in results if r['confidence'] < low_conf_threshold)
    
    print("\nConfidence analysis:")
    print("-" * 30)
    print(f"High confidence (≥{high_conf_threshold}): {high_conf_count} images ({high_conf_count/total_images*100:.1f}%)")
    print(f"Low confidence (<{low_conf_threshold}): {low_conf_count} images ({low_conf_count/total_images*100:.1f}%)")


def print_inference_time_statistics(results):
    """Print inference time statistics"""
    if not results:
        print("No inference time data to analyze.")
        return
    
    # Extract inference times
    inference_times = [r['inference_time'] for r in results]
    
    # Calculate statistics
    mean_time = np.mean(inference_times)
    median_time = np.median(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    percentile_95 = np.percentile(inference_times, 95)
    percentile_99 = np.percentile(inference_times, 99)
    
    # Calculate throughput (images per second)
    total_time = np.sum(inference_times)
    throughput = len(inference_times) / total_time
    
    print("\n" + "="*60)
    print("Inference Time Statistics")
    print("="*60)
    
    print(f"Total images processed: {len(inference_times)}")
    print(f"Total inference time: {total_time:.3f}s")
    print(f"Throughput: {throughput:.2f} images/second")
    
    print("\nPer-image inference time statistics:")
    print("-" * 40)
    print(f"Mean:           {mean_time:.4f}s ({mean_time*1000:.2f}ms)")
    print(f"Median:         {median_time:.4f}s ({median_time*1000:.2f}ms)")
    print(f"Standard dev:   {std_time:.4f}s ({std_time*1000:.2f}ms)")
    print(f"Min:            {min_time:.4f}s ({min_time*1000:.2f}ms)")
    print(f"Max:            {max_time:.4f}s ({max_time*1000:.2f}ms)")
    print(f"95th percentile: {percentile_95:.4f}s ({percentile_95*1000:.2f}ms)")
    print(f"99th percentile: {percentile_99:.4f}s ({percentile_99*1000:.2f}ms)")
    
    # Find fastest and slowest images
    fastest_idx = np.argmin(inference_times)
    slowest_idx = np.argmax(inference_times)
    
    print("\nFastest inference:")
    print(f"  Image: {results[fastest_idx]['image_name']}")
    print(f"  Time: {results[fastest_idx]['inference_time']:.4f}s")
    
    print("\nSlowest inference:")
    print(f"  Image: {results[slowest_idx]['image_name']}")
    print(f"  Time: {results[slowest_idx]['inference_time']:.4f}s")


def get_class_names_from_checkpoint(model_path):
    """Try to extract class names from model checkpoint"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'class_names' in checkpoint:
            return checkpoint['class_names']
        elif 'class_to_idx' in checkpoint:
            # Reverse the class_to_idx mapping
            class_to_idx = checkpoint['class_to_idx']
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            return [idx_to_class[i] for i in range(len(idx_to_class))]
    except:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description='Classify images using trained model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model file')
    parser.add_argument('--image-dir', type=str, required=True, help='Directory containing images to classify')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--max-vis-per-class', type=int, default=10, 
                       help='Maximum number of images to visualize per class')
    parser.add_argument('--class-names', type=str, nargs='+', default=None,
                       help='List of class names (optional)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Auto select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, num_classes = load_model(args.model_path, device)
    
    # Get class names
    class_names = args.class_names
    if class_names is None:
        class_names = get_class_names_from_checkpoint(args.model_path)
    
    if class_names is None:
        print("No class names provided, using generic names (Class_0, Class_1, ...)")
        class_names = [f"Class_{i}" for i in range(num_classes)]
    else:
        print(f"Using class names: {class_names}")
    
    # Get image transform
    transform = get_image_transform()
    
    # Classify all images
    print(f"\nStarting classification of images in: {args.image_dir}")
    results = classify_images_in_directory(model, args.image_dir, transform, device, class_names)
    
    if not results:
        print("No images were successfully classified.")
        return
    
    # Print summary
    print_classification_summary(results)
    
    # Print inference time statistics
    print_inference_time_statistics(results)
    
    # Save results to JSON
    json_path = os.path.join(args.output_dir, 'classification_results.json')
    save_results_json(results, json_path)
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_predictions(results, args.output_dir, class_names, args.max_vis_per_class)
    create_summary_visualization(results, args.output_dir)
    
    print("\nClassification completed!")
    print(f"Results saved to: {args.output_dir}")
    print("- JSON results: classification_results.json")
    print("- Summary plots: classification_summary.png, confidence_by_class.png")
    print("- Class visualizations: class_*_predictions.png")


if __name__ == '__main__':
    main()