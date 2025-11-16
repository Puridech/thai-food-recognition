"""
🍜 Thai Food Recognition - Batch Testing
ทดสอบ Layer 2 Model แบบทั้ง Folder พร้อมสร้าง Report

Usage:
    python version2_batch_test_layer2.py --folder D:\images_test\images_external
    python version2_batch_test_layer2.py --folder D:\images_test\images_external/ --save_report
    python version2_batch_test_layer2.py --folder D:\images_test\images_external/ --top_k 3 --confidence_threshold 80
    python version2_batch_test_layer2.py --folder D:\images_test\images_external --device cuda
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import csv

# ================================
# Model Architecture
# ================================

class CLIPClassifier(nn.Module):
    """
    Supports both old and new architectures:
    - Old: With BatchNorm layers (from early training)
    - New: Without BatchNorm (current version)
    """
    def __init__(self, clip_model, num_classes, use_batchnorm=False):
        super(CLIPClassifier, self).__init__()
        self.clip = clip_model
        
        # Get embedding dimension from CLIP model
        embedding_dim = clip_model.vision_model.config.hidden_size
        
        # Classification head
        if use_batchnorm:
            # Old architecture (with BatchNorm)
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            # New architecture (without BatchNorm)
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, pixel_values):
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.pooler_output
        logits = self.classifier(image_embeds)
        return logits


# ================================
# Image Preprocessing
# ================================

def preprocess_image(image_path, model_name="openai/clip-vit-base-patch32"):
    """Load and preprocess image for CLIP model"""
    try:
        image = Image.open(image_path).convert('RGB')
        processor = CLIPProcessor.from_pretrained(model_name)
        inputs = processor(images=image, return_tensors="pt")
        return inputs['pixel_values']
    except Exception as e:
        raise Exception(f"Error loading image: {e}")


# ================================
# Model Loading
# ================================

def load_model(model_path, device='cpu'):
    """Load the fine-tuned Layer 2 model"""
    
    # Try multiple possible locations
    possible_paths = [
        model_path,
        'models/layer2_finetuned/model_final.pth',
        '../models/layer2_finetuned/model_final.pth',
        'model_final.pth',
        'layer2_finetuned/model_final.pth'
    ]
    
    actual_path = None
    for path in possible_paths:
        if os.path.exists(path):
            actual_path = path
            break
    
    if actual_path is None:
        print(f"❌ Model file not found")
        sys.exit(1)
    
    print(f"📦 Loading model from: {actual_path}")
    checkpoint = torch.load(actual_path, map_location=device, weights_only=False)
    
    classes = checkpoint['classes']
    num_classes = len(classes)
    
    # Auto-detect CLIP variant and architecture
    model_name = checkpoint.get('model_name', None)
    use_batchnorm = False
    
    if 'model_state_dict' in checkpoint:
        state_dict_keys = checkpoint['model_state_dict'].keys()
        
        # Detect BatchNorm
        has_batchnorm = any('BatchNorm' in str(key) or 'running_mean' in str(key) 
                           for key in state_dict_keys)
        if has_batchnorm:
            use_batchnorm = True
            print(f"🔍 Detected: Old architecture (with BatchNorm)")
        
        # Detect CLIP variant from patch embedding
        if 'clip.vision_model.embeddings.patch_embedding.weight' in state_dict_keys:
            patch_weight = checkpoint['model_state_dict']['clip.vision_model.embeddings.patch_embedding.weight']
            patch_size = patch_weight.shape[2]
            
            if patch_size == 32:
                model_name = "openai/clip-vit-base-patch32"
                print(f"🔍 Detected: CLIP ViT-B/32")
            elif patch_size == 16:
                model_name = "openai/clip-vit-base-patch16"
                print(f"🔍 Detected: CLIP ViT-B/16")
    
    if model_name is None:
        model_name = "openai/clip-vit-base-patch32"
    
    print(f"📥 Loading base CLIP model: {model_name}...")
    clip_model = CLIPModel.from_pretrained(model_name)
    
    model = CLIPClassifier(clip_model, num_classes, use_batchnorm=use_batchnorm)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded! Classes: {num_classes}\n")
    
    return model, classes, model_name


# ================================
# Prediction
# ================================

def predict(model, image_tensor, classes, device='cpu', top_k=3):
    """Make prediction on a single image"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, min(top_k, len(classes)))
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            results.append({
                'class': classes[idx],
                'confidence': prob.item() * 100,
                'index': idx.item()
            })
        
        return results


# ================================
# Batch Processing
# ================================

def batch_predict(model, image_folder, classes, model_name, device='cpu', top_k=3):
    """Process all images in a folder"""
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_folder).glob(f'*{ext}'))
        image_files.extend(Path(image_folder).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ No images found in: {image_folder}")
        sys.exit(1)
    
    print(f"📁 Found {len(image_files)} images")
    print("🔄 Processing...\n")
    
    results = []
    
    for i, image_path in enumerate(image_files, 1):
        try:
            print(f"[{i:3d}/{len(image_files)}] {image_path.name:<40}...", end=' ')
            
            # Preprocess and predict
            image_tensor = preprocess_image(str(image_path), model_name)
            predictions = predict(model, image_tensor, classes, device, top_k)
            
            # Store result
            result = {
                'image': image_path.name,
                'path': str(image_path),
                'predictions': predictions,
                'top_class': predictions[0]['class'],
                'top_confidence': predictions[0]['confidence']
            }
            results.append(result)
            
            # Print result
            conf = predictions[0]['confidence']
            emoji = "✅" if conf >= 80 else "✓ " if conf >= 60 else "⚠️ "
            print(f"{emoji} {predictions[0]['class']:<20} ({conf:5.1f}%)")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({
                'image': image_path.name,
                'path': str(image_path),
                'error': str(e)
            })
    
    return results


# ================================
# Report Generation
# ================================

def generate_summary(results):
    """Generate summary statistics"""
    
    successful = [r for r in results if 'predictions' in r]
    failed = [r for r in results if 'error' in r]
    
    print("\n" + "="*70)
    print("📊 BATCH TESTING SUMMARY")
    print("="*70)
    print(f"Total Images: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        # Confidence statistics
        confidences = [r['top_confidence'] for r in successful]
        avg_conf = sum(confidences) / len(confidences)
        max_conf = max(confidences)
        min_conf = min(confidences)
        
        print(f"\n📈 Confidence Statistics:")
        print(f"   Average: {avg_conf:.2f}%")
        print(f"   Max: {max_conf:.2f}%")
        print(f"   Min: {min_conf:.2f}%")
        
        # Confidence distribution
        very_high = len([c for c in confidences if c >= 80])
        high = len([c for c in confidences if 60 <= c < 80])
        medium = len([c for c in confidences if 40 <= c < 60])
        low = len([c for c in confidences if c < 40])
        
        print(f"\n🎯 Confidence Distribution:")
        print(f"   Very High (≥80%): {very_high:3d} ({very_high/len(successful)*100:5.1f}%)")
        print(f"   High (60-79%):    {high:3d} ({high/len(successful)*100:5.1f}%)")
        print(f"   Medium (40-59%):  {medium:3d} ({medium/len(successful)*100:5.1f}%)")
        print(f"   Low (<40%):       {low:3d} ({low/len(successful)*100:5.1f}%)")
        
        # Top predicted classes
        class_counts = {}
        for r in successful:
            cls = r['top_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        print(f"\n🍜 Top Predicted Dishes:")
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        for cls, count in sorted_classes[:5]:
            print(f"   {cls:<25} {count:3d} images ({count/len(successful)*100:5.1f}%)")
    
    if failed:
        print(f"\n❌ Failed Images:")
        for r in failed:
            print(f"   {r['image']}: {r['error']}")
    
    print("="*70 + "\n")


def save_reports(results, output_dir='test_results'):
    """Save detailed reports to files"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON report
    json_file = os.path.join(output_dir, f'results_{timestamp}.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 JSON report saved: {json_file}")
    
    # Save CSV report
    csv_file = os.path.join(output_dir, f'results_{timestamp}.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Top Prediction', 'Confidence (%)', 'Status'])
        
        for r in results:
            if 'predictions' in r:
                writer.writerow([
                    r['image'],
                    r['top_class'],
                    f"{r['top_confidence']:.2f}",
                    'Success'
                ])
            else:
                writer.writerow([
                    r['image'],
                    'N/A',
                    'N/A',
                    f"Failed: {r.get('error', 'Unknown error')}"
                ])
    print(f"💾 CSV report saved: {csv_file}")
    
    # Save detailed text report
    txt_file = os.path.join(output_dir, f'results_{timestamp}.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("🍜 Thai Food Recognition - Batch Testing Results\n")
        f.write("="*70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Images: {len(results)}\n\n")
        
        for i, r in enumerate(results, 1):
            f.write(f"\n[{i}] {r['image']}\n")
            f.write("-" * 70 + "\n")
            
            if 'predictions' in r:
                f.write(f"Top Prediction: {r['top_class']}\n")
                f.write(f"Confidence: {r['top_confidence']:.2f}%\n")
                f.write(f"\nAll Predictions:\n")
                for j, pred in enumerate(r['predictions'], 1):
                    f.write(f"  {j}. {pred['class']:<25} {pred['confidence']:6.2f}%\n")
            else:
                f.write(f"Status: Failed\n")
                f.write(f"Error: {r.get('error', 'Unknown error')}\n")
    
    print(f"💾 Text report saved: {txt_file}")
    print(f"\n📂 All reports saved in: {output_dir}/\n")


# ================================
# Main Function
# ================================

def main():
    parser = argparse.ArgumentParser(description='Batch Test Thai Food Recognition Model')
    parser.add_argument('--folder', type=str, required=True,
                        help='Path to the folder containing images')
    parser.add_argument('--model', type=str, default='models/layer2_finetuned/model_final.pth',
                        help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top predictions per image (default: 3)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='Device to use (cpu, cuda, or mps)')
    parser.add_argument('--save_report', action='store_true',
                        help='Save detailed reports to files')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Directory to save reports (default: test_results)')
    
    args = parser.parse_args()
    
    # Check if folder exists
    if not os.path.exists(args.folder):
        print(f"❌ Error: Folder not found: {args.folder}")
        sys.exit(1)
    
    # Setup device
    device = args.device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU instead")
        device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, using CPU instead")
        device = 'cpu'
    
    print(f"🖥️  Device: {device.upper()}")
    print()
    
    # Load model
    model, classes, model_name = load_model(args.model, device)
    
    # Process all images
    results = batch_predict(model, args.folder, classes, model_name, device, args.top_k)
    
    # Generate summary
    generate_summary(results)
    
    # Save reports if requested
    if args.save_report:
        save_reports(results, args.output_dir)
    
    print("✨ Batch testing complete!\n")


if __name__ == "__main__":
    main()