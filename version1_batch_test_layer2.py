"""
🍜 Thai Food Recognition - Batch Testing Script (FINAL FIXED VERSION)
Test multiple images at once and generate comprehensive reports

FIXED:
- Auto-detects model path from multiple locations
- Auto-detects CLIP variant (ViT-B/16 or ViT-B/32) from checkpoint
- Handles architecture mismatch

Usage:
    python batch_test.py --folder test_images/
    python batch_test.py --folder D:\\images_test\\images_external\\
    python batch_test.py --folder test_images/ --save_report
    python batch_test.py --folder D:\images_test\images_external
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
    def __init__(self, clip_model, num_classes):
        super(CLIPClassifier, self).__init__()
        self.clip = clip_model
        
        # Get embedding dimension from CLIP model
        embedding_dim = clip_model.vision_model.config.hidden_size
        
        # Classification head
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
    
    # Auto-detect CLIP variant
    model_name = checkpoint.get('model_name', None)
    if model_name is None and 'model_state_dict' in checkpoint:
        if 'classifier.0.weight' in checkpoint['model_state_dict']:
            first_layer_weight = checkpoint['model_state_dict']['classifier.0.weight']
            embedding_dim = first_layer_weight.shape[1]
            
            if embedding_dim == 768:
                model_name = "openai/clip-vit-base-patch16"
                print(f"🔍 Auto-detected: CLIP ViT-B/16")
            else:
                model_name = "openai/clip-vit-base-patch32"
                print(f"🔍 Auto-detected: CLIP ViT-B/32")
    
    print(f"📥 Loading base CLIP model: {model_name}...")
    clip_model = CLIPModel.from_pretrained(model_name)
    
    model = CLIPClassifier(clip_model, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded! Classes: {num_classes}\n")
    
    return model, classes, model_name


# ================================
# Prediction
# ================================

def predict(model, image_tensor, classes, device='cpu', top_k=3):
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
            
            image_tensor = preprocess_image(str(image_path), model_name)
            predictions = predict(model, image_tensor, classes, device, top_k)
            
            result = {
                'filename': image_path.name,
                'path': str(image_path),
                'predictions': predictions,
                'top_prediction': predictions[0]['class'],
                'top_confidence': predictions[0]['confidence']
            }
            results.append(result)
            
            conf_icon = "✅" if predictions[0]['confidence'] >= 80 else "⚠️" if predictions[0]['confidence'] >= 60 else "🔴"
            print(f"{conf_icon} {predictions[0]['class']:<20} ({predictions[0]['confidence']:5.1f}%)")
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            results.append({
                'filename': image_path.name,
                'path': str(image_path),
                'error': str(e)
            })
    
    return results


# ================================
# Display Summary
# ================================

def display_summary(results):
    print("\n" + "="*80)
    print("📊 BATCH TEST SUMMARY")
    print("="*80)
    
    successful = [r for r in results if 'predictions' in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    if failed:
        print(f"❌ Failed: {len(failed)}")
    
    if not successful:
        return
    
    # Confidence stats
    confidences = [r['top_confidence'] for r in successful]
    print(f"\n📈 Confidence Statistics:")
    print(f"   Average: {sum(confidences)/len(confidences):.2f}%")
    print(f"   Maximum: {max(confidences):.2f}%")
    print(f"   Minimum: {min(confidences):.2f}%")
    
    # Distribution
    high = len([c for c in confidences if c >= 80])
    med = len([c for c in confidences if 60 <= c < 80])
    low = len([c for c in confidences if c < 60])
    
    print(f"\n🎯 Confidence Distribution:")
    print(f"   High (≥80%):     {high:3d} ({high/len(successful)*100:5.1f}%)")
    print(f"   Medium (60-80%): {med:3d} ({med/len(successful)*100:5.1f}%)")
    print(f"   Low (<60%):      {low:3d} ({low/len(successful)*100:5.1f}%)")
    
    # Class distribution
    from collections import Counter
    class_counts = Counter([r['top_prediction'] for r in successful])
    
    print(f"\n🍜 Predicted Classes:")
    for cls, count in class_counts.most_common():
        bar = "█" * (count * 40 // max(class_counts.values()))
        print(f"   {cls:<25} {count:3d} {bar}")
    
    print("\n" + "="*80)


# ================================
# Display Details
# ================================

def display_detailed_results(results, limit=20):
    print("\n" + "="*100)
    print("📋 DETAILED RESULTS (sorted by confidence)")
    print("="*100)
    
    successful = [r for r in results if 'predictions' in r]
    successful.sort(key=lambda x: x['top_confidence'], reverse=True)
    
    display_results = successful[:limit] if limit > 0 else successful
    
    print(f"{'#':<5} {'Filename':<35} {'Prediction':<20} {'Conf.':<10} {'2nd':<15} {'3rd':<15}")
    print("-"*100)
    
    for i, result in enumerate(display_results, 1):
        filename = result['filename'][:33] + '..' if len(result['filename']) > 35 else result['filename']
        pred = result['top_prediction'][:18] if len(result['top_prediction']) > 20 else result['top_prediction']
        conf = f"{result['top_confidence']:.1f}%"
        
        pred_2 = f"{result['predictions'][1]['class'][:10]}({result['predictions'][1]['confidence']:.0f}%)" if len(result['predictions']) > 1 else "-"
        pred_3 = f"{result['predictions'][2]['class'][:10]}({result['predictions'][2]['confidence']:.0f}%)" if len(result['predictions']) > 2 else "-"
        
        print(f"{i:<5} {filename:<35} {pred:<20} {conf:<10} {pred_2:<15} {pred_3:<15}")
    
    if limit > 0 and len(successful) > limit:
        print(f"\n... and {len(successful) - limit} more results")
    
    print("="*100)


# ================================
# Save Reports
# ================================

def save_reports(results, output_prefix='batch_test_report'):
    # JSON
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_images': len(results),
        'successful': len([r for r in results if 'predictions' in r]),
        'failed': len([r for r in results if 'error' in r]),
        'results': results
    }
    
    json_file = f"{output_prefix}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n💾 JSON report: {json_file}")
    
    # CSV
    csv_file = f"{output_prefix}.csv"
    successful = [r for r in results if 'predictions' in r]
    
    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Top Prediction', 'Confidence (%)', 
                        '2nd Prediction', '2nd Conf (%)', '3rd Prediction', '3rd Conf (%)'])
        
        for r in successful:
            row = [r['filename'], r['top_prediction'], f"{r['top_confidence']:.2f}"]
            for i in range(1, min(3, len(r['predictions']))):
                row.extend([r['predictions'][i]['class'], f"{r['predictions'][i]['confidence']:.2f}"])
            while len(row) < 7:
                row.extend(['', ''])
            writer.writerow(row)
    
    print(f"💾 CSV report: {csv_file}")


# ================================
# Main
# ================================

def main():
    parser = argparse.ArgumentParser(
        description='Batch test Layer 2 model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_test.py --folder test_images/
  python batch_test.py --folder D:\\images_test\\images_external\\
  python batch_test.py --folder photos/ --save_report --limit 50
        """
    )
    
    parser.add_argument('--folder', type=str, required=True, help='Folder with images')
    parser.add_argument('--model', type=str, default='models/layer2_finetuned/model_final.pth')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--save_report', action='store_true')
    parser.add_argument('--limit', type=int, default=20, help='Display limit (0 for all)')
    parser.add_argument('--output', type=str, default='batch_test_report')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🍜 Thai Food Recognition - Batch Testing")
    print("="*80 + "\n")
    
    if not os.path.exists(args.folder):
        print(f"❌ Folder not found: {args.folder}")
        sys.exit(1)
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    print(f"🖥️  Device: {device.upper()}")
    
    # Load model
    model, classes, model_name = load_model(args.model, device)
    
    # Batch predict
    results = batch_predict(model, args.folder, classes, model_name, device, args.top_k)
    
    # Display results
    display_summary(results)
    display_detailed_results(results, args.limit)
    
    # Save reports
    if args.save_report:
        save_reports(results, args.output)
    
    print("\n✨ Batch testing completed!\n")


if __name__ == "__main__":
    main()