"""
🍜 Thai Food Recognition - Layer 2 Model Testing Script (FINAL FIXED VERSION)
Test your fine-tuned CLIP model with images outside the dataset!

FIXED:
- Auto-detects model path from multiple locations
- Auto-detects CLIP variant (ViT-B/16 or ViT-B/32) from checkpoint
- Handles architecture mismatch

Usage:
    python test_model.py --image path/to/food.jpg
    python test_model.py --image path/to/food.jpg --top_k 3
    python test_layer2.py --image D:\images_test\images_external\foithong.jpg
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import argparse
import os
import sys
import numpy as np
from pathlib import Path

# ================================
# Model Architecture (Same as Training)
# ================================

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(CLIPClassifier, self).__init__()
        self.clip = clip_model
        
        # Get embedding dimension from CLIP model
        embedding_dim = clip_model.vision_model.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),  # Auto-adjust based on CLIP variant
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
    """
    Load and preprocess image for CLIP model
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Use CLIP processor (match the model variant)
        processor = CLIPProcessor.from_pretrained(model_name)
        inputs = processor(images=image, return_tensors="pt")
        
        return inputs['pixel_values']
    
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        sys.exit(1)


# ================================
# Model Loading (FINAL FIXED VERSION)
# ================================

def load_model(model_path, device='cpu'):
    """
    Load the fine-tuned Layer 2 model
    Auto-detects:
    1. Model file from multiple locations
    2. CLIP variant (ViT-B/16 or ViT-B/32) from checkpoint
    """
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
        print(f"❌ Model file not found in any of these locations:")
        for path in possible_paths:
            print(f"   - {path}")
        print("\n💡 Please download your model from Google Colab:")
        print("   1. models/layer2_finetuned/model_final.pth")
        print("   2. Place it in one of the locations above")
        print("\n💡 Or specify the path explicitly:")
        print("   python test_model.py --image food.jpg --model path/to/model_final.pth")
        sys.exit(1)
    
    print(f"📦 Loading model from: {actual_path}")
    
    # Load checkpoint
    checkpoint = torch.load(actual_path, map_location=device, weights_only=False)
    
    # Get classes
    classes = checkpoint['classes']
    num_classes = len(classes)
    
    # Auto-detect CLIP model variant from checkpoint
    model_name = checkpoint.get('model_name', None)
    
    # If model_name not in checkpoint, detect from weight shape
    if model_name is None and 'model_state_dict' in checkpoint:
        # Check the shape of first classifier layer
        if 'classifier.0.weight' in checkpoint['model_state_dict']:
            first_layer_weight = checkpoint['model_state_dict']['classifier.0.weight']
            embedding_dim = first_layer_weight.shape[1]  # 512 for patch32, 768 for patch16
            
            if embedding_dim == 768:
                model_name = "openai/clip-vit-base-patch16"
                print(f"🔍 Auto-detected: CLIP ViT-B/16 (embedding_dim={embedding_dim})")
            elif embedding_dim == 512:
                model_name = "openai/clip-vit-base-patch32"
                print(f"🔍 Auto-detected: CLIP ViT-B/32 (embedding_dim={embedding_dim})")
            else:
                print(f"⚠️  Unknown embedding dimension: {embedding_dim}, defaulting to ViT-B/32")
                model_name = "openai/clip-vit-base-patch32"
        else:
            print("⚠️  Could not detect model variant, defaulting to ViT-B/32")
            model_name = "openai/clip-vit-base-patch32"
    
    # Load base CLIP model
    print(f"📥 Loading base CLIP model: {model_name}...")
    clip_model = CLIPModel.from_pretrained(model_name)
    
    # Create classifier (will auto-adjust to correct embedding dim)
    model = CLIPClassifier(clip_model, num_classes)
    
    # Load trained weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded successfully!")
    except RuntimeError as e:
        print(f"❌ Error loading model weights: {e}")
        print(f"\n💡 This usually means the model was trained with a different CLIP variant.")
        print(f"   Current model: {model_name}")
        sys.exit(1)
    
    model.to(device)
    model.eval()
    
    print(f"📊 Classes: {num_classes}")
    print(f"📊 Class names: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}")
    
    # Show test accuracy if available
    if 'test_accuracy' in checkpoint:
        print(f"🎯 Test Accuracy: {checkpoint['test_accuracy']:.2f}%\n")
    else:
        print()
    
    return model, classes, model_name


# ================================
# Prediction
# ================================

def predict(model, image_tensor, classes, device='cpu', top_k=5):
    """
    Make prediction on a single image
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        
        # Get probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        
        # Get top-k predictions
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
# Display Results
# ================================

def display_results(image_path, predictions, show_image=True):
    """
    Display prediction results
    """
    print("\n" + "="*60)
    print("🍜 PREDICTION RESULTS")
    print("="*60)
    print(f"📷 Image: {os.path.basename(image_path)}")
    
    try:
        img_size = Image.open(image_path).size
        print(f"📏 Size: {img_size}")
    except:
        pass
    
    print("-"*60)
    
    # Top prediction
    top_pred = predictions[0]
    print(f"\n🏆 TOP PREDICTION:")
    print(f"   Class: {top_pred['class']}")
    print(f"   Confidence: {top_pred['confidence']:.2f}%")
    
    # All predictions
    print(f"\n📊 TOP {len(predictions)} PREDICTIONS:")
    for i, pred in enumerate(predictions, 1):
        bar_length = int(pred['confidence'] / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"   {i}. {pred['class']:<20} {pred['confidence']:>6.2f}% {bar}")
    
    print("\n" + "="*60)
    
    # Show image (optional)
    if show_image:
        try:
            import matplotlib.pyplot as plt
            
            img = Image.open(image_path)
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Predicted: {top_pred['class']} ({top_pred['confidence']:.1f}%)", 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        except ImportError:
            pass
        except Exception as e:
            print(f"⚠️  Could not display image: {e}")


# ================================
# Confidence Interpretation
# ================================

def interpret_confidence(confidence):
    """
    Provide interpretation of confidence score
    """
    if confidence >= 90:
        return "🟢 Very High - Model is very confident"
    elif confidence >= 75:
        return "🟡 High - Model is confident"
    elif confidence >= 60:
        return "🟠 Medium - Model has moderate confidence"
    else:
        return "🔴 Low - Model is uncertain, consider using Layer 1"


# ================================
# Main Function
# ================================

def main():
    parser = argparse.ArgumentParser(
        description='Test Layer 2 Thai Food Recognition Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_model.py --image test_images/pad_thai.jpg
  python test_model.py --image somtum.jpg --top_k 3
  python test_model.py --image food.jpg --model path/to/model.pth
        """
    )
    
    parser.add_argument('--image', type=str, required=True,
                       help='Path to the food image')
    parser.add_argument('--model', type=str, default='models/layer2_finetuned/model_final.pth',
                       help='Path to the model file (will auto-detect if not found)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to show (default: 5)')
    parser.add_argument('--no_display', action='store_true',
                       help='Do not display image (useful for batch processing)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use (cpu/cuda/mps)')
    
    args = parser.parse_args()
    
    # Header
    print("\n" + "="*60)
    print("🍜 Thai Food Recognition - Layer 2 Model Testing")
    print("="*60 + "\n")
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)
    
    # Determine device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU instead")
        device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, using CPU instead")
        device = 'cpu'
    else:
        device = args.device
    
    print(f"🖥️  Device: {device.upper()}")
    
    # Load model (auto-detects CLIP variant)
    model, classes, model_name = load_model(args.model, device)
    
    # Preprocess image (use correct CLIP variant)
    print(f"🖼️  Processing image: {os.path.basename(args.image)}")
    image_tensor = preprocess_image(args.image, model_name)
    
    # Make prediction
    print("🔮 Making prediction...")
    predictions = predict(model, image_tensor, classes, device, args.top_k)
    
    # Display results
    display_results(args.image, predictions, show_image=not args.no_display)
    
    # Confidence interpretation
    top_confidence = predictions[0]['confidence']
    interpretation = interpret_confidence(top_confidence)
    print(f"\n💡 Confidence Interpretation: {interpretation}")
    
    # Hybrid recommendation
    print("\n🔄 Hybrid System Recommendation:")
    if top_confidence >= 80:
        print("   ✅ Use Layer 2 result directly (High confidence)")
    else:
        print("   ⚠️  Consider comparing with Layer 1 (Pre-trained) for validation")
    
    print("\n✨ Done!\n")


if __name__ == "__main__":
    main()