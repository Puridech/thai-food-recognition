"""
🍜 Thai Food Recognition - Single Image Testing
ทดสอบ Layer 2 Model แบบทีละรูป

Usage:
    python test_single_image.py --image path/to/food.jpg
    python test_single_image.py --image path/to/food.jpg --top_k 5
    python test_single_image.py --image D:\\test\\pad_thai.jpg --model path/to/model_final.pth
    python version2_single_test_layer2.py --image D:\images_test\images_external\foithong.jpg --device cuda
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import argparse
import os
import sys
from pathlib import Path

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
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Use CLIP processor
        processor = CLIPProcessor.from_pretrained(model_name)
        inputs = processor(images=image, return_tensors="pt")
        
        return inputs['pixel_values']
    
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        sys.exit(1)


# ================================
# Model Loading
# ================================

def load_model(model_path, device='cpu'):
    """Load the fine-tuned Layer 2 model with auto-detection"""
    
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
        print("   - models/layer2_finetuned/model_final.pth")
        print("\n💡 Or specify the path explicitly:")
        print("   python test_single_image.py --image food.jpg --model path/to/model_final.pth")
        sys.exit(1)
    
    print(f"📦 Loading model from: {actual_path}")
    
    # Load checkpoint
    checkpoint = torch.load(actual_path, map_location=device, weights_only=False)
    
    # Get classes
    classes = checkpoint['classes']
    num_classes = len(classes)
    
    # Auto-detect CLIP model variant and architecture
    model_name = checkpoint.get('model_name', None)
    use_batchnorm = False
    
    # Check if checkpoint has BatchNorm layers (old architecture)
    if 'model_state_dict' in checkpoint:
        state_dict_keys = checkpoint['model_state_dict'].keys()
        
        # Detect BatchNorm
        has_batchnorm = any('BatchNorm' in str(key) or 'running_mean' in str(key) 
                           for key in state_dict_keys)
        if has_batchnorm:
            use_batchnorm = True
            print(f"🔍 Detected: Old architecture (with BatchNorm)")
        else:
            print(f"🔍 Detected: New architecture (without BatchNorm)")
        
        # Detect CLIP variant from embedding dimensions
        if 'clip.vision_model.embeddings.patch_embedding.weight' in state_dict_keys:
            patch_weight = checkpoint['model_state_dict']['clip.vision_model.embeddings.patch_embedding.weight']
            patch_size = patch_weight.shape[2]  # [channels, 3, patch_size, patch_size]
            
            if patch_size == 32:
                model_name = "openai/clip-vit-base-patch32"
                print(f"🔍 Detected: CLIP ViT-B/32 (patch_size={patch_size})")
            elif patch_size == 16:
                model_name = "openai/clip-vit-base-patch16"
                print(f"🔍 Detected: CLIP ViT-B/16 (patch_size={patch_size})")
        
        # Fallback: detect from classifier input dimension
        if model_name is None and 'classifier.0.weight' in state_dict_keys:
            first_layer_weight = checkpoint['model_state_dict']['classifier.0.weight']
            input_dim = first_layer_weight.shape[1]
            
            if input_dim == 768:
                model_name = "openai/clip-vit-base-patch16"
                print(f"🔍 Auto-detected: CLIP ViT-B/16 (input_dim={input_dim})")
            elif input_dim == 512:
                model_name = "openai/clip-vit-base-patch32"
                print(f"🔍 Auto-detected: CLIP ViT-B/32 (input_dim={input_dim})")
    
    # Default to ViT-B/32 if couldn't detect
    if model_name is None:
        model_name = "openai/clip-vit-base-patch32"
        print("⚠️  Could not detect model variant, defaulting to ViT-B/32")
    
    # Load base CLIP model
    print(f"📥 Loading base CLIP model: {model_name}...")
    clip_model = CLIPModel.from_pretrained(model_name)
    
    # Create classifier with correct architecture
    model = CLIPClassifier(clip_model, num_classes, use_batchnorm=use_batchnorm)
    
    # Load trained weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded successfully!")
    except RuntimeError as e:
        print(f"❌ Error loading model weights: {e}")
        print(f"\n💡 Debug info:")
        print(f"   Model variant: {model_name}")
        print(f"   Architecture: {'Old (with BatchNorm)' if use_batchnorm else 'New (without BatchNorm)'}")
        print(f"   Classes: {num_classes}")
        sys.exit(1)
    
    model.to(device)
    model.eval()
    
    print(f"📊 Classes: {num_classes}")
    print(f"📊 Architecture: {'Old (with BatchNorm)' if use_batchnorm else 'New (without BatchNorm)'}")
    print(f"📊 Class names: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}\n")
    
    return model, classes, model_name


# ================================
# Prediction
# ================================

def predict(model, image_tensor, classes, device='cpu', top_k=5):
    """Make prediction on a single image"""
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

def display_results(image_path, predictions):
    """Display prediction results"""
    print("\n" + "="*70)
    print("🍜 PREDICTION RESULTS")
    print("="*70)
    print(f"📷 Image: {os.path.basename(image_path)}")
    
    try:
        img = Image.open(image_path)
        print(f"📏 Size: {img.size[0]}x{img.size[1]} pixels")
    except:
        pass
    
    print("-"*70)
    
    # Top prediction
    top_pred = predictions[0]
    print(f"\n🏆 TOP PREDICTION:")
    print(f"   Dish: {top_pred['class']}")
    print(f"   Confidence: {top_pred['confidence']:.2f}%")
    
    # Confidence bar
    bar_length = int(top_pred['confidence'] / 2)
    bar = "█" * bar_length + "░" * (50 - bar_length)
    print(f"   [{bar}]")
    
    # All predictions
    print(f"\n📊 TOP {len(predictions)} PREDICTIONS:")
    for i, pred in enumerate(predictions, 1):
        bar_length = int(pred['confidence'] / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"   {i}. {pred['class']:<25} {pred['confidence']:6.2f}% [{bar}]")
    
    # Confidence interpretation
    print(f"\n💡 CONFIDENCE INTERPRETATION:")
    top_conf = top_pred['confidence']
    if top_conf >= 80:
        print(f"   ✅ Very High ({top_conf:.1f}%) - Model is very confident")
    elif top_conf >= 60:
        print(f"   ✓  High ({top_conf:.1f}%) - Model is confident")
    elif top_conf >= 40:
        print(f"   ⚠️  Medium ({top_conf:.1f}%) - Model has some uncertainty")
    else:
        print(f"   ❌ Low ({top_conf:.1f}%) - Model is uncertain")
    
    # Hybrid recommendation
    print(f"\n🔄 HYBRID SYSTEM RECOMMENDATION:")
    if top_conf >= 80:
        print(f"   Use Layer 2 result directly")
    elif top_conf >= 60:
        print(f"   Use Layer 2, but consider Layer 1 validation")
    else:
        print(f"   Compare with Layer 1 (pre-trained) for better results")
    
    print("\n" + "="*70 + "\n")


# ================================
# Main Function
# ================================

def main():
    parser = argparse.ArgumentParser(description='Test Thai Food Recognition Model (Single Image)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image file')
    parser.add_argument('--model', type=str, default='models/layer2_finetuned/model_final.pth',
                        help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to show (default: 5)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='Device to use (cpu, cuda, or mps)')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Error: Image not found: {args.image}")
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
    
    # Preprocess image
    print(f"🖼️  Processing image: {os.path.basename(args.image)}")
    image_tensor = preprocess_image(args.image, model_name)
    
    # Make prediction
    print("🔮 Making prediction...")
    predictions = predict(model, image_tensor, classes, device, args.top_k)
    
    # Display results
    display_results(args.image, predictions)
    
    print("✨ Done!\n")


if __name__ == "__main__":
    main()