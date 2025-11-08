"""
🖥️ Local Inference Script - Thai Food Recognition
Run บนเครื่องตัวเอง (CPU/GPU)

Requirements:
  pip install torch torchvision transformers pillow albumentations

Usage:
  python local_inference.py --image path/to/image.jpg
  python local_inference.py --folder path/to/folder
  python local_inference.py --interactive
"""

import torch
import torch.nn as nn
from transformers import CLIPModel
from PIL import Image
import numpy as np
import argparse
import os
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

# 20 Thai dishes
CLASSES = [
    'Som Tum', 'Tom Yum Goong', 'Larb', 'Pad Thai', 'Kaeng Khiao Wan',
    'Khao Soi', 'Kaeng Massaman', 'Pad Krapow', 'Khao Man Gai',
    'Khao Kha Mu', 'Tom Kha Gai', 'Gai Pad Med Ma Muang Himmaphan',
    'Kai Palo', 'Gung Ob Woon Sen', 'Khao Kluk Kapi', 'Por Pia Tod',
    'Hor Mok', 'Khao Niao Ma Muang', 'Khanom Krok', 'Foi Thong'
]

THAI_NAMES = {
    'Som Tum': 'ส้มตำ',
    'Tom Yum Goong': 'ต้มยำกุ้ง',
    'Larb': 'ลาบ',
    'Pad Thai': 'ผัดไทย',
    'Kaeng Khiao Wan': 'แกงเขียวหวาน',
    'Khao Soi': 'ข้าวซอย',
    'Kaeng Massaman': 'แกงมัสมั่น',
    'Pad Krapow': 'ผัดกะเพรา',
    'Khao Man Gai': 'ข้าวมันไก่',
    'Khao Kha Mu': 'ข้าวขาหมู',
    'Tom Kha Gai': 'ต้มข่าไก่',
    'Gai Pad Med Ma Muang Himmaphan': 'ไก่ผัดเม็ดมะม่วงหิมพานต์',
    'Kai Palo': 'ไข่พะโล้',
    'Gung Ob Woon Sen': 'กุ้งอบวุ้นเส้น',
    'Khao Kluk Kapi': 'ข้าวคลุกกะปิ',
    'Por Pia Tod': 'ปอเปี๊ยะทอด',
    'Hor Mok': 'ห่อหมก',
    'Khao Niao Ma Muang': 'ข้าวเหนียวมะม่วง',
    'Khanom Krok': 'ขนมครก',
    'Foi Thong': 'ฝอยทอง'
}

# Model paths
MODEL_PATH = 'models/layer2_finetuned/model_final10.pth'
CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'

# ============================================================================
# Model Definition
# ============================================================================

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(CLIPClassifier, self).__init__()
        self.clip = clip_model
        
        # Freeze CLIP encoder
        for param in self.clip.vision_model.parameters():
            param.requires_grad = False
        
        # Classification head (768-dim for ViT-B/32)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, pixel_values):
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.pooler_output
        logits = self.classifier(image_embeds)
        return logits

# ============================================================================
# Transform
# ============================================================================

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]),
    ToTensorV2()
])

# ============================================================================
# Load Model
# ============================================================================

def load_model(model_path, device):
    """Load trained model"""
    print(f"🔄 Loading model from: {model_path}")
    
    # Load CLIP base model
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    
    # Create classifier
    model = CLIPClassifier(clip_model, num_classes=len(CLASSES))
    
    # Load trained weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded successfully!")
        
        # Show model info
        if 'test_accuracy' in checkpoint:
            print(f"   Test Accuracy: {checkpoint['test_accuracy']:.2f}%")
    else:
        print(f"⚠️  Warning: Model file not found at {model_path}")
        print(f"   Using pre-trained CLIP without fine-tuning")
    
    model.to(device)
    model.eval()
    
    return model

# ============================================================================
# Inference Functions
# ============================================================================

def predict_image(model, image_path, device, top_k=5):
    """Predict single image"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Transform
    transformed = transform(image=image_np)['image']
    transformed = transformed.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(transformed)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k, dim=1)
    
    # Results
    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        class_name = CLASSES[idx]
        thai_name = THAI_NAMES.get(class_name, class_name)
        results.append({
            'class_en': class_name,
            'class_th': thai_name,
            'probability': prob.item() * 100
        })
    
    return results, image

# ============================================================================
# Display Functions
# ============================================================================

def display_results_cli(image_path, results):
    """Display results in terminal"""
    print("\n" + "="*70)
    print(f"📸 Image: {os.path.basename(image_path)}")
    print("="*70)
    
    for i, result in enumerate(results, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        bar = "█" * int(result['probability'] / 2)
        
        print(f"{emoji} {i}. {result['class_th']:15s} ({result['class_en']:30s})")
        print(f"     {result['probability']:6.2f}% {bar}")
    
    print("="*70)

def display_results_gui(image, results):
    """Display results with matplotlib"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Show image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Input Image', fontsize=14, fontweight='bold')
    
    # Show predictions
    labels = [f"{r['class_th']}\n({r['class_en']})" for r in results]
    probs = [r['probability'] for r in results]
    
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results))]
    bars = ax2.barh(labels, probs, color=colors)
    
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Top Predictions', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    
    # Add percentage labels
    for bar, prob in zip(bars, probs):
        ax2.text(prob + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# Batch Processing
# ============================================================================

def predict_folder(model, folder_path, device, show_gui=False):
    """Predict all images in folder"""
    image_exts = ['.jpg', '.jpeg', '.png', '.webp']
    image_files = []
    
    for ext in image_exts:
        image_files.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(folder_path, f'*{ext.upper()}')))
    
    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return
    
    print(f"\n🔍 Found {len(image_files)} images")
    print("="*70)
    
    results_summary = []
    
    for i, img_path in enumerate(image_files, 1):
        try:
            results, image = predict_image(model, img_path, device, top_k=3)
            
            # Display
            if show_gui:
                display_results_gui(image, results)
            else:
                print(f"{i:3d}. {os.path.basename(img_path):30s} → " +
                      f"{results[0]['class_th']:15s} ({results[0]['probability']:.1f}%)")
            
            results_summary.append({
                'file': os.path.basename(img_path),
                'prediction': results[0]['class_en'],
                'confidence': results[0]['probability']
            })
            
        except Exception as e:
            print(f"{i:3d}. {os.path.basename(img_path):30s} → ❌ Error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 Summary:")
    print("="*70)
    
    if results_summary:
        avg_conf = np.mean([r['confidence'] for r in results_summary])
        print(f"   Total images: {len(results_summary)}")
        print(f"   Average confidence: {avg_conf:.1f}%")
        
        high_conf = sum(1 for r in results_summary if r['confidence'] >= 80)
        print(f"   High confidence (≥80%): {high_conf}/{len(results_summary)}")

# ============================================================================
# Interactive Mode
# ============================================================================

def interactive_mode(model, device):
    """Interactive testing mode"""
    while True:
        print("\n" + "="*70)
        print("🎮 Interactive Mode")
        print("="*70)
        print("Options:")
        print("  [1] Test single image")
        print("  [2] Test folder")
        print("  [3] Show GUI (matplotlib)")
        print("  [0] Exit")
        print("="*70)
        
        choice = input("\nYour choice: ").strip()
        
        if choice == '1':
            img_path = input("\n📁 Enter image path: ").strip()
            if os.path.exists(img_path):
                results, image = predict_image(model, img_path, device)
                display_results_cli(img_path, results)
            else:
                print(f"❌ File not found: {img_path}")
        
        elif choice == '2':
            folder = input("\n📁 Enter folder path: ").strip()
            if os.path.exists(folder):
                predict_folder(model, folder, device)
            else:
                print(f"❌ Folder not found: {folder}")
        
        elif choice == '3':
            img_path = input("\n📁 Enter image path: ").strip()
            if os.path.exists(img_path):
                results, image = predict_image(model, img_path, device)
                display_results_gui(image, results)
            else:
                print(f"❌ File not found: {img_path}")
        
        elif choice == '0':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("\n⚠️  Invalid choice")

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Thai Food Recognition - Local Inference')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder with images')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to model')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--gui', action='store_true', help='Show GUI (matplotlib)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode')
    
    args = parser.parse_args()
    
    # Setup device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("🍜 Thai Food Recognition - Local Inference")
    print("="*70)
    print(f"Device: {device}")
    print(f"Classes: {len(CLASSES)}")
    print("="*70)
    
    # Load model
    model = load_model(args.model, device)
    
    # Run inference
    if args.interactive:
        interactive_mode(model, device)
    
    elif args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            return
        
        results, image = predict_image(model, args.image, device)
        
        if args.gui:
            display_results_gui(image, results)
        else:
            display_results_cli(args.image, results)
    
    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"❌ Folder not found: {args.folder}")
            return
        
        predict_folder(model, args.folder, device, show_gui=args.gui)
    
    else:
        print("\n⚠️  No input specified. Use --help for options")
        print("\n💡 Quick start:")
        print("   python local_inference.py --interactive")
        print("   python local_inference.py --image path/to/image.jpg")
        print("   python local_inference.py --folder path/to/folder")

if __name__ == "__main__":
    main()