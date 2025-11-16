"""
🍜 Thai Food Recognition - Hybrid 2-Layer System
ระบบจดจำอาหารไทยแบบ Hybrid 2 ชั้น

System Logic:
    1. Layer 1 (Pre-trained) ทำนายก่อน - เร็ว (0.5-1s)
    2. ถ้า confidence ≥ 80% → ใช้ผลจาก Layer 1
    3. ถ้า confidence < 80% → ส่งต่อ Layer 2 - แม่นยำ (1-2s)

Usage:
    python hybrid_prediction.py --image path/to/food.jpg
    python hybrid_prediction.py --image path/to/food.jpg --threshold 0.75
    python hybrid_prediction.py --image path/to/food.jpg --verbose
    python hybrid_prediction.py --folder test_images/ --save_report
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

# ================================
# Model Architectures
# ================================

class CLIPClassifierLayer2(nn.Module):
    """
    Layer 2: Fine-tuned Thai Food Specialist
    Supports both old (with BatchNorm) and new (without BatchNorm) architectures
    """
    def __init__(self, clip_model, num_classes, use_batchnorm=False):
        super(CLIPClassifierLayer2, self).__init__()
        self.clip = clip_model
        
        embedding_dim = clip_model.vision_model.config.hidden_size
        
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
# Hybrid Prediction System
# ================================

class HybridFoodRecognition:
    """
    Hybrid 2-Layer Food Recognition System
    
    Layer 1: Fast baseline (pre-trained)
    Layer 2: Thai food specialist (fine-tuned)
    """
    
    def __init__(
        self,
        layer1_model_name: str = "openai/clip-vit-base-patch32",
        layer2_model_path: str = "models/layer2_finetuned/model_final.pth",
        confidence_threshold: float = 0.80,
        device: str = None
    ):
        """
        Initialize hybrid system
        
        Args:
            layer1_model_name: HuggingFace model name for Layer 1
            layer2_model_path: Path to fine-tuned Layer 2 model
            confidence_threshold: Threshold to decide Layer 1 vs Layer 2 (default: 0.80)
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 Initializing Hybrid Food Recognition System")
        print(f"   Device: {self.device}")
        print(f"   Confidence Threshold: {confidence_threshold * 100}%")
        print()
        
        # Load Layer 1 (Pre-trained)
        self.layer1_model, self.layer1_processor = self._load_layer1(layer1_model_name)
        
        # Load Layer 2 (Fine-tuned)
        self.layer2_model, self.class_names = self._load_layer2(layer2_model_path)
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'layer1_used': 0,
            'layer2_used': 0,
            'layer1_time': 0.0,
            'layer2_time': 0.0
        }
        
        print("✅ Hybrid system ready!\n")
    
    def _load_layer1(self, model_name: str) -> Tuple[CLIPModel, CLIPProcessor]:
        """Load Layer 1 pre-trained model"""
        print(f"📥 Loading Layer 1 (Pre-trained)...")
        print(f"   Model: {model_name}")
        
        try:
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            model = model.to(self.device)
            model.eval()
            
            print(f"   ✅ Layer 1 loaded successfully")
            return model, processor
            
        except Exception as e:
            print(f"   ❌ Error loading Layer 1: {e}")
            sys.exit(1)
    
    def _load_layer2(self, model_path: str) -> Tuple[CLIPClassifierLayer2, list]:
        """Load Layer 2 fine-tuned model with auto-detection"""
        print(f"📥 Loading Layer 2 (Fine-tuned)...")
        
        # Try multiple possible paths
        possible_paths = [
            model_path,
            'models/layer2_finetuned/model_final.pth',
            '../models/layer2_finetuned/model_final.pth',
            'model_final.pth'
        ]
        
        actual_path = None
        for path in possible_paths:
            if os.path.exists(path):
                actual_path = path
                break
        
        if actual_path is None:
            print(f"   ❌ Model file not found!")
            print(f"   Searched paths:")
            for path in possible_paths:
                print(f"      - {path}")
            sys.exit(1)
        
        print(f"   Path: {actual_path}")
        
        try:
            # Load checkpoint
            # PyTorch 2.6+ requires weights_only=False for models with custom objects
            checkpoint = torch.load(actual_path, map_location=self.device, weights_only=False)
            
            # Extract class names (with fallback)
            # Try both 'class_names' and 'classes' keys
            class_names = checkpoint.get('class_names', None)
            if class_names is None:
                class_names = checkpoint.get('classes', None)
            
            # Fallback: Try to import from class_names.py
            if class_names is None or len(class_names) == 0:
                try:
                    from class_names import get_class_names
                    class_names = get_class_names()
                    print(f"   ⚠️  Using class names from class_names.py (checkpoint has no class_names)")
                except ImportError:
                    print(f"   ❌ Error: No class_names in checkpoint and class_names.py not found!")
                    print(f"   Please create class_names.py with your class names.")
                    sys.exit(1)
            
            num_classes = len(class_names)
            state_dict = checkpoint['model_state_dict']
            
            # Detect architecture - improved detection
            # Method 1: Look for BatchNorm in keys
            has_batchnorm_keys = any('BatchNorm' in key for key in state_dict.keys())
            
            # Method 2: Look for running_mean/running_var (BatchNorm specific)
            has_running_stats = any('running_mean' in key or 'running_var' in key for key in state_dict.keys())
            
            # Method 3: Check first layer size
            first_layer_key = 'classifier.0.weight'
            if first_layer_key in state_dict:
                first_layer_shape = state_dict[first_layer_key].shape
                # Old: 512 hidden, New: 256 hidden
                has_512_hidden = (first_layer_shape[0] == 512)
            else:
                has_512_hidden = False
            
            # Decision: any of these means old architecture
            has_batchnorm = has_batchnorm_keys or has_running_stats or has_512_hidden
            arch_type = "Old (with BatchNorm)" if has_batchnorm else "New (without BatchNorm)"
            
            # Detect CLIP variant
            patch_size = checkpoint.get('clip_config', {}).get('vision_config', {}).get('patch_size', 32)
            clip_variant = f"openai/clip-vit-base-patch{patch_size}"
            
            print(f"   🔍 Detected: {arch_type}")
            if has_512_hidden:
                print(f"      → 512 hidden units (Old architecture)")
            if has_running_stats:
                print(f"      → BatchNorm running stats found")
            print(f"   🔍 Detected: CLIP ViT-B/{patch_size}")
            print(f"   📊 Classes: {num_classes}")
            
            # Load base CLIP model
            print(f"   📥 Loading base CLIP model: {clip_variant}...")
            clip_model = CLIPModel.from_pretrained(clip_variant)
            
            # Create model with detected architecture
            model = CLIPClassifierLayer2(
                clip_model=clip_model,
                num_classes=num_classes,
                use_batchnorm=has_batchnorm
            )
            
            # Load weights
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            
            print(f"   ✅ Layer 2 loaded successfully")
            
            return model, class_names
            
        except Exception as e:
            print(f"   ❌ Error loading Layer 2: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _predict_layer1(self, image_path: str, top_k: int = 5) -> Dict:
        """Predict using Layer 1 (zero-shot with class names)"""
        start_time = time.time()
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare inputs
        inputs = self.layer1_processor(
            text=self.class_names,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.layer1_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.class_names)))
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'class': self.class_names[idx],
                'confidence': prob.item()
            })
        
        elapsed_time = time.time() - start_time
        
        return {
            'layer': 'Layer 1',
            'predictions': predictions,
            'top_prediction': predictions[0],
            'time': elapsed_time
        }
    
    def _predict_layer2(self, image_path: str, top_k: int = 5) -> Dict:
        """Predict using Layer 2 (fine-tuned)"""
        start_time = time.time()
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare inputs
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.layer2_model(pixel_values)
            probs = F.softmax(logits, dim=1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.class_names)))
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'class': self.class_names[idx],
                'confidence': prob.item()
            })
        
        elapsed_time = time.time() - start_time
        
        return {
            'layer': 'Layer 2',
            'predictions': predictions,
            'top_prediction': predictions[0],
            'time': elapsed_time
        }
    
    def predict(self, image_path: str, top_k: int = 5, verbose: bool = False) -> Dict:
        """
        Hybrid prediction with automatic layer selection
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            verbose: Print detailed info
            
        Returns:
            Dictionary with prediction results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🖼️  Image: {os.path.basename(image_path)}")
            print(f"{'='*60}\n")
        
        # Step 1: Try Layer 1 first
        if verbose:
            print("🔍 Step 1: Trying Layer 1 (Fast Baseline)...")
        
        layer1_result = self._predict_layer1(image_path, top_k)
        layer1_confidence = layer1_result['top_prediction']['confidence']
        
        if verbose:
            print(f"   Prediction: {layer1_result['top_prediction']['class']}")
            print(f"   Confidence: {layer1_confidence*100:.2f}%")
            print(f"   Time: {layer1_result['time']:.3f}s")
        
        # Step 2: Decide whether to use Layer 2
        if layer1_confidence >= self.confidence_threshold:
            # Use Layer 1 result (high confidence)
            if verbose:
                print(f"\n✅ High confidence (≥{self.confidence_threshold*100}%)!")
                print(f"   Using Layer 1 result")
            
            self.stats['layer1_used'] += 1
            self.stats['layer1_time'] += layer1_result['time']
            
            result = layer1_result
            result['decision'] = 'Layer 1 (high confidence)'
            
        else:
            # Use Layer 2 (low confidence, need specialist)
            if verbose:
                print(f"\n⚠️  Low confidence (<{self.confidence_threshold*100}%)")
                print(f"   Forwarding to Layer 2 (Thai Specialist)...\n")
            
            layer2_result = self._predict_layer2(image_path, top_k)
            layer2_confidence = layer2_result['top_prediction']['confidence']
            
            if verbose:
                print(f"🔍 Step 2: Layer 2 Prediction")
                print(f"   Prediction: {layer2_result['top_prediction']['class']}")
                print(f"   Confidence: {layer2_confidence*100:.2f}%")
                print(f"   Time: {layer2_result['time']:.3f}s")
            
            self.stats['layer2_used'] += 1
            self.stats['layer2_time'] += layer2_result['time']
            
            result = layer2_result
            result['decision'] = f'Layer 2 (Layer 1 confidence: {layer1_confidence*100:.2f}%)'
            result['layer1_result'] = layer1_result
        
        self.stats['total_predictions'] += 1
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🏆 FINAL RESULT")
            print(f"{'='*60}")
            print(f"Dish: {result['top_prediction']['class']}")
            print(f"Confidence: {result['top_prediction']['confidence']*100:.2f}%")
            print(f"Decision: {result['decision']}")
            print(f"Time: {result['time']:.3f}s")
            print(f"{'='*60}\n")
        
        return result
    
    def print_statistics(self):
        """Print system statistics"""
        if self.stats['total_predictions'] == 0:
            print("No predictions made yet.")
            return
        
        total = self.stats['total_predictions']
        layer1_pct = (self.stats['layer1_used'] / total) * 100
        layer2_pct = (self.stats['layer2_used'] / total) * 100
        
        avg_layer1_time = self.stats['layer1_time'] / self.stats['layer1_used'] if self.stats['layer1_used'] > 0 else 0
        avg_layer2_time = self.stats['layer2_time'] / self.stats['layer2_used'] if self.stats['layer2_used'] > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"📊 HYBRID SYSTEM STATISTICS")
        print(f"{'='*60}")
        print(f"Total Predictions: {total}")
        print(f"")
        print(f"Layer 1 Usage: {self.stats['layer1_used']} ({layer1_pct:.1f}%)")
        print(f"  Avg Time: {avg_layer1_time:.3f}s")
        print(f"")
        print(f"Layer 2 Usage: {self.stats['layer2_used']} ({layer2_pct:.1f}%)")
        print(f"  Avg Time: {avg_layer2_time:.3f}s")
        print(f"")
        print(f"Expected Performance:")
        print(f"  • {layer1_pct:.1f}% queries: Fast (~{avg_layer1_time:.1f}s)")
        print(f"  • {layer2_pct:.1f}% queries: Accurate (~{avg_layer2_time:.1f}s)")
        print(f"{'='*60}\n")


# ================================
# Main Function
# ================================

def main():
    parser = argparse.ArgumentParser(
        description='🍜 Thai Food Recognition - Hybrid 2-Layer System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python hybrid_prediction.py --image food.jpg
  
  # With custom threshold
  python hybrid_prediction.py --image food.jpg --threshold 0.75
  
  # Verbose mode
  python hybrid_prediction.py --image food.jpg --verbose
  
  # Batch processing
  python hybrid_prediction.py --folder test_images/ --save_report
        """
    )
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to single image')
    group.add_argument('--folder', type=str, help='Path to folder containing images')
    
    # System options
    parser.add_argument('--threshold', type=float, default=0.80,
                       help='Confidence threshold for Layer 1 (default: 0.80)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions (default: 5)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information')
    parser.add_argument('--save_report', action='store_true',
                       help='Save prediction report (for batch mode)')
    
    args = parser.parse_args()
    
    # Initialize hybrid system
    system = HybridFoodRecognition(
        confidence_threshold=args.threshold,
        device=args.device
    )
    
    # Single image mode
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Error: Image not found: {args.image}")
            sys.exit(1)
        
        result = system.predict(args.image, top_k=args.top_k, verbose=True)
        
    # Batch mode
    else:
        if not os.path.exists(args.folder):
            print(f"❌ Error: Folder not found: {args.folder}")
            sys.exit(1)
        
        # Get all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [
            os.path.join(args.folder, f)
            for f in os.listdir(args.folder)
            if os.path.splitext(f.lower())[1] in image_extensions
        ]
        
        if not image_files:
            print(f"❌ No images found in folder: {args.folder}")
            sys.exit(1)
        
        print(f"\n📂 Processing {len(image_files)} images from: {args.folder}\n")
        
        results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
            result = system.predict(image_path, top_k=args.top_k, verbose=args.verbose)
            results.append({
                'image': os.path.basename(image_path),
                'prediction': result['top_prediction']['class'],
                'confidence': result['top_prediction']['confidence'],
                'layer_used': result['layer'],
                'time': result['time']
            })
            
            if not args.verbose:
                print(f"   → {result['top_prediction']['class']} ({result['top_prediction']['confidence']*100:.1f}%) via {result['layer']}")
        
        # Print statistics
        system.print_statistics()
        
        # Save report
        if args.save_report:
            report_path = 'hybrid_prediction_report.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'folder': args.folder,
                    'total_images': len(image_files),
                    'threshold': args.threshold,
                    'statistics': system.stats,
                    'results': results
                }, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Report saved: {report_path}")


if __name__ == '__main__':
    main()
