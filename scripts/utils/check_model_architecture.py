"""
🔍 Model Architecture Checker
เครื่องมือตรวจสอบโครงสร้าง model Layer 1 และ Layer 2

Usage:
    python check_model_architecture.py
    python check_model_architecture.py --layer2 path/to/model.pth
    python check_model_architecture.py --detailed
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import argparse
import os
import sys
from pathlib import Path

# ================================
# Architecture Checker
# ================================

class ModelArchitectureChecker:
    """ตรวจสอบโครงสร้าง model และ compatibility"""
    
    def __init__(self):
        self.results = {
            'layer1': None,
            'layer2': None,
            'compatibility': None
        }
    
    def check_layer1(self, model_name: str = "openai/clip-vit-base-patch32") -> dict:
        """ตรวจสอบ Layer 1 (Pre-trained CLIP)"""
        print(f"\n{'='*70}")
        print(f"🔍 LAYER 1 ARCHITECTURE CHECK")
        print(f"{'='*70}\n")
        
        print(f"📥 Loading model: {model_name}...")
        
        try:
            # Load model and processor
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)
            
            # Get configuration
            config = model.config
            vision_config = config.vision_config
            text_config = config.text_config
            
            # Extract info
            info = {
                'model_name': model_name,
                'model_type': config.model_type,
                'vision_model': {
                    'hidden_size': vision_config.hidden_size,
                    'image_size': vision_config.image_size,
                    'patch_size': vision_config.patch_size,
                    'num_hidden_layers': vision_config.num_hidden_layers,
                    'num_attention_heads': vision_config.num_attention_heads,
                    'intermediate_size': vision_config.intermediate_size
                },
                'text_model': {
                    'hidden_size': text_config.hidden_size,
                    'vocab_size': text_config.vocab_size,
                    'max_position_embeddings': text_config.max_position_embeddings,
                    'num_hidden_layers': text_config.num_hidden_layers,
                    'num_attention_heads': text_config.num_attention_heads
                },
                'projection_dim': config.projection_dim
            }
            
            # Print info
            print("✅ Model loaded successfully!\n")
            print(f"📊 Vision Model Configuration:")
            print(f"   Image Size: {info['vision_model']['image_size']}x{info['vision_model']['image_size']}")
            print(f"   Patch Size: {info['vision_model']['patch_size']}x{info['vision_model']['patch_size']}")
            print(f"   Hidden Size: {info['vision_model']['hidden_size']}")
            print(f"   Layers: {info['vision_model']['num_hidden_layers']}")
            print(f"   Attention Heads: {info['vision_model']['num_attention_heads']}")
            print()
            print(f"📊 Text Model Configuration:")
            print(f"   Vocab Size: {info['text_model']['vocab_size']:,}")
            print(f"   Hidden Size: {info['text_model']['hidden_size']}")
            print(f"   Max Length: {info['text_model']['max_position_embeddings']}")
            print(f"   Layers: {info['text_model']['num_hidden_layers']}")
            print()
            print(f"📊 Projection Dimension: {info['projection_dim']}")
            
            self.results['layer1'] = info
            return info
            
        except Exception as e:
            print(f"❌ Error loading Layer 1: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def check_layer2(self, model_path: str = "models/layer2_finetuned/model_final.pth") -> dict:
        """ตรวจสอบ Layer 2 (Fine-tuned model)"""
        print(f"\n{'='*70}")
        print(f"🔍 LAYER 2 ARCHITECTURE CHECK")
        print(f"{'='*70}\n")
        
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
            print(f"❌ Model file not found!")
            print(f"\nSearched paths:")
            for path in possible_paths:
                print(f"   - {path}")
            return None
        
        print(f"📁 Model Path: {actual_path}")
        print(f"📏 File Size: {os.path.getsize(actual_path) / (1024**2):.2f} MB\n")
        
        try:
            # Load checkpoint
            print("📥 Loading checkpoint...")
            # PyTorch 2.6+ requires weights_only=False for models with custom objects
            checkpoint = torch.load(actual_path, map_location='cpu', weights_only=False)
            
            # Extract class names (with fallback)
            # Try both 'class_names' and 'classes' keys
            class_names = checkpoint.get('class_names', None)
            if class_names is None:
                class_names = checkpoint.get('classes', [])
            
            # Fallback: Try to import from class_names.py if no classes found
            if len(class_names) == 0:
                try:
                    import sys
                    sys.path.insert(0, '.')
                    from class_names import get_class_names
                    class_names = get_class_names()
                    print(f"   ⚠️  Using class names from class_names.py")
                except ImportError:
                    print(f"   ⚠️  No class_names found (neither in checkpoint nor class_names.py)")
            
            # Extract info
            info = {
                'path': actual_path,
                'file_size_mb': os.path.getsize(actual_path) / (1024**2),
                'checkpoint_keys': list(checkpoint.keys()),
                'class_names': class_names,
                'num_classes': len(class_names),
                'state_dict_keys': list(checkpoint['model_state_dict'].keys()) if 'model_state_dict' in checkpoint else [],
                'training_info': {},
                'clip_config': checkpoint.get('clip_config', {}),
                'architecture_info': {}
            }
            
            # Check architecture type - improved detection
            state_dict = checkpoint.get('model_state_dict', {})
            
            # Method 1: BatchNorm in keys
            has_batchnorm_keys = any('BatchNorm' in key or 'bn' in key for key in state_dict.keys())
            
            # Method 2: running_mean/running_var
            has_running_stats = any('running_mean' in key or 'running_var' in key for key in state_dict.keys())
            
            # Method 3: Check first layer size
            first_layer_key = 'classifier.0.weight'
            if first_layer_key in state_dict:
                first_layer_shape = state_dict[first_layer_key].shape
                has_512_hidden = (first_layer_shape[0] == 512)
            else:
                has_512_hidden = False
            
            # Final decision
            has_batchnorm = has_batchnorm_keys or has_running_stats or has_512_hidden
            
            # Count parameters
            total_params = sum(p.numel() for p in state_dict.values())
            
            # Detect CLIP variant
            clip_config = checkpoint.get('clip_config', {})
            vision_config = clip_config.get('vision_config', {})
            patch_size = vision_config.get('patch_size', 32)
            hidden_size = vision_config.get('hidden_size', 768)
            
            # Architecture details
            info['architecture_info'] = {
                'has_batchnorm': has_batchnorm,
                'architecture_type': 'Old (with BatchNorm)' if has_batchnorm else 'New (without BatchNorm)',
                'total_parameters': total_params,
                'clip_variant': f'ViT-B/{patch_size}',
                'embedding_dim': hidden_size,
                'patch_size': patch_size
            }
            
            # Training info
            if 'epoch' in checkpoint:
                info['training_info']['last_epoch'] = checkpoint['epoch']
            if 'best_val_acc' in checkpoint:
                info['training_info']['best_val_accuracy'] = checkpoint['best_val_acc']
            if 'train_losses' in checkpoint:
                info['training_info']['final_train_loss'] = checkpoint['train_losses'][-1] if checkpoint['train_losses'] else None
            
            # Print info
            print("✅ Checkpoint loaded successfully!\n")
            
            print(f"📊 Model Information:")
            print(f"   Classes: {info['num_classes']}")
            print(f"   Total Parameters: {info['architecture_info']['total_parameters']:,}")
            print()
            
            print(f"📊 Architecture:")
            print(f"   Type: {info['architecture_info']['architecture_type']}")
            print(f"   CLIP Variant: {info['architecture_info']['clip_variant']}")
            print(f"   Embedding Dim: {info['architecture_info']['embedding_dim']}")
            print(f"   Patch Size: {info['architecture_info']['patch_size']}")
            print()
            
            if info['training_info']:
                print(f"📊 Training Info:")
                for key, value in info['training_info'].items():
                    if value is not None:
                        if 'accuracy' in key.lower():
                            print(f"   {key.replace('_', ' ').title()}: {value*100:.2f}%")
                        elif 'loss' in key.lower():
                            print(f"   {key.replace('_', ' ').title()}: {value:.4f}")
                        else:
                            print(f"   {key.replace('_', ' ').title()}: {value}")
                print()
            
            print(f"📋 Classes ({info['num_classes']}):")
            for i, class_name in enumerate(info['class_names'], 1):
                print(f"   {i:2d}. {class_name}")
            
            self.results['layer2'] = info
            return info
            
        except Exception as e:
            print(f"❌ Error loading Layer 2: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def check_compatibility(self) -> dict:
        """ตรวจสอบความเข้ากันได้ระหว่าง Layer 1 และ Layer 2"""
        print(f"\n{'='*70}")
        print(f"🔗 COMPATIBILITY CHECK")
        print(f"{'='*70}\n")
        
        if not self.results['layer1'] or not self.results['layer2']:
            print("❌ Cannot check compatibility: Missing model info")
            return None
        
        layer1 = self.results['layer1']
        layer2 = self.results['layer2']
        
        compatibility = {
            'embedding_dim_match': False,
            'clip_variant_match': False,
            'issues': [],
            'recommendations': []
        }
        
        # Check embedding dimension
        layer1_embed = layer1['vision_model']['hidden_size']
        layer2_embed = layer2['architecture_info']['embedding_dim']
        
        if layer1_embed == layer2_embed:
            compatibility['embedding_dim_match'] = True
            print(f"✅ Embedding Dimension: MATCH ({layer1_embed})")
        else:
            compatibility['embedding_dim_match'] = False
            compatibility['issues'].append(
                f"Embedding dimension mismatch: Layer 1 ({layer1_embed}) vs Layer 2 ({layer2_embed})"
            )
            print(f"❌ Embedding Dimension: MISMATCH")
            print(f"   Layer 1: {layer1_embed}")
            print(f"   Layer 2: {layer2_embed}")
        
        # Check CLIP variant
        layer1_patch = layer1['vision_model']['patch_size']
        layer2_patch = layer2['architecture_info']['patch_size']
        
        if layer1_patch == layer2_patch:
            compatibility['clip_variant_match'] = True
            print(f"✅ CLIP Variant: MATCH (ViT-B/{layer1_patch})")
        else:
            compatibility['clip_variant_match'] = False
            compatibility['issues'].append(
                f"CLIP variant mismatch: Layer 1 (ViT-B/{layer1_patch}) vs Layer 2 (ViT-B/{layer2_patch})"
            )
            print(f"❌ CLIP Variant: MISMATCH")
            print(f"   Layer 1: ViT-B/{layer1_patch}")
            print(f"   Layer 2: ViT-B/{layer2_patch}")
        
        print()
        
        # Overall assessment
        if compatibility['embedding_dim_match'] and compatibility['clip_variant_match']:
            print("🎉 OVERALL: FULLY COMPATIBLE")
            print("   ✅ Both layers can work together in hybrid system")
        else:
            print("⚠️  OVERALL: COMPATIBILITY ISSUES DETECTED")
            print("\nIssues:")
            for issue in compatibility['issues']:
                print(f"   • {issue}")
            
            print("\nRecommendations:")
            if not compatibility['clip_variant_match']:
                print(f"   • Use 'openai/clip-vit-base-patch{layer2_patch}' for Layer 1")
                compatibility['recommendations'].append(f"Use openai/clip-vit-base-patch{layer2_patch}")
            if not compatibility['embedding_dim_match']:
                print(f"   • Re-train Layer 2 with matching CLIP model")
                compatibility['recommendations'].append("Re-train Layer 2 with matching CLIP model")
        
        self.results['compatibility'] = compatibility
        return compatibility
    
    def print_summary(self):
        """แสดงสรุปผลการตรวจสอบ"""
        print(f"\n{'='*70}")
        print(f"📋 SUMMARY")
        print(f"{'='*70}\n")
        
        if self.results['layer1']:
            print("✅ Layer 1 (Pre-trained): OK")
        else:
            print("❌ Layer 1 (Pre-trained): Failed to load")
        
        if self.results['layer2']:
            print("✅ Layer 2 (Fine-tuned): OK")
            print(f"   Architecture: {self.results['layer2']['architecture_info']['architecture_type']}")
            print(f"   Classes: {self.results['layer2']['num_classes']}")
        else:
            print("❌ Layer 2 (Fine-tuned): Failed to load")
        
        if self.results['compatibility']:
            comp = self.results['compatibility']
            if comp['embedding_dim_match'] and comp['clip_variant_match']:
                print("\n🎉 Hybrid System: READY TO USE")
            else:
                print(f"\n⚠️  Hybrid System: COMPATIBILITY ISSUES ({len(comp['issues'])})")
        
        print()


# ================================
# Main Function
# ================================

def main():
    parser = argparse.ArgumentParser(
        description='🔍 Model Architecture Checker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check both layers
  python check_model_architecture.py
  
  # Check with custom Layer 2 path
  python check_model_architecture.py --layer2 path/to/model.pth
  
  # Detailed output
  python check_model_architecture.py --detailed
        """
    )
    
    parser.add_argument('--layer1', type=str, default="openai/clip-vit-base-patch32",
                       help='HuggingFace model name for Layer 1')
    parser.add_argument('--layer2', type=str, default="models/layer2_finetuned/model_final.pth",
                       help='Path to Layer 2 model file')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed information')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"🔍 MODEL ARCHITECTURE CHECKER")
    print(f"{'='*70}")
    
    # Create checker
    checker = ModelArchitectureChecker()
    
    # Check Layer 1
    layer1_info = checker.check_layer1(args.layer1)
    
    # Check Layer 2
    layer2_info = checker.check_layer2(args.layer2)
    
    # Check compatibility
    if layer1_info and layer2_info:
        compatibility = checker.check_compatibility()
    
    # Print summary
    checker.print_summary()
    
    # Print detailed info if requested
    if args.detailed and layer2_info:
        print(f"\n{'='*70}")
        print(f"📊 DETAILED LAYER 2 INFO")
        print(f"{'='*70}\n")
        
        print("State Dict Keys:")
        for i, key in enumerate(layer2_info['state_dict_keys'][:10], 1):
            print(f"   {i:2d}. {key}")
        if len(layer2_info['state_dict_keys']) > 10:
            print(f"   ... and {len(layer2_info['state_dict_keys']) - 10} more")


if __name__ == '__main__':
    main()
