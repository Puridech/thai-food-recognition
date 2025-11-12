"""
Test script to verify both Layer 1 and Layer 2 models can load and run inference
This ensures model architectures are correct before building hybrid system

python test_models_load.py
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

print("=" * 80)
print("THAI FOOD RECOGNITION - MODEL LOADING TEST")
print("=" * 80)
print()

# ================================================================================
# CONFIGURATION
# ================================================================================

# Layer 1: Pretrained CLIP paths
LAYER1_MODEL_DIR = project_root / "models" / "layer1_pretrained"

# Layer 2: Finetuned model paths
LAYER2_MODEL_PATH = project_root / "models" / "layer2_finetuned" / "model_final.pth"
LAYER2_BASE_MODEL = "openai/clip-vit-base-patch32"  # Base model ที่ใช้ fine-tune

# Test image (ใช้รูปจาก training data หรือรูปตัวอย่าง)
# คุณต้องเปลี่ยนเป็น path ที่มีรูปจริง
#TEST_IMAGE_PATH = project_root / "data" / "training" / "test" / "Pad_Thai" / "0289.jpg"  # เปลี่ยนเป็นรูปจริง
TEST_IMAGE_PATH = project_root / "test_image.jpg"

# Thai food classes (20 dishes from your training)
THAI_FOOD_CLASSES = [
    'Som Tum',
    'Tom Yum Goong',
    'Larb',
    'Pad Thai',
    'Kaeng Khiao Wan',
    'Khao Soi',
    'Kaeng Massaman',
    'Pad Krapow',
    'Khao Man Gai',
    'Khao Kha Mu',
    'Tom Kha Gai',
    'Gai Pad Med Ma Muang Himmaphan',
    'Kai Palo',
    'Gung Ob Woon Sen',
    'Khao Kluk Kapi',
    'Por Pia Tod',
    'Hor Mok',
    'Khao Niao Ma Muang',
    'Khanom Krok',
    'Foi Thong'
]

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Device: {device}")
print()

# ================================================================================
# LAYER 1: PRETRAINED CLIP MODEL
# ================================================================================

print("=" * 80)
print("LAYER 1: LOADING PRETRAINED CLIP MODEL")
print("=" * 80)

try:
    # Load pretrained CLIP from local directory
    print(f"📂 Loading from: {LAYER1_MODEL_DIR}")
    
    layer1_model = CLIPModel.from_pretrained(LAYER1_MODEL_DIR).to(device)
    layer1_processor = CLIPProcessor.from_pretrained(LAYER1_MODEL_DIR)
    
    layer1_model.eval()
    
    print("✅ Layer 1 loaded successfully!")
    print(f"   Model: {layer1_model.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in layer1_model.parameters()):,}")
    print()
    
    layer1_loaded = True
    
except Exception as e:
    print(f"❌ Failed to load Layer 1: {str(e)}")
    print("   Make sure pretrained CLIP model is in models/layer1_pretrained/")
    layer1_loaded = False
    print()

# ================================================================================
# LAYER 2: FINETUNED MODEL
# ================================================================================

print("=" * 80)
print("LAYER 2: LOADING FINETUNED MODEL")
print("=" * 80)

try:
    print(f"📂 Loading base model: {LAYER2_BASE_MODEL}")
    
    # Load base CLIP model
    layer2_base_model = CLIPModel.from_pretrained(LAYER2_BASE_MODEL).to(device)
    layer2_processor = CLIPProcessor.from_pretrained(LAYER2_BASE_MODEL)
    
    # Load finetuned weights
    print(f"📂 Loading finetuned weights: {LAYER2_MODEL_PATH}")
    
    if not LAYER2_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {LAYER2_MODEL_PATH}")
    
    # PyTorch 2.6+ requires weights_only=False for models saved with older PyTorch versions
    checkpoint = torch.load(LAYER2_MODEL_PATH, map_location=device, weights_only=False)
    
    # Check checkpoint structure
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("   Found model_state_dict in checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("   Found state_dict in checkpoint")
        else:
            # Checkpoint is the state dict itself
            state_dict = checkpoint
            print("   Using checkpoint as state_dict")
    else:
        state_dict = checkpoint
        print("   Using checkpoint directly")
    
    # Load state dict
    layer2_base_model.load_state_dict(state_dict, strict=False)
    layer2_base_model.eval()
    
    print("✅ Layer 2 loaded successfully!")
    print(f"   Model: {layer2_base_model.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in layer2_base_model.parameters()):,}")
    
    # Check if there's additional info in checkpoint
    if isinstance(checkpoint, dict):
        if 'epoch' in checkpoint:
            print(f"   Trained epochs: {checkpoint['epoch']}")
        if 'accuracy' in checkpoint:
            print(f"   Best accuracy: {checkpoint['accuracy']:.2f}%")
    
    print()
    
    layer2_loaded = True
    
except Exception as e:
    print(f"❌ Failed to load Layer 2: {str(e)}")
    print("   Make sure finetuned model is in models/layer2_finetuned/model_final.pth")
    layer2_loaded = False
    print()

# ================================================================================
# INFERENCE TEST
# ================================================================================

if layer1_loaded or layer2_loaded:
    print("=" * 80)
    print("INFERENCE TEST")
    print("=" * 80)
    
    # Check if test image exists
    if not TEST_IMAGE_PATH.exists():
        print(f"⚠️  Test image not found: {TEST_IMAGE_PATH}")
        print("   Please provide a test image to run inference test")
        print()
        print("   Options:")
        print("   1. Copy a test image to project root and name it 'test_image.jpg'")
        print("   2. Or modify TEST_IMAGE_PATH variable in this script")
    else:
        try:
            # Load test image
            print(f"📷 Loading test image: {TEST_IMAGE_PATH}")
            test_image = Image.open(TEST_IMAGE_PATH).convert("RGB")
            print(f"   Image size: {test_image.size}")
            print()
            
            # ====================================================================
            # Test Layer 1
            # ====================================================================
            if layer1_loaded:
                print("-" * 80)
                print("Testing Layer 1 (Pretrained CLIP)")
                print("-" * 80)
                
                # Prepare inputs
                inputs = layer1_processor(
                    text=THAI_FOOD_CLASSES,
                    images=test_image,
                    return_tensors="pt",
                    padding=True
                ).to(device)
                
                # Run inference
                with torch.no_grad():
                    outputs = layer1_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                
                # Get top 3 predictions
                top3_probs, top3_indices = torch.topk(probs[0], k=3)
                
                print("\n🎯 Layer 1 Predictions:")
                for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
                    print(f"   {i+1}. {THAI_FOOD_CLASSES[idx]:20s} - {prob.item()*100:.2f}%")
                
                print(f"\n   Confidence: {top3_probs[0].item()*100:.2f}%")
                print(f"   Predicted: {THAI_FOOD_CLASSES[top3_indices[0]]}")
                print()
            
            # ====================================================================
            # Test Layer 2
            # ====================================================================
            if layer2_loaded:
                print("-" * 80)
                print("Testing Layer 2 (Finetuned Model)")
                print("-" * 80)
                
                # Prepare inputs
                inputs = layer2_processor(
                    text=THAI_FOOD_CLASSES,
                    images=test_image,
                    return_tensors="pt",
                    padding=True
                ).to(device)
                
                # Run inference
                with torch.no_grad():
                    outputs = layer2_base_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                
                # Get top 3 predictions
                top3_probs, top3_indices = torch.topk(probs[0], k=3)
                
                print("\n🎯 Layer 2 Predictions:")
                for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
                    print(f"   {i+1}. {THAI_FOOD_CLASSES[idx]:20s} - {prob.item()*100:.2f}%")
                
                print(f"\n   Confidence: {top3_probs[0].item()*100:.2f}%")
                print(f"   Predicted: {THAI_FOOD_CLASSES[top3_indices[0]]}")
                print()
            
            # ====================================================================
            # Compare results
            # ====================================================================
            if layer1_loaded and layer2_loaded:
                print("-" * 80)
                print("📊 Comparison")
                print("-" * 80)
                print("Both models loaded successfully! Ready for hybrid system.")
                print()
                
        except Exception as e:
            print(f"❌ Inference test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            print()

# ================================================================================
# SUMMARY
# ================================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)

status_layer1 = "✅ LOADED" if layer1_loaded else "❌ FAILED"
status_layer2 = "✅ LOADED" if layer2_loaded else "❌ FAILED"

print(f"Layer 1 (Pretrained): {status_layer1}")
print(f"Layer 2 (Finetuned):  {status_layer2}")
print()

if layer1_loaded and layer2_loaded:
    print("🎉 SUCCESS! Both models are ready.")
    print("   Next step: Create hybrid system (hybrid_model.py)")
elif layer1_loaded or layer2_loaded:
    print("⚠️  PARTIAL SUCCESS: Only one model loaded.")
    print("   Fix the failed model before proceeding to hybrid system.")
else:
    print("❌ FAILED: No models loaded.")
    print("   Please check model paths and files.")

print()
print("=" * 80)