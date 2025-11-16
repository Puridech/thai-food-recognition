"""
🔍 Checkpoint Inspector
ตรวจสอบเนื้อหาใน model checkpoint
"""

import torch

# Load checkpoint
checkpoint_path = "models/layer2_finetuned/model_final.pth"

print("📥 Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("\n" + "="*70)
print("📋 CHECKPOINT CONTENTS")
print("="*70)

# Show all keys
print("\n🔑 Top-level Keys:")
for i, key in enumerate(checkpoint.keys(), 1):
    value = checkpoint[key]
    value_type = type(value).__name__
    
    if isinstance(value, dict):
        print(f"   {i}. {key} ({value_type}, {len(value)} items)")
    elif isinstance(value, list):
        print(f"   {i}. {key} ({value_type}, {len(value)} items)")
    elif isinstance(value, (int, float, str)):
        print(f"   {i}. {key} = {value}")
    else:
        print(f"   {i}. {key} ({value_type})")

# Check for class names in different possible locations
print("\n" + "="*70)
print("🔍 SEARCHING FOR CLASS NAMES")
print("="*70)

possible_keys = ['class_names', 'classes', 'labels', 'class_to_idx', 'idx_to_class']

found_classes = False
for key in possible_keys:
    if key in checkpoint:
        print(f"\n✅ Found: '{key}'")
        value = checkpoint[key]
        print(f"   Type: {type(value).__name__}")
        
        if isinstance(value, list):
            print(f"   Length: {len(value)}")
            if len(value) > 0:
                print(f"   First 5: {value[:5]}")
                found_classes = True
        elif isinstance(value, dict):
            print(f"   Items: {len(value)}")
            items = list(value.items())[:5]
            print(f"   First 5: {items}")
            found_classes = True

if not found_classes:
    print("\n⚠️  No class names found in standard locations")
    print("\nChecking other possibilities...")
    
    # Check if it's in nested structures
    for key, value in checkpoint.items():
        if isinstance(value, dict) and 'class' in str(key).lower():
            print(f"\n   Found dict with 'class' in name: {key}")
            print(f"   Keys: {list(value.keys())[:10]}")

print("\n" + "="*70)
print("💡 RECOMMENDATION")
print("="*70)

if not found_classes:
    print("""
⚠️  Class names not found in checkpoint!

This means you need to manually specify class names when loading the model.
The class names should match the order used during training.

Example:
    class_names = [
        'Pad Thai',
        'Tom Yum Goong',
        'Som Tam',
        ...
    ]
""")
else:
    print("\n✅ Class names found! Model should work correctly.")

print("\n" + "="*70)
print("📊 CHECKPOINT DETAILS")
print("="*70)

# Show more details about model_state_dict
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print(f"\nModel State Dict:")
    print(f"   Total parameters: {len(state_dict)}")
    print(f"   First 10 keys:")
    for i, key in enumerate(list(state_dict.keys())[:10], 1):
        print(f"      {i}. {key}")

# Show training info if available
if 'epoch' in checkpoint:
    print(f"\nTraining Info:")
    print(f"   Last Epoch: {checkpoint['epoch']}")
    
if 'best_val_acc' in checkpoint:
    print(f"   Best Val Accuracy: {checkpoint['best_val_acc']*100:.2f}%")

if 'train_losses' in checkpoint and len(checkpoint.get('train_losses', [])) > 0:
    losses = checkpoint['train_losses']
    print(f"   Final Train Loss: {losses[-1]:.4f}")

if 'val_losses' in checkpoint and len(checkpoint.get('val_losses', [])) > 0:
    losses = checkpoint['val_losses']
    print(f"   Final Val Loss: {losses[-1]:.4f}")

print()
