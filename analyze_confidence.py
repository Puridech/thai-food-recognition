# analyze_confidence.py
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import sys
import os

def analyze_image(image_path):
    """วิเคราะห์ความมั่นใจของ Layer 1"""
    
    #เช็ค environment
    print("\n📊 Environment Check:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU name: {torch.cuda.get_device_name(0)}")
    
    # โหลด model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    
    # 21 เมนู
    thai_dishes = [
        "Som Tum - Papaya Salad",
        "Tom Yum Goong - Spicy Shrimp Soup",
        "Larb - Spicy Minced Meat Salad",
        "Pad Thai - Thai Stir-Fried Noodles",
        "Kaeng Khiao Wan - Green Curry",
        "Khao Soi - Northern Thai Curry Noodles",
        "Kaeng Massaman - Massaman Curry",
        "Pad Krapow - Stir-Fried Holy Basil",
        "Khao Man Gai - Chicken Rice",
        "Khao Kha Mu - Stewed Pork Leg with Rice",
        "Tom Kha Gai - Chicken Coconut Galangal Soup",
        "Gai Pad Med Ma Muang - Thai Cashew Chicken",
        "Kai Palo - Five-Spice Stewed Eggs",
        "Gung Ob Woon Sen - Baked Prawns with Glass Noodles",
        "Khao Kluk Kapi - Rice with Shrimp Paste",
        "Por Pia Tod - Fried Spring Rolls",
        "Hor Mok - Steamed Curry in Banana Leaves",
        "Khao Niao Ma Muang - Mango Sticky Rice",
        "Khanom Krok - Coconut Pancake",
        "Foi Thong - Golden Threads"
    ]
    
    thai_names = [
        "Som Tum - Papaya Salad",
        "Tom Yum Goong - Spicy Shrimp Soup",
        "Larb - Spicy Minced Meat Salad",
        "Pad Thai - Thai Stir-Fried Noodles",
        "Kaeng Khiao Wan - Green Curry",
        "Khao Soi - Northern Thai Curry Noodles",
        "Kaeng Massaman - Massaman Curry",
        "Pad Krapow - Stir-Fried Holy Basil",
        "Khao Man Gai - Chicken Rice",
        "Khao Kha Mu - Stewed Pork Leg with Rice",
        "Tom Kha Gai - Chicken Coconut Galangal Soup",
        "Gai Pad Med Ma Muang - Thai Cashew Chicken",
        "Kai Palo - Five-Spice Stewed Eggs",
        "Gung Ob Woon Sen - Baked Prawns with Glass Noodles",
        "Khao Kluk Kapi - Rice with Shrimp Paste",
        "Por Pia Tod - Fried Spring Rolls",
        "Hor Mok - Steamed Curry in Banana Leaves",
        "Khao Niao Ma Muang - Mango Sticky Rice",
        "Khanom Krok - Coconut Pancake",
        "Foi Thong - Golden Threads"
    ]
    
    # โหลดรูป
    image = Image.open(image_path)
    
    # Predict
    inputs = processor(text=thai_dishes, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)
    
    # วิเคราะห์
    top_confidence = probs.max().item() * 100
    top_idx = probs.argmax().item()
    
    print(f"\n{'='*70}")
    print(f"📸 Image: {os.path.basename(image_path)}")
    print(f"{'='*70}")
    
    # แสดง Top 5
    print(f"\n🎯 Top 5 Predictions:")
    sorted_indices = probs[0].argsort(descending=True)
    for rank, idx in enumerate(sorted_indices[:5], 1):
        dish_th = thai_names[idx]
        dish_en = thai_dishes[idx].split(" - ")[0]
        confidence = probs[0][idx].item() * 100
        bar = "█" * int(confidence / 2)
        print(f"{rank}. {dish_th:15s} {confidence:6.2f}% {bar}")
    
    # Decision
    print(f"\n{'='*70}")
    print(f"🤖 Layer 1 Decision:")
    print(f"{'='*70}")
    print(f"Top Prediction: {thai_names[top_idx]} ({top_confidence:.2f}%)")
    print()
    
    if top_confidence >= 80:
        print("✅ DECISION: USE LAYER 1 RESULT")
        print(f"   Reason: High confidence ({top_confidence:.2f}% ≥ 80%)")
        print(f"   Status: ⚡ FAST (Layer 1 only)")
    elif top_confidence >= 60:
        print("⚠️  DECISION: SEND TO LAYER 2")
        print(f"   Reason: Medium confidence ({top_confidence:.2f}% = 60-79%)")
        print(f"   Status: 🔄 HYBRID (Layer 1 → Layer 2)")
    else:
        print("❌ DECISION: MUST USE LAYER 2")
        print(f"   Reason: Low confidence ({top_confidence:.2f}% < 60%)")
        print(f"   Status: 🎯 ACCURATE (Layer 2 required)")
    
    print(f"{'='*70}\n")
    
    return top_confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_confidence.py <image_path>")
        print("Example: python analyze_confidence.py D:\\food.jpg")
    else:
        analyze_image(sys.argv[1])