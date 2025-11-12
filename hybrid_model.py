"""
Hybrid Thai Food Recognition System
Combines Layer 1 (pretrained CLIP) and Layer 2 (finetuned) for optimal speed and accuracy

python hybrid_model.py
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import time
from pathlib import Path
from typing import Tuple, Dict, List
import json

class HybridThaiFood:
    """
    Hybrid AI model for Thai food recognition
    Uses 2-layer architecture for optimal speed/accuracy trade-off
    """
    
    def __init__(
        self,
        layer1_model_dir: str = "models/layer1_pretrained",
        layer2_model_path: str = "models/layer2_finetuned/model_final.pth",
        layer2_base_model: str = "openai/clip-vit-base-patch32",
        confidence_threshold: float = 0.80,
        device: str = None
    ):
        """
        Initialize hybrid model with both layers
        
        Args:
            layer1_model_dir: Path to pretrained CLIP model directory
            layer2_model_path: Path to finetuned model weights (.pth file)
            layer2_base_model: Base model used for fine-tuning
            confidence_threshold: Minimum confidence to accept Layer 1 result (default: 0.80)
            device: Device to run models on ('cuda' or 'cpu', auto-detect if None)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Thai food classes (20 dishes)
        self.classes = [
            "Som Tum",
            "Tom Yum Goong",
            "Larb",
            "Pad Thai",
            "Kaeng Khiao Wan",
            "Khao Soi",
            "Kaeng Massaman",
            "Pad Krapow",
            "Khao Man Gai",
            "Khao Kha Mu",
            "Tom Kha Gai",
            "Gai Pad Med Ma Muang Himmaphan",
            "Kai Palo",
            "Gung Ob Woon Sen",
            "Khao Kluk Kapi",
            "Por Pia Tod",
            "Hor Mok",
            "Khao Niao Ma Muang",
            "Khanom Krok",
            "Foi Thong"
        ]
        
        print(f"🔧 Initializing Hybrid Thai Food Recognition System")
        print(f"   Device: {self.device}")
        print(f"   Confidence threshold: {self.confidence_threshold}")
        print()
        
        # Load models
        self._load_layer1(layer1_model_dir)
        self._load_layer2(layer2_model_path, layer2_base_model)
        
        # Statistics tracking
        self.stats = {
            "total_predictions": 0,
            "layer1_used": 0,
            "layer2_used": 0,
            "total_time": 0.0
        }
        
        print("✅ Hybrid system ready!")
        print()
    
    def _load_layer1(self, model_dir: str):
        """Load Layer 1 (pretrained CLIP)"""
        print("📦 Loading Layer 1 (Pretrained CLIP)...")
        try:
            self.layer1_model = CLIPModel.from_pretrained(model_dir).to(self.device)
            self.layer1_processor = CLIPProcessor.from_pretrained(model_dir)
            self.layer1_model.eval()
            print("   ✅ Layer 1 loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load Layer 1: {str(e)}")
    
    def _load_layer2(self, model_path: str, base_model: str):
        """Load Layer 2 (finetuned model)"""
        print("📦 Loading Layer 2 (Finetuned Thai Specialist)...")
        try:
            # Load base model
            self.layer2_model = CLIPModel.from_pretrained(base_model).to(self.device)
            self.layer2_processor = CLIPProcessor.from_pretrained(base_model)
            
            # Load finetuned weights
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # PyTorch 2.6+ requires weights_only=False for full checkpoint loading
            # This is safe since we're loading our own trained model
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Extract state dict
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load weights
            self.layer2_model.load_state_dict(state_dict, strict=False)
            self.layer2_model.eval()
            
            print("   ✅ Layer 2 loaded successfully")
            
            # Print training info if available
            if isinstance(checkpoint, dict):
                if 'epoch' in checkpoint:
                    print(f"   📊 Trained epochs: {checkpoint['epoch']}")
                if 'accuracy' in checkpoint:
                    print(f"   📊 Best accuracy: {checkpoint['accuracy']:.2f}%")
                    
        except Exception as e:
            raise RuntimeError(f"Failed to load Layer 2: {str(e)}")
    
    def _predict_with_layer(
        self, 
        image: Image.Image, 
        model, 
        processor,
        layer_name: str
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Run prediction with a specific layer
        
        Returns:
            predicted_class: Top predicted class name
            confidence: Confidence score (0-1)
            top3_results: List of (class_name, confidence) tuples
        """
        # Prepare inputs
        inputs = processor(
            text=self.classes,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            start_time = time.time()
            outputs = model(**inputs)
            inference_time = time.time() - start_time
            
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probs[0], k=3)
        
        predicted_class = self.classes[top3_indices[0]]
        confidence = top3_probs[0].item()
        
        top3_results = [
            (self.classes[idx], prob.item())
            for prob, idx in zip(top3_probs, top3_indices)
        ]
        
        return predicted_class, confidence, top3_results, inference_time
    
    def predict(
        self, 
        image: Image.Image, 
        return_details: bool = False
    ) -> Dict:
        """
        Main prediction method using hybrid architecture
        
        Args:
            image: PIL Image object
            return_details: If True, return detailed information about the prediction
        
        Returns:
            Dictionary containing:
                - predicted_class: Top predicted class
                - confidence: Confidence score (0-1)
                - layer_used: Which layer was used ('layer1' or 'layer2')
                - inference_time: Time taken for prediction (seconds)
                - top3_predictions: Top 3 predictions (if return_details=True)
                - layer1_result: Layer 1 prediction details (if return_details=True)
                - layer2_result: Layer 2 prediction details (if return_details=True and used)
        """
        start_time = time.time()
        
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Step 1: Try Layer 1 first
        layer1_class, layer1_conf, layer1_top3, layer1_time = self._predict_with_layer(
            image, self.layer1_model, self.layer1_processor, "Layer 1"
        )
        
        # Step 2: Check confidence threshold
        if layer1_conf >= self.confidence_threshold:
            # High confidence - use Layer 1 result
            layer_used = "layer1"
            final_class = layer1_class
            final_confidence = layer1_conf
            final_top3 = layer1_top3
            total_time = time.time() - start_time
            
            # Update statistics
            self.stats["layer1_used"] += 1
            
            result = {
                "predicted_class": final_class,
                "confidence": final_confidence,
                "layer_used": layer_used,
                "inference_time": total_time
            }
            
            if return_details:
                result["top3_predictions"] = final_top3
                result["layer1_result"] = {
                    "class": layer1_class,
                    "confidence": layer1_conf,
                    "top3": layer1_top3,
                    "time": layer1_time
                }
            
        else:
            # Low confidence - use Layer 2
            layer2_class, layer2_conf, layer2_top3, layer2_time = self._predict_with_layer(
                image, self.layer2_model, self.layer2_processor, "Layer 2"
            )
            
            layer_used = "layer2"
            final_class = layer2_class
            final_confidence = layer2_conf
            final_top3 = layer2_top3
            total_time = time.time() - start_time
            
            # Update statistics
            self.stats["layer2_used"] += 1
            
            result = {
                "predicted_class": final_class,
                "confidence": final_confidence,
                "layer_used": layer_used,
                "inference_time": total_time
            }
            
            if return_details:
                result["top3_predictions"] = final_top3
                result["layer1_result"] = {
                    "class": layer1_class,
                    "confidence": layer1_conf,
                    "top3": layer1_top3,
                    "time": layer1_time
                }
                result["layer2_result"] = {
                    "class": layer2_class,
                    "confidence": layer2_conf,
                    "top3": layer2_top3,
                    "time": layer2_time
                }
        
        # Update overall statistics
        self.stats["total_predictions"] += 1
        self.stats["total_time"] += total_time
        
        return result
    
    def predict_batch(
        self,
        images: List[Image.Image],
        return_details: bool = False
    ) -> List[Dict]:
        """
        Predict multiple images
        
        Args:
            images: List of PIL Image objects
            return_details: If True, return detailed information for each prediction
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        print(f"🔄 Processing {len(images)} images...")
        
        for i, image in enumerate(images, 1):
            result = self.predict(image, return_details=return_details)
            results.append(result)
            
            if i % 10 == 0 or i == len(images):
                print(f"   Progress: {i}/{len(images)}")
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get usage statistics"""
        total = self.stats["total_predictions"]
        
        if total == 0:
            return self.stats
        
        stats = self.stats.copy()
        stats["layer1_percentage"] = (self.stats["layer1_used"] / total) * 100
        stats["layer2_percentage"] = (self.stats["layer2_used"] / total) * 100
        stats["avg_time"] = self.stats["total_time"] / total
        
        return stats
    
    def print_statistics(self):
        """Print usage statistics in a readable format"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 80)
        print("HYBRID SYSTEM STATISTICS")
        print("=" * 80)
        print(f"Total predictions:       {stats['total_predictions']}")
        print(f"Layer 1 used:           {stats['layer1_used']} ({stats.get('layer1_percentage', 0):.1f}%)")
        print(f"Layer 2 used:           {stats['layer2_used']} ({stats.get('layer2_percentage', 0):.1f}%)")
        print(f"Average inference time: {stats.get('avg_time', 0):.3f}s")
        print(f"Total time:             {stats['total_time']:.3f}s")
        print("=" * 80)
    
    def reset_statistics(self):
        """Reset usage statistics"""
        self.stats = {
            "total_predictions": 0,
            "layer1_used": 0,
            "layer2_used": 0,
            "total_time": 0.0
        }


# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def load_hybrid_model(
    confidence_threshold: float = 0.80,
    device: str = None
) -> HybridThaiFood:
    """
    Convenience function to load hybrid model with default settings
    
    Args:
        confidence_threshold: Minimum confidence to accept Layer 1 result
        device: Device to run on ('cuda' or 'cpu', auto-detect if None)
    
    Returns:
        Initialized HybridThaiFood model
    """
    return HybridThaiFood(
        confidence_threshold=confidence_threshold,
        device=device
    )


if __name__ == "__main__":
    """
    Test the hybrid model with a sample image
    """
    print("Testing Hybrid Thai Food Recognition System")
    print()
    
    # Initialize model
    model = HybridThaiFood(confidence_threshold=0.80)
    
    # Test with a sample image
    test_image_path = "test_image.jpg"
    
    if Path(test_image_path).exists():
        print(f"Testing with image: {test_image_path}")
        test_image = Image.open(test_image_path)
        
        # Make prediction with details
        result = model.predict(test_image, return_details=True)
        
        print("\n" + "=" * 80)
        print("PREDICTION RESULT")
        print("=" * 80)
        print(f"Predicted: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Layer used: {result['layer_used'].upper()}")
        print(f"Inference time: {result['inference_time']:.3f}s")
        
        if 'top3_predictions' in result:
            print("\nTop 3 predictions:")
            for i, (cls, conf) in enumerate(result['top3_predictions'], 1):
                print(f"   {i}. {cls:20s} - {conf*100:.2f}%")
        
        if 'layer1_result' in result:
            print(f"\nLayer 1 prediction: {result['layer1_result']['class']} ({result['layer1_result']['confidence']*100:.2f}%)")
        
        if 'layer2_result' in result:
            print(f"Layer 2 prediction: {result['layer2_result']['class']} ({result['layer2_result']['confidence']*100:.2f}%)")
        
        print("=" * 80)
        
        # Show statistics
        model.print_statistics()
        
    else:
        print(f"⚠️  Test image not found: {test_image_path}")
        print("   Please provide a test image to run the demo")