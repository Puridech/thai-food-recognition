"""
Model Service - AI Model Management for Backend
Adapted from scripts/hybrid/hybrid_prediction.py

Handles:
- Layer 1 (Pre-trained CLIP)
- Layer 2 (Fine-tuned Thai Food Specialist)
- Hybrid prediction logic
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import time
from typing import Dict, Tuple, Optional
from pathlib import Path
import sys


# ================================
# Model Architecture (from hybrid_prediction.py)
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
# Model Service
# ================================

class ModelService:
    """
    Model Service for Backend API
    Manages Layer 1, Layer 2, and Hybrid prediction logic
    """
    
    def __init__(
        self,
        layer1_model_name: str = "openai/clip-vit-base-patch32",
        layer2_model_path: Optional[Path] = None,
        confidence_threshold: float = 0.80,
        device: Optional[str] = None
    ):
        """
        Initialize Model Service
        
        Args:
            layer1_model_name: HuggingFace model name for Layer 1
            layer2_model_path: Path to fine-tuned Layer 2 model
            confidence_threshold: Threshold for Layer 1 vs Layer 2 (default: 0.80)
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set default Layer 2 path if not provided
        if layer2_model_path is None:
            # Assuming we're in backend/app/services/
            # Go up to root: ../../../models/layer2_finetuned/model_final.pth
            base_path = Path(__file__).parent.parent.parent.parent
            layer2_model_path = base_path / "models" / "layer2_finetuned" / "model_final.pth"
        
        self.layer2_model_path = layer2_model_path
        
        print(f"🚀 Initializing Model Service")
        print(f"   Device: {self.device}")
        print(f"   Confidence Threshold: {confidence_threshold * 100}%")
        print()
        
        # Models (will be loaded on demand)
        self.layer1_model = None
        self.layer1_processor = None
        self.layer2_model = None
        self.class_names = None
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'layer1_used': 0,
            'layer2_used': 0,
            'layer1_time': 0.0,
            'layer2_time': 0.0
        }
    
    def load_models(self):
        """Load both Layer 1 and Layer 2 models"""
        print("📥 Loading AI Models...")
        
        # Load Layer 1
        self.layer1_model, self.layer1_processor = self._load_layer1()
        
        # Load Layer 2
        self.layer2_model, self.class_names = self._load_layer2()
        
        print("✅ All models loaded successfully!\n")
    
    def _load_layer1(self) -> Tuple[CLIPModel, CLIPProcessor]:
        """Load Layer 1 pre-trained CLIP model"""
        print(f"📥 Loading Layer 1 (Pre-trained CLIP)...")
        
        try:
            model_name = "openai/clip-vit-base-patch32"
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            model = model.to(self.device)
            model.eval()
            
            print(f"   ✅ Layer 1 loaded")
            return model, processor
            
        except Exception as e:
            print(f"   ❌ Error loading Layer 1: {e}")
            raise
    
    def _load_layer2(self) -> Tuple[CLIPClassifierLayer2, list]:
        """Load Layer 2 fine-tuned model with auto-detection"""
        print(f"📥 Loading Layer 2 (Fine-tuned Thai Specialist)...")
        print(f"   Path: {self.layer2_model_path}")
        
        if not self.layer2_model_path.exists():
            raise FileNotFoundError(
                f"Layer 2 model not found: {self.layer2_model_path}\n"
                f"Please ensure model_final.pth exists in models/layer2_finetuned/"
            )
        
        try:
            # Load checkpoint (PyTorch 2.6+ requires weights_only=False)
            checkpoint = torch.load(
                self.layer2_model_path, 
                map_location=self.device,
                weights_only=False
            )
            
            # Extract class names
            class_names = checkpoint.get('class_names', None)
            if class_names is None:
                class_names = checkpoint.get('classes', None)
            
            if class_names is None or len(class_names) == 0:
                raise ValueError("No class_names found in checkpoint!")
            
            num_classes = len(class_names)
            state_dict = checkpoint['model_state_dict']
            
            # Auto-detect architecture
            has_batchnorm = any('BatchNorm' in key for key in state_dict.keys())
            has_running_stats = any('running_mean' in key or 'running_var' in key 
                                   for key in state_dict.keys())
            
            first_layer_key = 'classifier.0.weight'
            has_512_hidden = False
            if first_layer_key in state_dict:
                first_layer_shape = state_dict[first_layer_key].shape
                has_512_hidden = (first_layer_shape[0] == 512)
            
            use_batchnorm = has_batchnorm or has_running_stats or has_512_hidden
            
            print(f"   Classes: {num_classes}")
            print(f"   Architecture: {'Old (with BatchNorm)' if use_batchnorm else 'New (without BatchNorm)'}")
            
            # Load base CLIP model
            clip_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            # Create classifier with detected architecture
            model = CLIPClassifierLayer2(
                clip_model=clip_base,
                num_classes=num_classes,
                use_batchnorm=use_batchnorm
            )
            
            # Load weights
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            
            print(f"   ✅ Layer 2 loaded")
            return model, class_names
            
        except Exception as e:
            print(f"   ❌ Error loading Layer 2: {e}")
            raise
    
    def predict_layer1(self, image: Image.Image) -> Dict:
        """Predict using Layer 1 (fast baseline)"""
        start_time = time.time()
        
        # Prepare text labels (class names from Layer 2)
        text_inputs = [f"a photo of {name}" for name in self.class_names]
        
        # Process inputs
        inputs = self.layer1_processor(
            text=text_inputs,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.layer1_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0]
        
        # Get top prediction
        top_prob, top_idx = torch.max(probs, dim=0)
        
        elapsed_time = time.time() - start_time
        
        return {
            'food_name': self.class_names[top_idx.item()],
            'confidence': float(top_prob.item()),
            'layer_used': 1,
            'processing_time': elapsed_time
        }
    
    def predict_layer2(self, image: Image.Image) -> Dict:
        """Predict using Layer 2 (accurate specialist)"""
        start_time = time.time()
        
        # Process image (use same processor as Layer 1)
        inputs = self.layer1_processor(
            images=image,
            return_tensors="pt"
        )
        pixel_values = inputs['pixel_values'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.layer2_model(pixel_values)
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
        
        # Get top prediction
        top_prob, top_idx = torch.max(probs, dim=0)
        
        elapsed_time = time.time() - start_time
        
        return {
            'food_name': self.class_names[top_idx.item()],
            'confidence': float(top_prob.item()),
            'layer_used': 2,
            'processing_time': elapsed_time
        }
    
    def predict_hybrid(self, image: Image.Image) -> Dict:
        """
        Hybrid prediction logic
        
        1. Try Layer 1 first (fast)
        2. If confidence >= threshold: return Layer 1 result
        3. If confidence < threshold: use Layer 2 (accurate)
        
        Args:
            image: PIL Image
            
        Returns:
            Dict with prediction results
        """
        # Ensure models are loaded
        if self.layer1_model is None or self.layer2_model is None:
            raise RuntimeError("Models not loaded! Call load_models() first.")
        
        # Step 1: Try Layer 1
        layer1_result = self.predict_layer1(image)
        layer1_confidence = layer1_result['confidence']
        
        # Step 2: Decide
        if layer1_confidence >= self.confidence_threshold:
            # High confidence - use Layer 1
            self.stats['layer1_used'] += 1
            self.stats['layer1_time'] += layer1_result['processing_time']
            self.stats['total_predictions'] += 1
            
            return {
                'success': True,
                'food_name': layer1_result['food_name'],
                'confidence': layer1_result['confidence'],
                'layer_used': 1,
                'processing_time': layer1_result['processing_time'],
                'decision': f'Layer 1 (high confidence: {layer1_confidence*100:.1f}%)'
            }
        else:
            # Low confidence - use Layer 2
            layer2_result = self.predict_layer2(image)
            
            self.stats['layer2_used'] += 1
            self.stats['layer2_time'] += layer2_result['processing_time']
            self.stats['total_predictions'] += 1
            
            total_time = layer1_result['processing_time'] + layer2_result['processing_time']
            
            return {
                'success': True,
                'food_name': layer2_result['food_name'],
                'confidence': layer2_result['confidence'],
                'layer_used': 2,
                'processing_time': total_time,
                'decision': f'Layer 2 (Layer 1 confidence was {layer1_confidence*100:.1f}%)',
                'layer1_prediction': layer1_result['food_name'],
                'layer1_confidence': layer1_confidence
            }
    
    def get_statistics(self) -> Dict:
        """Get prediction statistics"""
        if self.stats['total_predictions'] == 0:
            return {
                'total_predictions': 0,
                'layer1_used': 0,
                'layer2_used': 0
            }
        
        total = self.stats['total_predictions']
        
        return {
            'total_predictions': total,
            'layer1_used': self.stats['layer1_used'],
            'layer1_percentage': (self.stats['layer1_used'] / total) * 100,
            'layer2_used': self.stats['layer2_used'],
            'layer2_percentage': (self.stats['layer2_used'] / total) * 100,
            'avg_layer1_time': (self.stats['layer1_time'] / self.stats['layer1_used'] 
                               if self.stats['layer1_used'] > 0 else 0),
            'avg_layer2_time': (self.stats['layer2_time'] / self.stats['layer2_used'] 
                               if self.stats['layer2_used'] > 0 else 0)
        }


# ================================
# Singleton Instance
# ================================

# Global model service instance (initialized in main.py on startup)
_model_service_instance: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Get the global model service instance"""
    global _model_service_instance
    
    if _model_service_instance is None:
        raise RuntimeError(
            "Model service not initialized! "
            "Call initialize_model_service() on app startup."
        )
    
    return _model_service_instance


def initialize_model_service(
    confidence_threshold: float = 0.80,
    device: Optional[str] = None
) -> ModelService:
    """Initialize the global model service instance"""
    global _model_service_instance
    
    _model_service_instance = ModelService(
        confidence_threshold=confidence_threshold,
        device=device
    )
    
    # Load models immediately
    _model_service_instance.load_models()
    
    return _model_service_instance