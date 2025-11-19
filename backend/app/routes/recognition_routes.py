"""
Recognition Routes
Endpoints for AI food recognition
"""

from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import Optional
from ..services.model_service import get_model_service

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "success": True,
        "status": "healthy",
        "message": "Thai Food Recognition API is running"
    }


@router.post("/recognize")
async def recognize_food(
    file: UploadFile = File(...),
    top_k: Optional[int] = 3
):
    """
    Recognize Thai food from image
    
    - **file**: Image file (jpg, png, etc.)
    - **top_k**: Number of top predictions to return (default: 3)
    
    Returns:
    - Top predictions with confidence scores
    - Layer used (Layer 1 or Layer 2)
    - Processing time
    """
    try:
        # Read image
        image_data = await file.read()
        
        # Get model service
        model_service = get_model_service()
        
        # Predict
        result = model_service.predict_from_bytes(
            image_data=image_data,
            top_k=top_k
        )
        
        return {
            "success": True,
            "predictions": result['predictions'],
            "top_prediction": result['predictions'][0] if result['predictions'] else None,
            "layer_used": result.get('layer', 'unknown'),
            "processing_time": result.get('time', 0),
            "message": None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error recognizing food: {str(e)}"
        )


@router.get("/stats")
async def get_statistics():
    """
    Get AI model statistics
    
    Returns usage statistics for Layer 1 and Layer 2
    """
    try:
        model_service = get_model_service()
        stats = model_service.get_statistics()
        
        return {
            "success": True,
            "statistics": stats
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting statistics: {str(e)}"
        )
