"""
Thai Food Recognition Backend API
FastAPI Application - Week 5-6: Model Integration

Updated: Phase 1 - Model Integration Complete
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
from PIL import Image
import io
import uvicorn

# Import model service
try:
    from app.services import initialize_model_service, get_model_service
    MODEL_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import model service: {e}")
    print(f"   Running in mock mode only.")
    MODEL_SERVICE_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title="Thai Food Recognition API",
    description="AI-Powered Thai Food Recognition with Cultural Information",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    global MODEL_SERVICE_AVAILABLE
    
    print("\n" + "="*60)
    print("🍜 Thai Food Recognition API - Starting Up")
    print("="*60 + "\n")
    
    # Initialize AI Models
    if MODEL_SERVICE_AVAILABLE:
        try:
            print("🤖 Initializing AI Models...")
            initialize_model_service(confidence_threshold=0.80)
            print("✅ Models loaded successfully!\n")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print(f"⚠️  Server will run in MOCK MODE only\n")
            MODEL_SERVICE_AVAILABLE = False
    else:
        print("⚠️  Running in MOCK MODE (no models loaded)\n")
    
    # Initialize Data Service
    try:
        print("📚 Initializing Data Service...")
        from app.services import initialize_data_service
        initialize_data_service()
        print("✅ Data service ready!\n")
    except Exception as e:
        print(f"⚠️  Data service error: {e}\n")
    
    print("="*60)
    print("✅ Server Ready!")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n👋 Shutting down Thai Food Recognition API...")
    
    if MODEL_SERVICE_AVAILABLE:
        try:
            model_service = get_model_service()
            stats = model_service.get_statistics()
            
            if stats['total_predictions'] > 0:
                print("\n📊 Session Statistics:")
                print(f"   Total Predictions: {stats['total_predictions']}")
                print(f"   Layer 1 Used: {stats['layer1_used']} ({stats['layer1_percentage']:.1f}%)")
                print(f"   Layer 2 Used: {stats['layer2_used']} ({stats['layer2_percentage']:.1f}%)")
        except:
            pass
    
    print("\n✅ Shutdown complete\n")


# ==================== Response Models ====================

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str
    models_loaded: bool


class RecognitionResponse(BaseModel):
    success: bool
    food_name: str
    confidence: float
    layer_used: int
    processing_time: float
    decision: Optional[str] = None
    message: Optional[str] = None


class FoodInfoResponse(BaseModel):
    success: bool
    food_name: str
    language: str
    cultural_story: Optional[Dict[str, Any]] = None
    recipe: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class RestaurantResponse(BaseModel):
    success: bool
    food_name: str
    restaurants: list
    total_count: int
    message: Optional[str] = None


class StatsResponse(BaseModel):
    success: bool
    statistics: Dict[str, Any]


# ==================== API Endpoints ====================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Thai Food Recognition API! 🍜",
        "docs": "/docs",
        "health": "/api/health",
        "version": "1.0.0",
        "models_loaded": MODEL_SERVICE_AVAILABLE
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Thai Food Recognition API is running! 🚀",
        version="1.0.0",
        models_loaded=MODEL_SERVICE_AVAILABLE
    )


@app.post("/api/recognize", response_model=RecognitionResponse, tags=["Recognition"])
async def recognize_food(file: UploadFile = File(...)):
    """
    Recognize Thai food from uploaded image
    
    Hybrid 2-Layer System:
    1. Layer 1 predicts first (fast)
    2. If confidence >= 80%: return Layer 1 result
    3. If confidence < 80%: use Layer 2 (accurate)
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, WebP)"
        )
    
    # Check if models are loaded
    if not MODEL_SERVICE_AVAILABLE:
        return RecognitionResponse(
            success=True,
            food_name="Pad Thai",
            confidence=0.95,
            layer_used=1,
            processing_time=0.7,
            decision="Mock mode",
            message="⚠️ Models not loaded - this is a mock response"
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Get model service
        model_service = get_model_service()
        
        # Predict using hybrid logic
        result = model_service.predict_hybrid(image)
        
        return RecognitionResponse(
            success=result['success'],
            food_name=result['food_name'],
            confidence=result['confidence'],
            layer_used=result['layer_used'],
            processing_time=result['processing_time'],
            decision=result.get('decision', '')
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/api/food/{food_name}", response_model=FoodInfoResponse, tags=["Food Information"])
async def get_food_info(food_name: str, lang: str = "en"):
    """
    Get detailed food information
    
    Args:
        food_name: Food name (e.g., "foi_thong", "pad_thai")
        lang: Language code ("th" or "en")
    """
    try:
        from app.services import get_data_service
        data_service = get_data_service()
        
        # Get food info
        food_data = data_service.get_food_info(food_name, lang)
        
        return FoodInfoResponse(
            success=True,
            food_name=food_data['food_name'],
            language=food_data['language'],
            cultural_story={
                'title': food_data['title'],
                'general_info': food_data['general_info'],
                'story': food_data['cultural_story']
            },
            recipe=food_data['recipe']
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Food information not found: {food_name}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting food info: {str(e)}"
        )


@app.get("/api/restaurants/{food_name}", response_model=RestaurantResponse, tags=["Restaurants"])
async def get_restaurants(
    food_name: str, 
    region: Optional[str] = None,
    limit: int = 10
):
    """
    Get restaurants that serve a specific dish
    
    Args:
        food_name: Food name (e.g., "foi_thong", "pad_thai")
        region: Optional region filter
        limit: Maximum number of results (default: 10)
    """
    try:
        from app.services import get_data_service
        data_service = get_data_service()
        
        # Get restaurants
        restaurants = data_service.get_restaurants_by_food(
            food_name=food_name,
            region=region,
            limit=limit
        )
        
        return RestaurantResponse(
            success=True,
            food_name=food_name,
            restaurants=restaurants,
            total_count=len(restaurants)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting restaurants: {str(e)}"
        )


@app.get("/api/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_statistics():
    """Get prediction statistics"""
    if not MODEL_SERVICE_AVAILABLE:
        return StatsResponse(
            success=False,
            statistics={"message": "Models not loaded"}
        )
    
    try:
        model_service = get_model_service()
        stats = model_service.get_statistics()
        
        return StatsResponse(
            success=True,
            statistics=stats
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting statistics: {str(e)}"
        )


# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# ==================== Run Server ====================

if __name__ == "__main__":
    print("🍜 Starting Thai Food Recognition API...")
    print("📚 Swagger UI: http://localhost:8000/docs")
    print("📖 ReDoc: http://localhost:8000/redoc")
    print("🏥 Health Check: http://localhost:8000/api/health")
    print("\n✨ Server is starting...\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )