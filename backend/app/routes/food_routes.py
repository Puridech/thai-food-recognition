"""
Food Routes
Endpoints for food information (cultural stories, recipes, tips)
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from ..services.data_service import get_data_service

router = APIRouter()


@router.get("/food/{food_name}")
async def get_food_info(
    food_name: str,
    lang: str = Query(default="en", regex="^(en|th)$")
):
    """
    Get detailed food information including cultural story, recipe, and tips
    
    - **food_name**: Name of the Thai dish (e.g., 'pad_thai', 'tom_yum_goong', 'foi_thong')
    - **lang**: Language code ('en' or 'th')
    
    Returns:
    - Cultural story with general information
    - Recipe with ingredients, steps, cooking time, difficulty
    - Tips and notes for cooking
    """
    try:
        data_service = get_data_service()
        food_info = data_service.get_food_info(food_name, language=lang)
        
        # Get tips (with fallback)
        tips = food_info.get('tips', '')
        
        # Format response with ALL fields including tips
        response = {
            "success": True,
            "food_name": food_info['food_name'],
            "language": food_info['language'],
            "cultural_story": {
                "title": food_info['title'],
                "general_info": food_info['general_info'],
                "story": food_info['cultural_story']
            },
            "recipe": food_info['recipe'],
            "tips": tips,  # MUST HAVE!
            "message": None
        }
        
        return response
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting food info: {str(e)}"
        )


@router.get("/foods")
async def list_foods():
    """
    List all available Thai dishes
    
    Returns a list of food names that can be queried
    """
    try:
        data_service = get_data_service()
        foods = data_service.list_available_foods()
        
        return {
            "success": True,
            "foods": foods,
            "total_count": len(foods)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing foods: {str(e)}"
        )
