"""
Food Routes - HARDCODED TIPS TEST
Test if tips field can be sent at all
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
    
    - **food_name**: Name of the Thai dish (e.g., 'pad_thai', 'tom_yum_goong')
    - **lang**: Language code ('en' or 'th')
    """
    try:
        data_service = get_data_service()
        food_info = data_service.get_food_info(food_name, language=lang)
        
        # Get tips from data service
        tips_from_service = food_info.get('tips', '')
        
        # HARDCODE for testing - if empty, use test text
        if not tips_from_service:
            tips_from_service = "TEST: This is a hardcoded tips text to verify the field is being sent."
        
        print(f"🔍 DEBUG: Tips from service = {len(tips_from_service)} chars")
        print(f"🔍 DEBUG: Tips preview = {tips_from_service[:100]}")
        
        # Format response
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
            "tips": tips_from_service,  # EXPLICITLY HERE
            "message": None
        }
        
        print(f"🔍 DEBUG: Response has 'tips' key = {'tips' in response}")
        print(f"🔍 DEBUG: Response tips value = {response['tips'][:100]}")
        
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
    """List all available Thai dishes"""
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