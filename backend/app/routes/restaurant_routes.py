"""
Restaurant Routes
Endpoints for restaurant recommendations
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from ..services.data_service import get_data_service

router = APIRouter()


@router.get("/restaurants/{food_name}")
async def get_restaurants_by_food(
    food_name: str,
    region: Optional[str] = Query(default=None),
    limit: Optional[int] = Query(default=10)
):
    """
    Get restaurants that serve a specific food
    
    - **food_name**: Name of the Thai dish
    - **region**: Filter by region (optional)
    - **limit**: Maximum number of results (default: 10)
    """
    try:
        data_service = get_data_service()
        restaurants = data_service.get_restaurants_by_food(
            food_name=food_name,
            region=region,
            limit=limit
        )
        
        return {
            "success": True,
            "food_name": food_name,
            "restaurants": restaurants,
            "total_count": len(restaurants)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting restaurants: {str(e)}"
        )


@router.get("/restaurants")
async def search_restaurants(
    query: Optional[str] = Query(default=None),
    region: Optional[str] = Query(default=None),
    min_rating: Optional[float] = Query(default=None),
    limit: Optional[int] = Query(default=10)
):
    """
    Search restaurants with filters
    
    - **query**: Search by restaurant name (optional)
    - **region**: Filter by region (optional)
    - **min_rating**: Minimum rating (optional)
    - **limit**: Maximum number of results (default: 10)
    """
    try:
        data_service = get_data_service()
        restaurants = data_service.search_restaurants(
            query=query,
            region=region,
            min_rating=min_rating,
            limit=limit
        )
        
        return {
            "success": True,
            "restaurants": restaurants,
            "total_count": len(restaurants)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching restaurants: {str(e)}"
        )
