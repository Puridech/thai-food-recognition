"""
Routes Package
API endpoint definitions
"""

from .food_routes import router as food_router
from .restaurant_routes import router as restaurant_router
from .recognition_routes import router as recognition_router

__all__ = [
    'food_router',
    'restaurant_router', 
    'recognition_router'
]
