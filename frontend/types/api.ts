/**
 * API Response Types
 * Types for Backend API responses
 */

// ==================== Recognition API ====================
export interface RecognitionResponse {
  success: boolean;
  food_name: string;
  confidence: number;
  layer_used: 1 | 2;
  processing_time: number;
  decision?: string;
  message?: string;
}

// ==================== Food Info API ====================
export interface CulturalStory {
  title: string;
  origin: string;
  history: string;
  cultural_significance: string;
  season?: string;
  region?: string;
}

export interface Recipe {
  title: string;
  description: string;
  servings: number;
  prep_time: string;
  cook_time: string;
  total_time: string;
  difficulty: 'easy' | 'medium' | 'hard';
  ingredients: {
    category: string;
    items: string[];
  }[];
  instructions: {
    step: number;
    description: string;
  }[];
  tips?: string[];
  variations?: string[];
}

export interface FoodInfoResponse {
  success: boolean;
  food_name: string;
  language: string;
  cultural_story?: CulturalStory;
  recipe?: Recipe;
  message?: string;
}

// ==================== Restaurant API ====================
export interface Restaurant {
  name: string;
  location: string;
  region?: string;
  specialty: string;
  description: string;
  address?: string;
  rating?: number;
  price_range?: string;
}

export interface RestaurantResponse {
  success: boolean;
  food_name: string;
  restaurants: Restaurant[];
  total_count: number;
  message?: string;
}

// ==================== Health Check API ====================
export interface HealthResponse {
  status: string;
  message: string;
  version: string;
  models_loaded: boolean;
}

// ==================== Error Response ====================
export interface ErrorResponse {
  success: false;
  error: string;
  detail?: string;
}
