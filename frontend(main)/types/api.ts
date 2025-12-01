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
  // New format
  general_info?: Record<string, string>;
  story?: string;
  // Legacy format  
  origin?: string;
  history?: string;
  cultural_significance?: string;
  season?: string;
  region?: string;
}

export interface Recipe {
  title?: string;
  description?: string;
  servings?: number;
  prep_time?: string;
  cook_time?: string;
  total_time?: string;
  difficulty?: number;
  difficulty_text?: string;
  // New format: flat array of strings
  ingredients?: string[];
  // New format: steps with title and content
  steps?: {
    title?: string;
    content: string;
    full_text?: string;
  }[];
  // Legacy format: categorized ingredients
  categorized_ingredients?: {
    category: string;
    items: string[];
  }[];
  // Legacy format: numbered instructions
  instructions?: {
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
  // New format
  id?: string;
  name_th?: string;
  name_en?: string;
  specialty?: string | string[];
  region?: string;
  latitude?: number;
  longitude?: number;
  price_range?: string;
  rating?: number;
  rating_source?: string;
  phone?: string;
  opening_hours?: string;
  website?: string;
  // Legacy format
  name?: string;
  location?: string;
  description?: string;
  address?: string;
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
