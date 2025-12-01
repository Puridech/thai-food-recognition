/**
 * Environment Configuration
 * Backend API URL and other environment variables
 */

export const ENV = {
  // Backend API URL
  API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  
  // API Endpoints
  ENDPOINTS: {
    HEALTH: '/api/health',
    RECOGNIZE: '/api/recognize',
    FOOD_INFO: '/api/food',
    RESTAURANTS: '/api/restaurants',
  },
  
  // Supported Languages
  LANGUAGES: {
    TH: 'th',
    EN: 'en',
  },
  
  // Default Language
  DEFAULT_LANGUAGE: 'en',
  
  // Storage Keys
  STORAGE_KEYS: {
    LANGUAGE: 'thai-food-language',
    FAVORITES: 'thai-food-favorites',
    HISTORY: 'thai-food-history',
  },
} as const;

export default ENV;
