/**
 * API Client
 * Axios client for Backend API communication
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import ENV from '@/config/env';
import type {
  RecognitionResponse,
  FoodInfoResponse,
  RestaurantResponse,
  HealthResponse,
  ErrorResponse,
} from '@/types/api';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: ENV.API_URL,
      timeout: 30000, // 30 seconds
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        console.error('API Error:', error.message);
        return Promise.reject(error);
      }
    );
  }

  /**
   * Health Check
   */
  async healthCheck(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>(ENV.ENDPOINTS.HEALTH);
    return response.data;
  }

  /**
   * Recognize Food from Image
   */
  async recognizeFood(imageFile: File): Promise<RecognitionResponse> {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await this.client.post<RecognitionResponse>(
      ENV.ENDPOINTS.RECOGNIZE,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  }

  /**
   * Get Food Information
   */
  async getFoodInfo(
    foodName: string,
    language: string = ENV.DEFAULT_LANGUAGE
  ): Promise<FoodInfoResponse> {
    const response = await this.client.get<FoodInfoResponse>(
      `${ENV.ENDPOINTS.FOOD_INFO}/${foodName}`,
      {
        params: { lang: language },
      }
    );

    return response.data;
  }

  /**
   * Get Restaurants
   */
  async getRestaurants(
    foodName: string,
    region?: string
  ): Promise<RestaurantResponse> {
    const response = await this.client.get<RestaurantResponse>(
      `${ENV.ENDPOINTS.RESTAURANTS}/${foodName}`,
      {
        params: region ? { region } : {},
      }
    );

    return response.data;
  }

  /**
   * Generic error handler
   */
  handleError(error: unknown): ErrorResponse {
    if (axios.isAxiosError(error)) {
      return {
        success: false,
        error: error.message,
        detail: error.response?.data?.detail || 'Unknown error occurred',
      };
    }

    return {
      success: false,
      error: 'Unexpected error',
      detail: String(error),
    };
  }
}

// Export singleton instance
export const apiClient = new ApiClient();
export default apiClient;
