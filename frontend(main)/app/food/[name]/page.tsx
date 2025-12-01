'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { useTranslation } from 'react-i18next';
import '@/lib/i18n';
import Navbar from '@/components/layout/Navbar';
import { apiClient } from '@/lib/api-client';
import Loading from '@/components/ui/Loading';
import ErrorMessage from '@/components/ui/ErrorMessage';
import type { FoodInfoResponse, RestaurantResponse } from '@/types/api';

export default function FoodDetailPage() {
  const params = useParams();
  const router = useRouter();
  const { t, i18n } = useTranslation();
  const [foodInfo, setFoodInfo] = useState<FoodInfoResponse | null>(null);
  const [restaurants, setRestaurants] = useState<RestaurantResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'story' | 'recipe' | 'restaurants'>('story');
  const [foodImage, setFoodImage] = useState<string | null>(null);

  const foodName = params.name as string;

  // Convert URL-encoded food name to backend format (lowercase with underscores)
  const normalizedFoodName = decodeURIComponent(foodName)
    .toLowerCase()
    .replace(/\s+/g, '_'); // Replace spaces with underscores

  // Format food name for display
  const formatFoodName = (name: string): string => {
    return name
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Fetch food info and restaurants
  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      setError(null);

      try {
        // Load food image from history if available
        try {
          const historyData = localStorage.getItem('thai_food_history');
          if (historyData) {
            const history = JSON.parse(historyData);
            const item = history.find((h: any) => h.food_name.toLowerCase() === normalizedFoodName);
            if (item && item.image_url) {
              setFoodImage(item.image_url);
            }
          }
        } catch (err) {
          console.log('No image found in history');
        }

        // Fetch food info using normalized name
        const info = await apiClient.getFoodInfo(normalizedFoodName, i18n.language);
        console.log('📊 Food Info Response:', JSON.stringify(info, null, 2));
        setFoodInfo(info);

        // Fetch restaurants using normalized name
        const rest = await apiClient.getRestaurants(normalizedFoodName);
        console.log('🏪 Restaurants Response:', JSON.stringify(rest, null, 2));
        setRestaurants(rest);
      } catch (err: unknown) {
        console.error('Error fetching food data:', err);
        setError((err as Error).message || t('errorOccurred'));
      } finally {
        setIsLoading(false);
      }
    };

    if (normalizedFoodName) {
      fetchData();
    }
  }, [normalizedFoodName, i18n.language, t]);

  // Loading state
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-orange-50 via-red-50 to-yellow-50 flex items-center justify-center">
        <Loading message={t('pleaseWait')} size="lg" />
      </div>
    );
  }

  // Error state
  if (error || !foodInfo) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-orange-50 via-red-50 to-yellow-50">
        <div className="container-custom py-12">
          <ErrorMessage
            title={t('error')}
            message={error || 'Failed to load food information'}
            onRetry={() => window.location.reload()}
          />
          <div className="text-center mt-6">
            <button
              onClick={() => router.push('/')}
              className="btn btn-primary"
            >
              <FiArrowLeft className="inline mr-2" />
              {t('home')}
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-red-50 to-yellow-50">
      {/* Navbar */}
      <Navbar />

      {/* Main Content */}
      <main className="container-custom py-8">
        <div className="max-w-6xl mx-auto">
          {/* Hero Section with Image */}
          <div className="relative overflow-hidden rounded-2xl shadow-2xl mb-8">
            {/* Background Gradient */}
            <div className="absolute inset-0 bg-gradient-to-br from-orange-400 via-red-400 to-pink-400 opacity-90"></div>
            
            {/* Content */}
            <div className="relative z-10 p-6 md:p-12">
              <div className="flex flex-col md:flex-row items-center gap-6 md:gap-12">
                {/* Food Image */}
                <div className="flex-shrink-0">
                  <div className="w-40 h-40 md:w-56 md:h-56 rounded-full overflow-hidden ring-8 ring-white/30 shadow-2xl bg-white">
                    {foodImage ? (
                      <img
                        src={foodImage}
                        alt={formatFoodName(foodInfo.food_name)}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-8xl">
                        🍜
                      </div>
                    )}
                  </div>
                </div>

                {/* Text Content */}
                <div className="flex-1 text-center md:text-left">
                  <div className="inline-block mb-4">
                    <span className="px-4 py-2 bg-white/20 backdrop-blur-sm text-white rounded-full text-sm font-semibold">
                      🍽️ Thai Cuisine
                    </span>
                  </div>
                  
                  <h1 className="text-4xl md:text-6xl font-extrabold text-white mb-4 drop-shadow-lg">
                    {formatFoodName(foodInfo.food_name)}
                  </h1>
                  
                  {foodInfo.cultural_story?.region && (
                    <div className="flex items-center justify-center md:justify-start gap-2 text-white/90 text-lg mb-4">
                      <span className="text-2xl">📍</span>
                      <span className="font-medium">{foodInfo.cultural_story.region}</span>
                    </div>
                  )}

                  <div className="flex flex-wrap gap-3 justify-center md:justify-start">
                    <span className="px-4 py-2 bg-white/20 backdrop-blur-sm text-white rounded-lg text-sm font-medium">
                      ⭐ Popular Dish
                    </span>
                    <span className="px-4 py-2 bg-white/20 backdrop-blur-sm text-white rounded-lg text-sm font-medium">
                      🔥 Authentic Recipe
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Tabs - Responsive */}
          <div className="card mb-6 overflow-hidden">
            {/* Mobile: Vertical Tabs, Desktop: Horizontal */}
            <div className="flex flex-col sm:flex-row border-b border-gray-200">
              <button
                onClick={() => setActiveTab('story')}
                className={`flex-1 py-3 px-4 sm:py-4 sm:px-6 font-semibold transition-all text-left sm:text-center ${
                  activeTab === 'story'
                    ? 'bg-primary-50 text-primary-600 border-l-4 sm:border-l-0 sm:border-b-4 border-primary-600'
                    : 'text-gray-600 hover:bg-gray-50'
                }`}
              >
                <span className="text-lg sm:text-xl mr-2">📖</span>
                <span className="text-sm sm:text-base">{t('culturalStory')}</span>
              </button>

              <button
                onClick={() => setActiveTab('recipe')}
                className={`flex-1 py-3 px-4 sm:py-4 sm:px-6 font-semibold transition-all text-left sm:text-center ${
                  activeTab === 'recipe'
                    ? 'bg-primary-50 text-primary-600 border-l-4 sm:border-l-0 sm:border-b-4 border-primary-600'
                    : 'text-gray-600 hover:bg-gray-50'
                }`}
              >
                <span className="text-lg sm:text-xl mr-2">👨‍🍳</span>
                <span className="text-sm sm:text-base">{t('recipe')}</span>
              </button>

              <button
                onClick={() => setActiveTab('restaurants')}
                className={`flex-1 py-3 px-4 sm:py-4 sm:px-6 font-semibold transition-all text-left sm:text-center ${
                  activeTab === 'restaurants'
                    ? 'bg-primary-50 text-primary-600 border-l-4 sm:border-l-0 sm:border-b-4 border-primary-600'
                    : 'text-gray-600 hover:bg-gray-50'
                }`}
              >
                <span className="text-lg sm:text-xl mr-2">🏪</span>
                <span className="text-sm sm:text-base">{t('restaurants')}</span>
              </button>
            </div>

            {/* Tab Content */}
            <div className="p-6">
              {/* Cultural Story Tab */}
              {activeTab === 'story' && (
                <div className="prose max-w-none">
                  {foodInfo.cultural_story ? (
                    <>
                      <h2 className="text-2xl font-bold text-gray-900 mb-6">
                        {foodInfo.cultural_story.title || formatFoodName(foodInfo.food_name)}
                      </h2>
                      
                      {/* General Info Cards - New Format */}
                      {foodInfo.cultural_story.general_info && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8 not-prose">
                          {Object.entries(foodInfo.cultural_story.general_info).map(([key, value]) => {
                            const displayKey = key
                              .replace(/_/g, ' ')
                              .split(' ')
                              .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                              .join(' ');
                            
                            return (
                              <div key={key} className="p-4 bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg border border-amber-200">
                                <div className="text-sm text-gray-600 font-medium mb-1">{displayKey}</div>
                                <div className="text-gray-800 font-semibold">{String(value)}</div>
                              </div>
                            );
                          })}
                        </div>
                      )}

                      {/* Story Content - New Format */}
                      {foodInfo.cultural_story.story && (
                        <div className="mb-6">
                          <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                            <span>📖</span>
                            Story
                          </h3>
                          <div className="space-y-4">
                            {foodInfo.cultural_story.story.split('\n\n').map((paragraph, idx) => (
                              <p key={idx} className="text-gray-700 leading-relaxed">
                                {paragraph.trim()}
                              </p>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Legacy Format Support */}
                      {foodInfo.cultural_story.origin && (
                        <div className="mb-6">
                          <h3 className="text-xl font-semibold text-gray-800 mb-2">
                            🌏 Origin
                          </h3>
                          <p className="text-gray-700 leading-relaxed">
                            {foodInfo.cultural_story.origin}
                          </p>
                        </div>
                      )}

                      {foodInfo.cultural_story.history && (
                        <div className="mb-6">
                          <h3 className="text-xl font-semibold text-gray-800 mb-2">
                            📜 History
                          </h3>
                          <p className="text-gray-700 leading-relaxed">
                            {foodInfo.cultural_story.history}
                          </p>
                        </div>
                      )}

                      {foodInfo.cultural_story.cultural_significance && (
                        <div className="mb-6">
                          <h3 className="text-xl font-semibold text-gray-800 mb-2">
                            ⭐ Cultural Significance
                          </h3>
                          <p className="text-gray-700 leading-relaxed">
                            {foodInfo.cultural_story.cultural_significance}
                          </p>
                        </div>
                      )}

                      {foodInfo.cultural_story.season && (
                        <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-100 text-blue-700 rounded-lg">
                          <span>🗓️ Best Season:</span>
                          <span className="font-semibold">{foodInfo.cultural_story.season}</span>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="text-center py-12">
                      <div className="text-6xl mb-4">📖</div>
                      <p className="text-gray-600 text-lg">
                        Cultural story information is not available yet.
                      </p>
                      <p className="text-gray-500 text-sm mt-2">
                        Check back later for interesting stories about {formatFoodName(foodInfo.food_name)}!
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Recipe Tab */}
              {activeTab === 'recipe' && (
                <div>
                  {foodInfo.recipe ? (
                    <>
                      {/* Recipe Header */}
                      <div className="mb-6">
                        <h2 className="text-2xl font-bold text-gray-900 mb-2">
                          {foodInfo.recipe.title || formatFoodName(foodInfo.food_name)}
                        </h2>
                        {foodInfo.recipe.description && (
                          <p className="text-gray-600">{foodInfo.recipe.description}</p>
                        )}
                      </div>

                      {/* Recipe Info */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                        {foodInfo.recipe.servings && (
                          <div className="p-4 bg-blue-50 rounded-lg text-center">
                            <div className="text-2xl mb-1">👥</div>
                            <div className="text-sm text-gray-600">{t('servings')}</div>
                            <div className="font-bold text-gray-900">{foodInfo.recipe.servings}</div>
                          </div>
                        )}

                        {foodInfo.recipe.prep_time && (
                          <div className="p-4 bg-green-50 rounded-lg text-center">
                            <div className="text-2xl mb-1">⏱️</div>
                            <div className="text-sm text-gray-600">{t('prepTime')}</div>
                            <div className="font-bold text-gray-900">{foodInfo.recipe.prep_time}</div>
                          </div>
                        )}

                        {foodInfo.recipe.cook_time && (
                          <div className="p-4 bg-orange-50 rounded-lg text-center">
                            <div className="text-2xl mb-1">🔥</div>
                            <div className="text-sm text-gray-600">{t('cookTime')}</div>
                            <div className="font-bold text-gray-900">{foodInfo.recipe.cook_time}</div>
                          </div>
                        )}

                        {(foodInfo.recipe.difficulty || foodInfo.recipe.difficulty_text) && (
                          <div className="p-4 bg-purple-50 rounded-lg text-center">
                            <div className="text-2xl mb-1">📊</div>
                            <div className="text-sm text-gray-600">{t('difficulty')}</div>
                            <div className="font-bold text-gray-900 capitalize">
                              {foodInfo.recipe.difficulty_text || 
                               (typeof foodInfo.recipe.difficulty === 'number' 
                                 ? `Level ${foodInfo.recipe.difficulty}`
                                 : String(foodInfo.recipe.difficulty))}
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Ingredients */}
                      {foodInfo.recipe.ingredients && foodInfo.recipe.ingredients.length > 0 && (
                        <div className="mb-8">
                          <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                            <span>🛒</span>
                            {t('ingredients')}
                          </h3>
                          <ul className="space-y-2">
                            {foodInfo.recipe.ingredients.map((ingredient, idx) => (
                              <li
                                key={idx}
                                className="flex items-start gap-2 text-gray-700"
                              >
                                <span className="text-primary-600 mt-1">•</span>
                                <span>{ingredient}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Instructions */}
                      {foodInfo.recipe.steps && foodInfo.recipe.steps.length > 0 && (
                        <div className="mb-8">
                          <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                            <span>📝</span>
                            {t('instructions')}
                          </h3>
                          <div className="space-y-4">
                            {foodInfo.recipe.steps.map((step, idx) => (
                              <div
                                key={idx}
                                className="flex gap-4 p-4 bg-gray-50 rounded-lg"
                              >
                                <div className="flex-shrink-0 w-8 h-8 bg-primary-600 text-white rounded-full flex items-center justify-center font-bold">
                                  {idx + 1}
                                </div>
                                <div className="flex-1">
                                  {step.title && (
                                    <h4 className="font-semibold text-gray-800 mb-1">
                                      {step.title}
                                    </h4>
                                  )}
                                  <p className="text-gray-700">
                                    {step.content}
                                  </p>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Tips */}
                      {foodInfo.recipe.tips && foodInfo.recipe.tips.length > 0 && (
                        <div className="p-6 bg-yellow-50 border-2 border-yellow-200 rounded-lg">
                          <h3 className="text-xl font-bold text-gray-900 mb-3 flex items-center gap-2">
                            <span>💡</span>
                            {t('tips')}
                          </h3>
                          <ul className="space-y-2">
                            {foodInfo.recipe.tips.map((tip, idx) => (
                              <li key={idx} className="flex items-start gap-2 text-gray-700">
                                <span className="text-yellow-600 mt-1">★</span>
                                <span>{tip}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="text-center py-12">
                      <div className="text-6xl mb-4">👨‍🍳</div>
                      <p className="text-gray-600 text-lg">
                        Recipe information is not available yet.
                      </p>
                      <p className="text-gray-500 text-sm mt-2">
                        Check back later for detailed cooking instructions!
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Restaurants Tab */}
              {activeTab === 'restaurants' && restaurants && (
                <div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-6">
                    Recommended Restaurants ({restaurants.total_count})
                  </h2>
                  
                  {restaurants.restaurants.length === 0 ? (
                    <div className="text-center py-12 text-gray-500">
                      <div className="text-6xl mb-4">🏪</div>
                      <p>{t('noRestaurants')}</p>
                    </div>
                  ) : (
                    <div className="grid gap-6">
                      {restaurants.restaurants.map((restaurant, idx) => {
                        const mapsUrl = restaurant.latitude && restaurant.longitude
                          ? `https://www.google.com/maps/search/?api=1&query=${restaurant.latitude},${restaurant.longitude}`
                          : null;
                        
                        return (
                          <div
                            key={idx}
                            className="p-6 bg-white border-2 border-gray-200 rounded-xl hover:border-primary-300 hover:shadow-lg transition-all"
                          >
                            {/* Header: Name + Rating */}
                            <div className="flex items-start justify-between mb-4">
                              <div className="flex-1">
                                <h3 className="text-2xl font-bold text-gray-900 mb-1">
                                  {restaurant.name_en || restaurant.name || restaurant.name_th || 'Unknown Restaurant'}
                                </h3>
                                {restaurant.name_th && restaurant.name_en !== restaurant.name_th && (
                                  <p className="text-lg text-gray-600 mb-2">
                                    {restaurant.name_th}
                                  </p>
                                )}
                              </div>
                              {restaurant.rating && (
                                <div className="flex items-center gap-1 px-4 py-2 bg-yellow-100 text-yellow-700 rounded-full flex-shrink-0">
                                  <span className="text-xl">⭐</span>
                                  <span className="font-bold text-lg">{restaurant.rating}</span>
                                </div>
                              )}
                            </div>

                            {/* Info Grid */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
                              {/* Region */}
                              {(restaurant.region || restaurant.location) && (
                                <div className="flex items-center gap-2 text-gray-700">
                                  <span className="text-xl">📍</span>
                                  <span className="font-medium">{restaurant.region || restaurant.location}</span>
                                </div>
                              )}

                              {/* Price Range */}
                              {restaurant.price_range && (
                                <div className="flex items-center gap-2 text-gray-700">
                                  <span className="text-xl">💰</span>
                                  <span className="font-medium">{restaurant.price_range}</span>
                                </div>
                              )}

                              {/* Phone - Full width */}
                              {restaurant.phone && (
                                <div className="flex items-center gap-2 text-gray-700 md:col-span-2">
                                  <span className="text-xl">📞</span>
                                  <a 
                                    href={`tel:${restaurant.phone}`} 
                                    className="hover:text-primary-600 font-medium underline decoration-dotted"
                                  >
                                    {restaurant.phone}
                                  </a>
                                </div>
                              )}

                              {/* Opening Hours - Full width */}
                              {restaurant.opening_hours && (
                                <div className="flex items-start gap-2 text-gray-700 md:col-span-2">
                                  <span className="text-xl flex-shrink-0">🕐</span>
                                  <span className="font-medium">{restaurant.opening_hours}</span>
                                </div>
                              )}

                              {/* Website - Full width */}
                              {restaurant.website && (
                                <div className="flex items-center gap-2 text-gray-700 md:col-span-2">
                                  <span className="text-xl">🌐</span>
                                  <a 
                                    href={restaurant.website} 
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="hover:text-primary-600 font-medium underline decoration-dotted break-all"
                                  >
                                    {restaurant.website}
                                  </a>
                                </div>
                              )}
                            </div>

                            {/* Description (legacy field) */}
                            {restaurant.description && (
                              <p className="text-gray-700 mb-3">{restaurant.description}</p>
                            )}

                            {/* Specialty tags (legacy field) */}
                            {restaurant.specialty && typeof restaurant.specialty === 'string' && (
                              <div className="flex flex-wrap items-center gap-3 mb-4">
                                <span className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm font-medium">
                                  🍴 {restaurant.specialty}
                                </span>
                              </div>
                            )}

                            {/* Google Maps Button */}
                            {mapsUrl && (
                              <div className="mt-4 pt-4 border-t border-gray-200">
                                <a
                                  href={mapsUrl}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center gap-2 px-6 py-3 bg-primary-600 text-white font-medium rounded-lg hover:bg-primary-700 transition-colors shadow-md hover:shadow-lg"
                                >
                                  <span className="text-xl">🗺️</span>
                                  <span>Open in Google Maps</span>
                                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                  </svg>
                                </a>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Back to Home Button */}
          <div className="text-center">
            <button
              onClick={() => router.push('/')}
              className="btn btn-outline text-lg px-8 py-3"
            >
              🔄 Try Another Dish
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
