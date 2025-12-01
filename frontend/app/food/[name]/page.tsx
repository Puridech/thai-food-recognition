'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { useTranslation } from 'react-i18next';
import { FiArrowLeft } from 'react-icons/fi';
import '@/lib/i18n';
import Navbar from '@/components/layout/Navbar';
import { apiClient } from '@/lib/api-client';
import { getFoodImage } from '@/lib/food-images';
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
        // Load food image from predefined images
        const imageData = getFoodImage(normalizedFoodName);
        if (imageData.type === 'image') {
          setFoodImage(imageData.value);
        }

        // Fetch food info using normalized name
        const info = await apiClient.getFoodInfo(normalizedFoodName, i18n.language);
        console.log('📊 Food Info Response:', JSON.stringify(info, null, 2));
        
        // Debug: Check recipe structure
        if (info.recipe) {
          console.log('🍳 Recipe has ingredients_by_section?', !!info.recipe.ingredients_by_section);
          console.log('🍳 Recipe has categorized_ingredients?', !!info.recipe.categorized_ingredients);
          console.log('🍳 Recipe has ingredients?', !!info.recipe.ingredients);
          if (info.recipe.ingredients_by_section) {
            console.log('🍳 Ingredients by section:', info.recipe.ingredients_by_section);
          }
          if (info.recipe.categorized_ingredients) {
            console.log('🍳 Categorized ingredients:', info.recipe.categorized_ingredients);
          }
          if (info.recipe.ingredients) {
            console.log('🍳 Flat ingredients:', info.recipe.ingredients);
          }
        }
        
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
                      
                      {/* General Info Cards - New Format with Beautiful Icons */}
                      {foodInfo.cultural_story.general_info && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8 not-prose">
                          {Object.entries(foodInfo.cultural_story.general_info).map(([key, value]) => {
                            const displayKey = key
                              .replace(/_/g, ' ')
                              .split(' ')
                              .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                              .join(' ');
                            
                            // Icon and color mapping based on key
                            const getCardStyle = (k: string) => {
                              if (k.includes('region') || k.includes('origin')) {
                                return {
                                  icon: '🌍',
                                  gradient: 'from-blue-50 to-cyan-50',
                                  border: 'border-blue-200',
                                  label: 'text-blue-600',
                                  value: 'text-blue-900'
                                };
                              }
                              if (k.includes('taste') || k.includes('flavor')) {
                                return {
                                  icon: '😋',
                                  gradient: 'from-orange-50 to-amber-50',
                                  border: 'border-orange-200',
                                  label: 'text-orange-600',
                                  value: 'text-orange-900'
                                };
                              }
                              if (k.includes('ingredient')) {
                                return {
                                  icon: '🥘',
                                  gradient: 'from-green-50 to-emerald-50',
                                  border: 'border-green-200',
                                  label: 'text-green-600',
                                  value: 'text-green-900'
                                };
                              }
                              if (k.includes('serving') || k.includes('style')) {
                                return {
                                  icon: '🍽️',
                                  gradient: 'from-purple-50 to-pink-50',
                                  border: 'border-purple-200',
                                  label: 'text-purple-600',
                                  value: 'text-purple-900'
                                };
                              }
                              if (k.includes('price')) {
                                return {
                                  icon: '💵',
                                  gradient: 'from-emerald-50 to-teal-50',
                                  border: 'border-emerald-200',
                                  label: 'text-emerald-600',
                                  value: 'text-emerald-900'
                                };
                              }
                              if (k.includes('history') || k.includes('born')) {
                                return {
                                  icon: '📅',
                                  gradient: 'from-amber-50 to-yellow-50',
                                  border: 'border-amber-200',
                                  label: 'text-amber-600',
                                  value: 'text-amber-900'
                                };
                              }
                              if (k.includes('difficulty')) {
                                return {
                                  icon: '⚡',
                                  gradient: 'from-violet-50 to-purple-50',
                                  border: 'border-violet-200',
                                  label: 'text-violet-600',
                                  value: 'text-violet-900'
                                };
                              }
                              // Default style
                              return {
                                icon: '📋',
                                gradient: 'from-gray-50 to-slate-50',
                                border: 'border-gray-200',
                                label: 'text-gray-600',
                                value: 'text-gray-900'
                              };
                            };

                            const style = getCardStyle(key);
                            
                            return (
                              <div 
                                key={key} 
                                className={`p-5 bg-gradient-to-br ${style.gradient} rounded-xl border-2 ${style.border} shadow-sm hover:shadow-md transition-shadow`}
                              >
                                <div className="flex items-start gap-3">
                                  <span className="text-3xl flex-shrink-0">{style.icon}</span>
                                  <div className="flex-1 min-w-0">
                                    <div className={`text-sm font-semibold mb-1.5 ${style.label}`}>
                                      {displayKey}
                                    </div>
                                    <div className={`text-base font-bold ${style.value} leading-snug`}>
                                      {String(value)}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}

                      {/* Story Content - New Format with Border */}
                      {foodInfo.cultural_story.story && (
                        <div className="border-l-4 border-amber-400 pl-6 pr-4 py-4 bg-gradient-to-r from-amber-50/50 to-transparent rounded-r-lg mb-6">
                          <h3 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-3">
                            <span className="text-3xl">📖</span>
                            <span>Story</span>
                          </h3>
                          <div className="space-y-4">
                            {foodInfo.cultural_story.story.split('\n\n').map((paragraph, idx) => (
                              <p key={idx} className="text-gray-700 leading-relaxed text-base">
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

                      {/* Recipe Info - Beautiful Cards */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                        {foodInfo.recipe.servings && (
                          <div className="p-5 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-xl border-2 border-blue-200 text-center shadow-sm hover:shadow-md transition-shadow">
                            <div className="text-3xl mb-2">👥</div>
                            <div className="text-xs font-semibold text-blue-600 mb-1">{t('servings')}</div>
                            <div className="font-bold text-blue-900 text-lg">{foodInfo.recipe.servings}</div>
                          </div>
                        )}

                        {foodInfo.recipe.prep_time && (
                          <div className="p-5 bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl border-2 border-green-200 text-center shadow-sm hover:shadow-md transition-shadow">
                            <div className="text-3xl mb-2">⏱️</div>
                            <div className="text-xs font-semibold text-green-600 mb-1">{t('prepTime')}</div>
                            <div className="font-bold text-green-900 text-lg">
                              {foodInfo.recipe.prep_time}
                              {!foodInfo.recipe.prep_time.includes('นาที') && 
                               !foodInfo.recipe.prep_time.toLowerCase().includes('min') && 
                               !foodInfo.recipe.prep_time.toLowerCase().includes('minute') && (
                                <span className="text-sm font-normal ml-1">{t('minutes')}</span>
                              )}
                            </div>
                          </div>
                        )}

                        {foodInfo.recipe.cook_time && (
                          <div className="p-5 bg-gradient-to-br from-orange-50 to-amber-50 rounded-xl border-2 border-orange-200 text-center shadow-sm hover:shadow-md transition-shadow">
                            <div className="text-3xl mb-2">🔥</div>
                            <div className="text-xs font-semibold text-orange-600 mb-1">{t('cookTime')}</div>
                            <div className="font-bold text-orange-900 text-lg">
                              {foodInfo.recipe.cook_time}
                              {!foodInfo.recipe.cook_time.includes('นาที') && 
                               !foodInfo.recipe.cook_time.toLowerCase().includes('min') && 
                               !foodInfo.recipe.cook_time.toLowerCase().includes('minute') && (
                                <span className="text-sm font-normal ml-1">{t('minutes')}</span>
                              )}
                            </div>
                          </div>
                        )}

                        {(foodInfo.recipe.difficulty || foodInfo.recipe.difficulty_text) && (
                          <div className="p-5 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border-2 border-purple-200 text-center shadow-sm hover:shadow-md transition-shadow">
                            <div className="text-3xl mb-2">📊</div>
                            <div className="text-xs font-semibold text-purple-600 mb-1">{t('difficulty')}</div>
                            <div className="font-bold text-purple-900 text-lg capitalize">
                              {foodInfo.recipe.difficulty_text || 
                               (typeof foodInfo.recipe.difficulty === 'number' 
                                 ? `Level ${foodInfo.recipe.difficulty}`
                                 : String(foodInfo.recipe.difficulty))}
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Ingredients - Support multiple formats */}
                      {(foodInfo.recipe.ingredients_by_section || 
                        foodInfo.recipe.categorized_ingredients || 
                        foodInfo.recipe.ingredients) && (
                        <div className="mb-8">
                          {/* Format 1: ingredients_by_section (from Backend API) */}
                          {foodInfo.recipe.ingredients_by_section ? (
                            <div className="space-y-6">
                              {Object.entries(foodInfo.recipe.ingredients_by_section).map(([sectionName, items], idx) => {
                                // Determine icon and color based on section name
                                const isEquipment = 
                                  sectionName.toLowerCase().includes('equipment') ||
                                  sectionName.includes('อุปกรณ์') || 
                                  sectionName.includes('เครื่องมือ');
                                
                                const icon = isEquipment ? '🔪' : '🥘';
                                const borderColor = isEquipment ? 'border-purple-400' : 'border-red-400';
                                const bgGradient = isEquipment ? 'from-purple-50/50' : 'from-red-50/50';
                                const hoverBg = isEquipment ? 'hover:bg-purple-50/50' : 'hover:bg-red-50/50';
                                const bulletColor = isEquipment ? 'text-purple-500' : 'text-red-500';
                                
                                return (
                                  <div 
                                    key={idx}
                                    className={`border-l-4 ${borderColor} pl-6 pr-4 py-4 bg-gradient-to-r ${bgGradient} to-transparent rounded-r-xl`}
                                  >
                                    <h3 className="text-2xl font-bold text-gray-900 mb-5 flex items-center gap-3">
                                      <span className="text-3xl">{icon}</span>
                                      <span>{sectionName}</span>
                                      {!isEquipment && foodInfo.recipe?.servings && (
                                        <span className="text-sm font-normal text-gray-500">
                                          ({foodInfo.recipe.servings} {i18n.language === 'th' ? 'ท่าน' : 'servings'})
                                        </span>
                                      )}
                                    </h3>
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-3">
                                      {items.map((item, itemIdx) => (
                                        <div
                                          key={itemIdx}
                                          className={`flex items-start gap-3 text-gray-700 p-2 ${hoverBg} rounded-lg transition-colors`}
                                        >
                                          <span className={`${bulletColor} text-lg mt-0.5`}>•</span>
                                          <span className="flex-1 leading-relaxed">{item}</span>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          ) : foodInfo.recipe.categorized_ingredients && foodInfo.recipe.categorized_ingredients.length > 0 ? (
                            /* Format 2: categorized_ingredients (legacy format) */
                            <div className="space-y-6">
                              {foodInfo.recipe.categorized_ingredients.map((category, catIdx) => {
                                const isEquipment = category.category.includes('อุปกรณ์') || 
                                                   category.category.toLowerCase().includes('equipment');
                                const icon = isEquipment ? '🔪' : '🥘';
                                const borderColor = isEquipment ? 'border-purple-400' : 'border-red-400';
                                const bgGradient = isEquipment ? 'from-purple-50/50' : 'from-red-50/50';
                                const hoverBg = isEquipment ? 'hover:bg-purple-50/50' : 'hover:bg-red-50/50';
                                const bulletColor = isEquipment ? 'text-purple-500' : 'text-red-500';
                                
                                return (
                                  <div 
                                    key={catIdx}
                                    className={`border-l-4 ${borderColor} pl-6 pr-4 py-4 bg-gradient-to-r ${bgGradient} to-transparent rounded-r-xl`}
                                  >
                                    <h3 className="text-2xl font-bold text-gray-900 mb-5 flex items-center gap-3">
                                      <span className="text-3xl">{icon}</span>
                                      <span>{category.category}</span>
                                      {category.category.includes('วัตถุดิบ') && foodInfo.recipe?.servings && (
                                        <span className="text-sm font-normal text-gray-500">
                                          ({foodInfo.recipe.servings} ท่าน)
                                        </span>
                                      )}
                                    </h3>
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-3">
                                      {category.items.map((item, itemIdx) => (
                                        <div
                                          key={itemIdx}
                                          className={`flex items-start gap-3 text-gray-700 p-2 ${hoverBg} rounded-lg transition-colors`}
                                        >
                                          <span className={`${bulletColor} text-lg mt-0.5`}>•</span>
                                          <span className="flex-1 leading-relaxed">{item}</span>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          ) : (
                            /* Format 3: Fallback - flat ingredients array */
                            foodInfo.recipe.ingredients && foodInfo.recipe.ingredients.length > 0 && (
                              <div className="border-l-4 border-red-400 pl-6 pr-4 py-4 bg-gradient-to-r from-red-50/50 to-transparent rounded-r-xl">
                                <h3 className="text-2xl font-bold text-gray-900 mb-5 flex items-center gap-3">
                                  <span className="text-3xl">🥘</span>
                                  <span>{t('ingredients')}</span>
                                  <span className="text-sm font-normal text-gray-500">
                                    ({foodInfo.recipe.servings || 4} servings)
                                  </span>
                                </h3>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-3">
                                  {foodInfo.recipe.ingredients.map((ingredient, idx) => (
                                    <div
                                      key={idx}
                                      className="flex items-start gap-3 text-gray-700 p-2 hover:bg-red-50/50 rounded-lg transition-colors"
                                    >
                                      <span className="text-red-500 text-lg mt-0.5">•</span>
                                      <span className="flex-1 leading-relaxed">{ingredient}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )
                          )}
                        </div>
                      )}

                      {/* Instructions/Steps - Beautiful Numbered List */}
                      {foodInfo.recipe.steps && foodInfo.recipe.steps.length > 0 && (
                        <div className="mb-8 border-l-4 border-blue-400 pl-6 pr-4 py-4 bg-gradient-to-r from-blue-50/50 to-transparent rounded-r-xl">
                          <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-3">
                            <span className="text-3xl">📝</span>
                            <span>{t('instructions')}</span>
                          </h3>
                          <div className="space-y-5">
                            {foodInfo.recipe.steps.map((step, idx) => (
                              <div
                                key={idx}
                                className="flex gap-4 p-5 bg-white rounded-xl border-2 border-blue-100 hover:border-blue-300 hover:shadow-md transition-all"
                              >
                                {/* Step Number Circle */}
                                <div className="flex-shrink-0">
                                  <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-full flex items-center justify-center font-bold text-xl shadow-lg">
                                    {idx + 1}
                                  </div>
                                </div>
                                
                                {/* Step Content */}
                                <div className="flex-1 min-w-0">
                                  {step.title && (
                                    <h4 className="font-bold text-gray-900 mb-2 text-lg">
                                      {step.title}
                                    </h4>
                                  )}
                                  <p className="text-gray-700 leading-relaxed">
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

                              {/* Phone - Text only (not link) */}
                              {restaurant.phone && (
                                <div className="flex items-center gap-2 text-gray-700 md:col-span-2">
                                  <span className="text-xl">📞</span>
                                  <span className="font-medium">{restaurant.phone}</span>
                                </div>
                              )}

                              {/* Opening Hours - Multi-line display */}
                              {restaurant.opening_hours && (
                                <div className="flex items-start gap-2 text-gray-700 md:col-span-2">
                                  <span className="text-xl flex-shrink-0 mt-0.5">🕐</span>
                                  <div className="font-medium leading-relaxed whitespace-pre-line">
                                    {restaurant.opening_hours}
                                  </div>
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
