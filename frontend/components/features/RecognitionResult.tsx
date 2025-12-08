'use client';

import { useTranslation } from 'react-i18next';
import { FiCheckCircle, FiClock, FiLayers, FiTrendingUp } from 'react-icons/fi';
import type { RecognitionResponse } from '@/types/api';

interface RecognitionResultProps {
  result: RecognitionResponse;
  onViewDetails: () => void;
}

export default function RecognitionResult({
  result,
  onViewDetails,
}: RecognitionResultProps) {
  const { t } = useTranslation();

  // Format food name for display
  const formatFoodName = (name: string): string => {
    return name
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Get confidence color
  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.9) return 'text-green-600';
    if (confidence >= 0.75) return 'text-yellow-600';
    return 'text-orange-600';
  };

  // Get confidence background
  const getConfidenceBg = (confidence: number): string => {
    if (confidence >= 0.9) return 'bg-green-100';
    if (confidence >= 0.75) return 'bg-yellow-100';
    return 'bg-orange-100';
  };

  return (
    <div className="card p-6 animate-fadeIn">
      {/* Success Header */}
      <div className="flex items-center gap-3 mb-6">
        <div className="text-4xl">✅</div>
        <div>
          <h3 className="text-2xl font-bold text-gray-900">
            {t('recognitionResult')}
          </h3>
          <p className="text-gray-600">
            {result.success ? t('successfullyIdentified') : t('recognitionResult')}
          </p>
        </div>
      </div>

      {/* Food Name */}
      <div className="mb-6 p-4 bg-gradient-to-r from-primary-50 to-orange-50 rounded-lg border-2 border-primary-200">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-3xl">🍜</span>
          <h4 className="text-lg font-semibold text-gray-700">
            {t('recognizedAs')}
          </h4>
        </div>
        <p className="text-3xl font-bold text-primary-700">
          {formatFoodName(result.food_name)}
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        {/* Confidence */}
        <div className={`p-4 rounded-lg ${getConfidenceBg(result.confidence)}`}>
          <div className="flex items-center gap-2 mb-2">
            <FiTrendingUp className={getConfidenceColor(result.confidence)} />
            <span className="text-sm font-medium text-gray-700">
              {t('confidence')}
            </span>
          </div>
          <p className={`text-2xl font-bold ${getConfidenceColor(result.confidence)}`}>
            {(result.confidence * 100).toFixed(1)}%
          </p>
        </div>

        {/* Layer Used */}
        <div className="p-4 bg-blue-100 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <FiLayers className="text-blue-600" />
            <span className="text-sm font-medium text-gray-700">
              AI Layer
            </span>
          </div>
          <p className="text-2xl font-bold text-blue-600">
            Layer {result.layer_used}
          </p>
          <p className="text-xs text-blue-600 mt-1">
            {result.layer_used === 1 ? 'Fast Baseline' : 'Thai Specialist'}
          </p>
        </div>

        {/* Processing Time */}
        <div className="p-4 bg-purple-100 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <FiClock className="text-purple-600" />
            <span className="text-sm font-medium text-gray-700">
              {t('processingTime')}
            </span>
          </div>
          <p className="text-2xl font-bold text-purple-600">
            {result.processing_time.toFixed(2)}s
          </p>
        </div>

        {/* Status */}
        <div className="p-4 bg-green-100 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <FiCheckCircle className="text-green-600" />
            <span className="text-sm font-medium text-gray-700">
              {t('status')}
            </span>
          </div>
          <p className="text-lg font-bold text-green-600">
            ✓ {t('success')}
          </p>
        </div>
      </div>

      {/* Decision Info */}
      {result.decision && (
        <div className="mb-6 p-3 bg-gray-50 rounded-lg border border-gray-200">
          <p className="text-sm text-gray-700">
            <span className="font-semibold">Decision:</span> {result.decision}
          </p>
        </div>
      )}

      {/* Action Button */}
      <button
        onClick={onViewDetails}
        className="w-full btn btn-primary text-lg py-4 flex items-center justify-center gap-2"
      >
        <span className="text-2xl">📖</span>
        <span>{t('viewFullDetails')}</span>
      </button>

      {/* Info cards */}
      <div className="mt-6 grid grid-cols-3 gap-3">
        <button 
          onClick={() => {
            onViewDetails();
            // Scroll to story tab after navigation
            setTimeout(() => {
              const storyTab = document.querySelector('[data-tab="story"]') as HTMLElement;
              if (storyTab) storyTab.click();
            }, 100);
          }}
          className="p-3 bg-white border-2 border-gray-200 hover:border-primary-300 hover:bg-primary-50 rounded-lg transition-all text-center cursor-pointer"
        >
          <div className="text-2xl mb-1">📖</div>
          <p className="text-xs font-medium text-gray-700">{t('culturalStory')}</p>
        </button>
        
        <button 
          onClick={() => {
            onViewDetails();
            // Scroll to recipe tab after navigation
            setTimeout(() => {
              const recipeTab = document.querySelector('[data-tab="recipe"]') as HTMLElement;
              if (recipeTab) recipeTab.click();
            }, 100);
          }}
          className="p-3 bg-white border-2 border-gray-200 hover:border-primary-300 hover:bg-primary-50 rounded-lg transition-all text-center cursor-pointer"
        >
          <div className="text-2xl mb-1">👨‍🍳</div>
          <p className="text-xs font-medium text-gray-700">{t('recipe')}</p>
        </button>
        
        <button 
          onClick={() => {
            onViewDetails();
            // Scroll to restaurants tab after navigation
            setTimeout(() => {
              const restaurantsTab = document.querySelector('[data-tab="restaurants"]') as HTMLElement;
              if (restaurantsTab) restaurantsTab.click();
            }, 100);
          }}
          className="p-3 bg-white border-2 border-gray-200 hover:border-primary-300 hover:bg-primary-50 rounded-lg transition-all text-center cursor-pointer"
        >
          <div className="text-2xl mb-1">🏪</div>
          <p className="text-xs font-medium text-gray-700">{t('restaurants')}</p>
        </button>
      </div>
    </div>
  );
}
