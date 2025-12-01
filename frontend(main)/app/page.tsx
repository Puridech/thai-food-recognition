'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useTranslation } from 'react-i18next';
import '@/lib/i18n';
import Navbar from '@/components/layout/Navbar';
import ImageUpload from '@/components/features/ImageUpload';
import RecognitionResult from '@/components/features/RecognitionResult';
import { apiClient } from '@/lib/api-client';
import { addToHistory } from '@/lib/history';
import type { RecognitionResponse } from '@/types/api';

export default function Home() {
  const router = useRouter();
  const { t, i18n } = useTranslation();
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<RecognitionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  useEffect(() => {
    // Initialize i18n on client side
    console.log('i18n initialized:', i18n.language);
  }, [i18n]);

  // Compress and convert image to base64
  const compressImage = async (file: File, maxWidth: number = 400, quality: number = 0.7): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          // Create canvas
          const canvas = document.createElement('canvas');
          
          // Calculate new dimensions (maintain aspect ratio)
          let width = img.width;
          let height = img.height;
          
          if (width > maxWidth) {
            height = (height * maxWidth) / width;
            width = maxWidth;
          }
          
          canvas.width = width;
          canvas.height = height;
          
          // Draw and compress
          const ctx = canvas.getContext('2d');
          if (!ctx) {
            reject(new Error('Failed to get canvas context'));
            return;
          }
          
          ctx.drawImage(img, 0, 0, width, height);
          
          // Convert to compressed base64
          const compressedBase64 = canvas.toDataURL('image/jpeg', quality);
          resolve(compressedBase64);
        };
        img.onerror = () => reject(new Error('Failed to load image'));
        img.src = e.target?.result as string;
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsDataURL(file);
    });
  };

  // Handle image selection and recognition
  const handleImageSelect = async (file: File) => {
    setSelectedFile(file);
    setError(null);
    setResult(null);
    setIsLoading(true);

    try {
      // Compress image to reduce storage size
      const imageBase64 = await compressImage(file, 400, 0.7);
      console.log('📸 Image compressed:', {
        original: file.size,
        compressed: imageBase64.length,
        reduction: `${Math.round((1 - imageBase64.length / file.size) * 100)}%`
      });

      // Call recognition API
      const response = await apiClient.recognizeFood(file);
      setResult(response);

      // Add to history if successful
      if (response.success && response.food_name) {
        try {
          addToHistory({
            food_name: response.food_name,
            image_url: imageBase64,
            confidence: response.confidence,
            layer_used: response.layer_used,
          });

          // Dispatch event to update other components
          window.dispatchEvent(new Event('historyUpdated'));
          console.log('✅ Added to history:', response.food_name);
        } catch (historyError) {
          console.error('❌ Failed to save to history:', historyError);
          // Don't show error to user - main recognition succeeded
        }
      }
    } catch (err: unknown) {
      console.error('Recognition error:', err);
      setError((err as Error).message || t('errorOccurred'));
    } finally {
      setIsLoading(false);
    }
  };

  // Clear all and start over
  const handleClear = () => {
    setSelectedFile(null);
    setResult(null);
    setError(null);
  };

  // View full details (navigate to detail page)
  const handleViewDetails = () => {
    if (result) {
      // Ensure food_name is in correct format (lowercase with underscores)
      const foodName = result.food_name.toLowerCase();
      router.push(`/food/${foodName}`);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-red-50 to-yellow-50">
      {/* Navbar */}
      <Navbar />

      {/* Main Content */}
      <main className="container-custom py-12">
        <div className="max-w-4xl mx-auto">
          {/* Hero Section */}
          <div className="text-center mb-12">
            <div className="text-8xl mb-6 animate-bounce">🍛</div>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              {t('appTitle')}
            </h2>
            <p className="text-xl text-gray-600">
              {t('appSubtitle')}
            </p>
          </div>

          {/* Upload Section or Result */}
          <div className="mb-8">
            {!result ? (
              <ImageUpload
                onImageSelect={handleImageSelect}
                onClear={handleClear}
                isLoading={isLoading}
                maxSizeMB={10}
              />
            ) : (
              <RecognitionResult
                result={result}
                onViewDetails={handleViewDetails}
              />
            )}
          </div>

          {/* Error Display */}
          {error && (
            <div className="card p-6 bg-red-50 border-2 border-red-200 mb-8">
              <div className="flex items-center gap-3">
                <div className="text-4xl">⚠️</div>
                <div>
                  <h3 className="text-xl font-bold text-red-800">
                    {t('error')}
                  </h3>
                  <p className="text-red-700">{error}</p>
                  <button
                    onClick={handleClear}
                    className="mt-3 btn btn-outline border-red-600 text-red-600 hover:bg-red-50"
                  >
                    {t('tryAgain')}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Try Another Button (when showing result) */}
          {result && (
            <div className="text-center mb-8">
              <button
                onClick={handleClear}
                className="btn btn-outline text-lg px-8 py-3"
              >
                🔄 {t('tryAnother')}
              </button>
            </div>
          )}

          {/* Features */}
          <div className="grid md:grid-cols-3 gap-6">
            <div className="card p-6 text-center hover:shadow-xl transition-shadow">
              <div className="text-5xl mb-4">🎯</div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">
                AI Recognition
              </h3>
              <p className="text-gray-600">
                96% accuracy with hybrid 2-layer AI system
              </p>
            </div>
            
            <div className="card p-6 text-center hover:shadow-xl transition-shadow">
              <div className="text-5xl mb-4">📖</div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">
                {t('culturalStory')}
              </h3>
              <p className="text-gray-600">
                Learn the history and traditions behind each dish
              </p>
            </div>
            
            <div className="card p-6 text-center hover:shadow-xl transition-shadow">
              <div className="text-5xl mb-4">👨‍🍳</div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">
                {t('recipe')}
              </h3>
              <p className="text-gray-600">
                Authentic recipes with step-by-step instructions
              </p>
            </div>
          </div>

          {/* Supported Dishes Preview */}
          <div className="mt-12 text-center">
            <h3 className="text-2xl font-bold text-gray-800 mb-6">
              Supported Thai Dishes (20 Menus)
            </h3>
            <div className="flex flex-wrap justify-center gap-3">
              {[
                '🍜 Pad Thai',
                '🍲 Tom Yum Goong',
                '🥗 Som Tam',
                '🍛 Green Curry',
                '🍗 Gai Yang',
                '🥘 Massaman Curry',
                '🍚 Khao Pad',
                '🍜 Pad Krapow',
              ].map((dish, index) => (
                <span
                  key={index}
                  className="px-4 py-2 bg-white rounded-full shadow-md text-gray-700 font-medium"
                >
                  {dish}
                </span>
              ))}
              <span className="px-4 py-2 bg-primary-100 text-primary-700 rounded-full shadow-md font-medium">
                + 12 more...
              </span>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="container-custom py-6 text-center text-gray-600">
          <p>
            {t('madeWith')} ❤️ {t('by')} Hokkaido Information University
          </p>
          <p className="text-sm mt-2">
            8-Week Internship Project • 2024-2025
          </p>
        </div>
      </footer>
    </div>
  );
}
