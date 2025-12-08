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

// Supported Dishes Component
function SupportedDishes() {
  const { t, i18n } = useTranslation();
  const [showAll, setShowAll] = useState(false);

  const allDishes = [
    { emoji: '🥗', en: 'Som Tam', th: 'ส้มตำ', name: 'som_tam' },
    { emoji: '🍲', en: 'Tom Yum Goong', th: 'ต้มยำกุ้ง', name: 'tom_yum_goong' },
    { emoji: '🥙', en: 'Larb', th: 'ลาบ', name: 'larb' },
    { emoji: '🍜', en: 'Pad Thai', th: 'ผัดไทย', name: 'pad_thai' },
    { emoji: '🍛', en: 'Green Curry', th: 'แกงเขียวหวาน', name: 'kaeng_khiao_wan' },
    { emoji: '🍜', en: 'Khao Soi', th: 'ข้าวซอย', name: 'khao_soi' },
    { emoji: '🍛', en: 'Massaman Curry', th: 'แกงมัสมั่น', name: 'kaeng_matsaman' },
    { emoji: '🍛', en: 'Pad Kra Pao', th: 'ผัดกะเพรา', name: 'pad_kra_pao' },
    { emoji: '🍗', en: 'Khao Man Gai', th: 'ข้าวมันไก่', name: 'khao_man_gai' },
    { emoji: '🍖', en: 'Khao Kha Mu', th: 'ข้าวขาหมู', name: 'khao_kha_mu' },
    { emoji: '🥥', en: 'Tom Kha Gai', th: 'ต้มข่าไก่', name: 'tom_kha_gai' },
    { emoji: '🥜', en: 'Thai Cashew Chicken', th: 'ไก่ผัดเม็ดมะม่วง', name: 'gai_pad_med_ma_muang_himmaphan' },
    { emoji: '🥚', en: 'Kai Palo', th: 'ไข่พะโล้', name: 'kai_palo' },
    { emoji: '🦐', en: 'Gung Ob Woon Sen', th: 'กุ้งอบวุ้นเส้น', name: 'gung_ob_woon_sen' },
    { emoji: '🍚', en: 'Khao Kluk Kapi', th: 'ข้าวคลุกกะปิ', name: 'khao_kluk_kapi' },
    { emoji: '🥟', en: 'Por Pia Tod', th: 'ปอเปี๊ยะทอด', name: 'por_pia_tod' },
    { emoji: '🫔', en: 'Hor Mok', th: 'ห่อหมก', name: 'hor_mok' },
    { emoji: '🥭', en: 'Mango Sticky Rice', th: 'ข้าวเหนียวมะม่วง', name: 'khao_niao_mamuang' },
    { emoji: '🥞', en: 'Khanom Krok', th: 'ขนมครก', name: 'khanom_krok' },
    { emoji: '🍯', en: 'Foi Thong', th: 'ฝอยทอง', name: 'foi_thong' },
  ];

  const visibleDishes = showAll ? allDishes : allDishes.slice(0, 8);

  return (
    <div>
      <div className="flex flex-wrap justify-center gap-3">
        {visibleDishes.map((dish, index) => (
          <span
            key={index}
            className="px-4 py-2 bg-white rounded-full shadow-md text-gray-700 font-medium hover:shadow-lg transition-shadow cursor-default"
          >
            {dish.emoji} {i18n.language === 'th' ? dish.th : dish.en}
          </span>
        ))}
        {!showAll && (
          <button
            onClick={() => setShowAll(true)}
            className="px-4 py-2 bg-primary-100 text-primary-700 rounded-full shadow-md font-medium hover:bg-primary-200 transition-colors"
          >
            +{allDishes.length - 8} {t('showMore')}
          </button>
        )}
      </div>
      {showAll && (
        <button
          onClick={() => setShowAll(false)}
          className="mt-4 px-6 py-2 text-gray-600 hover:text-gray-800 transition-colors"
        >
          {i18n.language === 'th' ? 'แสดงน้อยลง' : 'Show Less'}
        </button>
      )}
    </div>
  );
}

export default function Home() {
  const router = useRouter();
  const { t, i18n } = useTranslation();
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<RecognitionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

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
                {t('aiRecognition')}
              </h3>
              <p className="text-gray-600">
                {t('aiRecognitionDesc')}
              </p>
            </div>
            
            <div className="card p-6 text-center hover:shadow-xl transition-shadow">
              <div className="text-5xl mb-4">📖</div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">
                {t('culturalStories')}
              </h3>
              <p className="text-gray-600">
                {t('culturalStoriesDesc')}
              </p>
            </div>
            
            <div className="card p-6 text-center hover:shadow-xl transition-shadow">
              <div className="text-5xl mb-4">👨‍🍳</div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">
                {t('authenticRecipes')}
              </h3>
              <p className="text-gray-600">
                {t('authenticRecipesDesc')}
              </p>
            </div>
          </div>

          {/* Supported Dishes Preview */}
          <div className="mt-12 text-center">
            <h3 className="text-2xl font-bold text-gray-800 mb-6">
              {t('supportedDishes')}
            </h3>
            <SupportedDishes />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="container-custom py-6 text-center text-gray-600">
          <p>
            ⛩️{t('madeWith')} {t('by')} Puridech Khunpring⛩️
          </p>
          <p className="text-sm mt-2">
            ⛩️13-Week Internship Project at Hokkaido Information University⛩️
          </p>
        </div>
      </footer>
    </div>
  );
}
