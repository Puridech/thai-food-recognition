'use client';

import { useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import '@/lib/i18n';

export default function Home() {
  const { t, i18n } = useTranslation();

  useEffect(() => {
    // Initialize i18n on client side
    console.log('i18n initialized:', i18n.language);
  }, [i18n]);

  const toggleLanguage = () => {
    const newLang = i18n.language === 'en' ? 'th' : 'en';
    i18n.changeLanguage(newLang);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-red-50 to-yellow-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="container-custom py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="text-4xl">🍜</div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  {t('appTitle')}
                </h1>
                <p className="text-sm text-gray-600">{t('appSubtitle')}</p>
              </div>
            </div>
            
            {/* Language Switcher */}
            <button
              onClick={toggleLanguage}
              className="btn btn-outline flex items-center gap-2"
            >
              <span className="text-lg">🌐</span>
              <span>{i18n.language === 'en' ? 'ไทย' : 'English'}</span>
            </button>
          </div>
        </div>
      </header>

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

          {/* Upload Section */}
          <div className="card p-8 mb-8">
            <div className="border-4 border-dashed border-gray-300 rounded-xl p-12 text-center hover:border-primary-400 transition-colors cursor-pointer">
              <div className="text-6xl mb-4">📸</div>
              <h3 className="text-2xl font-semibold text-gray-800 mb-2">
                {t('uploadImage')}
              </h3>
              <p className="text-gray-600 mb-4">{t('dragDrop')}</p>
              <p className="text-sm text-gray-500 mb-6">{t('or')}</p>
              <div className="flex gap-4 justify-center flex-wrap">
                <button className="btn btn-primary text-lg px-6 py-3">
                  📁 {t('uploadImage')}
                </button>
                <button className="btn btn-secondary text-lg px-6 py-3">
                  📷 {t('takePhoto')}
                </button>
              </div>
            </div>
          </div>

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
