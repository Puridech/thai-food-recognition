'use client';

import { useRouter } from 'next/navigation';
import { useTranslation } from 'react-i18next';

interface NavbarProps {
  showBackButton?: boolean;
  onBack?: () => void;
}

export default function Navbar({ showBackButton = false, onBack }: NavbarProps) {
  const router = useRouter();
  const { t, i18n } = useTranslation();

  const toggleLanguage = () => {
    const newLang = i18n.language === 'en' ? 'th' : 'en';
    i18n.changeLanguage(newLang);
  };

  const handleLogoClick = () => {
    if (onBack) {
      onBack();
    } else {
      router.push('/');
    }
  };

  return (
    <header className="bg-white shadow-sm sticky top-0 z-20">
      <div className="container-custom py-3 sm:py-4">
        <div className="flex items-center justify-between gap-2">
          {/* Left: Logo/Title (Clickable to go home) */}
          <button
            onClick={handleLogoClick}
            className="flex items-center gap-2 sm:gap-3 hover:opacity-80 transition-opacity min-w-0 flex-1 sm:flex-initial"
          >
            <div className="text-3xl sm:text-4xl flex-shrink-0">🍜</div>
            <div className="text-left min-w-0">
              <h1 className="text-lg sm:text-2xl font-bold text-gray-900 truncate">
                Thai Food Recognition
              </h1>
              <p className="text-xs sm:text-sm text-gray-600 hidden sm:block">
                Discover Thai Cuisine with AI
              </p>
            </div>
          </button>

          {/* Right: Actions (History + Language) */}
          <div className="flex items-center gap-2 sm:gap-3 flex-shrink-0">
            {/* History Button - Icon only on mobile */}
            <button
              onClick={() => router.push('/history')}
              className="btn btn-outline flex items-center gap-1 sm:gap-2 px-3 sm:px-4 py-2"
            >
              <span className="text-lg sm:text-xl">📜</span>
              <span className="hidden sm:inline text-sm sm:text-base">History</span>
            </button>

            {/* Language Switcher - Icon only on mobile */}
            <button
              onClick={toggleLanguage}
              className="btn btn-outline flex items-center gap-1 sm:gap-2 px-3 sm:px-4 py-2"
            >
              <span className="text-lg sm:text-xl">🌐</span>
              <span className="hidden sm:inline text-sm sm:text-base">
                {i18n.language === 'en' ? 'ไทย' : 'EN'}
              </span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
