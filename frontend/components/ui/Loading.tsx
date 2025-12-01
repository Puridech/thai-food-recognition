'use client';

import { useTranslation } from 'react-i18next';

interface LoadingProps {
  message?: string;
  size?: 'sm' | 'md' | 'lg';
}

export default function Loading({ message, size = 'md' }: LoadingProps) {
  const { t } = useTranslation();

  const sizeClasses = {
    sm: 'text-3xl',
    md: 'text-5xl',
    lg: 'text-7xl',
  };

  return (
    <div className="flex flex-col items-center justify-center p-8">
      <div className={`animate-spin ${sizeClasses[size]} mb-4`}>⏳</div>
      <p className="text-lg font-semibold text-gray-700">
        {message || t('pleaseWait')}
      </p>
      <div className="flex gap-1 mt-3">
        <div className="w-2 h-2 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
        <div className="w-2 h-2 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
        <div className="w-2 h-2 bg-primary-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
      </div>
    </div>
  );
}
