'use client';

import { FiX } from 'react-icons/fi';

interface ErrorMessageProps {
  title?: string;
  message: string;
  onClose?: () => void;
  onRetry?: () => void;
}

export default function ErrorMessage({
  title = 'Error',
  message,
  onClose,
  onRetry,
}: ErrorMessageProps) {
  return (
    <div className="card p-6 bg-red-50 border-2 border-red-200 animate-fadeIn">
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 text-4xl">⚠️</div>
        
        <div className="flex-1">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-xl font-bold text-red-800">{title}</h3>
            {onClose && (
              <button
                onClick={onClose}
                className="text-red-600 hover:text-red-800 transition-colors"
              >
                <FiX className="text-xl" />
              </button>
            )}
          </div>
          
          <p className="text-red-700 mb-4">{message}</p>
          
          {onRetry && (
            <button
              onClick={onRetry}
              className="btn btn-outline border-red-600 text-red-600 hover:bg-red-50"
            >
              🔄 Try Again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
