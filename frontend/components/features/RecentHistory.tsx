'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { FiClock, FiTrash2, FiChevronRight } from 'react-icons/fi';
import { getRecentHistory, removeFromHistory, formatRelativeTime, formatFoodName, type HistoryItem } from '@/lib/history';

interface RecentHistoryProps {
  limit?: number;
  showViewAll?: boolean;
}

export default function RecentHistory({ limit = 5, showViewAll = true }: RecentHistoryProps) {
  const router = useRouter();
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [isClient, setIsClient] = useState(false);

  // Load history on client side only
  useEffect(() => {
    setIsClient(true);
    loadHistory();

    // Listen for storage changes (when new item added)
    const handleStorageChange = () => {
      loadHistory();
    };

    window.addEventListener('storage', handleStorageChange);
    window.addEventListener('historyUpdated', handleStorageChange);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('historyUpdated', handleStorageChange);
    };
  }, []);

  const loadHistory = () => {
    const recentHistory = getRecentHistory(limit);
    setHistory(recentHistory);
  };

  const handleRemove = (id: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent navigation
    
    if (confirm('Remove this item from history?')) {
      removeFromHistory(id);
      loadHistory();
      
      // Dispatch custom event to notify other components
      window.dispatchEvent(new Event('historyUpdated'));
    }
  };

  const handleItemClick = (item: HistoryItem) => {
    router.push(`/food/${encodeURIComponent(item.food_name)}`);
  };

  // Don't render on server (localStorage not available)
  if (!isClient) {
    return null;
  }

  // No history
  if (history.length === 0) {
    return (
      <div className="text-center py-8 px-4 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
        <FiClock className="text-4xl text-gray-400 mx-auto mb-2" />
        <p className="text-gray-600 font-medium">No search history yet</p>
        <p className="text-sm text-gray-500 mt-1">Upload an image to start recognizing Thai food!</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
          <FiClock className="text-primary-600" />
          Recent History ({history.length})
        </h2>
        {showViewAll && history.length > 0 && (
          <button
            onClick={() => router.push('/history')}
            className="text-primary-600 hover:text-primary-700 font-medium text-sm flex items-center gap-1 transition-colors"
          >
            View All
            <FiChevronRight />
          </button>
        )}
      </div>

      {/* History Items */}
      <div className="space-y-3">
        {history.map((item) => (
          <div
            key={item.id}
            onClick={() => handleItemClick(item)}
            className="flex items-center gap-4 p-4 bg-white border-2 border-gray-200 rounded-lg hover:border-primary-300 hover:shadow-md transition-all cursor-pointer group"
          >
            {/* Thumbnail */}
            <div className="flex-shrink-0 w-20 h-20 rounded-lg overflow-hidden bg-gray-100 border border-gray-200">
              <img
                src={item.image_url}
                alt={formatFoodName(item.food_name)}
                className="w-full h-full object-cover"
              />
            </div>

            {/* Info */}
            <div className="flex-1 min-w-0">
              <h3 className="font-bold text-gray-900 text-lg mb-1 truncate group-hover:text-primary-600 transition-colors">
                {formatFoodName(item.food_name)}
              </h3>
              
              <div className="flex items-center gap-3 text-sm text-gray-600">
                <span className="flex items-center gap-1">
                  <span className="text-yellow-500">⭐</span>
                  <span className="font-medium">{Math.round(item.confidence)}%</span>
                </span>
                
                <span className="text-gray-400">•</span>
                
                <span>{formatRelativeTime(item.timestamp)}</span>
                
                {item.layer_used === 2 && (
                  <>
                    <span className="text-gray-400">•</span>
                    <span className="px-2 py-0.5 bg-purple-100 text-purple-700 rounded text-xs font-medium">
                      AI Layer 2
                    </span>
                  </>
                )}
              </div>
            </div>

            {/* Delete Button */}
            <button
              onClick={(e) => handleRemove(item.id, e)}
              className="flex-shrink-0 p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors opacity-0 group-hover:opacity-100"
              title="Remove from history"
            >
              <FiTrash2 className="text-xl" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
