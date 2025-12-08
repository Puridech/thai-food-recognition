'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useTranslation } from 'react-i18next';
import '@/lib/i18n';
import { FiTrash2, FiSearch } from 'react-icons/fi';
import Navbar from '@/components/layout/Navbar';
import { 
  getHistory, 
  removeFromHistory, 
  clearHistory,
  formatRelativeTime, 
  formatFoodName,
  type HistoryItem 
} from '@/lib/history';

export default function HistoryPage() {
  const router = useRouter();
  const { t, i18n } = useTranslation();
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [filteredHistory, setFilteredHistory] = useState<HistoryItem[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [isClient, setIsClient] = useState(false);

  // Helper to show both EN and TH names
  const getBilingualFoodName = (foodName: string) => {
    const enName = formatFoodName(foodName, 'en');
    const thName = formatFoodName(foodName, 'th');
    
    // If names are different, show both
    if (enName !== thName) {
      return `${enName} / ${thName}`;
    }
    // If same (no Thai translation), show only English
    return enName;
  };

  useEffect(() => {
    setIsClient(true);
    loadHistory();

    // Listen for updates
    const handleUpdate = () => loadHistory();
    window.addEventListener('historyUpdated', handleUpdate);
    return () => window.removeEventListener('historyUpdated', handleUpdate);
  }, []);

  useEffect(() => {
    filterHistory();
  }, [history, searchQuery]);

  const loadHistory = () => {
    const allHistory = getHistory();
    setHistory(allHistory);
  };

  const filterHistory = () => {
    let filtered = [...history];

    // Apply search filter
    if (searchQuery) {
      filtered = filtered.filter(item =>
        formatFoodName(item.food_name, i18n.language as 'en' | 'th').toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    setFilteredHistory(filtered);
  };

  const handleRemove = (id: string) => {
    if (confirm(t('removeConfirm'))) {
      removeFromHistory(id);
      loadHistory();
      window.dispatchEvent(new Event('historyUpdated'));
    }
  };

  const handleClearAll = () => {
    if (confirm(t('clearAllConfirm'))) {
      clearHistory();
      loadHistory();
      window.dispatchEvent(new Event('historyUpdated'));
    }
  };

  const handleItemClick = (item: HistoryItem) => {
    router.push(`/food/${encodeURIComponent(item.food_name)}`);
  };

  if (!isClient) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-red-50 to-yellow-50">
      {/* Navbar */}
      <Navbar />

      <main className="container-custom py-8">
        <div className="max-w-5xl mx-auto">
          {/* Title */}
          <div className="mb-6">
            <h1 className="text-4xl font-bold text-gray-900 mb-2">
              📜 {t('searchHistory')}
            </h1>
            <p className="text-gray-600">
              {history.length} {history.length === 1 ? t('searchFound') : t('searchesFound')}
            </p>
          </div>

          {/* Search + Clear All */}
          <div className="bg-white rounded-lg p-4 border-2 border-gray-200 mb-6">
            <div className="flex gap-3">
              {/* Search */}
              <div className="flex-1 relative">
                <FiSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                <input
                  type="text"
                  placeholder={t('searchFoodNames')}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border-2 border-gray-300 rounded-lg focus:border-primary-500 focus:outline-none"
                />
              </div>

              {/* Clear All Button */}
              {history.length > 0 && (
                <button
                  onClick={handleClearAll}
                  className="flex items-center gap-2 px-6 py-2 text-red-600 bg-red-50 hover:bg-red-100 border-2 border-red-200 rounded-lg transition-colors font-medium"
                >
                  <FiTrash2 />
                  <span>{t('clearAll')}</span>
                </button>
              )}
            </div>
          </div>

          {/* Results Count */}
          {searchQuery && (
            <div className="mb-4 text-gray-600">
              {t('found')} {filteredHistory.length} {filteredHistory.length !== 1 ? t('results') : t('result')}
            </div>
          )}

          {/* History List */}
          {filteredHistory.length === 0 ? (
            <div className="text-center py-16 bg-white rounded-lg border-2 border-dashed border-gray-300">
              <div className="text-6xl mb-4">📭</div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">
                {searchQuery ? t('noResultsFound') : t('noHistoryYet')}
              </h3>
              <p className="text-gray-600">
                {searchQuery 
                  ? t('tryDifferentSearch')
                  : t('startRecognizing')}
              </p>
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="mt-4 btn btn-outline"
                >
                  {t('clearSearch')}
                </button>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              {filteredHistory.map((item) => (
                <div
                  key={item.id}
                  onClick={() => handleItemClick(item)}
                  className="flex items-center gap-4 p-4 bg-white border-2 border-gray-200 rounded-lg hover:border-primary-300 hover:shadow-lg transition-all cursor-pointer group"
                >
                  {/* Thumbnail */}
                  <div className="flex-shrink-0 w-24 h-24 rounded-lg overflow-hidden bg-gray-100 border border-gray-200">
                    <img
                      src={item.image_url}
                      alt={formatFoodName(item.food_name, i18n.language as 'en' | 'th')}
                      className="w-full h-full object-cover"
                    />
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <h3 className="font-bold text-gray-900 text-xl mb-2 group-hover:text-primary-600 transition-colors">
                      {getBilingualFoodName(item.food_name)}
                    </h3>
                    
                    <div className="flex flex-wrap items-center gap-3 text-sm text-gray-600">
                      <span className="flex items-center gap-1">
                        <span className="text-xs">📅</span>
                        {formatRelativeTime(item.timestamp, i18n.language as 'en' | 'th')}
                      </span>
                      
                      {item.layer_used === 2 && (
                        <>
                          <span className="text-gray-400">•</span>
                          <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded text-xs font-medium">
                            🤖 {t('aiSpecialist')}
                          </span>
                        </>
                      )}
                    </div>

                    {/* Exact timestamp on hover */}
                    <div className="text-xs text-gray-500 mt-1">
                      {new Date(item.timestamp).toLocaleString()}
                    </div>
                  </div>

                  {/* Delete Button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRemove(item.id);
                    }}
                    className="flex-shrink-0 p-3 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors opacity-0 group-hover:opacity-100"
                    title="Remove from history"
                  >
                    <FiTrash2 className="text-xl" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
