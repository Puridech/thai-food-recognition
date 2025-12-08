/**
 * History Management Utilities
 * Handles localStorage operations for search history
 */

export interface HistoryItem {
  id: string;
  food_name: string;
  image_url: string; // Base64 data URL
  confidence: number;
  timestamp: number; // Unix timestamp
  layer_used: 1 | 2;
  display_name?: string; // Formatted name for display
}

const HISTORY_KEY = 'thai_food_history';
const MAX_HISTORY_ITEMS = 50;

/**
 * Get all history items from localStorage
 */
export function getHistory(): HistoryItem[] {
  try {
    const data = localStorage.getItem(HISTORY_KEY);
    if (!data) return [];
    
    const history = JSON.parse(data) as HistoryItem[];
    
    // Sort by timestamp (newest first)
    return history.sort((a, b) => b.timestamp - a.timestamp);
  } catch (error) {
    console.error('Error reading history:', error);
    return [];
  }
}

/**
 * Add item to history
 */
export function addToHistory(item: Omit<HistoryItem, 'id' | 'timestamp'>): HistoryItem {
  try {
    const history = getHistory();
    const now = Date.now();
    
    const newItem: HistoryItem = {
      ...item,
      id: `${now}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: now,
    };
    
    // Enhanced duplicate prevention:
    // 1. Check if same food within last 10 seconds (increased from 5)
    // 2. Check if similar confidence (within 5% range)
    const recentDuplicate = history.find(
      h => h.food_name === newItem.food_name && 
           Math.abs(h.timestamp - newItem.timestamp) < 10000 && // 10 seconds
           Math.abs(h.confidence - newItem.confidence) < 5 // Within 5% confidence
    );
    
    if (recentDuplicate) {
      console.log('⚠️ Duplicate entry prevented:', {
        food: newItem.food_name,
        timeSince: `${Math.round((now - recentDuplicate.timestamp) / 1000)}s ago`,
        reason: 'Same food within 10 seconds with similar confidence'
      });
      return recentDuplicate;
    }
    
    // Add to beginning of array
    history.unshift(newItem);
    
    // Limit to MAX_HISTORY_ITEMS
    const limitedHistory = history.slice(0, MAX_HISTORY_ITEMS);
    
    // Save to localStorage with error handling
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(limitedHistory));
      console.log('💾 History saved:', {
        total: limitedHistory.length,
        size: `${Math.round(JSON.stringify(limitedHistory).length / 1024)}KB`
      });
    } catch (error) {
      // localStorage quota exceeded - remove oldest items and retry
      console.warn('⚠️ Storage quota exceeded, reducing history size');
      console.error('Storage error:', error);
      
      // Try with fewer items
      const reducedHistory = limitedHistory.slice(0, 30); // Keep only 30 instead of 50
      localStorage.setItem(HISTORY_KEY, JSON.stringify(reducedHistory));
      
      console.log('💾 History saved (reduced):', {
        total: reducedHistory.length,
        size: `${Math.round(JSON.stringify(reducedHistory).length / 1024)}KB`
      });
    }
    
    return newItem;
  } catch (error) {
    console.error('❌ Error adding to history:', error);
    throw error;
  }
}

/**
 * Remove item from history by ID
 */
export function removeFromHistory(id: string): void {
  try {
    const history = getHistory();
    const filtered = history.filter(item => item.id !== id);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(filtered));
  } catch (error) {
    console.error('Error removing from history:', error);
    throw error;
  }
}

/**
 * Clear all history
 */
export function clearHistory(): void {
  try {
    localStorage.removeItem(HISTORY_KEY);
  } catch (error) {
    console.error('Error clearing history:', error);
    throw error;
  }
}

/**
 * Get recent history items (limited number)
 */
export function getRecentHistory(limit: number = 5): HistoryItem[] {
  const history = getHistory();
  return history.slice(0, limit);
}

/**
 * Format timestamp to relative time
 */
export function formatRelativeTime(timestamp: number, language: 'en' | 'th' = 'en'): string {
  const now = Date.now();
  const diff = now - timestamp;
  
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  
  if (language === 'th') {
    // Thai translations
    if (seconds < 60) return 'เมื่อสักครู่';
    if (minutes < 60) return `${minutes} นาทีที่แล้ว`;
    if (hours < 24) return `${hours} ชั่วโมงที่แล้ว`;
    if (days < 7) return `${days} วันที่แล้ว`;
  } else {
    // English
    if (seconds < 60) return 'Just now';
    if (minutes < 60) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
    if (hours < 24) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    if (days < 7) return `${days} day${days > 1 ? 's' : ''} ago`;
  }
  
  // Format as date for older items
  const date = new Date(timestamp);
  const locale = language === 'th' ? 'th-TH' : 'en-US';
  return date.toLocaleDateString(locale, { 
    month: 'short', 
    day: 'numeric',
    year: date.getFullYear() !== new Date().getFullYear() ? 'numeric' : undefined
  });
}

/**
 * Format food name for display
 */
export function formatFoodName(name: string, language: 'en' | 'th' = 'en'): string {
  // Thai dish names mapping
  const thaiNames: Record<string, string> = {
    'som_tam': 'ส้มตำ',
    'tom_yum_goong': 'ต้มยำกุ้ง',
    'larb': 'ลาบ',
    'pad_thai': 'ผัดไทย',
    'kaeng_khiao_wan': 'แกงเขียวหวาน',
    'khao_soi': 'ข้าวซอย',
    'kaeng_matsaman': 'แกงมัสมั่น',
    'pad_kra_pao': 'ผัดกะเพรา',
    'khao_man_gai': 'ข้าวมันไก่',
    'khao_kha_mu': 'ข้าวขาหมู',
    'tom_kha_gai': 'ต้มข่าไก่',
    'gai_pad_med_ma_muang_himmaphan': 'ไก่ผัดเม็ดมะม่วงหิมพานต์',
    'kai_palo': 'ไข่พะโล้',
    'gung_ob_woon_sen': 'กุ้งอบวุ้นเส้น',
    'khao_kluk_kapi': 'ข้าวคลุกกะปิ',
    'por_pia_tod': 'ปอเปี๊ยะทอด',
    'hor_mok': 'ห่อหมก',
    'khao_niao_mamuang': 'ข้าวเหนียวมะม่วง',
    'khanom_krok': 'ขนมครก',
    'foi_thong': 'ฝอยทอง',
  };
  
  if (language === 'th' && thaiNames[name.toLowerCase()]) {
    return thaiNames[name.toLowerCase()];
  }
  
  // English: Convert snake_case to Title Case
  return name
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Get history statistics
 */
export function getHistoryStats() {
  const history = getHistory();
  
  return {
    total: history.length,
    today: history.filter(h => {
      const today = new Date().setHours(0, 0, 0, 0);
      return h.timestamp >= today;
    }).length,
    thisWeek: history.filter(h => {
      const weekAgo = Date.now() - 7 * 24 * 60 * 60 * 1000;
      return h.timestamp >= weekAgo;
    }).length,
    avgConfidence: history.length > 0
      ? Math.round(history.reduce((sum, h) => sum + h.confidence, 0) / history.length)
      : 0,
  };
}

/**
 * Get storage size in KB
 */
export function getStorageSize(): number {
  try {
    const data = localStorage.getItem(HISTORY_KEY);
    if (!data) return 0;
    
    // Calculate size in KB
    const sizeInBytes = new Blob([data]).size;
    return Math.round(sizeInBytes / 1024);
  } catch (error) {
    console.error('Error getting storage size:', error);
    return 0;
  }
}

/**
 * Check if storage is near limit (> 8MB)
 */
export function isStorageNearLimit(): boolean {
  try {
    const totalSize = getStorageSize();
    const limitKB = 8 * 1024; // 8MB in KB
    return totalSize > limitKB;
  } catch {
    return false;
  }
}

/**
 * Clean up old items if storage is too large
 */
export function cleanupOldHistory(): void {
  try {
    if (!isStorageNearLimit()) return;
    
    console.log('🧹 Storage near limit, cleaning up old history...');
    
    const history = getHistory();
    
    // Keep only 25 most recent items (half of max)
    const reducedHistory = history.slice(0, 25);
    
    localStorage.setItem(HISTORY_KEY, JSON.stringify(reducedHistory));
    
    console.log('✅ Cleanup complete:', {
      before: history.length,
      after: reducedHistory.length,
      removed: history.length - reducedHistory.length
    });
  } catch (error) {
    console.error('Error during cleanup:', error);
  }
}
