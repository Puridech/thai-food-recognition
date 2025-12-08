/**
 * i18n Configuration
 * Multi-language support (Thai & English)
 */

import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import ENV from '@/config/env';

// Translation resources
const resources = {
  en: {
    translation: {
      // Navigation
      home: 'Home',
      about: 'About',
      favorites: 'Favorites',
      history: 'History',

      // Home Page
      appTitle: 'Thai Food Recognition',
      appSubtitle: 'Discover Thai Cuisine with AI',
      uploadImage: 'Upload Image',
      takePhoto: 'Take Photo',
      dragDrop: 'Drag & drop an image here',
      or: 'or',
      clickToUpload: 'Click to upload',

      // Recognition
      recognizing: 'Recognizing...',
      analyzing: 'Analyzing your image',
      pleaseWait: 'Please wait',
      
      // Results
      recognitionResult: 'Recognition Result',
      confidence: 'Confidence',
      recognizedAs: 'Recognized as',
      culturalStory: 'Cultural Story',
      recipe: 'Recipe',
      restaurants: 'Recommended Restaurants',
      
      // Recipe
      ingredients: 'Ingredients',
      instructions: 'Instructions',
      servings: 'Servings',
      prepTime: 'Prep Time',
      cookTime: 'Cook Time',
      totalTime: 'Total Time',
      difficulty: 'Difficulty',
      easy: 'Easy',
      medium: 'Medium',
      hard: 'Hard',
      tips: 'Tips & Tricks',
      minutes: 'min',
      
      // Restaurants
      noRestaurants: 'No restaurants found',
      viewOnMap: 'View on Map',
      location: 'Location',
      specialty: 'Specialty',
      
      // Actions
      tryAnother: 'Try Another Image',
      saveToFavorites: 'Save to Favorites',
      share: 'Share',
      viewRecipe: 'View Recipe',
      viewRestaurants: 'View Restaurants',
      
      // Errors
      error: 'Error',
      errorOccurred: 'An error occurred',
      tryAgain: 'Try Again',
      invalidImage: 'Invalid image file',
      uploadFailed: 'Upload failed',
      networkError: 'Network error. Please check your connection.',
      
      // Language
      language: 'Language',
      thai: 'ไทย',
      english: 'English',
      
      // Footer
      madeWith: 'Made with',
      by: 'by',
      
      // Home Page Features
      aiRecognition: 'AI Recognition',
      aiRecognitionDesc: '96% accuracy with hybrid 2-layer AI system',
      culturalStories: 'Cultural Stories',
      culturalStoriesDesc: 'Learn the history and traditions behind each dish',
      authenticRecipes: 'Authentic Recipes',
      authenticRecipesDesc: 'Authentic recipes with step-by-step instructions',
      supportedDishes: 'Supported Thai Dishes (20 Menus)',
      showMore: 'more...',
      
      // Recognition Page
      successfullyIdentified: 'Successfully Identified',
      processingTime: 'Processing Time',
      status: 'Status',
      success: 'Success',
      viewFullDetails: 'View Full Details',
      tryAnotherDish: 'Try Another Dish',
      
      // History Page
      searchHistory: 'Search History',
      searchesFound: 'searches found',
      searchFound: 'search found',
      searchFoodNames: 'Search food names...',
      clearAll: 'Clear All',
      found: 'Found',
      result: 'result',
      results: 'results',
      noResultsFound: 'No results found',
      noHistoryYet: 'No history yet',
      tryDifferentSearch: 'Try a different search term',
      startRecognizing: 'Start recognizing Thai food to build your history!',
      clearSearch: 'Clear Search',
      removeConfirm: 'Remove this item from history?',
      clearAllConfirm: 'Are you sure you want to clear all history? This cannot be undone.',
      aiSpecialist: 'AI Specialist (Layer 2)',
      
      // Food Detail Page
      story: 'Story',
      thaiCuisine: 'Thai Cuisine',
      popularDish: 'Popular Dish',
      authenticRecipe: 'Popular Recipes',
      recommendedRestaurants: 'Recommended Restaurants',
      openInGoogleMaps: 'Open in Google Maps',
    },
  },
  th: {
    translation: {
      // Navigation
      home: 'หน้าแรก',
      about: 'เกี่ยวกับ',
      favorites: 'รายการโปรด',
      history: 'ประวัติ',

      // Home Page
      appTitle: 'ระบบจดจำอาหารไทย',
      appSubtitle: 'ค้นพบอาหารไทยด้วย AI',
      uploadImage: 'อัพโหลดรูป',
      takePhoto: 'ถ่ายรูป',
      dragDrop: 'ลากและวางรูปภาพที่นี่',
      or: 'หรือ',
      clickToUpload: 'คลิกเพื่ออัพโหลด',

      // Recognition
      recognizing: 'กำลังจดจำ...',
      analyzing: 'กำลังวิเคราะห์รูปภาพ',
      pleaseWait: 'กรุณารอสักครู่',
      
      // Results
      recognitionResult: 'ผลการจดจำ',
      confidence: 'ความมั่นใจ',
      recognizedAs: 'จดจำได้ว่าเป็น',
      culturalStory: 'เรื่องราววัฒนธรรม',
      recipe: 'สูตรอาหาร',
      restaurants: 'ร้านอาหารแนะนำ',
      
      // Recipe
      ingredients: 'ส่วนผสม',
      instructions: 'วิธีทำ',
      servings: 'สำหรับ',
      prepTime: 'เวลาเตรียม',
      cookTime: 'เวลาปรุง',
      totalTime: 'เวลารวม',
      difficulty: 'ความยาก',
      easy: 'ง่าย',
      medium: 'ปานกลาง',
      hard: 'ยาก',
      tips: 'เคล็ดลับ',
      minutes: 'นาที',
      
      // Restaurants
      noRestaurants: 'ไม่พบร้านอาหาร',
      viewOnMap: 'ดูบนแผนที่',
      location: 'สถานที่',
      specialty: 'ความเชี่ยวชาญ',
      
      // Actions
      tryAnother: 'ลองรูปอื่น',
      saveToFavorites: 'บันทึกเป็นรายการโปรด',
      share: 'แชร์',
      viewRecipe: 'ดูสูตรอาหาร',
      viewRestaurants: 'ดูร้านอาหาร',
      
      // Errors
      error: 'ข้อผิดพลาด',
      errorOccurred: 'เกิดข้อผิดพลาด',
      tryAgain: 'ลองอีกครั้ง',
      invalidImage: 'ไฟล์รูปภาพไม่ถูกต้อง',
      uploadFailed: 'อัพโหลดล้มเหลว',
      networkError: 'เกิดข้อผิดพลาดเครือข่าย กรุณาตรวจสอบการเชื่อมต่อ',
      
      // Language
      language: 'ภาษา',
      thai: 'ไทย',
      english: 'English',
      
      // Footer
      madeWith: 'สร้างด้วย',
      by: 'โดย',
      
      // Home Page Features
      aiRecognition: 'การจดจำด้วย AI',
      aiRecognitionDesc: 'ความแม่นยำ 96% ด้วยระบบ AI 2 ชั้น',
      culturalStories: 'เรื่องราววัฒนธรรม',
      culturalStoriesDesc: 'เรียนรู้ประวัติและประเพณีที่อยู่เบื้องหลังอาหารแต่ละจาน',
      authenticRecipes: 'สูตรอาหาร',
      authenticRecipesDesc: 'สูตรอาหารต้นตำรับพร้อมคำแนะนำทีละขั้นตอน',
      supportedDishes: 'เมนูอาหารไทยที่รองรับ (20 เมนู)',
      showMore: 'เพิ่มเติม...',
      
      // Recognition Page
      successfullyIdentified: 'ผลการจดจำ',
      processingTime: 'เวลาประมวลผล',
      status: 'สถานะ',
      success: 'สำเร็จ',
      viewFullDetails: 'ดูรายละเอียดเพิ่มเติม',
      tryAnotherDish: 'ลองอีกครั้ง',
      
      // History Page
      searchHistory: 'ประวัติการค้นหา',
      searchesFound: 'ครั้งที่พบ',
      searchFound: 'ครั้งที่พบ',
      searchFoodNames: 'ค้นหาชื่ออาหาร...',
      clearAll: 'ลบทั้งหมด',
      found: 'พบ',
      result: 'ผลลัพธ์',
      results: 'ผลลัพธ์',
      noResultsFound: 'ไม่พบผลลัพธ์',
      noHistoryYet: 'ยังไม่มีประวัติ',
      tryDifferentSearch: 'ลองค้นหาคำอื่น',
      startRecognizing: 'เริ่มจดจำอาหารไทยเพื่อสร้างประวัติของคุณ!',
      clearSearch: 'ล้างการค้นหา',
      removeConfirm: 'ลบรายการนี้ออกจากประวัติ?',
      clearAllConfirm: 'คุณแน่ใจหรือไม่ว่าต้องการลบประวัติทั้งหมด? การกระทำนี้ไม่สามารถยกเลิกได้',
      aiSpecialist: 'ผู้เชี่ยวชาญ AI (Layer 2)',
      
      // Food Detail Page
      story: 'เรื่องราว',
      thaiCuisine: 'อาหารไทย',
      popularDish: 'เมนูยอดนิยม',
      authenticRecipe: 'สูตรยอดนิยม',
      recommendedRestaurants: 'ร้านอาหารแนะนำ',
      openInGoogleMaps: 'เปิดใน Google Maps',
    },
  },
};

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: ENV.DEFAULT_LANGUAGE,
    lng: ENV.DEFAULT_LANGUAGE,
    interpolation: {
      escapeValue: false,
    },
    detection: {
      order: ['localStorage', 'navigator'],
      caches: ['localStorage'],
      lookupLocalStorage: ENV.STORAGE_KEYS.LANGUAGE,
    },
  });

export default i18n;
