/**
 * Food Images Mapping
 * Maps food names to their respective image paths
 */

export const foodImages: Record<string, string> = {
  // Savory Dishes (17)
  'som_tum': '/food-images/som_tum.png',                                           // ส้มตำ
  'tom_yum_goong': '/food-images/tom_yum_goong.png',                              // ต้มยำกุ้ง
  'larb': '/food-images/larb.png',                                                 // ลาบ
  'pad_thai': '/food-images/pad_thai.png',                                         // ผัดไทย
  'kaeng_khiao_wan': '/food-images/kaeng_khiao_wan.png',                          // แกงเขียวหวาน
  'khao_soi': '/food-images/khao_soi.png',                                         // ข้าวซอย
  'kaeng_massaman': '/food-images/kaeng_massaman.png',                            // แกงมัสมั่น
  'pad_krapow': '/food-images/pad_krapow.png',                                  // ผัดกะเพรา
  'khao_man_gai': '/food-images/khao_man_gai.png',                                // ข้าวมันไก่
  'khao_kha_mu': '/food-images/khao_kha_mu.png',                                  // ข้าวขาหมู
  'tom_kha_gai': '/food-images/tom_kha_gai.png',                                  // ต้มข่าไก่
  'gai_pad_med_ma_muang_himmaphan': '/food-images/gai_pad_med_ma_muang_himmaphan.png', // ไก่ผัดเม็ดมะม่วงหิมพานต์
  'kai_palo': '/food-images/kai_palo.png',                                        // ไข่พะโล้
  'gung_ob_woon_sen': '/food-images/gung_ob_woon_sen.png',                       // กุ้งอบวุ้นเส้น
  'khao_kluk_kapi': '/food-images/khao_kluk_kapi.png',                           // ข้าวคลุกกะปิ
  'por_pia_tod': '/food-images/por_pia_tod.png',                                 // ปอเปี๊ยะทอด
  'hor_mok': '/food-images/hor_mok.png',                                         // ห่อหมก

  // Desserts (3)
  'khao_niao_ma_muang': '/food-images/khao_niao_ma_muang.png',                   // ข้าวเหนียวมะม่วง
  'khanom_krok': '/food-images/khanom_krok.png',                                 // ขนมครก
  'foi_thong': '/food-images/foi_thong.png',                                     // ฝอยทอง
};

/**
 * Get food image URL
 * Returns the image path or fallback emoji if image not found
 */
export function getFoodImage(foodName: string): { type: 'image' | 'emoji', value: string } {
  const normalizedName = foodName.toLowerCase().replace(/\s+/g, '_');
  
  if (foodImages[normalizedName]) {
    return {
      type: 'image',
      value: foodImages[normalizedName]
    };
  }
  
  // Fallback to emoji
  return {
    type: 'emoji',
    value: '🍜'
  };
}

/**
 * Preload food images
 */
export function preloadFoodImages(foodNames: string[]) {
  foodNames.forEach(name => {
    const { type, value } = getFoodImage(name);
    if (type === 'image') {
      const img = new Image();
      img.src = value;
    }
  });
}
