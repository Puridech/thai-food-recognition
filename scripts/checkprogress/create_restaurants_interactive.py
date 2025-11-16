"""
Interactive Restaurant Database Builder - Version 2.0
Simplified structure
"""

import json
import os

DISH_MAP = {
    '1': 'som_tum',
    '2': 'tom_yum_goong',
    '3': 'larb',
    '4': 'pad_thai',
    '5': 'kaeng_khiao_wan',
    '6': 'khao_soi',
    '7': 'kaeng_massaman',
    '8': 'pad_krapow',
    '9': 'khao_man_gai',
    '10': 'khao_kha_mu',
    '11': 'tom_kha_gai',
    '12': 'gai_pad_med_ma_muang_himmaphan',
    '13': 'kai_palo',
    '14': 'gung_ob_woon_sen',
    '15': 'khao_kluk_kapi',
    '16': 'por_pia_tod',
    '17': 'hor_mok',
    '18': 'khao_niao_ma_muang',
    '19': 'khanom_krok',
    '20': 'foi_thong'
}

DISH_NAMES = {
    'som_tum': 'ส้มตำ',
    'tom_yum_goong': 'ต้มยำกุ้ง',
    'larb': 'ลาบ',
    'pad_thai': 'ผัดไทย',
    'kaeng_khiao_wan': 'แกงเขียวหวาน',
    'khao_soi': 'ข้าวซอย',
    'kaeng_massaman': 'แกงมัสมั่น',
    'pad_krapow': 'ผัดกะเพรา',
    'khao_man_gai': 'ข้าวมันไก่',
    'khao_kha_mu': 'ข้าวขาหมู',
    'tom_kha_gai': 'ต้มข่าไก่',
    'gai_pad_med_ma_muang_himmaphan': 'ไก่ผัดเม็ดมะม่วงหิมพานต์',
    'kai_palo': 'ไข่พะโล้',
    'gung_ob_woon_sen': 'กุ้งอบวุ้นเส้น',
    'khao_kluk_kapi': 'ข้าวคลุกกะปิ',
    'por_pia_tod': 'ปอเปี๊ยะทอด',
    'hor_mok': 'ห่อหมก',
    'khao_niao_ma_muang': 'ข้าวเหนียวมะม่วง',
    'khanom_krok': 'ขนมครก',
    'foi_thong': 'ฝอยทอง'
}

def get_coordinates():
    """Helper to get coordinates from Google Maps"""
    print("\n📍 How to get coordinates:")
    print("   1. Go to Google Maps")
    print("   2. Right-click on the restaurant location")
    print("   3. Click the coordinates (first item)")
    print("   4. Paste here")
    print("\n   Example: 13.7563, 100.5018")
    
    coords_input = input("\n   Coordinates (lat, lng): ").strip()
    
    try:
        lat_str, lng_str = coords_input.split(',')
        lat = float(lat_str.strip())
        lng = float(lng_str.strip())
        return lat, lng
    except:
        print("   ⚠️  Invalid format, using default Bangkok coordinates")
        return 13.7563, 100.5018

def create_restaurant():
    """Interactive restaurant entry - Version 2.0"""
    
    print("\n" + "="*70)
    print("🏪 Add New Restaurant (Simplified)")
    print("="*70)
    
    restaurant = {}
    
    # Basic info
    print("\n📝 Basic Information:")
    restaurant['id'] = input("   Restaurant ID (e.g., raan_thipsamai_01): ").strip()
    restaurant['name_th'] = input("   ชื่อร้าน (ไทย): ").strip()
    restaurant['name_en'] = input("   Restaurant Name (English): ").strip()
    
    # Specialty
    print("\n🍜 Specialty Dishes:")
    print("   Enter dish numbers (comma-separated)")
    print("   Examples: 1=ส้มตำ, 2=ต้มยำกุ้ง, 4=ผัดไทย, 6=ข้าวซอย")
    
    specialty_input = input("   Dish numbers: ").strip()
    specialty_list = [DISH_MAP.get(n.strip(), '') for n in specialty_input.split(',')]
    specialty_list = [s for s in specialty_list if s]
    restaurant['specialty'] = specialty_list
    
    # Location (Region only)
    print("\n📍 Location:")
    print("   Examples:")
    print("   - กรุงเทพฯ - สุขุมวิท")
    print("   - กรุงเทพฯ - สีลม")
    print("   - เชียงใหม่ - นิมมาน")
    print("   - ภูเก็ต - ป่าตอง")
    restaurant['region'] = input("   Region: ").strip()
    
    lat, lng = get_coordinates()
    restaurant['latitude'] = lat
    restaurant['longitude'] = lng
    
    # Additional info
    print("\n💰 Additional Information:")
    print("   Price: $ (cheap), $$ (moderate), $$$ (expensive), $$$$ (luxury)")
    restaurant['price_range'] = input("   Price range: ").strip() or "$$"
    
    # Rating with source
    rating_input = input("   Rating (0-5): ").strip()
    if rating_input:
        restaurant['rating'] = float(rating_input)
        
        print("\n   Rating source:")
        print("   1. Wongnai")
        print("   2. Google Maps")
        print("   3. TripAdvisor")
        print("   4. Other")
        source_choice = input("   Choose (1-4): ").strip()
        
        source_map = {
            '1': 'wongnai',
            '2': 'google',
            '3': 'tripadvisor',
            '4': 'other'
        }
        restaurant['rating_source'] = source_map.get(source_choice, 'google')
    
    # Optional fields
    phone = input("\n   Phone (optional): ").strip()
    if phone:
        restaurant['phone'] = phone
    
    website = input("   Website (optional): ").strip()
    if website:
        restaurant['website'] = website
    
    hours = input("   Opening hours (optional): ").strip()
    if hours:
        restaurant['opening_hours'] = hours
    
    return restaurant

def display_restaurant(restaurant):
    """Display restaurant info for confirmation"""
    print("\n" + "="*70)
    print("📋 Restaurant Preview:")
    print("="*70)
    print(f"ID: {restaurant['id']}")
    print(f"Name: {restaurant['name_th']} / {restaurant['name_en']}")
    print(f"Specialty: {', '.join([DISH_NAMES.get(d, d) for d in restaurant['specialty']])}")
    print(f"Region: {restaurant['region']}")
    print(f"Price: {restaurant['price_range']}")
    if restaurant.get('rating'):
        source = restaurant.get('rating_source', 'unknown')
        print(f"Rating: {restaurant['rating']}/5.0 ({source})")
    print("="*70)

def load_database(db_path='../data/restaurants/thai_restaurants.json'):
    """Load existing database"""
    if os.path.exists(db_path):
        with open(db_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {'restaurants': []}

def save_database(data, db_path='../data/restaurants/thai_restaurants.json'):
    """Save database"""
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    print("="*70)
    print("🏪 Thai Restaurant Database Builder v2.0")
    print("="*70)
    
    db_path = '../data/restaurants/thai_restaurants.json'
    data = load_database(db_path)
    
    print(f"\nCurrent restaurants: {len(data['restaurants'])}")
    print(f"Target: 50+ restaurants")
    
    while True:
        print("\n" + "="*70)
        print("Options:")
        print("  [1] Add new restaurant")
        print("  [2] View all restaurants")
        print("  [3] Save and exit")
        print("  [0] Exit without saving")
        print("="*70)
        
        choice = input("\nYour choice: ").strip()
        
        if choice == '1':
            restaurant = create_restaurant()
            display_restaurant(restaurant)
            
            confirm = input("\nAdd this restaurant? (y/n): ").strip().lower()
            if confirm == 'y':
                data['restaurants'].append(restaurant)
                print(f"\n✅ Added! Total: {len(data['restaurants'])} restaurants")
            else:
                print("\n❌ Cancelled")
        
        elif choice == '2':
            print("\n" + "="*70)
            print(f"📋 All Restaurants ({len(data['restaurants'])})")
            print("="*70)
            
            for i, rest in enumerate(data['restaurants'], 1):
                specialty_names = ', '.join([DISH_NAMES.get(d, d) for d in rest['specialty'][:3]])
                if len(rest['specialty']) > 3:
                    specialty_names += '...'
                
                print(f"{i:3d}. {rest['name_th']:30s} | {rest['region']:20s} | {specialty_names}")
            
            input("\nPress Enter to continue...")
        
        elif choice == '3':
            save_database(data, db_path)
            print(f"\n💾 Saved {len(data['restaurants'])} restaurants to:")
            print(f"   {db_path}")
            print("\n✅ Done!")
            break
        
        elif choice == '0':
            confirm = input("\nExit without saving? (y/n): ").strip().lower()
            if confirm == 'y':
                print("\n❌ Exited without saving")
                break
        
        else:
            print("\n⚠️  Invalid choice")

if __name__ == "__main__":
    main()