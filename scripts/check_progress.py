"""
Check Week 1-2 Progress - Updated for 31 dishes with train/valid/test split
"""

import os
import json

# รายการอาหารทั้งหมด 31 เมนู (ตามชื่อโฟลเดอร์จริง)
DISHES = [
    'Bua Loi',
    'Foi Thong',
    'Gai Pad Med Ma Muang Himmaphan',
    'Gung Mae Nam Pao',
    'Gung Ob Woon Sen',
    'Hor Mok',
    'Kaeng Jued Tao Hoo Mu Sap',
    'Kaeng Khiao Wan',
    'Kaeng Massaman',
    'Kaeng Panang',
    'Kaeng Som',
    'Kai Look Keuy',
    'Kai Palo',
    'Khanom Krok',
    'Khao Kha Mu',
    'Khao Kluk Kapi',
    'Khao Man Gai',
    'Khao Niao Ma Muang',
    'Khao Soi',
    'Kluay Buat Chee',
    'Larb',
    'Pad Hoi Lai',
    'Pad Krapow',
    'Pad See Ew',
    'Pad Thai',
    'Por Pia Tod',
    'Sangkaya Fak Thong',
    'Som Tum',
    'Tom Kha Gai',
    'Tom Yum Goong',
    'Yum Woon Sen'
]

DISH_NAMES_TH = {
    'Bua Loi': 'บัวลอย',
    'Foi Thong': 'ฝอยทอง',
    'Gai Pad Med Ma Muang Himmaphan': 'ไก่ผัดเม็ดมะม่วงหิมพานต์',
    'Gung Mae Nam Pao': 'กุ้งแม่น้ำเผา',
    'Gung Ob Woon Sen': 'กุ้งอบวุ้นเส้น',
    'Hor Mok': 'ห่อหมก',
    'Kaeng Jued Tao Hoo Mu Sap': 'แกงจืดเต้าหู้หมูสับ',
    'Kaeng Khiao Wan': 'แกงเขียวหวาน',
    'Kaeng Massaman': 'แกงมัสมั่น',
    'Kaeng Panang': 'แกงพะแนง',
    'Kaeng Som': 'แกงส้ม',
    'Kai Look Keuy': 'ไข่ลูกเขย',
    'Kai Palo': 'ไข่พะโล้',
    'Khanom Krok': 'ขนมครก',
    'Khao Kha Mu': 'ข้าวขาหมู',
    'Khao Kluk Kapi': 'ข้าวคลุกกะปิ',
    'Khao Man Gai': 'ข้าวมันไก่',
    'Khao Niao Ma Muang': 'ข้าวเหนียวมะม่วง',
    'Khao Soi': 'ข้าวซอย',
    'Kluay Buat Chee': 'กล้วยบวชชี',
    'Larb': 'ลาบ',
    'Pad Hoi Lai': 'ผัดหอยลาย',
    'Pad Krapow': 'ผัดกะเพรา',
    'Pad See Ew': 'ผัดซีอิ๊ว',
    'Pad Thai': 'ผัดไทย',
    'Por Pia Tod': 'ปอเปี๊ยะทอด',
    'Sangkaya Fak Thong': 'สังขยาฟักทอง',
    'Som Tum': 'ส้มตำ',
    'Tom Kha Gai': 'ต้มข่าไก่',
    'Tom Yum Goong': 'ต้มยำกุ้ง',
    'Yum Woon Sen': 'ยำวุ้นเส้น'
}

def count_images_by_split(base_path='../data/training'):
    """Count images in train/valid/test folders"""
    results = {}
    total_by_split = {'train': 0, 'valid': 0, 'test': 0}
    missing_dishes = []
    
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(base_path, split)
        
        if not os.path.exists(split_path):
            print(f"⚠️  Warning: {split_path} does not exist!")
            continue
            
        for dish in DISHES:
            dish_path = os.path.join(split_path, dish)
            
            if dish not in results:
                results[dish] = {'train': 0, 'valid': 0, 'test': 0, 'total': 0}
            
            if os.path.isdir(dish_path):
                images = [f for f in os.listdir(dish_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                count = len(images)
                results[dish][split] = count
                results[dish]['total'] += count
                total_by_split[split] += count
            else:
                if dish not in missing_dishes:
                    missing_dishes.append(dish)
    
    return results, total_by_split, missing_dishes

def count_markdown(path='../data/foods'):
    """Count markdown files by language"""
    if not os.path.exists(path):
        return {'th': 0, 'en': 0, 'jp': 0, 'total': 0}
    
    files = [f for f in os.listdir(path) if f.endswith('.md') and not f.startswith('_')]
    
    th_count = len([f for f in files if '_th.md' in f])
    en_count = len([f for f in files if '_en.md' in f])
    jp_count = len([f for f in files if '_jp.md' in f])
    
    return {
        'th': th_count,
        'en': en_count,
        'jp': jp_count,
        'total': len(files)
    }

def count_restaurants(path='../data/restaurants/thai_restaurants.json'):
    """Count restaurants in JSON"""
    if not os.path.exists(path):
        return 0
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return len(data.get('restaurants', []))
    except:
        return 0

def print_progress_bar(current, target, width=30):
    """Print a progress bar"""
    percentage = min(current / target, 1.0) if target > 0 else 0
    filled = int(width * percentage)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {current:4d}/{target:4d} ({percentage*100:5.1f}%)"

def main():
    print("="*85)
    print(" "*28 + "📊 Week 1-2 Progress Report")
    print(" "*30 + "31 Thai Dishes")
    print(" "*23 + "(train/valid/test: 240/30/30 per dish)")
    print("="*85)
    
    # Task 1: Images
    print("\n📸 Task 1: Image Collection & Split")
    print("-"*85)
    
    results, total_by_split, missing = count_images_by_split()
    
    # Calculate totals
    grand_total = sum(total_by_split.values())
    target_total = 31 * 300  # 31 dishes × 300 images
    
    # Summary by split
    print(f"\n📊 Overall Statistics:")
    print(f"   Train:  {total_by_split['train']:5d} images (target: {31*240:5d})")
    print(f"   Valid:  {total_by_split['valid']:5d} images (target: {31*30:5d})")
    print(f"   Test:   {total_by_split['test']:5d} images (target: {31*30:5d})")
    print(f"   {'─'*50}")
    print(f"   Total:  {grand_total:5d} images (target: {target_total:5d})")
    print(f"   Progress: {(grand_total/target_total)*100:.1f}%")
    
    # Detailed per dish
    print(f"\n📋 Detailed Breakdown (31 dishes):")
    print("-"*85)
    print(f"{'Dish':<40} {'Train':>6} {'Valid':>6} {'Test':>6} {'Total':>6} {'Status':>8}")
    print("-"*85)
    
    complete_count = 0
    partial_count = 0
    empty_count = 0
    
    for dish in sorted(DISHES):
        if dish in results:
            data = results[dish]
            train = data['train']
            valid = data['valid']
            test = data['test']
            total = data['total']
            
            # Determine status
            if total >= 300:
                status = "✅ Done"
                complete_count += 1
            elif total >= 200:
                status = "⚠️  Near"
                partial_count += 1
            elif total > 0:
                status = "🔄 WIP"
                partial_count += 1
            else:
                status = "❌ Empty"
                empty_count += 1
            
            # Display name (Thai + English)
            thai_name = DISH_NAMES_TH.get(dish, dish)
            display_name = f"{thai_name} ({dish})"[:38]
            
            print(f"{display_name:<40} {train:6d} {valid:6d} {test:6d} {total:6d} {status:>8}")
        else:
            print(f"{dish:<40} {'─':>6} {'─':>6} {'─':>6} {'─':>6} {'❌ Missing':>8}")
            empty_count += 1
    
    print("-"*85)
    print(f"Summary: ✅ Complete: {complete_count} | ⚠️  Partial: {partial_count} | ❌ Empty: {empty_count}")
    
    # Task 2: Markdown
    print("\n\n📝 Task 2: Knowledge Base (Markdown Files)")
    print("-"*85)
    md_count = count_markdown()
    target_md = 31 * 2  # 31 dishes × 2 languages
    
    print(f"   ไทย (Thai):     {md_count['th']:3d} / 31 files")
    print(f"   English:        {md_count['en']:3d} / 31 files")
    print(f"   {'─'*50}")
    print(f"   Total:          {md_count['total']:3d} / {target_md} files ({(md_count['total']/target_md)*100:.1f}%)")
    
    # Task 3: Restaurants
    print("\n\n🏪 Task 3: Restaurant Database (JSON)")
    print("-"*85)
    rest_count = count_restaurants()
    target_rest = 50  # ~1-2 restaurants per dish
    
    print(f"   Restaurants:    {rest_count:3d} / {target_rest} entries ({(rest_count/target_rest)*100:.1f}%)")
    
    # Overall Progress
    print("\n" + "="*85)
    
    # Calculate weighted progress
    image_progress = (grand_total / target_total) * 100
    md_progress = (md_count['total'] / target_md) * 100
    rest_progress = (rest_count / target_rest) * 100
    
    overall = (image_progress * 0.6 + md_progress * 0.25 + rest_progress * 0.15)
    
    print(f"📈 Overall Week 1-2 Progress: {overall:.1f}%")
    print(f"   • Images:      {image_progress:.1f}% (weight: 60%)")
    print(f"   • Markdown:    {md_progress:.1f}% (weight: 25%)")
    print(f"   • Restaurants: {rest_progress:.1f}% (weight: 15%)")
    
    print("\n" + "="*85)
    
    # Status message
    if overall >= 100:
        print("🎉 COMPLETE! Week 1-2 finished! Ready for Week 3 (Model Training)!")
    elif overall >= 80:
        print("🔥 Almost there! Just a little more to complete Week 1-2!")
    elif overall >= 60:
        print("💪 Great progress! Keep going!")
    elif overall >= 40:
        print("🚀 Good start! Continue the momentum!")
    else:
        print("📌 Let's focus on completing the image collection first!")
    
    print("="*85)
    
    # Missing dishes warning
    if missing:
        print(f"\n⚠️  Warning: {len(missing)} dishes have missing folders:")
        for dish in missing[:5]:  # Show first 5
            print(f"   • {dish}")
        if len(missing) > 5:
            print(f"   ... and {len(missing)-5} more")

if __name__ == "__main__":
    main()