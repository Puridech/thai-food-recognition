"""
Check progress for Task 2 (Markdown) and Task 3 (Restaurants)
"""

import os
import json

def count_markdown_files(path='../data/foods'):
    """Count markdown files by language"""
    if not os.path.exists(path):
        return {'th': 0, 'en': 0, 'total': 0, 'files': []}
    
    files = [f for f in os.listdir(path) 
             if f.endswith('.md') and not f.startswith('_')]
    
    th_files = [f for f in files if f.endswith('_th.md')]
    en_files = [f for f in files if f.endswith('_en.md')]
    
    return {
        'th': len(th_files),
        'en': len(en_files),
        'total': len(files),
        'th_files': th_files,
        'en_files': en_files
    }

def count_restaurants(path='../data/restaurants/thai_restaurants.json'):
    """Count and analyze restaurants"""
    if not os.path.exists(path):
        return {'count': 0, 'by_region': {}, 'by_dish': {}}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        restaurants = data.get('restaurants', [])
        
        # Count by region
        by_region = {}
        for rest in restaurants:
            region = rest.get('region', 'Unknown')
            by_region[region] = by_region.get(region, 0) + 1
        
        # Count by dish
        by_dish = {}
        for rest in restaurants:
            for dish in rest.get('specialty', []):
                by_dish[dish] = by_dish.get(dish, 0) + 1
        
        return {
            'count': len(restaurants),
            'by_region': by_region,
            'by_dish': by_dish,
            'restaurants': restaurants
        }
    
    except:
        return {'count': 0, 'by_region': {}, 'by_dish': {}}

def main():
    print("="*80)
    print("📊 Task 2 & 3 Progress Report")
    print("="*80)
    
    # Task 2: Markdown
    print("\n📝 Task 2: Knowledge Base (Markdown Files)")
    print("-"*80)
    
    md_data = count_markdown_files()
    target_total = 31 * 2  # 31 dishes × 2 languages
    
    print(f"\nTarget: {target_total} files (31 dishes × 2 languages)")
    print(f"\nCurrent status:")
    print(f"   Thai files:    {md_data['th']:3d} / 31  ({md_data['th']/31*100:5.1f}%)")
    print(f"   English files: {md_data['en']:3d} / 31  ({md_data['en']/31*100:5.1f}%)")
    print(f"   {'─'*50}")
    print(f"   Total:         {md_data['total']:3d} / {target_total}  ({md_data['total']/target_total*100:5.1f}%)")
    
    # Missing files
    all_dishes = set([
        'som_tum', 'tom_yum_goong', 'larb', 'pad_thai', 'kaeng_khiao_wan',
        'khao_soi', 'kaeng_massaman', 'pad_krapow', 'kaeng_som', 'khao_man_gai',
        'khao_kha_mu', 'pad_see_ew', 'kaeng_panang', 'tom_kha_gai',
        'kaeng_jued_tao_hoo_mu_sap', 'gai_pad_med_ma_muang_himmaphan',
        'kai_palo', 'kai_look_keuy', 'gung_mae_nam_pao', 'gung_ob_woon_sen',
        'khao_kluk_kapi', 'por_pia_tod', 'pad_hoi_lai', 'yum_woon_sen',
        'hor_mok', 'kluay_buat_chee', 'khao_niao_ma_muang', 'bua_loi',
        'khanom_krok', 'foi_thong', 'sangkaya_fak_thong'
    ])
    
    existing_th = set([f.replace('_th.md', '') for f in md_data['th_files']])
    existing_en = set([f.replace('_en.md', '') for f in md_data['en_files']])
    
    missing_th = all_dishes - existing_th
    missing_en = all_dishes - existing_en
    
    if missing_th:
        print(f"\n⚠️  Missing Thai files ({len(missing_th)}):")
        for dish in sorted(list(missing_th)[:5]):
            print(f"   • {dish}_th.md")
        if len(missing_th) > 5:
            print(f"   ... and {len(missing_th)-5} more")
    
    if missing_en:
        print(f"\n⚠️  Missing English files ({len(missing_en)}):")
        for dish in sorted(list(missing_en)[:5]):
            print(f"   • {dish}_en.md")
        if len(missing_en) > 5:
            print(f"   ... and {len(missing_en)-5} more")
    
    # Task 3: Restaurants
    print("\n\n🏪 Task 3: Restaurant Database")
    print("-"*80)
    
    rest_data = count_restaurants()
    target_rest = 50
    
    print(f"\nTarget: {target_rest}+ restaurants")
    print(f"Current: {rest_data['count']} restaurants ({rest_data['count']/target_rest*100:.1f}%)")
    
    if rest_data['by_region']:
        print(f"\n📍 By Region:")
        for region, count in sorted(rest_data['by_region'].items(), 
                                    key=lambda x: x[1], reverse=True):
            print(f"   {region:25s}: {count:3d} restaurants")
    
    if rest_data['by_dish']:
        print(f"\n🍜 Coverage by Dish (Top 10):")
        for dish, count in sorted(rest_data['by_dish'].items(), 
                                   key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {dish:30s}: {count:2d} restaurants")
        
        # Dishes with no restaurants
        covered_dishes = set(rest_data['by_dish'].keys())
        uncovered = all_dishes - covered_dishes
        if uncovered:
            print(f"\n⚠️  Dishes with NO restaurants ({len(uncovered)}):")
            for dish in sorted(list(uncovered)[:5]):
                print(f"   • {dish}")
            if len(uncovered) > 5:
                print(f"   ... and {len(uncovered)-5} more")
    
    # Overall progress
    print("\n" + "="*80)
    
    md_progress = md_data['total'] / target_total * 100
    rest_progress = rest_data['count'] / target_rest * 100
    overall = (md_progress * 0.6 + rest_progress * 0.4)
    
    print(f"📈 Overall Task 2 & 3 Progress: {overall:.1f}%")
    print(f"   • Markdown:    {md_progress:.1f}% (weight: 60%)")
    print(f"   • Restaurants: {rest_progress:.1f}% (weight: 40%)")
    
    print("\n" + "="*80)
    
    if overall >= 100:
        print("🎉 COMPLETE! Ready for Week 3 (Model Training)!")
    elif overall >= 75:
        print("🔥 Almost there! Just a bit more!")
    elif overall >= 50:
        print("💪 Halfway done! Keep going!")
    elif overall >= 25:
        print("🚀 Good start! Continue!")
    else:
        print("📌 Let's focus on completing the markdown files first!")
    
    print("="*80)

if __name__ == "__main__":
    main()