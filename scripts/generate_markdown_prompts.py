"""
Generate AI prompts for creating markdown files
"""

# รายการอาหารทั้งหมด 31 เมนู
DISHES = {
    'Som Tum': 'ส้มตำ',
    'Tom Yum Goong': 'ต้มยำกุ้ง',
    'Larb': 'ลาบ',
    'Pad Thai': 'ผัดไทย',
    'Kaeng Khiao Wan': 'แกงเขียวหวาน',
    'Khao Soi': 'ข้าวซอย',
    'Kaeng Massaman': 'แกงมัสมั่น',
    'Pad Krapow': 'ผัดกะเพรา',
    'Kaeng Som': 'แกงส้ม',
    'Khao Man Gai': 'ข้าวมันไก่',
    'Khao Kha Mu': 'ข้าวขาหมู',
    'Pad See Ew': 'ผัดซีอิ๊ว',
    'Kaeng Panang': 'แกงพะแนง',
    'Tom Kha Gai': 'ต้มข่าไก่',
    'Kaeng Jued Tao Hoo Mu Sap': 'แกงจืดเต้าหู้หมูสับ',
    'Gai Pad Med Ma Muang Himmaphan': 'ไก่ผัดเม็ดมะม่วงหิมพานต์',
    'Kai Palo': 'ไข่พะโล้',
    'Kai Look Keuy': 'ไข่ลูกเขย',
    'Gung Mae Nam Pao': 'กุ้งแม่น้ำเผา',
    'Gung Ob Woon Sen': 'กุ้งอบวุ้นเส้น',
    'Khao Kluk Kapi': 'ข้าวคลุกกะปิ',
    'Por Pia Tod': 'ปอเปี๊ยะทอด',
    'Pad Hoi Lai': 'ผัดหอยลาย',
    'Yum Woon Sen': 'ยำวุ้นเส้น',
    'Hor Mok': 'ห่อหมก',
    'Kluay Buat Chee': 'กล้วยบวชชี',
    'Khao Niao Ma Muang': 'ข้าวเหนียวมะม่วง',
    'Bua Loi': 'บัวลอย',
    'Khanom Krok': 'ขนมครก',
    'Foi Thong': 'ฝอยทอง',
    'Sangkaya Fak Thong': 'สังขยาฟักทอง'
}

def generate_prompt_thai(dish_en, dish_th):
    """Generate Thai prompt"""
    
    prompt = f"""เขียนเอกสาร Markdown สำหรับอาหารไทย "{dish_th} ({dish_en})" ในรูปแบบต่อไปนี้:

# {dish_th} ({dish_en})

## 📖 ข้อมูลทั่วไป
ให้ระบุ:
- ชื่อภาษาไทย, อังกฤษ
- ภูมิภาคต้นกำเนิด (กลาง/เหนือ/ใต้/อีสาน)
- ประเภท (อาหารคาว/หวาน)
- รสชาติหลัก

## 🌟 เรื่องราวและวัฒนธรรม
เขียน 3-4 ย่อหน้าเกี่ยวกับ:
- ประวัติความเป็นมาของอาหารชนิดนี้
- ภูมิภาคที่เป็นถิ่นกำเนิด และการแพร่กระจาย
- ความสำคัญทางวัฒนธรรมหรือประเพณี
- เรื่องน่าสนใจหรือตำนานที่เกี่ยวข้อง
- โอกาสหรือช่วงเวลาที่นิยมรับประทาน

## 👨‍🍳 สูตรอาหาร

### ส่วนผสม (สำหรับ 2-3 ที่)
แยกเป็น:
- **วัตถุดิบหลัก**: (ระบุปริมาณชัดเจน)
- **เครื่องปรุง**: 
- **เครื่องเคียง/ผัก**:

### วิธีทำ
เขียนเป็นขั้นตอน 5-7 ขั้นตอน โดยละเอียดและชัดเจน

### เวลาในการทำ
- เวลาเตรียม: X นาที
- เวลาประกอบอาหาร: Y นาที
- รวม: Z นาที

### ระดับความยาก
ให้คะแนน ⭐ 1-5 ดาว

### 💡 เคล็ดลับและข้อควรรู้
- Tips สำหรับการทำให้อร่อยขึ้น
- ข้อควรระวัง
- วิธีเสิร์ฟและทานคู่กับอะไร

---
เขียนในรูปแบบ Markdown ที่สมบูรณ์ ภาษาไทยที่ถูกต้อง เป็นมิตร อ่านง่าย"""

    return prompt

def generate_prompt_english(dish_en, dish_th):
    """Generate English prompt"""
    
    prompt = f"""Write a comprehensive Markdown document for the Thai dish "{dish_en} ({dish_th})" in the following format:

# {dish_en} ({dish_th})

## 📖 General Information
Include:
- Thai name, English name
- Region of origin (Central/Northern/Southern/Northeastern Thailand)
- Category (savory/sweet)
- Main flavor profile

## 🌟 Cultural Story and History
Write 3-4 paragraphs about:
- Historical origins of this dish
- Regional background and how it spread
- Cultural or traditional significance
- Interesting stories or legends
- Typical occasions or seasons for eating this dish

## 👨‍🍳 Recipe

### Ingredients (serves 2-3)
Organize into:
- **Main ingredients**: (with specific measurements)
- **Seasonings**:
- **Garnishes/Vegetables**:

### Cooking Instructions
Write 5-7 detailed, clear steps

### Cooking Time
- Prep time: X minutes
- Cook time: Y minutes
- Total: Z minutes

### Difficulty Level
Rate with ⭐ 1-5 stars

### 💡 Tips and Notes
- Tips for better taste
- Important warnings
- Serving suggestions and pairings

---
Write in complete Markdown format, in clear, friendly English that's easy to understand."""

    return prompt

def save_prompts_to_file():
    """Save all prompts to a text file for easy copying"""
    
    with open('../data/foods/_ai_prompts.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("AI PROMPTS FOR GENERATING MARKDOWN FILES\n")
        f.write("Copy each prompt below and paste to ChatGPT/Claude\n")
        f.write("="*80 + "\n\n")
        
        for i, (dish_en, dish_th) in enumerate(DISHES.items(), 1):
            # Thai prompt
            f.write(f"\n{'='*80}\n")
            f.write(f"DISH {i}/31: {dish_th} ({dish_en})\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"--- PROMPT 1: THAI VERSION ---\n")
            f.write(f"Save to: {dish_en.lower().replace(' ', '_')}_th.md\n\n")
            f.write(generate_prompt_thai(dish_en, dish_th))
            f.write("\n\n")
            
            # English prompt
            f.write(f"--- PROMPT 2: ENGLISH VERSION ---\n")
            f.write(f"Save to: {dish_en.lower().replace(' ', '_')}_en.md\n\n")
            f.write(generate_prompt_english(dish_en, dish_th))
            f.write("\n\n")
    
    print(f"✅ Prompts saved to: data/foods/_ai_prompts.txt")
    print(f"   Total: {len(DISHES)} dishes × 2 languages = {len(DISHES)*2} prompts")

def generate_interactive():
    """Interactive prompt generator"""
    
    print("="*80)
    print("📝 AI Prompt Generator for Markdown Files")
    print("="*80)
    print(f"\nTotal dishes: {len(DISHES)}")
    print("\nOptions:")
    print("  1. Generate all prompts to file (recommended)")
    print("  2. Generate one by one (interactive)")
    print("  0. Exit")
    
    choice = input("\nYour choice: ").strip()
    
    if choice == '1':
        save_prompts_to_file()
        print("\n" + "="*80)
        print("Next steps:")
        print("1. Open: data/foods/_ai_prompts.txt")
        print("2. Copy each prompt")
        print("3. Paste to ChatGPT or Claude")
        print("4. Save the output to the specified filename")
        print("="*80)
        
    elif choice == '2':
        for i, (dish_en, dish_th) in enumerate(DISHES.items(), 1):
            print(f"\n{'='*80}")
            print(f"Dish {i}/{len(DISHES)}: {dish_th} ({dish_en})")
            print(f"{'='*80}\n")
            
            print("--- THAI PROMPT ---")
            print(generate_prompt_thai(dish_en, dish_th))
            print("\n" + "-"*80)
            
            input(f"\nPress Enter to see English prompt...")
            
            print("\n--- ENGLISH PROMPT ---")
            print(generate_prompt_english(dish_en, dish_th))
            print("\n" + "="*80)
            
            cont = input(f"\nContinue to next dish? (y/n): ").strip().lower()
            if cont != 'y':
                break
    
    else:
        print("Goodbye!")

if __name__ == "__main__":
    generate_interactive()