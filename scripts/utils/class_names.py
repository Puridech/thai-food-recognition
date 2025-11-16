"""
📋 Class Names Configuration
กำหนดรายชื่อ classes สำหรับ Thai Food Recognition

ใช้ไฟล์นี้เมื่อ checkpoint ไม่มี class_names
"""

"""
📋 Class Names Configuration
กำหนดรายชื่อ classes สำหรับ Thai Food Recognition

ใช้ไฟล์นี้เมื่อ checkpoint ไม่มี class_names
"""

# ===================================================================
# 20 Thai Food Classes (ตามที่เทรนจริง)
# ===================================================================
# NOTE: ลำดับนี้มาจาก checkpoint - เรียงตาม alphabetical order

CLASS_NAMES = [
    "Foi Thong",                      # ฝอยทอง
    "Gai Pad Med Ma Muang Himmaphan", # ไก่ผัดเม็ดมะม่วงหิมพานต์
    "Gung Ob Woon Sen",               # กุ้งอบวุ้นเส้น
    "Hor Mok",                        # ห่อหมก
    "Kaeng Khiao Wan",                # แกงเขียวหวาน
    "Kaeng Matsaman",                 # แกงมัสมั่น
    "Kaeng Phet Pet Yang",            # แกงเผ็ดเป็ดย่าง
    "Khanom Krok",                    # ขนมครก
    "Khao Niao Mamuang",              # ข้าวเหนียวมะม่วง
    "Khao Pad",                       # ข้าวผัด
    "Khao Soi",                       # ข้าวซอย
    "Larb",                           # ลาบ
    "Pad Kra Pao",                    # ผัดกระเพรา
    "Pad See Ew",                     # ผัดซีอิ๊ว
    "Pad Thai",                       # ผัดไทย
    "Panang",                         # พะแนง
    "Som Tam",                        # ส้มตำ
    "Tom Kha Gai",                    # ต้มข่าไก่
    "Tom Yum Goong",                  # ต้มยำกุ้ง
    "Yam Woon Sen",                   # ยำวุ้นเส้น
]

# Alternative: ถ้าเทรนด้วย class names ที่แตกต่าง
# แก้ไขตรงนี้ให้ตรงกับที่ใช้ตอน training

# ===================================================================
# Helper Functions
# ===================================================================

def get_class_names():
    """Get list of class names"""
    return CLASS_NAMES.copy()

def get_num_classes():
    """Get number of classes"""
    return len(CLASS_NAMES)

def get_class_index(class_name):
    """Get index of a class name"""
    try:
        return CLASS_NAMES.index(class_name)
    except ValueError:
        return -1

def get_class_name(index):
    """Get class name from index"""
    if 0 <= index < len(CLASS_NAMES):
        return CLASS_NAMES[index]
    return None

# ===================================================================
# Verification
# ===================================================================

if __name__ == "__main__":
    print("="*70)
    print("📋 THAI FOOD CLASSES")
    print("="*70)
    print(f"\nTotal Classes: {get_num_classes()}")
    print("\nClass Names:")
    for i, name in enumerate(CLASS_NAMES, 1):
        print(f"   {i:2d}. {name}")
    
    print("\n" + "="*70)
    print("✅ Configuration ready!")
    print("="*70)
    
    # Verify
    print("\n🔍 Verification:")
    print(f"   get_num_classes() = {get_num_classes()}")
    print(f"   get_class_name(0) = {get_class_name(0)}")
    print(f"   get_class_index('Pad Thai') = {get_class_index('Pad Thai')}")
