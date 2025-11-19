"""
Configuration Management
Handles all backend configuration settings

Path Configuration:
- ROOT_DIR points to thai-food-recognition/ (repository root)
- DATA_PATH points to thai-food-recognition/data/
- MODELS_PATH points to thai-food-recognition/models/
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List

# Get project root directory
# config.py location: thai-food-recognition/backend/app/config.py
# ROOT_DIR should be: thai-food-recognition/
ROOT_DIR = Path(__file__).parent.parent.parent  # Go up 3 levels: app/ -> backend/ -> root/

# Data and Models paths (at repository root)
DATA_PATH = ROOT_DIR / "data"
MODELS_PATH = ROOT_DIR / "models"


class Settings(BaseSettings):
    """Application settings"""
    
    # ==================== Server Settings ====================
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    LOG_LEVEL: str = "info"
    
    # ==================== API Settings ====================
    API_TITLE: str = "Thai Food Recognition API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "AI-Powered Thai Food Recognition with Cultural Information"
    
    # ==================== Model Settings ====================
    # Layer 1 (Pre-trained)
    LAYER1_MODEL_PATH: Path = MODELS_PATH / "layer1_pretrained"
    LAYER1_CONFIDENCE_THRESHOLD: float = 0.80
    
    # Layer 2 (Fine-tuned)
    LAYER2_MODEL_PATH: Path = MODELS_PATH / "layer2_finetuned"
    LAYER2_MODEL_NAME: str = "clip-vit-base-patch32"
    
    # ==================== Data Paths ====================
    KNOWLEDGE_BASE_PATH: Path = DATA_PATH / "foods"
    RESTAURANT_DATA_PATH: Path = DATA_PATH / "restaurants" / "thai_restaurants.json"
    
    # ==================== Upload Settings ====================
    MAX_UPLOAD_SIZE: int = 10485760  # 10MB in bytes
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]
    
    # ==================== Performance ====================
    MAX_WORKERS: int = 4
    BATCH_SIZE: int = 1
    DEVICE: str = "cpu"  # Will auto-detect GPU if available
    
    # ==================== CORS Settings ====================
    CORS_ORIGINS: List[str] = ["*"]  # In production: specify frontend URLs
    
    # ==================== Supported Languages ====================
    SUPPORTED_LANGUAGES: List[str] = ["th", "en"]
    DEFAULT_LANGUAGE: str = "en"
    
    # ==================== Thai Food Dishes ====================
    # จากรายชื่อเมนู 20 เมนู
    SUPPORTED_DISHES: List[str] = [
        "som_tam",                          # ส้มตำ
        "tom_yum_goong",                    # ต้มยำกุ้ง
        "larb",                             # ลาบ
        "pad_thai",                         # ผัดไทย
        "kaeng_khiao_wan",                  # แกงเขียวหวาน
        "khao_soi",                         # ข้าวซอย
        "kaeng_matsaman",                   # แกงมัสมั่น
        "pad_kra_pao",                      # ผัดกะเพรา
        "khao_man_gai",                     # ข้าวมันไก่
        "khao_kha_mu",                      # ข้าวขาหมู
        "tom_kha_gai",                      # ต้มข่าไก่
        "gai_pad_med_ma_muang_himmaphan",   # ไก่ผัดเม็ดมะม่วงหิมพานต์
        "kai_palo",                         # ไข่พะโล้
        "gung_ob_woon_sen",                 # กุ้งอบวุ้นเส้น
        "khao_kluk_kapi",                   # ข้าวคลุกกะปิ
        "por_pia_tod",                      # ปอเปี๊ยะทอด
        "hor_mok",                          # ห่อหมก
        "khao_niao_mamuang",                # ข้าวเหนียวมะม่วง
        "khanom_krok",                      # ขนมครก
        "foi_thong"                         # ฝอยทอง
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()


# ==================== Helper Functions ====================

def verify_paths():
    """
    Verify that all required paths exist
    Returns dict with path status
    """
    paths_status = {
        "root_dir": ROOT_DIR.exists(),
        "data_dir": DATA_PATH.exists(),
        "models_dir": MODELS_PATH.exists(),
        "foods_dir": settings.KNOWLEDGE_BASE_PATH.exists(),
        "restaurants_dir": (DATA_PATH / "restaurants").exists(),
        "layer1_model": settings.LAYER1_MODEL_PATH.exists(),
        "layer2_model": settings.LAYER2_MODEL_PATH.exists(),
    }
    return paths_status


def is_valid_language(lang: str) -> bool:
    """
    Check if language code is supported
    
    Args:
        lang: Language code (e.g., "th", "en")
        
    Returns:
        True if supported, False otherwise
    """
    return lang in settings.SUPPORTED_LANGUAGES


def is_valid_dish(dish_name: str) -> bool:
    """
    Check if dish name is supported
    
    Args:
        dish_name: Dish name (e.g., "pad_thai")
        
    Returns:
        True if supported, False otherwise
    """
    return dish_name.lower() in settings.SUPPORTED_DISHES


# ==================== Display Configuration ====================

def print_config():
    """Print current configuration (for debugging)"""
    print("\n" + "=" * 70)
    print("⚙️  Backend Configuration")
    print("=" * 70)
    
    print(f"\n📁 Paths:")
    print(f"   Root:        {ROOT_DIR}")
    print(f"   Data:        {DATA_PATH}")
    print(f"   Models:      {MODELS_PATH}")
    
    print(f"\n🌐 Server:")
    print(f"   Host:        {settings.HOST}")
    print(f"   Port:        {settings.PORT}")
    print(f"   Reload:      {settings.RELOAD}")
    
    print(f"\n🤖 Models:")
    print(f"   Layer 1:     {settings.LAYER1_MODEL_PATH}")
    print(f"   Layer 2:     {settings.LAYER2_MODEL_PATH}")
    print(f"   Threshold:   {settings.LAYER1_CONFIDENCE_THRESHOLD}")
    
    print(f"\n📚 Data:")
    print(f"   Foods:       {settings.KNOWLEDGE_BASE_PATH}")
    print(f"   Restaurants: {settings.RESTAURANT_DATA_PATH}")
    
    print(f"\n🍜 Supported:")
    print(f"   Dishes:      {len(settings.SUPPORTED_DISHES)} menus")
    print(f"   Languages:   {', '.join(settings.SUPPORTED_LANGUAGES)}")
    
    print(f"\n✅ Path Verification:")
    paths = verify_paths()
    for path_name, exists in paths.items():
        status = "✅" if exists else "❌"
        print(f"   {status} {path_name}")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Test configuration
    print_config()
    
    # Test paths
    print("\n🧪 Testing Paths:")
    print(f"ROOT_DIR exists: {ROOT_DIR.exists()}")
    print(f"DATA_PATH exists: {DATA_PATH.exists()}")
    print(f"MODELS_PATH exists: {MODELS_PATH.exists()}")
