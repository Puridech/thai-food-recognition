<<<<<<< HEAD
# 🍜 Thai Food Recognition & Cultural Discovery

**AI-Powered Thai Food Recognition System with Cultural Information**

> โปรเจค Internship ระหว่าง Hokkaido Information University (HIU) และ RMUTT (8 สัปดาห์)

---

## 📖 สารบัญ

- [ภาพรวมโปรเจค](#-ภาพรวมโปรเจค)
- [เทคโนโลยีที่ใช้](#-เทคโนโลยีที่ใช้)
- [โครงสร้างไฟล์](#-โครงสร้างไฟล์)
- [ระบบ AI และการทำงาน](#-ระบบ-ai-และการทำงาน)
- [Backend API](#-backend-api)
- [Frontend Web Application](#-frontend-web-application)
- [ฐานข้อมูลอาหาร](#-ฐานข้อมูลอาหาร)
- [วิธีการติดตั้งและใช้งาน](#-วิธีการติดตั้งและใช้งาน)
- [รายชื่ออาหารที่รองรับ](#-รายชื่ออาหาร-20-เมนู)

---

## 🎯 ภาพรวมโปรเจค

### คืออะไร?

ระบบ **Thai Food Recognition** เป็นระบบ AI ที่สามารถจดจำอาหารไทยจากรูปภาพ และแสดงข้อมูลทางวัฒนธรรมของอาหารนั้นๆ รวมถึงสูตรอาหาร และร้านอาหารแนะนำ

### ทำงานอย่างไร?

```
📷 อัปโหลดรูปอาหาร
     ↓
🤖 AI ทำนายชื่ออาหาร (Hybrid 2-Layer System)
     ↓
📚 แสดงข้อมูลวัฒนธรรม + สูตรอาหาร + ร้านอาหาร
```

### ฟีเจอร์หลัก

| ฟีเจอร์ | รายละเอียด |
|--------|------------|
| 🎯 **Image Recognition** | จดจำอาหารไทย 20 เมนูจากรูปภาพ |
| 🔥 **Hybrid AI System** | ระบบ 2 ชั้นที่รวดเร็วและแม่นยำ |
| 📖 **Cultural Stories** | เรื่องราววัฒนธรรมของแต่ละอาหาร |
| 🍳 **Recipes** | สูตรอาหารพร้อมขั้นตอน |
| 🏪 **Restaurant Finder** | ร้านอาหารแนะนำแยกตามภูมิภาค |
| 🌐 **Multi-Language** | รองรับภาษาไทยและอังกฤษ (i18next) |
| 📱 **Responsive Design** | ใช้งานได้บน Mobile และ Desktop |

---

## 🛠️ เทคโนโลยีที่ใช้

### AI / Machine Learning

| เทคโนโลยี | เวอร์ชัน | หน้าที่ |
|----------|---------|--------|
| **PyTorch** | 2.7+ | Deep Learning Framework |
| **CLIP (OpenAI)** | ViT-B/32 | Pre-trained Vision-Language Model |
| **Transformers** | 4.57+ | HuggingFace Tokenizer/Processor |
| **CUDA** | 11.8+ | GPU Acceleration (Optional) |

### Backend

| เทคโนโลยี | เวอร์ชัน | หน้าที่ |
|----------|---------|--------|
| **FastAPI** | 0.104+ | Modern Web Framework (Python) |
| **Uvicorn** | 0.24+ | ASGI Server |
| **Pydantic** | 2.10+ | Data Validation |
| **Pillow** | 12.0+ | Image Processing |

### Frontend

| เทคโนโลยี | เวอร์ชัน | หน้าที่ |
|----------|---------|--------|
| **Next.js** | 14.2 | React Framework |
| **React** | 18 | UI Library |
| **TypeScript** | 5 | Type Safety |
| **TailwindCSS** | 3.4 | Styling |
| **i18next** | 25.6 | Internationalization (TH/EN) |
| **Axios** | 1.13 | HTTP Client |
| **React Icons** | 5.5 | Icon Library |

### Tools & Infrastructure

| เครื่องมือ | หน้าที่ |
|----------|--------|
| **Git** | Version Control |
| **Python venv** | Virtual Environment |
| **npm** | Package Manager (Frontend) |

---

## 📁 โครงสร้างไฟล์

```
thai-food-recognition/
├── 📂 backend/                  # Backend API (FastAPI)
│   ├── main.py                  # Main application entry
│   ├── run_server.py            # Server launcher
│   ├── requirements.txt         # Python dependencies
│   ├── test_api.py              # API tests
│   └── 📂 app/
│       ├── config.py            # Configuration settings
│       └── 📂 services/
│           ├── model_service.py    # AI Model Service
│           └── data_service.py     # Data/Content Service
│
├── 📂 frontend/                 # Frontend Web App (Next.js)
│   ├── package.json             # npm dependencies
│   ├── tailwind.config.ts       # TailwindCSS config
│   ├── next.config.mjs          # Next.js config
│   ├── 📂 app/                  # Next.js App Router
│   │   ├── layout.tsx           # Root layout
│   │   ├── page.tsx             # Homepage
│   │   ├── globals.css          # Global styles
│   │   ├── 📂 food/             # Food detail page
│   │   └── 📂 history/          # Prediction history page
│   ├── 📂 components/
│   │   ├── 📂 features/         # Feature components
│   │   │   ├── ImageUpload.tsx       # Image upload component
│   │   │   ├── RecognitionResult.tsx # Result display
│   │   │   └── RecentHistory.tsx     # History display
│   │   ├── 📂 layout/           # Layout components
│   │   └── 📂 ui/               # UI components
│   │       ├── ErrorMessage.tsx
│   │       ├── Loading.tsx
│   │       └── Logo.tsx
│   ├── 📂 lib/
│   │   ├── api-client.ts        # Backend API client
│   │   ├── food-images.ts       # Food image mappings
│   │   ├── history.ts           # History management
│   │   └── i18n.ts              # i18next configuration
│   └── 📂 types/
│       └── index.ts             # TypeScript types
│
├── 📂 models/                   # Trained AI Models
│   ├── 📂 layer1_pretrained/    # Pre-trained CLIP model
│   │   ├── model.safetensors    # Model weights (~580MB)
│   │   ├── config.json          # Model config
│   │   ├── tokenizer.json       # Tokenizer
│   │   └── preprocessor_config.json
│   └── 📂 layer2_finetuned/     # Fine-tuned classifier
│       └── model_final.pth      # Fine-tuned weights (~690MB)
│
├── 📂 data/                     # Knowledge Base
│   ├── 📂 foods/                # Food information (Markdown)
│   │   ├── pad_thai_th.md       # ข้อมูลภาษาไทย
│   │   ├── pad_thai_en.md       # ข้อมูลภาษาอังกฤษ
│   │   └── ... (42 files, 21 foods x 2 languages)
│   ├── 📂 restaurants/          # Restaurant database (JSON)
│   └── 📂 training/             # Training images
│
├── 📂 scripts/                  # Utility Scripts
│   ├── 📂 hybrid/
│   │   └── hybrid_prediction.py # Standalone prediction script
│   ├── 📂 testing_model/        # Model testing scripts
│   ├── 📂 checkprogress/        # Training progress scripts
│   └── 📂 utils/                # Utility functions
│
└── requirements.txt             # Root Python dependencies
```

---

## 🧠 ระบบ AI และการทำงาน

### Hybrid 2-Layer System

โปรเจคนี้ใช้ระบบ AI แบบ **Hybrid 2 ชั้น** ที่รวมความเร็วและความแม่นยำ:

```
                    ┌─────────────────────────────────┐
                    │       📷 Input Image            │
                    └───────────────┬─────────────────┘
                                    ↓
                    ┌─────────────────────────────────┐
                    │     Layer 1: Pre-trained CLIP    │
                    │     (Zero-shot Classification)   │
                    │          ⚡ ~0.7 วินาที          │
                    └───────────────┬─────────────────┘
                                    ↓
                    ┌─────────────────────────────────┐
                    │   Confidence ≥ 80% ?            │
                    └───────────────┬─────────────────┘
                          ↓ YES            ↓ NO
              ┌───────────────────┐  ┌───────────────────┐
              │   ✅ ใช้ Layer 1   │  │ 🎯 ส่งต่อ Layer 2  │
              │   (เร็ว + มั่นใจ)  │  │ (Fine-tuned, แม่น) │
              └───────────────────┘  │    ⏱️ ~2 วินาที    │
                                     └────────┬──────────┘
                                              ↓
                    ┌─────────────────────────────────┐
                    │       🏆 Final Prediction        │
                    │   Food Name + Confidence Score   │
                    └─────────────────────────────────┘
```

### Layer 1: Pre-trained CLIP

- **Model**: OpenAI CLIP ViT-B/32
- **วิธีการ**: Zero-shot classification
- **ข้อดี**: เร็ว (~0.7 วินาที)
- **ข้อจำกัด**: อาจไม่แม่นยำกับอาหารไทยบางจาน

### Layer 2: Fine-tuned Thai Food Specialist

- **Base Model**: CLIP ViT-B/32
- **Training**: Fine-tuned บน 1,000+ รูปอาหารไทย 20 เมนู
- **Accuracy**: ~96.33%
- **Architecture**: CLIP + Custom Classifier Head

```python
# Layer 2 Architecture
CLIP Vision Encoder
     ↓
[Linear 768→256] → ReLU → Dropout(0.3)
     ↓
[Linear 256→20] → Output (20 classes)
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| Training Accuracy | **96.33%** |
| Number of Classes | 20 dishes |
| Layer 1 Speed | ~0.7 seconds |
| Layer 2 Speed | ~2.0 seconds |
| Hybrid Threshold | 80% confidence |

---

## ⚡ Backend API

### Endpoints Overview

| Endpoint | Method | หน้าที่ |
|----------|--------|--------|
| `/` | GET | Welcome message |
| `/api/health` | GET | Health check |
| `/api/recognize` | POST | 🎯 **Image recognition** |
| `/api/food/{name}` | GET | 📚 Food information |
| `/api/restaurants/{name}` | GET | 🏪 Restaurant list |
| `/api/stats` | GET | 📊 Prediction statistics |

### Main Endpoint: `/api/recognize`

**Request:**
```bash
POST /api/recognize
Content-Type: multipart/form-data

file: <image file>
```

**Response:**
```json
{
  "success": true,
  "food_name": "Pad Thai",
  "confidence": 0.92,
  "layer_used": 1,
  "processing_time": 0.73,
  "decision": "Layer 1 (high confidence)"
}
```

### Food Information: `/api/food/{food_name}`

**Request:**
```bash
GET /api/food/pad_thai?lang=th
```

**Response:**
```json
{
  "success": true,
  "food_name": "Pad Thai",
  "language": "th",
  "cultural_story": {
    "title": "ผัดไทย",
    "general_info": "...",
    "story": "..."
  },
  "recipe": {
    "ingredients": [...],
    "steps": [...]
  }
}
```

### API Documentation

เมื่อ run server แล้ว สามารถเข้าดู API docs ได้ที่:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 🌐 Frontend Web Application

### Pages

| Route | Component | หน้าที่ |
|-------|-----------|--------|
| `/` | `page.tsx` | หน้าหลัก - Upload และ Recognition |
| `/food/[id]` | `food/[id]/page.tsx` | รายละเอียดอาหาร |
| `/history` | `history/page.tsx` | ประวัติการทำนาย |

### Main Features

#### 1. Image Upload Component (`ImageUpload.tsx`)
- รองรับ drag & drop
- รองรับ click to upload
- Preview รูปก่อน submit
- Validation ไฟล์รูป

#### 2. Recognition Result (`RecognitionResult.tsx`)
- แสดงผลการทำนาย
- แสดง confidence score
- แสดง layer ที่ใช้
- Link ไปหน้ารายละเอียด

#### 3. Recent History (`RecentHistory.tsx`)
- บันทึกประวัติการทำนาย (localStorage)
- แสดงรายการล่าสุด
- Clear history

### Internationalization (i18n)

รองรับ 2 ภาษา:
- 🇹🇭 **ภาษาไทย** (default)
- 🇺🇸 **English**

Configuration: `lib/i18n.ts`

---

## 📚 ฐานข้อมูลอาหาร

### Food Knowledge Base

ไฟล์ข้อมูลอาหารอยู่ใน `data/foods/` แต่ละไฟล์มีโครงสร้าง:

```markdown
# [ชื่ออาหาร]

## ข้อมูลทั่วไป
...

## ประวัติและวัฒนธรรม
...

## ส่วนผสม
- ...

## วิธีทำ
1. ...
2. ...

## เกร็ดความรู้
...
```

**จำนวนไฟล์**: 42 files (21 อาหาร × 2 ภาษา)

### Restaurant Database

ไฟล์ `data/restaurants/restaurants.json` มีข้อมูลร้านอาหารแนะนำ:

```json
{
  "restaurants": [
    {
      "name": "ร้านอาหาร ABC",
      "dishes": ["pad_thai", "som_tum"],
      "region": "bangkok",
      "rating": 4.5,
      "address": "..."
    }
  ]
}
```

---

## 🚀 วิธีการติดตั้งและใช้งาน

### Prerequisites

- Python 3.9+
- Node.js 18+
- CUDA-compatible GPU (recommended)
- 4GB+ free disk space

### 1. Clone Repository

```bash
git clone https://github.com/[username]/thai-food-recognition.git
cd thai-food-recognition
```

### 2. Setup Backend

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Navigate to backend
cd backend
pip install -r requirements.txt
```

### 3. Setup Frontend

```bash
cd frontend
npm install
```

### 4. Run Application

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
# หรือ
python run_server.py
```
→ Server runs at: `http://localhost:8000`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```
→ Web app runs at: `http://localhost:3000`

### 5. Test Hybrid AI (Standalone)

```bash
cd scripts/hybrid
python hybrid_prediction.py --image ../../data/test_images/padthai.jpg --verbose
```

---

## 🍜 รายชื่ออาหาร (20 เมนู)

| # | ชื่อไทย | English Name | ID |
|---|--------|--------------|-----|
| 1 | ฝอยทอง | Foi Thong | `foi_thong` |
| 2 | ไก่ผัดเม็ดมะม่วงหิมพานต์ | Gai Pad Med Ma Muang | `gai_pad_med_ma_muang_himmaphan` |
| 3 | กุ้งอบวุ้นเส้น | Gung Ob Woon Sen | `gung_ob_woon_sen` |
| 4 | ห่อหมก | Hor Mok | `hor_mok` |
| 5 | แกงเขียวหวาน | Green Curry | `kaeng_khiao_wan` |
| 6 | แกงมัสมั่น | Massaman Curry | `kaeng_massaman` |
| 7 | ขนมครก | Khanom Krok | `khanom_krok` |
| 8 | ข้าวเหนียวมะม่วง | Mango Sticky Rice | `khao_niao_ma_muang` |
| 9 | ข้าวซอย | Khao Soi | `khao_soi` |
| 10 | ลาบ | Larb | `larb` |
| 11 | ผัดกระเพรา | Pad Kra Pao | `pad_krapow` |
| 12 | ผัดไทย | Pad Thai | `pad_thai` |
| 13 | ส้มตำ | Som Tum (Papaya Salad) | `som_tum` |
| 14 | ต้มข่าไก่ | Tom Kha Gai | `tom_kha_gai` |
| 15 | ต้มยำกุ้ง | Tom Yum Goong | `tom_yum_goong` |
| 16 | ไข่พะโล้ | Kai Palo | `kai_palo` |
| 17 | ข้าวขาหมู | Khao Kha Mu | `khao_kha_mu` |
| 18 | ข้าวคลุกกะปิ | Khao Kluk Kapi | `khao_kluk_kapi` |
| 19 | ข้าวมันไก่ | Khao Man Gai | `khao_man_gai` |
| 20 | ปอเปี๊ยะทอด | Por Pia Tod | `por_pia_tod` |

---

## 📊 Project Progress

### ✅ สัปดาห์ที่ 1-2: Data Collection & Knowledge Base
- [x] Selected 20 Thai dishes
- [x] Collected 1,000+ training images
- [x] Created knowledge base (Markdown files)
- [x] Created restaurant database (JSON)

### ✅ สัปดาห์ที่ 3: Pre-trained Model Setup
- [x] Tested pre-trained CLIP models
- [x] Evaluated Layer 1 baseline performance
- [x] Selected optimal model architecture

### ✅ สัปดาห์ที่ 4: Fine-tuning & Hybrid System
- [x] Fine-tuned Layer 2 model (96% accuracy)
- [x] Implemented Hybrid 2-Layer system
- [x] Auto-detection of model architecture

### ✅ สัปดาห์ที่ 5-6: Backend Development
- [x] Setup FastAPI project structure
- [x] Implement API endpoints
- [x] Integrate AI models
- [x] Parse Markdown/JSON data files

### ✅ สัปดาห์ที่ 7: Frontend Development
- [x] Setup Next.js + TailwindCSS
- [x] Build main pages
- [x] Implement i18next (Thai/English)
- [x] Connect to Backend API
- [x] Responsive design polish

### ✅ สัปดาห์ที่ 8: Testing & Documentation
- [x] User testing
- [x] Bug fixing & optimization
- [x] Technical documentation
- [x] Demo video

---

## 👥 Project Info

| | |
|---|---|
| **Status** | Finished |
| **Institution** | Hokkaido Information University (HIU) × RMUTT |
| **Duration** | 8 Weeks |
| **Type** | Internship Project in Japan |

---
=======
🍜 Thai Food Recognition & Cultural Discovery 🍜 
>>>>>>> fda92cbf9a5b952d56deeb7e9f817876b502c587
