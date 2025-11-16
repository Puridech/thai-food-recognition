# 🍜 Thai Food Recognition & Cultural Discovery

**AI-Powered Thai Food Recognition System with Cultural Information**

HIU Internship Project (8 Weeks)

---

## 📊 Project Progress

### Week 1-2: Data Collection & Knowledge Base ✅ **COMPLETED**
- [x] Selected 20 Thai dishes
- [x] Collected 1,000+ training images
- [x] Created knowledge base (Markdown files)
- [x] Created restaurant database (JSON)
- [x] Setup development environment

### Week 3: Pre-trained Model Setup ✅ **COMPLETED**
- [x] Tested pre-trained CLIP models
- [x] Evaluated Layer 1 baseline performance
- [x] Selected optimal model architecture
- [x] Prepared training pipeline

### Week 4: Fine-tuning & Hybrid System ✅ **COMPLETED**
- [x] Fine-tuned Layer 2 model (96% accuracy)
- [x] Implemented Hybrid 2-Layer system
- [x] Auto-detection of model architecture
- [x] Tested hybrid logic successfully
- [x] Fixed PyTorch 2.6 compatibility
- [x] Organized project structure
- [x] **First successful prediction!** 🎉

### Week 5-6: Backend Development 🔄 **IN PROGRESS**
- [ ] Setup FastAPI project structure
- [ ] Implement 4 API endpoints
- [ ] Integrate AI models (Layer 1 + 2)
- [ ] Implement hybrid prediction logic
- [ ] Parse Markdown/JSON files
- [ ] Multi-language content serving
- [ ] Error handling & validation
- [ ] API documentation (Swagger)
- [ ] Testing

### Week 7: Frontend Development ⏳ **PENDING**
- [ ] Setup Next.js + TailwindCSS
- [ ] Build main pages
- [ ] Implement camera capture + upload
- [ ] Implement i18next (Thai/English)
- [ ] Connect to Backend API
- [ ] Responsive design
- [ ] Favorites & History features

### Week 8: Testing & Documentation ⏳ **PENDING**
- [ ] User testing (5-10 people)
- [ ] Bug fixing & optimization
- [ ] UI/UX polish
- [ ] Performance optimization
- [ ] Technical documentation
- [ ] User guide
- [ ] Demo video (5-7 min)
- [ ] Presentation slides

---

## 🎯 Current Status

**📍 Week 4 Complete - Ready for Backend Development!**

### ✅ What's Working

- **Layer 1 (Pre-trained)**: Fast baseline recognition
- **Layer 2 (Fine-tuned)**: 96% accuracy on 20 Thai dishes
- **Hybrid System**: Smart decision making (80% confidence threshold)
- **Architecture**: Auto-detection (Old/New, with/without BatchNorm)
- **Test Result**: Foi Thong recognized at 96.90% confidence

### 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Training Accuracy | 96.33% |
| Number of Classes | 20 dishes |
| Layer 1 Speed | ~0.7s |
| Layer 2 Speed | ~2.0s |
| Hybrid Threshold | 80% |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)
- 2GB+ free disk space

### Installation

```bash
# Clone repository
git clone https://github.com/[your-username]/thai-food-recognition.git
cd thai-food-recognition

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🍜 Supported Dishes (20)

1. Foi Thong (ฝอยทอง)
2. Gai Pad Med Ma Muang Himmaphan (ไก่ผัดเม็ดมะม่วงหิมพานต์)
3. Gung Ob Woon Sen (กุ้งอบวุ้นเส้น)
4. Hor Mok (ห่อหมก)
5. Kaeng Khiao Wan (แกงเขียวหวาน)
6. Kaeng Matsaman (แกงมัสมั่น)
7. Kaeng Phet Pet Yang (แกงเผ็ดเป็ดย่าง)
8. Khanom Krok (ขนมครก)
9. Khao Niao Mamuang (ข้าวเหนียวมะม่วง)
10. Khao Pad (ข้าวผัด)
11. Khao Soi (ข้าวซอย)
12. Larb (ลาบ)
13. Pad Kra Pao (ผัดกระเพรา)
14. Pad See Ew (ผัดซีอิ๊ว)
15. Pad Thai (ผัดไทย)
16. Panang (พะแนง)
17. Som Tam (ส้มตำ)
18. Tom Kha Gai (ต้มข่าไก่)
19. Tom Yum Goong (ต้มยำกุ้ง)
20. Yam Woon Sen (ยำวุ้นเส้น)

---

## 🔬 Technical Architecture

### Hybrid 2-Layer System

```
Input Image → Layer 1 (Fast) → Confidence ≥ 80%?
                                    ↓
                          YES → Use L1  |  NO → Layer 2 (Accurate)
                                    ↓
                              Final Result
```

### Technology Stack

- **AI/ML**: PyTorch, CLIP, HuggingFace
- **Backend**: FastAPI (Week 5-6)
- **Frontend**: Next.js, React, TailwindCSS (Week 7)
- **Tools**: Python 3.9+, CUDA, Git

---

## 📈 Week 4 Results

**Test: Foi Thong (ฝอยทอง)**

```
Layer 1: Khao Kluk Kapi (26.78%) ❌ → Low confidence
Layer 2: Foi Thong (96.90%) ✅ → High confidence

✅ System correctly used Layer 2 for better accuracy!
```

---

## 📚 Documentation

- [Week 4 Quick Start](docs/WEEK4_QUICKSTART.md)
- [Hybrid System Guide](docs/HYBRID_GUIDE.md)
- [Project Organization](docs/PROJECT_ORGANIZATION.md)

---

## 👥 Project Info

**Status**: Week 4/8 Complete  
**Institution**: Hokkaido Information University (HIU)  
**Duration**: 8 Weeks  
**Next**: Week 5-6 Backend Development

---

**Last Updated**: Week 4 Complete ✅  
**Next Milestone**: Backend API Development 🚀