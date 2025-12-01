# 🍜 Thai Food Recognition - Frontend

AI-powered Thai food recognition web application with cultural stories, authentic recipes, and restaurant recommendations.

## 🚀 Features

- **AI Food Recognition**: Upload or capture Thai food images for instant identification
- **Cultural Stories**: Learn the history and traditions behind each dish
- **Authentic Recipes**: Step-by-step cooking instructions with tips
- **Restaurant Recommendations**: Find the best Thai restaurants
- **Multi-language**: Full support for Thai (ไทย) and English
- **Responsive Design**: Mobile-first, works on all devices
- **Favorites & History**: Save your favorite dishes and track history

## 🛠️ Tech Stack

- **Framework**: Next.js 14.x (App Router)
- **UI**: React 18.x + TailwindCSS 3.x
- **Language**: TypeScript
- **i18n**: i18next + react-i18next
- **HTTP Client**: Axios
- **Icons**: React Icons

## 📋 Prerequisites

- Node.js 18+ or 20+
- npm or yarn
- Backend API running on `http://localhost:8000`

## 🔧 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd thai-food-frontend
```

2. **Install dependencies**:
```bash
npm install
```

3. **Configure environment**:
Create `.env.local` file:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

4. **Run development server**:
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 📁 Project Structure

```
thai-food-frontend/
├── app/                    # Next.js App Router
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Home page
│   └── globals.css        # Global styles
├── components/            # React components
│   ├── ui/               # Reusable UI components
│   └── features/         # Feature-specific components
├── lib/                   # Utilities
│   ├── api-client.ts     # API client (Axios)
│   └── i18n.ts           # i18n configuration
├── config/               # Configuration files
│   └── env.ts            # Environment config
├── types/                # TypeScript types
│   └── api.ts            # API response types
├── public/               # Static assets
│   ├── images/          # Images
│   └── icons/           # Icons
└── package.json         # Dependencies
```

## 🌐 API Integration

The frontend connects to the Backend API with these endpoints:

- `POST /api/recognize` - Food recognition
- `GET /api/food/{name}?lang={th|en}` - Food information
- `GET /api/restaurants/{name}` - Restaurant recommendations
- `GET /api/health` - Health check

## 🎨 Styling

Uses TailwindCSS with custom Thai-inspired color palette:

- **Primary**: Red tones (Thai flag inspired)
- **Secondary**: Green tones (Fresh ingredients)
- **Accent**: Gold/Yellow tones (Royal Thai cuisine)

## 🌍 Multi-language

Supports 2 languages:
- English (en) - Default
- Thai (th)

Language is automatically detected from browser settings and stored in localStorage.

## 📦 Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

## 🔜 Next Steps

1. Implement image upload component
2. Create result display page
3. Add recipe detail page
4. Implement restaurant map view
5. Add favorites system
6. Implement history tracking

## 📝 License

This project is part of an 8-week internship at Hokkaido Information University (HIU).

## 👨‍💻 Author

Developed as part of Cooperative Education Internship Program
Hokkaido Information University, Japan • 2024-2025
