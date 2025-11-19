"""
Data Service - FIXED TIPS VERSION
Fixed _simple_extract_tips to actually work
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import re


class DataService:
    """Service for parsing and serving food and restaurant data"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize Data Service"""
        if data_dir is None:
            base_path = Path(__file__).parent.parent.parent.parent
            data_dir = base_path / "data"
        
        self.data_dir = Path(data_dir)
        self.foods_dir = self.data_dir / "foods"
        self.restaurants_file = self.data_dir / "restaurants" / "thai_restaurants.json"
        self._restaurants_cache = None
        
        print(f"📂 Data Service initialized")
    
    def _simple_extract_ingredients(self, content: str) -> tuple:
        """Simple extraction - get all lines after #### in ingredients section"""
        all_ingredients = []
        
        lines = content.split('\n')
        in_ingredients_section = False
        in_subsection = False
        current_section_name = ""
        sections = {}
        
        for i, line in enumerate(lines):
            # Check if we're entering ingredients section
            if re.search(r'###\s*(?:ส่วนผสม|Ingredients)', line):
                in_ingredients_section = True
                continue
            
            # Check if we're leaving ingredients section
            if in_ingredients_section and re.match(r'###\s+(?:วิธีทำ|Cooking Instructions|วิธี|Instructions)', line):
                in_ingredients_section = False
                break
            
            # Check for subsections (####)
            if in_ingredients_section and line.startswith('####'):
                in_subsection = True
                current_section_name = line.replace('####', '').strip()
                sections[current_section_name] = []
                continue
            
            # Collect ingredients
            if in_ingredients_section and line.strip().startswith('- '):
                ingredient = line.strip()[2:].strip()  # Remove "- "
                if ingredient and len(ingredient) > 3:
                    all_ingredients.append(ingredient)
                    if current_section_name:
                        sections[current_section_name].append(ingredient)
        
        return all_ingredients, sections
    
    def _simple_extract_tips(self, content: str) -> str:
        """FIXED: Simple extraction - get everything after 💡 until next major heading"""
        lines = content.split('\n')
        in_tips = False
        tips_lines = []
        skip_first_empty = True
        
        for line in lines:
            # Start collecting when we see 💡
            if '💡' in line and line.startswith('###'):
                in_tips = True
                skip_first_empty = True
                continue
            
            # If in tips section
            if in_tips:
                # Skip first empty line after heading
                if skip_first_empty and not line.strip():
                    skip_first_empty = False
                    continue
                
                # Stop when we hit another ### or ## (but make sure line is not empty)
                if line.strip() and (line.startswith('###') or line.startswith('##') or line.startswith('---')):
                    break
                
                # Collect this line
                tips_lines.append(line)
                skip_first_empty = False
        
        result = '\n'.join(tips_lines).strip()
        
        # Debug print
        if result:
            print(f"✅ Tips extracted: {len(result)} characters")
        else:
            print("⚠️  Tips extraction returned empty!")
        
        return result
    
    def parse_markdown(self, file_path: Path) -> Dict:
        """Parse markdown file - simple version"""
        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract ingredients using simple method
        ingredients, ingredients_by_section = self._simple_extract_ingredients(content)
        
        # Extract tips using simple method
        tips = self._simple_extract_tips(content)
        
        result = {
            'raw_content': content,
            'title': self._extract_title(content),
            'general_info': self._extract_general_info(content),
            'cultural_story': self._extract_cultural_story(content),
            'recipe': {
                'ingredients': ingredients,
                'ingredients_by_section': ingredients_by_section,
                'steps': self._extract_steps(content),
                **self._extract_time_and_difficulty(content)
            },
            'tips': tips
        }
        
        return result
    
    def _extract_title(self, content: str) -> str:
        """Extract main title"""
        match = re.search(r'^#\s+(.+?)(?:\s+\([^)]*\))?\s*$', content, re.MULTILINE)
        return match.group(1).strip() if match else ""
    
    def _extract_general_info(self, content: str) -> Dict:
        """Extract general information"""
        info = {}
        pattern = r'##\s*📖[^\n]*\n(.*?)(?=\n##|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            section = match.group(1).strip()
            lines = section.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('- **'):
                    if ':**' in line:
                        parts = line.split(':**', 1)
                    elif '**: ' in line:
                        parts = line.split('**: ', 1)
                    elif '**:' in line:
                        parts = line.split('**:', 1)
                    else:
                        continue
                    
                    if len(parts) == 2:
                        key = parts[0].replace('- **', '').replace('**', '').strip()
                        value = parts[1].strip()
                        if key and value:
                            info[key] = value
        
        return info
    
    def _extract_cultural_story(self, content: str) -> str:
        """Extract cultural story"""
        pattern = r'##\s*🌟[^\n]*\n\n(.*?)(?=\n##|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _extract_steps(self, content: str) -> List[Dict]:
        """Extract cooking steps"""
        steps = []
        step_pattern = r'\*\*(Step|ขั้นตอนที่)\s+\d+:\s+([^\*\n]+)\*\*\s*\n([^\*]+?)(?=\*\*(?:Step|ขั้นตอนที่)|\n###|\Z)'
        
        for match in re.finditer(step_pattern, content, re.DOTALL | re.IGNORECASE):
            step_title = match.group(2).strip()
            step_content = match.group(3).strip()
            step_content = re.sub(r'\n+', ' ', step_content)
            step_content = re.sub(r'\s+', ' ', step_content)
            
            steps.append({
                'title': step_title,
                'content': step_content,
                'full_text': f"{step_title}: {step_content}"
            })
        
        return steps
    
    def _extract_time_and_difficulty(self, content: str) -> Dict:
        """Extract time and difficulty"""
        result = {}
        
        # Extract time
        time_section = re.search(r'###\s+(?:Cooking Time|เวลาในการทำ)\s*\n(.*?)(?=\n###|\Z)', content, re.DOTALL)
        if time_section:
            time_content = time_section.group(1)
            
            prep_match = re.search(r'(?:Prep time|เวลาเตรียม):\s*(\d+)', time_content)
            if prep_match:
                result['prep_time'] = prep_match.group(1)
            
            cook_match = re.search(r'(?:Cook time|เวลาประกอบอาหาร|เวลาทำ):\s*(\d+)', time_content)
            if cook_match:
                result['cook_time'] = cook_match.group(1)
            
            total_match = re.search(r'(?:Total|รวม):\s*(\d+)', time_content)
            if total_match:
                result['total_time'] = total_match.group(1)
        
        # Extract difficulty
        diff_match = re.search(r'###\s+(?:Difficulty Level|ระดับความยาก)\s*\n([^\n]+)', content)
        if diff_match:
            diff_text = diff_match.group(1).strip()
            stars = diff_text.count('⭐') or diff_text.count('★')
            result['difficulty'] = stars if stars > 0 else None
            result['difficulty_text'] = diff_text
        
        return result
    
    def get_food_info(self, food_name: str, language: str = "en") -> Dict:
        """Get food information"""
        file_name = f"{food_name}_{language}.md"
        file_path = self.foods_dir / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"Food information not found: {food_name} ({language})")
        
        try:
            parsed_data = self.parse_markdown(file_path)
        except Exception as e:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                'food_name': food_name,
                'language': language,
                'title': food_name,
                'general_info': {},
                'cultural_story': content[:1000],
                'recipe': {'ingredients': [], 'steps': [], 'parse_error': str(e)},
                'tips': '',
                'raw_content': content
            }
        
        # Return with tips!
        print(f"📤 Returning food_info with tips: {len(parsed_data['tips'])} characters")
        
        return {
            'food_name': food_name,
            'language': language,
            'title': parsed_data['title'],
            'general_info': parsed_data['general_info'],
            'cultural_story': parsed_data['cultural_story'],
            'recipe': parsed_data['recipe'],
            'tips': parsed_data['tips']
        }
    
    def list_available_foods(self) -> List[str]:
        """List all available foods"""
        if not self.foods_dir.exists():
            return []
        
        md_files = self.foods_dir.glob("*.md")
        food_names = set()
        
        for file in md_files:
            name = file.stem
            if name.endswith('_th') or name.endswith('_en'):
                food_name = name.rsplit('_', 1)[0]
                food_names.add(food_name)
        
        return sorted(list(food_names))
    
    # ==================== Restaurant Information ====================
    
    def _load_restaurants(self) -> List[Dict]:
        """Load restaurants"""
        if self._restaurants_cache is not None:
            return self._restaurants_cache
        
        if not self.restaurants_file.exists():
            raise FileNotFoundError(f"Restaurants file not found")
        
        with open(self.restaurants_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._restaurants_cache = data.get('restaurants', [])
        return self._restaurants_cache
    
    def get_restaurants_by_food(
        self, 
        food_name: str, 
        region: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get restaurants by food"""
        restaurants = self._load_restaurants()
        filtered = [r for r in restaurants if food_name in r.get('specialty', [])]
        
        if region:
            filtered = [r for r in filtered if r.get('region') == region]
        
        if limit:
            filtered = filtered[:limit]
        
        return filtered
    
    def get_restaurant_by_id(self, restaurant_id: str) -> Optional[Dict]:
        """Get restaurant by ID"""
        restaurants = self._load_restaurants()
        for restaurant in restaurants:
            if restaurant.get('id') == restaurant_id:
                return restaurant
        return None
    
    def search_restaurants(
        self,
        query: Optional[str] = None,
        region: Optional[str] = None,
        min_rating: Optional[float] = None,
        limit: Optional[int] = 10
    ) -> List[Dict]:
        """Search restaurants"""
        restaurants = self._load_restaurants()
        
        if query:
            query_lower = query.lower()
            restaurants = [
                r for r in restaurants
                if query_lower in r.get('name_th', '').lower() 
                or query_lower in r.get('name_en', '').lower()
            ]
        
        if region:
            restaurants = [r for r in restaurants if r.get('region') == region]
        
        if min_rating is not None:
            restaurants = [r for r in restaurants if r.get('rating', 0) >= min_rating]
        
        if limit:
            restaurants = restaurants[:limit]
        
        return restaurants


# ==================== Singleton ====================

_data_service_instance: Optional[DataService] = None


def get_data_service() -> DataService:
    global _data_service_instance
    if _data_service_instance is None:
        raise RuntimeError("Data service not initialized!")
    return _data_service_instance


def initialize_data_service(data_dir: Optional[Path] = None) -> DataService:
    global _data_service_instance
    _data_service_instance = DataService(data_dir=data_dir)
    return _data_service_instance