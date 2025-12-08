#!/usr/bin/env python3
"""
Test All 20 Thai Foods API
Verifies that all dishes return complete data with ingredients, steps, and times
"""

import requests
import json
import sys
from datetime import datetime
from typing import Dict, List, Tuple

BASE_URL = "http://localhost:8000"

FOODS = [
    # Savory dishes (17)
    "som_tum",
    "tom_yum_goong",
    "larb",
    "pad_thai",
    "kaeng_khiao_wan",
    "khao_soi",
    "kaeng_massaman",
    "pad_krapow",
    "khao_man_gai",
    "khao_kha_mu",
    "tom_kha_gai",
    "gai_pad_med_ma_muang_himmaphan",
    "kai_palo",
    "gung_ob_woon_sen",
    "khao_kluk_kapi",
    "por_pia_tod",
    "hor_mok",
    # Desserts (3)
    "khao_niao_ma_muang",
    "khanom_krok",
    "foi_thong"
]

LANGUAGES = ["en", "th"]


def test_food(food_name: str, language: str = "en") -> Dict:
    """
    Test a single food endpoint
    Returns status dict with all info
    """
    url = f"{BASE_URL}/api/food/{food_name}?lang={language}"
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return {
                'food': food_name,
                'lang': language,
                'status': 'HTTP_ERROR',
                'code': response.status_code,
                'message': f"HTTP {response.status_code}",
                'complete': False
            }
        
        data = response.json()
        
        if not data.get('success'):
            return {
                'food': food_name,
                'lang': language,
                'status': 'API_ERROR',
                'message': data.get('message', 'Unknown API error'),
                'complete': False
            }
        
        recipe = data.get('recipe', {})
        
        # Check all required fields
        has_ingredients = bool(recipe.get('ingredients_by_section') and 
                              len(recipe.get('ingredients_by_section', {})) > 0)
        has_steps = bool(recipe.get('steps') and len(recipe.get('steps', [])) > 0)
        has_prep_time = recipe.get('prep_time') is not None
        has_cook_time = recipe.get('cook_time') is not None
        
        is_complete = has_ingredients and has_steps and has_prep_time and has_cook_time
        
        return {
            'food': food_name,
            'lang': language,
            'status': 'OK',
            'complete': is_complete,
            'has_ingredients': has_ingredients,
            'has_steps': has_steps,
            'has_prep_time': has_prep_time,
            'has_cook_time': has_cook_time,
            'details': {
                'ingredients_count': len(recipe.get('ingredients', [])),
                'sections': len(recipe.get('ingredients_by_section', {})),
                'steps': len(recipe.get('steps', [])),
                'prep_time': recipe.get('prep_time'),
                'cook_time': recipe.get('cook_time'),
                'difficulty': recipe.get('difficulty_text', 'N/A'),
            }
        }
    
    except requests.exceptions.ConnectionError as e:
        return {
            'food': food_name,
            'lang': language,
            'status': 'CONNECTION_ERROR',
            'message': 'Cannot reach API server at ' + BASE_URL,
            'complete': False
        }
    except requests.exceptions.Timeout:
        return {
            'food': food_name,
            'lang': language,
            'status': 'TIMEOUT',
            'message': 'Request timeout (>10s)',
            'complete': False
        }
    except Exception as e:
        return {
            'food': food_name,
            'lang': language,
            'status': 'EXCEPTION',
            'message': str(e),
            'complete': False
        }


def print_header(text: str, char: str = "="):
    """Print formatted header"""
    width = 90
    print()
    print(char * width)
    print(f"  {text}")
    print(char * width)


def print_result(result: Dict, show_details: bool = False):
    """Print test result"""
    food = result['food']
    lang = result['lang']
    status = result['status']
    complete = result.get('complete', False)
    
    # Status icon
    if status == 'OK':
        icon = '✅' if complete else '⚠️ '
        status_text = 'Complete' if complete else 'Incomplete'
    elif status == 'CONNECTION_ERROR':
        icon = '❌'
        status_text = 'Cannot connect'
    elif status == 'HTTP_ERROR':
        icon = '❌'
        status_text = f"HTTP {result['code']}"
    else:
        icon = '❌'
        status_text = status
    
    # Print line
    print(f"{icon} {food:40s} [{lang}] {status_text}")
    
    # Show details if requested and available
    if show_details and 'details' in result:
        d = result['details']
        print(f"     Ingredients: {d['ingredients_count']:2d} | "
              f"Sections: {d['sections']:2d} | "
              f"Steps: {d['steps']:2d} | "
              f"Time: {d['prep_time']}/{d['cook_time']} | "
              f"Difficulty: {d['difficulty']}")
    
    if not result.get('complete') and 'message' in result:
        print(f"     Error: {result['message']}")


def print_summary(results: List[Dict]):
    """Print test summary"""
    total = len(results)
    complete = sum(1 for r in results if r.get('complete'))
    incomplete = sum(1 for r in results if r['status'] == 'OK' and not r.get('complete'))
    errors = total - complete - incomplete
    
    print()
    print("-" * 90)
    print(f"Total:      {total:2d} tests")
    print(f"✅ Complete: {complete:2d} ({complete*100//total:3d}%)")
    print(f"⚠️  Incomplete: {incomplete:2d}")
    print(f"❌ Errors:   {errors:2d}")
    print("-" * 90)
    
    if complete == total:
        print("🎉 SUCCESS! All tests passed!")
    elif complete >= total * 0.9:
        print("✨ Great! 90%+ dishes are complete")
    else:
        print(f"⚠️  Only {complete}/{total} dishes are complete")


def main():
    """Run all tests"""
    print_header("🍜 Thai Food Recognition API - Complete Test Suite")
    print(f"Server: {BASE_URL}")
    print(f"Testing: {len(FOODS)} dishes in {len(LANGUAGES)} languages")
    print(f"Total: {len(FOODS) * len(LANGUAGES)} API calls")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    # Test each food in each language
    for language in LANGUAGES:
        print_header(f"Testing in {language.upper()}", "-")
        
        lang_results = []
        for i, food in enumerate(FOODS, 1):
            result = test_food(food, language)
            lang_results.append(result)
            all_results.append(result)
            
            # Show result immediately
            print_result(result, show_details=(not result.get('complete')))
            
            # Add delay to avoid overwhelming server
            if i < len(FOODS):
                import time
                time.sleep(0.1)
        
        # Summary for this language
        print()
        lang_complete = sum(1 for r in lang_results if r.get('complete'))
        print(f"Language Summary: {lang_complete}/{len(FOODS)} complete")
    
    # Overall summary
    print_header("📊 OVERALL RESULTS")
    print_summary(all_results)
    
    # Detailed incomplete list
    incomplete_results = [r for r in all_results if r['status'] == 'OK' and not r.get('complete')]
    if incomplete_results:
        print()
        print("⚠️  Incomplete Items (details):")
        print("-" * 90)
        for result in incomplete_results:
            food = result['food']
            lang = result['lang']
            has_ing = result.get('has_ingredients', False)
            has_steps = result.get('has_steps', False)
            has_prep = result.get('has_prep_time', False)
            has_cook = result.get('has_cook_time', False)
            
            missing = []
            if not has_ing:
                missing.append("ingredients_by_section")
            if not has_steps:
                missing.append("steps")
            if not has_prep:
                missing.append("prep_time")
            if not has_cook:
                missing.append("cook_time")
            
            print(f"  {food:40s} [{lang}] missing: {', '.join(missing)}")
    
    # Error list
    error_results = [r for r in all_results if r['status'] != 'OK']
    if error_results:
        print()
        print("❌ Error Items:")
        print("-" * 90)
        for result in error_results:
            food = result['food']
            lang = result['lang']
            message = result.get('message', 'Unknown error')
            print(f"  {food:40s} [{lang}] {message}")
    
    # Final status
    print()
    print_header("✅ TEST COMPLETE")
    total = len(all_results)
    complete = sum(1 for r in all_results if r.get('complete'))
    
    if complete == total:
        print(f"🎉 SUCCESS! All {total} tests passed!")
        return 0
    else:
        print(f"⚠️  {complete}/{total} tests passed ({complete*100//total}%)")
        print(f"📋 Fix {total - complete} remaining items and rerun test")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)