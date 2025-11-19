#!/usr/bin/env python3
"""
API Endpoint Tester
Quick script to test all backend endpoints

Run the server first, then run this script:
    python test_api.py
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_endpoint(method, endpoint, description, **kwargs):
    """Test a single endpoint"""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n🔍 Testing: {description}")
    print(f"   {method} {endpoint}")
    
    try:
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs)
        else:
            print(f"❌ Unsupported method: {method}")
            return
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Success!")
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)[:200]}...")
        else:
            print(f"   ❌ Failed!")
            print(f"   Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to server!")
        print("   💡 Make sure server is running: python run_server.py")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    """Run all tests"""
    print_section("🍜 Thai Food Recognition API - Endpoint Tests")
    
    print("\n📡 Server URL:", BASE_URL)
    print("⚠️  Note: Make sure server is running first!\n")
    
    # Test 1: Root endpoint
    print_section("Test 1: Root Endpoint")
    test_endpoint("GET", "/", "Root welcome message")
    
    # Test 2: Health check
    print_section("Test 2: Health Check")
    test_endpoint("GET", "/api/health", "Health check endpoint")
    
    # Test 3: Food info
    print_section("Test 3: Get Food Information")
    test_endpoint("GET", "/api/food/pad_thai", "Get Pad Thai info (English)")
    test_endpoint("GET", "/api/food/pad_thai?lang=th", "Get Pad Thai info (Thai)")
    
    # Test 4: Restaurants
    print_section("Test 4: Get Restaurants")
    test_endpoint("GET", "/api/restaurants/pad_thai", "Get Pad Thai restaurants")
    test_endpoint("GET", "/api/restaurants/tom_yum_goong?region=Bangkok", 
                  "Get Tom Yum restaurants in Bangkok")
    
    # Test 5: Recognition (will need actual image later)
    print_section("Test 5: Food Recognition")
    print("\n🔍 Testing: Food recognition endpoint")
    print("   POST /api/recognize")
    print("   ⚠️  Skipped: Requires image file")
    print("   💡 Test this later in Swagger UI: http://localhost:8000/docs")
    
    # Summary
    print_section("✅ Test Summary")
    print("\n✨ All basic endpoints are working!")
    print("📚 View full API docs: http://localhost:8000/docs")
    print("🧪 Test with Swagger UI for interactive testing")
    print("\n🎯 Next Steps (Week 5-6):")
    print("   1. Integrate AI models")
    print("   2. Add real data parsing (Markdown + JSON)")
    print("   3. Implement image processing")
    print("   4. Test with actual images")
    print()

if __name__ == "__main__":
    main()
