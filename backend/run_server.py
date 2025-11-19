#!/usr/bin/env python3
"""
Thai Food Recognition Backend Server
Simple startup script for development

Usage:
    python run_server.py
"""

import uvicorn
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    print("=" * 60)
    print("🍜 Thai Food Recognition API Server")
    print("=" * 60)
    print()
    print("📡 Server Information:")
    print("   Host: http://localhost:8000")
    print("   Swagger UI: http://localhost:8000/docs")
    print("   ReDoc: http://localhost:8000/redoc")
    print()
    print("🏥 Quick Test:")
    print("   curl http://localhost:8000/api/health")
    print()
    print("⚡ Auto-reload: ENABLED (save files to restart)")
    print("🛑 Stop server: Press CTRL+C")
    print()
    print("=" * 60)
    print()
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
