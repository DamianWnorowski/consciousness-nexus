#!/usr/bin/env python3
"""
Consciousness Computing Suite Launcher
=====================================

Proper launcher for the consciousness computing suite that handles
imports and initialization correctly.
"""

import sys
import os
from pathlib import Path

# Add the consciousness_suite directory to Python path
suite_dir = Path(__file__).parent / "consciousness_suite"
sys.path.insert(0, str(suite_dir))

try:
    # Import and run the main suite
    from consciousness_suite.main import main

    print("🔮 CONSCIOUSNESS COMPUTING SUITE LAUNCHER 🔮")
    print("=" * 50)
    print()

    if __name__ == "__main__":
        # Run the async main function
        import asyncio
        asyncio.run(main())

except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)
except Exception as e:
    print(f"❌ Runtime Error: {e}")
    sys.exit(1)
