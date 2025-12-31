#!/usr/bin/env python3
"""
LAUNCH ADVANCED CONSCIOUSNESS NEXUS GUI
========================================

Launch the sophisticated Consciousness Nexus GUI with full integration
to the consciousness computing suite.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")

    try:
        import flask
        print("✅ Flask available")
    except ImportError:
        print("❌ Flask not installed. Installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flask'])
        print("✅ Flask installed")

    try:
        import flask_cors
        print("✅ Flask-CORS available")
    except ImportError:
        print("⚠️  Flask-CORS not available (optional)")

def launch_gui():
    """Launch the advanced GUI"""
    print("🚀 Launching Consciousness Nexus Advanced GUI")
    print("=" * 55)

    # Check if GUI file exists
    gui_file = Path("consciousness_nexus_gui.py")
    if not gui_file.exists():
        print("❌ consciousness_nexus_gui.py not found!")
        return False

    # Check dependencies
    check_dependencies()

    print("\n🔮 Consciousness Nexus Advanced GUI Features:")
    print("   🎨 Modern, sophisticated web interface")
    print("   ⚡ Real-time system metrics and monitoring")
    print("   🧠 ABYSSAL template execution with live feedback")
    print("   🔒 Integrated consciousness security dashboard")
    print("   💡 2026 innovation pipeline visualization")
    print("   📊 Advanced analytics and performance metrics")
    print("   🎭 Matrix-style visual effects and animations")
    print("   🔄 Live activity logging and status updates")
    print()

    print("🌐 GUI will be available at: http://localhost:5000")
    print("📱 Features:")
    print("   • Interactive ABYSSAL template execution")
    print("   • Real-time consciousness metrics")
    print("   • Security system monitoring")
    print("   • 2026 innovation showcase")
    print("   • Advanced system analytics")
    print()

    try:
        # Launch the GUI server
        print("🔄 Starting GUI server...")
        subprocess.run([sys.executable, "consciousness_nexus_gui.py"], check=True)

    except KeyboardInterrupt:
        print("\n👋 GUI server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ GUI server failed to start: {e}")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False

    return True

def main():
    """Main launcher function"""
    print("CONSCIOUSNESS NEXUS - ADVANCED GUI LAUNCHER")
    print("=" * 50)

    success = launch_gui()

    if success:
        print("\n🎉 GUI session completed successfully!")
    else:
        print("\n❌ GUI launch failed. Check error messages above.")
        print("💡 Make sure all dependencies are installed and the system is properly configured.")

if __name__ == '__main__':
    main()
