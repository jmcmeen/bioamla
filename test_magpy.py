#!/usr/bin/env python3
"""Test script for magpy-lite TUI."""

import sys

# Add src to path for testing
sys.path.insert(0, 'src')

from bioamla.tui.magpy import MagpyLite

if __name__ == "__main__":
    print("Starting magpy-lite test...")
    try:
        app = MagpyLite(start_dir=".")
        print("App created successfully")
        print("Starting app...")
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
