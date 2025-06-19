#!/usr/bin/env python3
"""
Script to check and manage Silero VAD model cache.
"""

import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_cache import print_cache_info, clear_silero_cache

def main():
    """Main function to check and manage cache."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "clear":
            print("🗑️  Clearing Silero VAD cache...")
            clear_silero_cache()
        elif command == "info":
            print_cache_info()
        elif command == "help":
            print("Usage:")
            print("  python utils/check_cache.py          # Show cache info")
            print("  python utils/check_cache.py info     # Show cache info")
            print("  python utils/check_cache.py clear    # Clear cache")
            print("  python utils/check_cache.py help     # Show this help")
        else:
            print(f"Unknown command: {command}")
            print("Use 'help' to see available commands")
    else:
        print_cache_info()

if __name__ == "__main__":
    main() 