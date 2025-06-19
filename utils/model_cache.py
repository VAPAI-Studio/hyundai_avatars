"""
Utility functions for managing Silero VAD model cache.
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)

def get_silero_cache_info():
    """Get information about Silero VAD model cache."""
    cache_dir = torch.hub.get_dir()
    model_path = f"{cache_dir}/snakers4_silero-vad_master"
    
    info = {
        "cache_dir": cache_dir,
        "model_path": model_path,
        "is_cached": os.path.exists(model_path),
        "cache_size": 0
    }
    
    if info["is_cached"]:
        try:
            # Calculate cache size
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(model_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            info["cache_size"] = total_size
        except Exception as e:
            logger.warning(f"Could not calculate cache size: {e}")
    
    return info

def print_cache_info():
    """Print information about the Silero VAD model cache."""
    info = get_silero_cache_info()
    
    print("🔍 Silero VAD Model Cache Information:")
    print(f"   Cache Directory: {info['cache_dir']}")
    print(f"   Model Path: {info['model_path']}")
    print(f"   Is Cached: {'✅ Yes' if info['is_cached'] else '❌ No'}")
    
    if info['is_cached']:
        size_mb = info['cache_size'] / (1024 * 1024)
        print(f"   Cache Size: {size_mb:.1f} MB")
    else:
        print("   Cache Size: Not available")
    
    print()

def clear_silero_cache():
    """Clear the Silero VAD model cache."""
    info = get_silero_cache_info()
    
    if info["is_cached"]:
        try:
            import shutil
            shutil.rmtree(info["model_path"])
            print(f"✅ Cleared Silero VAD cache: {info['model_path']}")
            return True
        except Exception as e:
            print(f"❌ Failed to clear cache: {e}")
            return False
    else:
        print("ℹ️  No cache to clear")
        return True

if __name__ == "__main__":
    print_cache_info() 