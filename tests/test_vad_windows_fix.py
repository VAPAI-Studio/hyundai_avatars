#!/usr/bin/env python3
"""
Test script for Windows-specific Silero VAD loading issues.
"""

import sys
import os
import time
import logging
import torch

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_torch_hub_loading():
    """Test different approaches to load Silero VAD model."""
    print("🔍 Testing Silero VAD model loading on Windows...")
    
    # Test 1: Basic torch.hub.load
    print("\n1️⃣ Testing basic torch.hub.load...")
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', 
            model='silero_vad', 
            force_reload=False,
            trust_repo=True
        )
        print("✅ Basic torch.hub.load successful!")
        return True
    except Exception as e:
        print(f"❌ Basic torch.hub.load failed: {e}")
    
    # Test 2: With Windows-specific workarounds
    print("\n2️⃣ Testing with Windows workarounds...")
    try:
        import platform
        if platform.system() == "Windows":
            cache_dir = torch.hub.get_dir()
            os.environ['TORCH_HOME'] = cache_dir
            os.environ['HF_HOME'] = cache_dir
            
            import gc
            gc.collect()
            time.sleep(1)
        
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', 
            model='silero_vad', 
            force_reload=False,
            trust_repo=True
        )
        print("✅ Windows workarounds successful!")
        return True
    except Exception as e:
        print(f"❌ Windows workarounds failed: {e}")
    
    # Test 3: Direct import
    print("\n3️⃣ Testing direct silero_vad import...")
    try:
        from silero_vad import load_model, get_speech_timestamps
        model = load_model()
        print("✅ Direct import successful!")
        return True
    except ImportError:
        print("❌ silero_vad package not available")
    except Exception as e:
        print(f"❌ Direct import failed: {e}")
    
    # Test 4: Manual download
    print("\n4️⃣ Testing manual download...")
    try:
        import urllib.request
        import zipfile
        import tempfile
        
        cache_dir = torch.hub.get_dir()
        model_path = f"{cache_dir}/snakers4_silero-vad_master"
        
        if not os.path.exists(model_path):
            print("Downloading model manually...")
            model_url = "https://github.com/snakers4/silero-vad/archive/refs/heads/master.zip"
            
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                urllib.request.urlretrieve(model_url, temp_file.name)
                
                with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
                    zip_ref.extractall(cache_dir)
            
            os.unlink(temp_file.name)
        
        model, utils = torch.hub.load(
            repo_or_dir=model_path,
            model='silero_vad',
            source='local'
        )
        print("✅ Manual download successful!")
        return True
    except Exception as e:
        print(f"❌ Manual download failed: {e}")
    
    return False

def test_audio_recorder():
    """Test the AudioRecorder class with the new fixes."""
    print("\n🎤 Testing AudioRecorder with fixes...")
    try:
        from audio.audio_recorder import AudioRecorder
        recorder = AudioRecorder()
        print("✅ AudioRecorder created successfully!")
        return True
    except Exception as e:
        print(f"❌ AudioRecorder failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Windows Silero VAD Loading Test")
    print("=" * 50)
    
    # Check system info
    import platform
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Test torch.hub loading
    if test_torch_hub_loading():
        print("\n🎉 At least one loading method worked!")
    else:
        print("\n💥 All loading methods failed!")
    
    # Test AudioRecorder
    if test_audio_recorder():
        print("\n🎉 AudioRecorder works!")
    else:
        print("\n💥 AudioRecorder failed!")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    main() 