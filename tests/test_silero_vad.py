"""
Test script for Silero VAD implementation.
"""

import sys
import os
import time
import logging

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.audio_recorder import AudioRecorder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_silero_vad():
    """Test the Silero VAD implementation."""
    print("Testing Silero VAD implementation...")
    
    try:
        # Create audio recorder
        recorder = AudioRecorder()
        print("✓ AudioRecorder created successfully")
        
        # Start listening
        recorder.start_listening()
        print("✓ Started listening with Silero VAD")
        
        print("\n🎤 Speak into the microphone to test VAD...")
        print("Press Ctrl+C to stop")
        
        # Wait for speech detection
        while True:
            if recorder.wait_for_speech(timeout=1.0):
                print("🎯 Speech detected and recorded!")
                recorder.reset_detection_event()
                print("🔄 Ready for next speech...")
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Stopping test...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'recorder' in locals():
            recorder.stop_listening()
            print("✓ Stopped listening")

if __name__ == "__main__":
    test_silero_vad() 