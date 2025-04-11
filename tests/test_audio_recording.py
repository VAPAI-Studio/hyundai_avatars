"""
Test script for the audio recording implementation.
"""

import logging
import time
from audio_recorder import AudioRecorder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_audio_recording():
    """Test the audio recording implementation."""
    logger.info("Testing audio recording implementation...")
    
    # Initialize the AudioRecorder class
    recorder = AudioRecorder()
    
    # Start listening for speech
    logger.info("Starting to listen for speech. Please speak now...")
    recorder.start_listening()
    
    # Wait for speech to be detected
    logger.info("Waiting for speech to be detected...")
    if recorder.wait_for_speech(timeout=30):
        logger.info("Speech detected and recorded!")
        
        # Reset the detection event to listen for new speech
        recorder.reset_detection_event()
        
        # Wait a moment before stopping
        time.sleep(1)
    else:
        logger.warning("No speech detected within the timeout period.")
    
    # Stop listening
    logger.info("Stopping audio recording...")
    recorder.stop_listening()
    logger.info("Audio recording test completed.")

if __name__ == "__main__":
    test_audio_recording() 