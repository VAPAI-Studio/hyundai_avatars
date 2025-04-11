"""
Test script for the streaming text-to-speech implementation.
"""

import logging
import time
from text_to_speech import TextToSpeech

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_streaming_tts():
    """Test the streaming text-to-speech implementation."""
    logger.info("Testing streaming text-to-speech implementation...")
    
    # Initialize the TextToSpeech class
    tts = TextToSpeech()
    
    # Test text
    test_text = "This is a test of the streaming text-to-speech implementation. It should stream audio directly to Audio2Face with lower latency than the previous implementation."
    
    # Convert text to speech and stream to Audio2Face
    logger.info(f"Converting text to speech: '{test_text}'")
    start_time = time.time()
    success = tts.convert_text_to_speech(test_text)
    end_time = time.time()
    
    if success:
        logger.info(f"Successfully streamed audio to Audio2Face in {end_time - start_time:.2f} seconds")
    else:
        logger.error("Failed to stream audio to Audio2Face")

if __name__ == "__main__":
    test_streaming_tts() 