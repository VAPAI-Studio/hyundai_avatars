"""
Main application logic for the Hyundai Voice Assistant.
"""

import time
import logging
import os
import signal
import sys
import pyaudio
from datetime import datetime
import torch

# Import our modules
from audio.audio_recorder import AudioRecorder
from audio.speech_to_text import SpeechToText
from ai.ai_processor import AIProcessor
from audio.text_to_speech import TextToSpeech
from audio.audio_player import AudioPlayer
from utils.config import AUDIO_DEVICE
from utils.model_cache import print_cache_info

# Ensure the log directory exists
log_dir = "logs"
try:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created log directory: {log_dir}")
    else:
        print(f"Log directory already exists: {log_dir}")
        
    log_file = os.path.join(log_dir, f"voice_assistant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    print(f"Log file path: {log_file}")
    
    # Test if we can write to the log file
    with open(log_file, 'a') as f:
        f.write("=== Log file write test ===\n")
    print("Successfully wrote to log file")
    
except Exception as e:
    print(f"Error setting up logging: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def list_audio_devices():
    """List all available audio input devices."""
    print("DEBUG: Starting list_audio_devices()")
    try:
        p = pyaudio.PyAudio()
        devices = []
        
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:  # Only include input devices
                devices.append((i, device_info.get('name')))
        
        p.terminate()
        print(f"DEBUG: Found {len(devices)} audio input devices")
        return devices
    except Exception as e:
        print(f"DEBUG: Error in list_audio_devices: {e}")
        return []

def print_ai_models_info(ai_processor):
    """Print information about available AI models."""
    print("\n=== Available AI Models ===")
    models_info = ai_processor.get_available_models()
    
    for model_name, info in models_info.items():
        status = info["status"]
        if status == "available":
            model = info.get("model", "Unknown")
            if model_name == "local_llm" and "available_models" in info:
                available_models = info["available_models"]
                print(f"✅ {model_name.upper()}: {model} (Available models: {', '.join(available_models)})")
            else:
                print(f"✅ {model_name.upper()}: {model}")
        else:
            reason = info.get("reason", "Unknown reason")
            print(f"❌ {model_name.upper()}: {reason}")
    
    print("==========================\n")

class VoiceAssistant:
    def __init__(self):
        logger.info("Initializing Voice Assistant...")
        
        # Show Silero VAD model cache information
        print_cache_info()
        
        # List available audio devices
        devices = list_audio_devices()
        
        if devices:
            logger.info("\nAvailable audio input devices:")
            for idx, (device_id, device_name) in enumerate(devices):
                logger.info(f"  {idx+1}. ID {device_id}: {device_name}")
            
            if AUDIO_DEVICE is not None:
                logger.info(f"\nUsing configured device: {AUDIO_DEVICE}")
            else:
                logger.info("\nUsing default microphone")
        else:
            logger.warning("No audio input devices found!")
            
        self.recorder = AudioRecorder()
        self.speech_to_text = SpeechToText()
        self.ai_processor = AIProcessor()
        self.text_to_speech = TextToSpeech()
        self.audio_player = AudioPlayer()
        self.running = False
        
        # Show AI models information
        print_ai_models_info(self.ai_processor)
        
        # Conversation memory
        self.conversation_history = []
        self.last_conversation_time = time.time()
        self.conversation_timeout = 180  # 3 minutes in seconds
        print("DEBUG: VoiceAssistant initialization complete")
        
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            self.stop()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def start(self):
        """Start the voice assistant."""
        self.setup_signal_handlers()
        self.running = True
        self.recorder.start_listening()
        logger.info("Voice Assistant started. Listening for speech...")
        
        try:
            while self.running:
                # Wait for speech to be detected
                if self.recorder.wait_for_speech(timeout=None):
                    # Process detected speech
                    self._process_speech()
                    # Reset for next detection
                    self.recorder.reset_detection_event()
                    
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            self.stop()
            
    def stop(self):
        """Stop the voice assistant."""
        logger.info("Stopping Voice Assistant...")
        self.running = False
        self.recorder.stop_listening()
        
    def _check_conversation_timeout(self):
        """Check if the conversation has timed out and reset if needed."""
        current_time = time.time()
        if current_time - self.last_conversation_time > self.conversation_timeout:
            logger.info("Conversation timed out after 3 minutes of inactivity. Resetting memory.")
            self.conversation_history = []
            return True
        return False
        
    def _process_speech(self):
        """Process detected speech and generate a response."""
        try:
            # Convert speech to text
            stt_start = time.time()
            text = self.speech_to_text.convert_audio_to_text()
            stt_time = time.time() - stt_start
            
            if not text:
                return
                
            # Process with AI model
            ai_start = time.time()
            model, response = self.ai_processor.process_with_all_available(text, self.conversation_history)
            ai_time = time.time() - ai_start
            
            if not response or not model:
                return
                
            # Convert response to speech and stream directly to Audio2Face
            tts_start = time.time()
            if self.text_to_speech.convert_text_to_speech(response):
                tts_time = time.time() - tts_start
                
                # Log all timings
                total_time = time.time() - stt_start
                print("\n=== Processing Times ===")
                print(f"Speech to Text: {stt_time:.2f}s")
                print(f"AI Processing ({model}): {ai_time:.2f}s")
                print(f"Text to Speech & Streaming: {tts_time:.2f}s")
                print(f"Total Time: {total_time:.2f}s")
                print("=====================")
                
        except Exception as e:
            logger.error(f"Error processing speech: {e}")

def main():
    """Main entry point for the Hyundai Voice Assistant."""
    try:
        assistant = VoiceAssistant()
        assistant.start()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main()) 