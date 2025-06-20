#!/usr/bin/env python3
"""
Hyundai Avatars Voice Assistant
Main entry point for the voice assistant application.
"""

import sys
import os
import signal

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== Starting Hyundai Avatars Voice Assistant ===")
print("Testing imports...")

try:
    print("Importing SpeechToText...")
    from audio.speech_to_text import SpeechToText
    print("SpeechToText imported successfully")
except Exception as e:
    print(f"ERROR importing SpeechToText: {e}")
    import traceback
    traceback.print_exc()

try:
    print("Importing StreamingLLMProcessor...")
    from ai.streaming_llm_processor import StreamingLLMProcessor
    print("StreamingLLMProcessor imported successfully")
except Exception as e:
    print(f"ERROR importing StreamingLLMProcessor: {e}")
    import traceback
    traceback.print_exc()

try:
    print("Importing StreamingTTSProcessor...")
    from audio.streaming_tts_processor import StreamingTTSProcessor
    print("StreamingTTSProcessor imported successfully")
except Exception as e:
    print(f"ERROR importing StreamingTTSProcessor: {e}")
    import traceback
    traceback.print_exc()

try:
    print("Importing AudioRecorder...")
    from audio.audio_recorder import AudioRecorder
    print("AudioRecorder imported successfully")
except Exception as e:
    print(f"ERROR importing AudioRecorder: {e}")
    import traceback
    traceback.print_exc()

print("All imports successful!")

def main():
    """Main entry point for the Hyundai Voice Assistant."""
    print("=== Starting Hyundai Avatars Voice Assistant ===")
    
    try:
        print("Step 1: Creating VoiceAssistant instance...")
        assistant = VoiceAssistant()
        print("Step 2: VoiceAssistant created successfully")
        
        print("Step 3: Starting VoiceAssistant...")
        assistant.start()
        print("Step 4: VoiceAssistant started successfully")
        
    except Exception as e:
        print(f"ERROR during initialization: {e}")
        import traceback
        traceback.print_exc()
        print("Press Enter to exit...")
        input()
    return 0

class VoiceAssistant:
    def __init__(self):
        """Initialize the Voice Assistant with all components."""
        print("Initializing Voice Assistant...")
        
        try:
            print("  - Step 1: Initializing SpeechToText...")
            self.speech_to_text = SpeechToText()
            print("  - Step 1: SpeechToText initialized")
            
            print("  - Step 2: Initializing StreamingLLMProcessor...")
            self.streaming_llm_processor = StreamingLLMProcessor()
            print("  - Step 2: StreamingLLMProcessor initialized")
            
            print("  - Step 3: Initializing StreamingTTSProcessor...")
            self.streaming_tts_processor = StreamingTTSProcessor()
            print("  - Step 3: StreamingTTSProcessor initialized")
            
            print("  - Step 4: Initializing AudioRecorder...")
            self.recorder = AudioRecorder()
            print("  - Step 4: AudioRecorder initialized")
            
            print("  - Step 5: Setting up streaming callbacks...")
            self._setup_streaming_callbacks()
            print("  - Step 5: Streaming callbacks set up")
            
            # State management
            self.running = False
            
            print("Voice Assistant initialized successfully!")
            
        except Exception as e:
            print(f"ERROR in VoiceAssistant initialization: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    def _setup_streaming_callbacks(self):
        """Set up callbacks for the streaming TTS processor."""
        def on_chunk_processed(chunk, status):
            print(f"TTS Chunk processed: '{chunk.text[:30]}...' - {status}")
        
        def on_audio_ready(audio_data):
            print(f"Audio ready: {len(audio_data)} samples")
        
        def on_streaming_complete():
            print("Streaming TTS completed")
        
        self.streaming_tts_processor.set_callbacks(
            on_chunk_processed=on_chunk_processed,
            on_audio_ready=on_audio_ready,
            on_streaming_complete=on_streaming_complete
        )
        
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            print("Shutdown signal received")
            self.stop()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def start(self):
        """Start the voice assistant."""
        self.setup_signal_handlers()
        self.running = True
        self.recorder.start_listening()
        print("Voice Assistant started. Listening for speech...")
        
        try:
            while self.running:
                # Wait for speech to be detected
                if self.recorder.wait_for_speech(timeout=None):
                    # Process detected speech
                    self._process_speech()
                    # Reset for next detection
                    self.recorder.reset_detection_event()
                    
        except Exception as e:
            print(f"Error in main loop: {e}")
            self.stop()
            
    def _process_speech(self):
        """Process detected speech through the pipeline."""
        try:
            print("Speech detected! Processing...")
            
            # Convert speech to text
            audio_data = self.recorder.get_audio_data()
            if audio_data is None:
                print("No audio data available")
                return
                
            print("Converting speech to text...")
            text = self.speech_to_text.convert(audio_data)
            if not text or text.strip() == "":
                print("No text detected from speech")
                return
                
            print(f"Transcribed text: '{text}'")
            
            # Generate response using streaming LLM
            print("Generating response with streaming LLM...")
            response_text = self.streaming_llm_processor.generate_response(text)
            if not response_text:
                print("No response generated")
                return
                
            print(f"Generated response: '{response_text}'")
            
            # Convert response to speech using streaming TTS
            print("Converting response to speech...")
            self.streaming_tts_processor.process_text(response_text)
            
        except Exception as e:
            print(f"Error processing speech: {e}")
            import traceback
            traceback.print_exc()
            
    def stop(self):
        """Stop the voice assistant."""
        print("Stopping Voice Assistant...")
        self.running = False
        if self.recorder:
            self.recorder.stop()
        print("Voice Assistant stopped")

if __name__ == "__main__":
    print("Starting main function...")
    main()
    print("Main function completed") 