"""
Module for audio recording and speech detection using Silero VAD.
"""

import wave
import numpy as np
import time
import os
from threading import Thread, Event
import logging
import sounddevice as sd
import torch
from collections import deque
from utils.config import (
    SAMPLE_RATE, CHANNELS, MIN_PHRASE_DURATION, TEMP_AUDIO_PATH,
    SILENCE_TIMEOUT, MIN_SPEECH_DURATION
)
from audio.audio_player import AudioPlayer

logger = logging.getLogger(__name__)

# Constants
MAX_RECORDING_TIME = 10.0  # Maximum time to record after speech is detected
CHUNK_SIZE = 16000  # 1 second chunks for VAD processing
PRE_BUFFER_SIZE = 2  # Number of seconds to keep in pre-buffer

class AudioRecorder:
    def __init__(self):
        print("DEBUG: AudioRecorder __init__ started")
        self.rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.min_phrase_duration = MIN_PHRASE_DURATION
        self.stop_event = Event()
        self.audio_detected_event = Event()
        self.recording_thread = None
        self.audio_player = AudioPlayer()
        print("DEBUG: AudioRecorder basic initialization complete")
        
        # Silero VAD model
        self.model = None
        self.utils = None
        print("DEBUG: About to load VAD model")
        self._load_vad_model()
        print("DEBUG: VAD model loaded successfully")
        
        # Audio buffers
        self.audio_buffer = deque(maxlen=int(SAMPLE_RATE * PRE_BUFFER_SIZE))
        self.recording_frames = []
        self.is_recording = False
        print("DEBUG: AudioRecorder initialization complete")
        
    def _load_vad_model(self):
        """Load the Silero VAD model."""
        try:
            print("DEBUG: _load_vad_model started")
            logger.info("Loading Silero VAD model...")
            
            # Check if model is already cached
            cache_dir = torch.hub.get_dir()
            model_path = f"{cache_dir}/snakers4_silero-vad_master"
            if os.path.exists(model_path):
                logger.info("Using cached Silero VAD model")
                print("DEBUG: Using cached model")
            else:
                logger.info("Downloading Silero VAD model (this may take a few minutes on first run)...")
                print("DEBUG: Will download model")
            
            # Windows-specific workaround for torch.hub.load issues
            import platform
            if platform.system() == "Windows":
                print("DEBUG: Windows detected, using workaround for torch.hub.load")
                # Set environment variables to help with Windows file handling
                os.environ['TORCH_HOME'] = cache_dir
                os.environ['HF_HOME'] = cache_dir
                
                # Try to clear any existing file handles
                import gc
                gc.collect()
                
                # Add a small delay to let Windows file system settle
                import time
                time.sleep(1)
            
            print("DEBUG: About to call torch.hub.load")
            
            # Try multiple approaches for loading the model
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.model, self.utils = torch.hub.load(
                        repo_or_dir='snakers4/silero-vad', 
                        model='silero_vad', 
                        force_reload=False,  # Use cached model if available
                        trust_repo=True  # Trust the repository
                    )
                    print(f"DEBUG: torch.hub.load completed successfully on attempt {attempt + 1}")
                    break
                except Exception as e:
                    print(f"DEBUG: Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        print(f"DEBUG: Retrying in 2 seconds...")
                        time.sleep(2)
                        # Force garbage collection before retry
                        gc.collect()
                    else:
                        # If all torch.hub.load attempts fail, try alternative method
                        print("DEBUG: All torch.hub.load attempts failed, trying alternative method")
                        self._load_vad_model_alternative()
                        return
            
            self.get_speech_timestamps = self.utils[0]
            print("DEBUG: get_speech_timestamps assigned")
            logger.info("Silero VAD model loaded successfully")
            print("DEBUG: _load_vad_model completed successfully")
            
        except Exception as e:
            print(f"DEBUG: Error in _load_vad_model: {e}")
            logger.error(f"Failed to load Silero VAD model: {e}")
            
            # Provide helpful error message and suggestions
            error_msg = f"Failed to load Silero VAD model: {e}"
            if "Controlador no válido" in str(e) or "Invalid handle" in str(e):
                error_msg += "\n\nThis is a Windows-specific file handle issue. Try the following:"
                error_msg += "\n1. Restart your Python environment"
                error_msg += "\n2. Clear the model cache: python utils/check_cache.py clear"
                error_msg += "\n3. Run as administrator if the issue persists"
                error_msg += "\n4. Check if antivirus software is blocking the download"
            
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
    def _load_vad_model_alternative(self):
        """Alternative method to load Silero VAD model using direct import."""
        try:
            print("DEBUG: Using alternative VAD model loading method")
            logger.info("Loading Silero VAD model using alternative method...")
            
            # Try to import silero-vad directly
            try:
                from silero_vad import load_model, get_speech_timestamps
                self.model = load_model()
                self.get_speech_timestamps = get_speech_timestamps
                print("DEBUG: Successfully loaded using direct silero_vad import")
                logger.info("Silero VAD model loaded using direct import")
                return
            except ImportError:
                print("DEBUG: silero_vad package not available, trying manual download")
            
            # Manual download and setup
            import urllib.request
            import zipfile
            import tempfile
            
            # Download the model files manually
            model_url = "https://github.com/snakers4/silero-vad/archive/refs/heads/master.zip"
            cache_dir = torch.hub.get_dir()
            model_path = f"{cache_dir}/snakers4_silero-vad_master"
            
            if not os.path.exists(model_path):
                print("DEBUG: Downloading model manually...")
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
                    urllib.request.urlretrieve(model_url, temp_file.name)
                    
                    # Extract the zip file
                    with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
                        zip_ref.extractall(cache_dir)
                
                # Clean up temp file
                os.unlink(temp_file.name)
            
            # Now try to load using torch.hub.load with local path
            self.model, self.utils = torch.hub.load(
                repo_or_dir=model_path,
                model='silero_vad',
                source='local'
            )
            
            self.get_speech_timestamps = self.utils[0]
            print("DEBUG: Alternative method completed successfully")
            logger.info("Silero VAD model loaded using alternative method")
            
        except Exception as e:
            print(f"DEBUG: Alternative method also failed: {e}")
            raise RuntimeError(f"All methods to load Silero VAD model failed: {e}")
        
    def start_listening(self):
        """Start listening for audio in a background thread."""
        if self.recording_thread and self.recording_thread.is_alive():
            logger.warning("Already listening")
            return False
            
        self.stop_event.clear()
        self.audio_detected_event.clear()
        self.recording_thread = Thread(target=self._listen_for_speech)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        logger.info("Started listening for speech with Silero VAD")
        return True
        
    def stop_listening(self):
        """Stop the background listening thread."""
        if self.recording_thread and self.recording_thread.is_alive():
            self.stop_event.set()
            self.recording_thread.join(timeout=2.0)
            logger.info("Stopped listening")
            return True
        return False
        
    def wait_for_speech(self, timeout=None):
        """Wait until speech is detected and recorded."""
        return self.audio_detected_event.wait(timeout=timeout)
        
    def _listen_for_speech(self):
        """Background thread that listens for speech using Silero VAD."""
        logger.info("Starting Silero VAD speech detection...")
        
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
                
            # Convert to mono and float
            if indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata.flatten()
                
            # Add to rolling buffer
            self.audio_buffer.extend(audio_data)
            
            # Process for VAD
            self._process_audio_chunk(audio_data)
        
        try:
            with sd.InputStream(
                samplerate=self.rate,
                channels=self.channels,
                callback=audio_callback,
                blocksize=CHUNK_SIZE,
                dtype=np.float32
            ):
                logger.info("Microphone is open and listening with Silero VAD...")
                
                while not self.stop_event.is_set():
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Error in audio stream: {e}")
            
    def _process_audio_chunk(self, audio_data):
        """Process audio chunk for speech detection."""
        try:
            # Convert to tensor
            tensor_audio = torch.from_numpy(audio_data).float()
            
            # Get speech timestamps
            speech_segments = self.get_speech_timestamps(
                tensor_audio, 
                self.model, 
                sampling_rate=self.rate,
                min_speech_duration_ms=int(MIN_SPEECH_DURATION * 1000),
                min_silence_duration_ms=int(SILENCE_TIMEOUT * 1000)
            )
            
            # Check if avatar is speaking
            if self.audio_player.is_playing_audio():
                self._handle_interruption_detection(speech_segments, audio_data)
            else:
                self._handle_speech_detection(speech_segments, audio_data)
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            
    def _handle_interruption_detection(self, speech_segments, audio_data):
        """Handle speech detection when avatar is speaking (interruption)."""
        if speech_segments:
            logger.info("User interruption detected, stopping avatar speech")
            self.audio_player.stop_audio()
            
            # Start recording the interruption
            self._start_recording()
            
            # Add current audio data to recording
            self.recording_frames.append(audio_data.tobytes())
            
    def _handle_speech_detection(self, speech_segments, audio_data):
        """Handle normal speech detection."""
        if speech_segments and not self.is_recording:
            logger.info("Speech detected, starting recording...")
            self._start_recording()
            
        if self.is_recording:
            self.recording_frames.append(audio_data.tobytes())
            
            # Check if we should stop recording
            if not speech_segments:
                # No speech detected, check timeout
                if hasattr(self, 'last_speech_time'):
                    if time.time() - self.last_speech_time > SILENCE_TIMEOUT:
                        print("VAD detected user stopped speaking")
                        self._stop_recording()
            else:
                # Update last speech time
                self.last_speech_time = time.time()
                
    def _start_recording(self):
        """Start recording speech."""
        self.is_recording = True
        self.recording_frames = []
        self.last_speech_time = time.time()
        self.recording_start_time = time.time()
        
        # Add pre-buffer content
        if self.audio_buffer:
            pre_buffer_audio = np.array(list(self.audio_buffer))
            self.recording_frames.append(pre_buffer_audio.tobytes())
            
        logger.info("Started recording speech")
        
    def _stop_recording(self):
        """Stop recording and save audio."""
        if not self.is_recording:
            return
            
        self.is_recording = False
        recording_duration = time.time() - self.recording_start_time
        
        if recording_duration < self.min_phrase_duration:
            logger.info(f"Speech too short ({recording_duration:.2f}s), ignoring")
            return
            
        # Save the recorded audio
        self._save_audio()
        logger.info(f"Recording saved ({recording_duration:.2f}s)")
        
        # Signal that audio is ready for processing
        self.audio_detected_event.set()
        
        # Wait for the event to be cleared before continuing to listen
        while self.audio_detected_event.is_set() and not self.stop_event.is_set():
            time.sleep(0.1)
            
    def _save_audio(self):
        """Save recorded audio frames to a WAV file."""
        try:
            # Convert bytes back to numpy array
            audio_data = []
            for frame_bytes in self.recording_frames:
                frame_data = np.frombuffer(frame_bytes, dtype=np.float32)
                audio_data.extend(frame_data)
                
            audio_array = np.array(audio_data)
            
            # Convert to int16 for WAV file
            audio_int16 = (audio_array * 32767).astype(np.int16)
            
            with wave.open(TEMP_AUDIO_PATH, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.rate)
                wf.writeframes(audio_int16.tobytes())
                
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            
    def reset_detection_event(self):
        """Reset the audio detection event to listen for new speech."""
        self.audio_detected_event.clear()
        
    def get_audio_data(self):
        """Get the recorded audio data as numpy array."""
        try:
            if not self.recording_frames:
                return None
                
            # Convert bytes back to numpy array
            audio_data = []
            for frame_bytes in self.recording_frames:
                frame_data = np.frombuffer(frame_bytes, dtype=np.float32)
                audio_data.extend(frame_data)
                
            return np.array(audio_data)
            
        except Exception as e:
            logger.error(f"Error getting audio data: {e}")
            return None
            
    def stop(self):
        """Stop the audio recorder."""
        return self.stop_listening()
