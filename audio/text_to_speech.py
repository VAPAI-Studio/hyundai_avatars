"""
Module for converting text to speech using ElevenLabs with streaming support.
"""

import requests
import json
import logging
import os
import time
import traceback
import tempfile
import numpy as np
import grpc
import threading
from pynput import keyboard
from pydub import AudioSegment
from datetime import datetime
from utils.config import (
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL_ID, 
    VOICE_SETTINGS, RESPONSE_AUDIO_PATH, USE_GRPC
)
from proto import audio2face_pb2
from proto import audio2face_pb2_grpc
from audio.audio_player import AudioPlayer

logger = logging.getLogger(__name__)

# Target audio parameters
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # 16-bit

# Buffer management constants
MIN_BUFFER_SIZE = 32000  # Minimum MP3 bytes to process
FRAME_BUFFER_SIZE = 4096  # PCM audio frames per buffer


class AudioBufferManager:
    """
    Manages audio buffers for smooth playback.
    """
    def __init__(self, sample_rate):
        self.buffers = []
        self.sample_rate = sample_rate
        self.total_duration = 0
        self.current_position = 0
        self.is_playing = False
        self._lock = threading.Lock()
        self._paused = False
    
    def add_buffer(self, audio_data):
        """Add a new buffer of audio data (float32 array)"""
        if len(audio_data) == 0:
            return
            
        with self._lock:
            self.buffers.append(audio_data)
            self.total_duration += len(audio_data) / self.sample_rate
            logger.debug(f"Added buffer: {len(audio_data)} samples, total duration: {self.total_duration:.2f}s")
    
    def reset(self):
        """Clear all buffers and reset state"""
        with self._lock:
            self.buffers = []
            self.total_duration = 0
            self.current_position = 0
            self.is_playing = False
            self._paused = False
            logger.debug("Audio buffers reset")
    
    def get_all_audio(self):
        """Get all audio data as a single array"""
        with self._lock:
            if not self.buffers:
                return np.array([], dtype=np.float32)
            return np.concatenate(self.buffers)
            
    def pause(self):
        """Pause playback"""
        with self._lock:
            self._paused = True
            self.is_playing = False
            
    def resume(self):
        """Resume playback"""
        with self._lock:
            self._paused = False
            self.is_playing = True
            
    def is_paused(self):
        """Check if playback is paused"""
        with self._lock:
            return self._paused


def log_time(message):
    """Log message with timestamp for performance tracking"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    logger.debug(f"[{timestamp}] {message}")


class TextToSpeech:
    def __init__(self):
        self.api_key = ELEVENLABS_API_KEY
        self.voice_id = ELEVENLABS_VOICE_ID
        self.model_id = ELEVENLABS_MODEL_ID
        self.voice_settings = VOICE_SETTINGS
        self.api_url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        self.test_audios_dir = "test_audios"
        self._ensure_test_audios_dir()
        self.audio_player = AudioPlayer()
        self._streaming_thread = None
        self._stop_streaming = threading.Event()
        self._paused = False
        self._pause_lock = threading.Lock()
        
        # Set up keyboard listener
        self._keyboard_listener = keyboard.Listener(on_press=self._handle_key_press)
        self._keyboard_listener.start()
        
    def _handle_key_press(self, key):
        """Handle key press events"""
        try:
            if key.char == 'p':  # Check if 'p' key was pressed
                with self._pause_lock:
                    if self._paused:
                        self.resume_playback()
                    else:
                        self.pause_playback()
        except AttributeError:
            pass  # Ignore special keys
            
    def pause_playback(self):
        """Pause the current audio playback"""
        with self._pause_lock:
            if not self._paused and self._streaming_thread and self._streaming_thread.is_alive():
                self._paused = True
                self.audio_player.is_playing = False
                logger.info("Audio playback paused")
                return True
        return False
        
    def resume_playback(self):
        """Resume the paused audio playback"""
        with self._pause_lock:
            if self._paused and self._streaming_thread and self._streaming_thread.is_alive():
                self._paused = False
                self.audio_player.is_playing = True
                logger.info("Audio playback resumed")
                return True
        return False
        
    def _ensure_test_audios_dir(self):
        """Ensure the test_audios directory exists."""
        if not os.path.exists(self.test_audios_dir):
            os.makedirs(self.test_audios_dir)
            
    def _get_next_file_index(self):
        """Get the next available file index in the test_audios directory."""
        existing_files = [f for f in os.listdir(self.test_audios_dir) if f.endswith('.wav')]
        if not existing_files:
            return 1
            
        # Extract numbers from filenames and find the maximum
        indices = []
        for file in existing_files:
            try:
                # Extract number from filename (e.g., "audio_1.wav" -> 1)
                index = int(file.split('_')[1].split('.')[0])
                indices.append(index)
            except:
                continue
                
        return max(indices) + 1 if indices else 1
        
    def stream_audio_from_elevenlabs(self, text):
        """
        Stream audio from ElevenLabs TTS API.
        
        Parameters:
            text (str): Text to convert to speech
        """
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
        
        headers = {
            "Accept": "audio/mpeg",  # MP3 format
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": self.voice_settings
        }
        
        try:
            log_time(f"Requesting audio from ElevenLabs API for text: '{text[:30]}...'")
            response = requests.post(url, json=data, headers=headers, stream=True)
            
            if response.status_code != 200:
                error_msg = f"ElevenLabs API Error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            log_time("Successfully connected to ElevenLabs API, receiving audio stream...")
            return response
        except Exception as e:
            logger.error(f"Exception in ElevenLabs API request: {str(e)}")
            raise
            
    def process_mp3_data(self, mp3_data):
        """
        Convert MP3 data to the format required by Audio2Face (float32 PCM).
        
        Parameters:
            mp3_data (bytes): MP3 data to convert
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        try:
            log_time(f"Processing MP3 data chunk: {len(mp3_data)} bytes")
            
            # Save MP3 data to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(mp3_data)
                temp_path = temp_file.name
            
            # Load MP3 using pydub
            audio_segment = AudioSegment.from_file(temp_path, format="mp3")
            
            # Delete the temporary file
            os.unlink(temp_path)
            
            # Convert to target format
            audio_segment = audio_segment.set_frame_rate(TARGET_SAMPLE_RATE)
            audio_segment = audio_segment.set_channels(TARGET_CHANNELS)
            audio_segment = audio_segment.set_sample_width(TARGET_SAMPLE_WIDTH)
            
            # Get raw PCM data
            pcm_data = audio_segment.raw_data
            
            # Convert to numpy array of int16
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            
            # Normalize to float32 in range [-1.0, 1.0]
            float32_array = audio_array.astype(np.float32) / 32768.0
            
            log_time(f"Processed MP3 chunk: {len(float32_array)} samples at {TARGET_SAMPLE_RATE}Hz")
            return float32_array, TARGET_SAMPLE_RATE
            
        except Exception as e:
            logger.error(f"Error processing MP3 data: {str(e)}")
            traceback.print_exc()
            raise
            
    def _stream_audio_to_audio2face(self, text, instance_name="/World/audio2face/PlayerStreaming"):
        """
        Internal method to stream audio to Audio2Face in a separate thread.
        """
        try:
            response = self.stream_audio_from_elevenlabs(text)
            url = "localhost:50051"
            channel = grpc.insecure_channel(url)
            stub = audio2face_pb2_grpc.Audio2FaceStub(channel)
            
            start_marker = audio2face_pb2.PushAudioRequestStart(
                samplerate=TARGET_SAMPLE_RATE,
                instance_name=instance_name,
                block_until_playback_is_finished=False,
            )
            
            def request_generator():
                yield audio2face_pb2.PushAudioStreamRequest(start_marker=start_marker)
                
                mp3_buffer = bytearray()
                for chunk in response.iter_content(chunk_size=4096):
                    if self._stop_streaming.is_set():
                        break
                        
                    with self._pause_lock:
                        while self._paused:
                            time.sleep(0.1)
                            if self._stop_streaming.is_set():
                                break
                        
                    if not chunk:
                        continue
                        
                    mp3_buffer.extend(chunk)
                    
                    if len(mp3_buffer) >= MIN_BUFFER_SIZE:
                        try:
                            audio_data, _ = self.process_mp3_data(mp3_buffer)
                            yield audio2face_pb2.PushAudioStreamRequest(
                                audio_data=audio_data.tobytes()
                            )
                            mp3_buffer = bytearray()
                        except Exception as e:
                            logger.warning(f"Warning: Error processing buffer: {str(e)}")
                            mp3_buffer = bytearray()
                            
                    time.sleep(0.001)
            
            self.audio_player.is_playing = True
            response = stub.PushAudioStream(request_generator())
            
            if response.success:
                logger.info("Audio streaming completed successfully")
            else:
                logger.error(f"Error in audio streaming: {response.message}")
                
        except Exception as e:
            logger.error(f"Error in audio streaming thread: {str(e)}")
            traceback.print_exc()
            
        finally:
            self.audio_player.is_playing = False
            self._stop_streaming.clear()
            self._paused = False
            if channel:
                channel.close()
                
    def push_audio_stream_to_audio2face(self, text, instance_name="/World/audio2face/PlayerStreaming"):
        """
        Start streaming audio from ElevenLabs to Audio2Face in a separate thread.
        """
        # Stop any existing streaming
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._stop_streaming.set()
            self._streaming_thread.join(timeout=1.0)
            
        # Start new streaming thread
        self._stop_streaming.clear()
        self._streaming_thread = threading.Thread(
            target=self._stream_audio_to_audio2face,
            args=(text, instance_name)
        )
        self._streaming_thread.daemon = True
        self._streaming_thread.start()
        return True
        
    def stop_streaming(self):
        """Stop the current audio streaming."""
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._stop_streaming.set()
            self._streaming_thread.join(timeout=1.0)
            self.audio_player.is_playing = False
            return True
        return False
        
    def convert_text_to_speech(self, text):
        """
        Convert text to speech and handle the audio output based on USE_GRPC setting.
        
        Parameters:
            text (str): Text to convert to speech
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if USE_GRPC:
                self.push_audio_stream_to_audio2face(text)
            else:
                def audio_thread():
                    try:
                        response = self.stream_audio_from_elevenlabs(text)
                        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                            for chunk in response.iter_content(chunk_size=4096):
                                if chunk:
                                    temp_file.write(chunk)
                                with self._pause_lock:
                                    while self._paused:
                                        time.sleep(0.1)
                        temp_path = temp_file.name
                        audio_segment = AudioSegment.from_file(temp_path, format="mp3")
                        audio_segment.export(RESPONSE_AUDIO_PATH, format="wav")
                        os.unlink(temp_path)
                        self.audio_player.play_audio(RESPONSE_AUDIO_PATH)
                    except Exception as e:
                        logger.error(f"Error in audio thread: {str(e)}")
                        traceback.print_exc()
                
                audio_thread = threading.Thread(target=audio_thread)
                audio_thread.daemon = True
                audio_thread.start()
            return True
        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {str(e)}")
            traceback.print_exc()
            return False
            
    def test_voices(self):
        """
        List available voices from ElevenLabs.
        Useful for selecting a voice ID.
        """
        if not self.api_key or self.api_key == "your_elevenlabs_api_key_here":
            logger.error("ElevenLabs API key not configured")
            return None
            
        try:
            voices_url = "https://api.elevenlabs.io/v1/voices"
            headers = {
                "Accept": "application/json",
                "xi-api-key": self.api_key
            }
            
            response = requests.get(voices_url, headers=headers)
            
            if response.status_code == 200:
                voices = response.json()
                for voice in voices.get("voices", []):
                    logger.info(f"Voice ID: {voice['voice_id']}, Name: {voice['name']}")
                return voices
            else:
                logger.error(f"Error fetching voices: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error testing voices: {e}")
            return None
            
    def test_models(self):
        """
        List available models from ElevenLabs.
        Useful for selecting a model ID.
        """
        if not self.api_key or self.api_key == "your_elevenlabs_api_key_here":
            logger.error("ElevenLabs API key not configured")
            return None
            
        try:
            models_url = "https://api.elevenlabs.io/v1/models"
            headers = {
                "Accept": "application/json",
                "xi-api-key": self.api_key
            }
            
            response = requests.get(models_url, headers=headers)
            
            if response.status_code == 200:
                models = response.json()
                for model in models.get("models", []):
                    logger.info(f"Model ID: {model['model_id']}, Name: {model['name']}")
                return models
            else:
                logger.error(f"Error fetching models: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error testing models: {e}")
            return None

    def __del__(self):
        """Clean up keyboard listener"""
        if hasattr(self, '_keyboard_listener'):
            self._keyboard_listener.stop()
