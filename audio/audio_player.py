"""
Module for playing audio responses.
"""

import pygame
import time
import logging
import os
from utils.config import RESPONSE_AUDIO_PATH, USE_GRPC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.is_playing = False
        self._last_busy_check = 0
        self._busy_check_interval = 0.1  # Check every 100ms
        self._playback_start_time = 0
        self._playback_duration = 0
        self._current_audio_path = None
        
    def is_playing_audio(self):
        """Check if audio is currently playing."""
        current_time = time.time()
        
        # For pygame playback
        if not USE_GRPC and current_time - self._last_busy_check >= self._busy_check_interval:
            self._last_busy_check = current_time
            is_busy = pygame.mixer.music.get_busy()
            if is_busy != self.is_playing:
                logger.debug(f"Audio playing state changed: {self.is_playing} -> {is_busy}")
                self.is_playing = is_busy
                
        # For Audio2Face streaming
        if USE_GRPC and self.is_playing:
            # Estimate if audio should still be playing based on start time and duration
            if current_time - self._playback_start_time >= self._playback_duration:
                self.is_playing = False
                logger.debug("Audio2Face playback duration exceeded, marking as stopped")
                
        return self.is_playing
        
    def play_audio(self, audio_path=RESPONSE_AUDIO_PATH):
        """
        Play audio from the specified file.
        Returns True if successful, False otherwise.
        """
        try:
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return False
                
            logger.info(f"Playing audio response from {audio_path}")
            self._current_audio_path = audio_path
            
            if not USE_GRPC:
                # Use pygame for playback
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                self.is_playing = True
                self._last_busy_check = time.time()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy() and self.is_playing:
                    time.sleep(0.1)
                    
                # Unload the audio file to free it
                pygame.mixer.music.unload()
                self.is_playing = False
            else:
                # For Audio2Face streaming, estimate duration
                import wave
                with wave.open(audio_path, 'r') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    self._playback_duration = frames / float(rate)
                    self._playback_start_time = time.time()
                    self.is_playing = True
                    logger.debug(f"Audio2Face playback started, estimated duration: {self._playback_duration:.2f}s")
                
            logger.info("Audio playback completed")
            return True
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            self.is_playing = False
            return False
            
    def stop_audio(self):
        """Stop currently playing audio."""
        try:
            if self.is_playing:
                logger.info("Stopping audio playback")
                if not USE_GRPC:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()  # Make sure to unload after stopping
                self.is_playing = False
                self._playback_start_time = 0
                self._playback_duration = 0
                logger.info("Audio playback stopped")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error stopping audio: {e}")
            self.is_playing = False
            return False
