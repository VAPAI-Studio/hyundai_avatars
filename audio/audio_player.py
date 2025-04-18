"""
Module for playing audio responses.
"""

import pygame
import time
import logging
import os
from utils.config import RESPONSE_AUDIO_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioPlayer:
    def __init__(self):
        pygame.mixer.init()
        
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
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            # Unload the audio file to free it
            pygame.mixer.music.unload()
            logger.info("Audio playback completed")
            return True
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            return False
            
    def stop_audio(self):
        """Stop currently playing audio."""
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()  # Make sure to unload after stopping
                logger.info("Audio playback stopped")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error stopping audio: {e}")
            return False
