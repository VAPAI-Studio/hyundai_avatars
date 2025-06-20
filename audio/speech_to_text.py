"""
Module for converting speech to text.
"""

import speech_recognition as sr
import logging
from utils.config import TEMP_AUDIO_PATH, LANGUAGE

logger = logging.getLogger(__name__)

class SpeechToText:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
    def convert_audio_to_text(self, audio_path=TEMP_AUDIO_PATH):
        """
        Convert audio file to text using Google's Speech Recognition API.
        Returns the recognized text or None if recognition failed.
        """
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                
            text = self.recognizer.recognize_google(audio_data, language=LANGUAGE)
            logger.info(f"Recognized text: {text}")
            print(f"Speech-to-text finished: '{text}'")
            return text
            
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            print("Speech-to-text finished: No speech detected")
            return None
            
        except sr.RequestError as e:
            logger.error(f"Could not request results from service; {e}")
            print(f"Speech-to-text finished: Error - {e}")
            return None
            
        except Exception as e:
            logger.error(f"Error converting speech to text: {e}")
            print(f"Speech-to-text finished: Error - {e}")
            return None
