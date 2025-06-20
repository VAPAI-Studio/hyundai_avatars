"""
Module for converting speech to text.
"""

import speech_recognition as sr
import logging
import numpy as np
import wave
import tempfile
import os
from utils.config import TEMP_AUDIO_PATH, LANGUAGE, SAMPLE_RATE

logger = logging.getLogger(__name__)

class SpeechToText:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
    def convert(self, audio_data):
        """
        Convert audio data (numpy array) to text using Google's Speech Recognition API.
        Returns the recognized text or None if recognition failed.
        """
        try:
            # Create a temporary WAV file from the audio data
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                
            # Save audio data as WAV file
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(SAMPLE_RATE)
                
                # Convert float32 to int16
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            
            # Use the existing method to convert the WAV file
            text = self.convert_audio_to_text(temp_path)
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
                
            return text
            
        except Exception as e:
            logger.error(f"Error converting audio data to text: {e}")
            print(f"Speech-to-text finished: Error - {e}")
            return None
        
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
