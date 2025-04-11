"""
Module for audio recording and speech detection.
"""

import pyaudio
import wave
import numpy as np
import time
from array import array
from threading import Thread, Event
import logging
from utils.config import (
    SAMPLE_RATE, CHANNELS, SILENCE_THRESHOLD,
    SILENCE_DURATION, MIN_PHRASE_DURATION, TEMP_AUDIO_PATH
)

logger = logging.getLogger(__name__)

# Constants
SPEECH_TIMEOUT = 4.0  # Time in seconds to record after speech is detected

class AudioRecorder:
    def __init__(self):
        self.format = pyaudio.paInt16
        self.chunk = 1024
        self.rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.silence_threshold = SILENCE_THRESHOLD
        self.silence_duration = SILENCE_DURATION
        self.min_phrase_duration = MIN_PHRASE_DURATION
        self.stop_event = Event()
        self.audio_detected_event = Event()
        self.recording_thread = None
        self.p = None
        self.stream = None
        
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
        logger.info("Started listening for speech")
        return True
        
    def stop_listening(self):
        """Stop the background listening thread."""
        if self.recording_thread and self.recording_thread.is_alive():
            self.stop_event.set()
            self.recording_thread.join(timeout=2.0)
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()
            logger.info("Stopped listening")
            return True
        return False
        
    def wait_for_speech(self, timeout=None):
        """Wait until speech is detected and recorded."""
        return self.audio_detected_event.wait(timeout=timeout)
        
    def _is_silent(self, data_chunk):
        """Check if the audio chunk is below the silence threshold."""
        as_ints = array('h', data_chunk)
        return max(abs(x) for x in as_ints) < self.silence_threshold
        
    def _listen_for_speech(self):
        """Background thread that listens for speech and records it when detected."""
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        logger.info("Microphone is open and listening...")
        
        while not self.stop_event.is_set():
            # Wait for speech to begin
            silent_chunks = 0
            speech_detected = False
            
            while not speech_detected and not self.stop_event.is_set():
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                if not self._is_silent(data):
                    speech_detected = True
                    logger.info("Speech detected, recording...")
                    break
                    
            if not speech_detected:
                continue
                
            # Record for SPEECH_TIMEOUT seconds after speech is detected
            frames = []
            recording_start_time = time.time()
            
            while not self.stop_event.is_set():
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)
                
                # Check if we've reached the speech timeout
                if time.time() - recording_start_time >= SPEECH_TIMEOUT:
                    logger.info(f"Speech timeout ({SPEECH_TIMEOUT}s) reached, stopping recording")
                    break
                    
            recording_duration = time.time() - recording_start_time
            
            if recording_duration < self.min_phrase_duration:
                logger.info(f"Speech too short ({recording_duration:.2f}s), ignoring")
                continue
                
            # Save the recorded audio to a file
            self._save_audio(frames)
            logger.info(f"Recording saved ({recording_duration:.2f}s)")
            
            # Signal that audio is ready for processing
            self.audio_detected_event.set()
            
            # Wait for the event to be cleared before continuing to listen
            while self.audio_detected_event.is_set() and not self.stop_event.is_set():
                time.sleep(0.1)
                
    def _save_audio(self, frames):
        """Save recorded audio frames to a WAV file."""
        with wave.open(TEMP_AUDIO_PATH, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            
    def reset_detection_event(self):
        """Reset the audio detection event to listen for new speech."""
        self.audio_detected_event.clear()
