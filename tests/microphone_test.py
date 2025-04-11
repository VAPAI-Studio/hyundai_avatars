#!/usr/bin/env python3
"""
Enhanced microphone test script with device selection and visual audio level display.
Works better with headphones by allowing you to select your input device.
"""

import pyaudio
import wave
import speech_recognition as sr
import numpy as np
import time
from array import array
import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Audio recording parameters (will be adjustable)
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
SILENCE_THRESHOLD = 500  # Will be adjustable
SILENCE_DURATION = 1.0  # Seconds of silence to consider speech ended
MIN_PHRASE_DURATION = 0.5  # Minimum seconds of audio to consider valid
TEMP_AUDIO_PATH = "mic_test_audio.wav"

def is_silent(data_chunk, threshold=SILENCE_THRESHOLD):
    """Check if the audio chunk is below the silence threshold."""
    as_ints = array('h', data_chunk)
    # Return the max volume along with the silence check
    max_volume = 0
    if len(as_ints) > 0:
        max_volume = max(abs(x) for x in as_ints)
    return max_volume < threshold, max_volume

def display_volume_meter(volume, threshold):
    """Display a simple volume meter in the console."""
    meter_width = 50
    normalized_volume = min(1.0, volume / (threshold * 3))  # Scale it for better visibility
    filled_blocks = int(normalized_volume * meter_width)
    
    meter = '|' + '█' * filled_blocks + ' ' * (meter_width - filled_blocks) + '|'
    
    # Add threshold marker
    threshold_position = int((threshold / (threshold * 3)) * meter_width)
    threshold_marker = ' ' * threshold_position + '▼' + ' ' * (meter_width - threshold_position - 1)
    
    # Add indicator if sound is above threshold
    indicator = "LISTENING" if volume < threshold else "RECORDING"
    
    sys.stdout.write(f"\r{meter} {volume:5d} {indicator}")
    sys.stdout.flush()

def record_audio(device_index=None, threshold=SILENCE_THRESHOLD, show_volume=True, timeout=10):
    """
    Record audio from the specified microphone device.
    Returns the path to the recorded audio file or None if no speech detected.
    """
    p = pyaudio.PyAudio()
    
    try:
        # Open audio stream with specified device if provided
        stream_kwargs = {
            'format': FORMAT,
            'channels': CHANNELS,
            'rate': SAMPLE_RATE,
            'input': True,
            'frames_per_buffer': CHUNK
        }
        
        if device_index is not None:
            stream_kwargs['input_device_index'] = device_index
            
        stream = p.open(**stream_kwargs)
        
        if not show_volume:
            logger.info("Listening... (speak now)")
        
        # Wait for speech to begin
        silent_chunks = 0
        speech_detected = False
        start_time = time.time()
        
        # Add a buffer to store audio even before speech is officially detected
        pre_buffer = []
        PRE_BUFFER_SIZE = 5  # Number of chunks to keep before speech is detected
        
        while not speech_detected and time.time() - start_time < timeout:
            data = stream.read(CHUNK, exception_on_overflow=False)
            is_silent_chunk, volume = is_silent(data, threshold)
            
            # Keep a rolling buffer of recent audio
            pre_buffer.append(data)
            if len(pre_buffer) > PRE_BUFFER_SIZE:
                pre_buffer.pop(0)
                
            if show_volume:
                display_volume_meter(volume, threshold)
                
            if not is_silent_chunk:
                speech_detected = True
                if not show_volume:
                    logger.info("Speech detected, recording...")
                break
        
        if not speech_detected:
            if not show_volume:
                logger.info("No speech detected within timeout period.")
            else:
                print("\nNo speech detected within timeout period.")
            return None
        
        # Record until silence is detected for SILENCE_DURATION
        # Start with the pre-buffer content to capture the beginning of speech
        frames = list(pre_buffer)
        silent_time = None
        recording_start_time = time.time()
        
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            is_silent_chunk, volume = is_silent(data, threshold)
            if show_volume:
                display_volume_meter(volume, threshold)
            
            if is_silent_chunk:
                if silent_time is None:
                    silent_time = time.time()
                elif time.time() - silent_time >= SILENCE_DURATION:
                    break
            else:
                silent_time = None
                
            # Safety timeout (30 seconds max recording)
            if time.time() - recording_start_time > 30:
                if not show_volume:
                    logger.warning("Recording timed out (30 seconds max)")
                break
        
        recording_duration = time.time() - recording_start_time
        
        if recording_duration < MIN_PHRASE_DURATION:
            if not show_volume:
                logger.info(f"Audio too short ({recording_duration:.2f}s), ignoring")
            else:
                print(f"\nAudio too short ({recording_duration:.2f}s), ignoring")
            return None
        
        if show_volume:
            print(f"\nFinished recording ({recording_duration:.2f}s)")
        else:    
            logger.info(f"Finished recording ({recording_duration:.2f}s)")
        
        # Save recorded audio to file
        with wave.open(TEMP_AUDIO_PATH, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))
            
        return TEMP_AUDIO_PATH
        
    finally:
        # Clean up
        try:
            stream.stop_stream()
            stream.close()
        except:
            pass
        p.terminate()

def recognize_speech(audio_file, language="en-US"):
    """Convert audio file to text using Google's Speech Recognition."""
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            
        text = recognizer.recognize_google(audio_data, language=language)
        return text
        
    except sr.UnknownValueError:
        logger.warning("Speech recognition could not understand audio")
        return None
        
    except sr.RequestError as e:
        logger.error(f"Could not request results from speech recognition service; {e}")
        return None
        
    except Exception as e:
        logger.error(f"Error recognizing speech: {e}")
        return None

def list_audio_devices():
    """List all available audio input devices and return them as a list."""
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    devices = []
    
    for i in range(numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            devices.append((i, device_info.get('name')))
    
    p.terminate()
    return devices

def calibrate_threshold(device_index=None):
    """
    Run a calibration to determine the appropriate silence threshold.
    """
    print("\nLet's calibrate the silence threshold.")
    print("Please remain silent for 5 seconds to measure ambient noise...")
    
    p = pyaudio.PyAudio()
    
    try:
        stream_kwargs = {
            'format': FORMAT,
            'channels': CHANNELS,
            'rate': SAMPLE_RATE,
            'input': True,
            'frames_per_buffer': CHUNK
        }
        
        if device_index is not None:
            stream_kwargs['input_device_index'] = device_index
            
        stream = p.open(**stream_kwargs)
        
        # Collect ambient noise samples
        ambient_values = []
        start_time = time.time()
        
        while time.time() - start_time < 5:
            data = stream.read(CHUNK, exception_on_overflow=False)
            _, volume = is_silent(data, 1)  # Use 1 as threshold to get raw volume
            ambient_values.append(volume)
            display_volume_meter(volume, 500)  # Just for display
            time.sleep(0.05)
            
        # Calculate the threshold based on ambient noise
        ambient_avg = sum(ambient_values) / len(ambient_values) if ambient_values else 0
        ambient_max = max(ambient_values) if ambient_values else 0
        
        # Set threshold to a value above the maximum ambient noise
        recommended_threshold = int(ambient_max * 1.5) + 50
        
        print(f"\n\nAmbient noise level: avg={ambient_avg:.1f}, max={ambient_max}")
        print(f"Recommended threshold: {recommended_threshold}")
        
        return recommended_threshold
        
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def main():
    """Main function that runs the microphone test."""
    try:
        # First list available devices
        devices = list_audio_devices()
        
        if not devices:
            print("No audio input devices found!")
            return
            
        print("\nAvailable audio input devices:")
        for idx, (device_id, device_name) in enumerate(devices):
            print(f"  {idx+1}. ID {device_id}: {device_name}")
        
        # Let user select a device
        device_idx = 0  # Default to first device
        try:
            selection = input("\nSelect input device (or press Enter for default): ")
            if selection.strip():
                device_idx = int(selection) - 1
                if device_idx < 0 or device_idx >= len(devices):
                    print("Invalid selection, using default device")
                    device_idx = 0
        except ValueError:
            print("Invalid input, using default device")
            device_idx = 0
            
        selected_device_id, selected_device_name = devices[device_idx]
        print(f"Using device: {selected_device_name} (ID: {selected_device_id})")
        
        # Calibrate the threshold
        threshold = calibrate_threshold(selected_device_id)
        
        # Optionally let user adjust the threshold
        try:
            custom_threshold = input(f"\nUse recommended threshold ({threshold}) or enter custom value: ")
            if custom_threshold.strip():
                threshold = int(custom_threshold)
        except ValueError:
            print(f"Invalid input, using recommended threshold: {threshold}")
        
        # Select language
        language = "en-US"  # Default
        language_selection = input("\nSelect language (1 for English, 2 for Spanish, 3 for Other): ")
        if language_selection == "2":
            language = "es-ES"
        elif language_selection == "3":
            language_code = input("Enter language code (e.g., fr-FR, de-DE): ")
            if language_code.strip():
                language = language_code
        
        print("\n" + "-"*50)
        print("MICROPHONE TEST")
        print(f"Device: {selected_device_name}")
        print(f"Threshold: {threshold}")
        print(f"Language: {language}")
        print("The volume meter will show your audio level.")
        print("Speak when ready, and press Ctrl+C to exit.")
        print("-"*50 + "\n")
        
        time.sleep(1)  # Brief pause before starting
        
        while True:
            # Record audio with visual feedback
            audio_file = record_audio(
                device_index=selected_device_id,
                threshold=threshold,
                show_volume=True
            )
            
            if audio_file:
                # Try to recognize speech
                text = recognize_speech(audio_file, language=language)
                
                if text:
                    print(f"\n🎤 Recognized: \"{text}\"\n")
                else:
                    print("\n❌ Could not recognize speech\n")
                    
                # Clean up temp file
                try:
                    os.remove(audio_file)
                except:
                    pass
            
            print("-"*50)
            print("Listening again... (Ctrl+C to exit)")
            
    except KeyboardInterrupt:
        print("\nMicrophone test ended by user.")
    except Exception as e:
        logger.error(f"Error in microphone test: {e}")
    
if __name__ == "__main__":
    main()