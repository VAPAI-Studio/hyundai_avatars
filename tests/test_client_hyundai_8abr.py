"""
This script streams audio from ElevenLabs Text-to-Speech API to Audio2Face Streaming Audio Player via gRPC.
The script implements a buffered streaming approach inspired by AVAudioEngine patterns:
 * Connects to ElevenLabs API and requests TTS with streaming enabled
 * Manages audio buffers for smooth playback
 * Processes chunks incrementally and sends to Audio2Face
 * Uses a single gRPC stream to ensure compatibility
"""

import sys
import time
import os
import io
import traceback
from datetime import datetime
import requests
import numpy as np
import grpc
import audio2face_pb2
import audio2face_pb2_grpc
from pydub import AudioSegment
import tempfile
from collections import deque

# Hardcoded ElevenLabs API key
ELEVENLABS_API_KEY = "sk_33b55d275d2b0bceceeac63bdb5e981870b1b1864e4fdbf7"

# Target audio parameters
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # 16-bit

# Buffer management constants
MIN_BUFFER_SIZE = 32000  # Minimum MP3 bytes to process
FRAME_BUFFER_SIZE = 4096  # PCM audio frames per buffer


class AudioBufferManager:
    """
    Manages audio buffers similar to AVAudioPlayerNode in the Swift implementation.
    """
    def __init__(self, sample_rate):
        self.buffers = []
        self.sample_rate = sample_rate
        self.total_duration = 0
        self.current_position = 0
        self.is_playing = False
    
    def add_buffer(self, audio_data):
        """Add a new buffer of audio data (float32 array)"""
        if len(audio_data) == 0:
            return
            
        self.buffers.append(audio_data)
        self.total_duration += len(audio_data) / self.sample_rate
        print(f"Added buffer: {len(audio_data)} samples, total duration: {self.total_duration:.2f}s")
    
    def reset(self):
        """Clear all buffers and reset state"""
        self.buffers = []
        self.total_duration = 0
        self.current_position = 0
        self.is_playing = False
        print("Audio buffers reset")
    
    def get_all_audio(self):
        """Get all audio data as a single array"""
        if not self.buffers:
            return np.array([], dtype=np.float32)
            
        return np.concatenate(self.buffers)


def log_time(message):
    """Log message with timestamp for performance tracking"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")


def stream_audio_from_elevenlabs(text, voice_id="21m00Tcm4TlvDq8ikWAM", model_id="eleven_monolingual_v1"):
    """
    Stream audio from ElevenLabs TTS API.
    
    Parameters:
        text (str): Text to convert to speech
        voice_id (str): ElevenLabs voice ID
        model_id (str): ElevenLabs model ID
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    
    headers = {
        "Accept": "audio/mpeg",  # MP3 format
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    
    data = {
        "text": text,
        "model_id": model_id,
    }
    
    try:
        log_time(f"Requesting audio from ElevenLabs API for text: '{text[:30]}...'")
        response = requests.post(url, json=data, headers=headers, stream=True)
        
        if response.status_code != 200:
            error_msg = f"ElevenLabs API Error: {response.status_code} - {response.text}"
            print(error_msg)
            raise Exception(error_msg)
        
        log_time("Successfully connected to ElevenLabs API, receiving audio stream...")
        return response
    except Exception as e:
        print(f"Exception in ElevenLabs API request: {str(e)}")
        raise


def process_mp3_data(mp3_data):
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
        print(f"Error processing MP3 data: {str(e)}")
        traceback.print_exc()
        raise


def push_audio_stream_from_elevenlabs(url, text, instance_name, voice_id="21m00Tcm4TlvDq8ikWAM"):
    """
    Stream audio from ElevenLabs to Audio2Face using buffer management.
    
    Parameters:
        url (str): Audio2Face gRPC server URL
        text (str): Text to convert to speech
        instance_name (str): Prim path of A2F Streaming Audio Player
        voice_id (str): ElevenLabs voice ID
    """
    log_time("Starting audio streaming process")
    buffer_manager = AudioBufferManager(TARGET_SAMPLE_RATE)
    block_until_playback_is_finished = True
    mp3_buffer = bytearray()
    
    try:
        # Get streaming response from ElevenLabs
        response = stream_audio_from_elevenlabs(text, voice_id)
        
        # First phase: Process audio into buffers
        log_time("Processing audio data into buffers")
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            
            # Add to MP3 buffer
            mp3_buffer.extend(chunk)
            
            # Process when we have enough data
            if len(mp3_buffer) >= MIN_BUFFER_SIZE:
                try:
                    log_time(f"Processing MP3 buffer: {len(mp3_buffer)} bytes")
                    audio_data, _ = process_mp3_data(mp3_buffer)
                    buffer_manager.add_buffer(audio_data)
                    mp3_buffer = bytearray()
                except Exception as e:
                    print(f"Warning: Error processing buffer: {str(e)}")
                    # Continue even if processing this chunk failed
                    mp3_buffer = bytearray()
        
        # Process any remaining data
        if mp3_buffer:
            try:
                log_time(f"Processing final MP3 buffer: {len(mp3_buffer)} bytes")
                audio_data, _ = process_mp3_data(mp3_buffer)
                buffer_manager.add_buffer(audio_data)
            except Exception as e:
                print(f"Warning: Error processing final buffer: {str(e)}")
        
        # Second phase: Send to Audio2Face
        with grpc.insecure_channel(url) as channel:
            log_time(f"Channel created to Audio2Face at {url}")
            stub = audio2face_pb2_grpc.Audio2FaceStub(channel)
            
            # Get all processed audio for streaming
            all_audio = buffer_manager.get_all_audio()
            log_time(f"Preparing to send {len(all_audio)} samples to Audio2Face")
            
            def request_generator():
                # First request contains the start marker
                start_marker = audio2face_pb2.PushAudioRequestStart(
                    samplerate=TARGET_SAMPLE_RATE,
                    instance_name=instance_name,
                    block_until_playback_is_finished=block_until_playback_is_finished,
                )
                yield audio2face_pb2.PushAudioStreamRequest(start_marker=start_marker)
                log_time(f"Sent start marker with sample rate {TARGET_SAMPLE_RATE}Hz")
                
                # Break audio into manageable chunks
                for i in range(0, len(all_audio), FRAME_BUFFER_SIZE):
                    chunk = all_audio[i:i+FRAME_BUFFER_SIZE]
                    if i % (FRAME_BUFFER_SIZE * 10) == 0:  # Log every 10 chunks
                        log_time(f"Sending audio chunk at position {i}/{len(all_audio)}")
                    yield audio2face_pb2.PushAudioStreamRequest(
                        audio_data=chunk.tobytes()
                    )
                    # Small sleep to avoid overwhelming the receiver
                    time.sleep(0.005)
                
                log_time("All audio data sent")
            
            # Send all audio data in a single gRPC streaming call
            log_time("Starting Audio2Face streaming")
            response = stub.PushAudioStream(request_generator())
            
            if response.success:
                log_time("SUCCESS: Audio playback completed")
            else:
                log_time(f"ERROR: {response.message}")
    
    except Exception as e:
        print(f"Error in push_audio_stream_from_elevenlabs: {str(e)}")
        traceback.print_exc()
    
    buffer_manager.reset()
    log_time("Audio streaming process completed")


def main():
    """
    Main function to run the ElevenLabs to Audio2Face streaming client.
    """
    try:
        # URL of the Audio2Face Streaming Audio Player server
        url = "localhost:50051"  # Adjust if needed
        
        # Hardcoded text to convert to speech
        text_to_speak = "Welcome to Audio2Face integration with ElevenLabs. This is a demonstration of real-time audio streaming and facial animation."
        print(f"Text to speak: {text_to_speak}")
        
        # Hardcoded Prim path of the Audio2Face Streaming Audio Player on the stage
        instance_name = "/World/audio2face/PlayerStreaming"
        print(f"Audio2Face instance path: {instance_name}")
        
        # Default ElevenLabs voice ID
        voice_id = "21m00Tcm4TlvDq8ikWAM"
        print(f"Using voice ID: {voice_id}")
        
        # Stream audio from ElevenLabs to Audio2Face
        push_audio_stream_from_elevenlabs(url, text_to_speak, instance_name, voice_id)
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        print("Traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()