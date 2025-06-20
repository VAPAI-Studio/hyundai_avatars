"""
Module for streaming text chunks to ElevenLabs TTS with parallel processing.
"""

import requests
import json
import logging
import time
import threading
import queue
import tempfile
import os
import numpy as np
import grpc
import sounddevice as sd
from typing import Generator, Optional, Callable, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
import wave
import struct
import pygame
import io

from utils.config import (
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL_ID,
    USE_GRPC, AUDIO2FACE_HOST, AUDIO2FACE_PORT, TARGET_SAMPLE_RATE, VOICE_SETTINGS
)
from proto import audio2face_pb2
from proto import audio2face_pb2_grpc
from audio.text_chunker import TextChunk, TextChunker
from audio.audio_player import AudioPlayer

logger = logging.getLogger(__name__)

# Target audio parameters
TARGET_SAMPLE_RATE = 24000  # Default sample rate
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # 16-bit

# Buffer management constants
MIN_BUFFER_SIZE = 32000  # Minimum MP3 bytes to process
FRAME_BUFFER_SIZE = 4096

class StreamingTTSProcessor:
    def __init__(self, chunk_strategy: str = "sentence", max_workers: int = 3):
        """
        Initialize the streaming TTS processor.
        
        Args:
            chunk_strategy: Strategy for text chunking ("semantic", "sentence", "phrase", "pause")
            max_workers: Maximum number of parallel TTS requests
        """
        self.api_key = ELEVENLABS_API_KEY
        self.voice_id = ELEVENLABS_VOICE_ID
        self.model_id = ELEVENLABS_MODEL_ID
        self.voice_settings = VOICE_SETTINGS
        
        # Initialize TTS processor
        from audio.text_to_speech import TextToSpeech
        self.tts_processor = TextToSpeech()
        
        self.text_chunker = TextChunker(chunk_strategy="sentence")
        self.text_chunker.set_chunk_limits(min_size=40, max_size=200)
        self.audio_player = AudioPlayer()
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Threading and queue management
        self.chunk_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.processing_threads = []
        self.is_processing = False
        self.stop_event = threading.Event()
        
        # Audio buffer management
        self.audio_buffers = []
        self.buffer_lock = threading.Lock()
        
        # Callbacks
        self.on_chunk_processed: Optional[Callable[[TextChunk, str], None]] = None
        self.on_audio_ready: Optional[Callable[[np.ndarray], None]] = None
        self.on_streaming_complete: Optional[Callable[[], None]] = None
        
    def set_callbacks(self, 
                     on_chunk_processed: Optional[Callable[[TextChunk, str], None]] = None,
                     on_audio_ready: Optional[Callable[[np.ndarray], None]] = None,
                     on_streaming_complete: Optional[Callable[[], None]] = None):
        """Set callback functions for various events."""
        self.on_chunk_processed = on_chunk_processed
        self.on_audio_ready = on_audio_ready
        self.on_streaming_complete = on_streaming_complete
    
    def stream_text_to_speech(self, text_stream: Generator[str, None, None], 
                             conversation_history: Optional[list] = None) -> bool:
        """
        Stream text from LLM to ElevenLabs TTS with parallel processing.
        
        Args:
            text_stream: Generator yielding text chunks from LLM
            conversation_history: Optional conversation history for context
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Reset state
            self.stop_event.clear()
            self.is_processing = True
            self.audio_buffers = []
            
            # Timing metrics
            start_time = time.time()
            first_token_time = None
            first_audio_time = None
            audio_playback_start_time = None
            last_audio_time = None
            
            # Create a wrapper for on_audio_ready to track audio timing
            original_on_audio_ready = self.on_audio_ready
            
            def audio_ready_wrapper(audio_data):
                nonlocal first_audio_time, last_audio_time
                current_time = time.time()
                
                if first_audio_time is None:
                    first_audio_time = current_time
                    time_to_first_audio = first_audio_time - start_time
                    print(f"🔊 Time to first audio chunk: {time_to_first_audio:.3f}s")
                    print(f"🎵 Audio chunk size: {len(audio_data)} samples")
                
                # Update last audio time
                last_audio_time = current_time
                
                if original_on_audio_ready:
                    original_on_audio_ready(audio_data)
            
            # Set the wrapper as the callback
            self.on_audio_ready = audio_ready_wrapper
            
            # Start processing threads
            self._start_processing_threads()
            
            # Start audio streaming thread
            audio_thread = threading.Thread(target=self._audio_streaming_worker)
            audio_thread.daemon = True
            audio_thread.start()
            
            # Process text stream
            accumulated_text = ""
            chunk_count = 0
            
            for text_chunk in text_stream:
                if self.stop_event.is_set():
                    break
                    
                # Record time to first token
                if first_token_time is None:
                    first_token_time = time.time()
                    time_to_first_token = first_token_time - start_time
                    print(f"⏱️  Time to first token (LLM): {time_to_first_token:.3f}s")
                    print(f"📝 First token: '{text_chunk}'")
                
                accumulated_text += text_chunk
                
                # Check if we have enough text to chunk
                if len(accumulated_text) >= 50:  # Minimum chunk size
                    # Chunk the accumulated text
                    for chunk in self.text_chunker.chunk_text(accumulated_text):
                        if self.stop_event.is_set():
                            break
                            
                        # Print chunk info
                        chunk_preview = chunk.text[:60].replace('\n', ' ')
                        print(f"🟦 Chunk {chunk_count}: '{chunk_preview}{'...' if len(chunk.text) > 60 else ''}' (type: {chunk.chunk_type})")
                        # Add chunk to processing queue
                        self.chunk_queue.put((chunk_count, chunk))
                        chunk_count += 1
                        
                        # Call callback if set
                        if self.on_chunk_processed:
                            self.on_chunk_processed(chunk, "queued")
                    
                    # Reset accumulated text
                    accumulated_text = ""
            
            # Process any remaining text
            if accumulated_text.strip() and not self.stop_event.is_set():
                for chunk in self.text_chunker.chunk_text(accumulated_text):
                    if self.stop_event.is_set():
                        break
                    self.chunk_queue.put((chunk_count, chunk))
                    chunk_count += 1
                    if self.on_chunk_processed:
                        self.on_chunk_processed(chunk, "queued")
            
            # Wait for all chunks to be processed
            self.chunk_queue.join()
            
            # Signal completion
            self.audio_queue.put(None)  # Sentinel value
            
            # Wait for audio streaming to complete
            audio_thread.join(timeout=30)
            
            # Call completion callback
            if self.on_streaming_complete:
                self.on_streaming_complete()
            
            # Print final timing metrics
            if first_token_time:
                total_time = time.time() - start_time
                print(f"📊 Streaming Metrics:")
                print(f"   ⏱️  Time to first token: {time_to_first_token:.3f}s")
                if first_audio_time:
                    time_to_first_audio = first_audio_time - start_time
                    print(f"   🔊 Time to first audio: {time_to_first_audio:.3f}s")
                if audio_playback_start_time:
                    time_to_playback = audio_playback_start_time - start_time
                    print(f"   ▶️  Time to start playing: {time_to_playback:.3f}s")
                if last_audio_time:
                    time_to_last_audio = last_audio_time - start_time
                    print(f"   🏁 Time to last audio chunk: {time_to_last_audio:.3f}s")
                print(f"   ⏱️  Total streaming time: {total_time:.3f}s")
                print(f"   📝 Total chunks processed: {chunk_count}")
            
            return True
            
        except Exception as e:
            print(f"Error in stream_text_to_speech: {e}")
            return False
        finally:
            self.is_processing = False
            self._stop_processing_threads()
    
    def _start_processing_threads(self):
        """Start worker threads for processing text chunks."""
        for i in range(self.max_workers):
            thread = threading.Thread(target=self._text_processing_worker, args=(i,))
            thread.daemon = True
            thread.start()
            self.processing_threads.append(thread)
    
    def _stop_processing_threads(self):
        """Stop all processing threads."""
        for _ in self.processing_threads:
            self.chunk_queue.put(None)  # Sentinel value
        
        for thread in self.processing_threads:
            thread.join(timeout=5)
        
        self.processing_threads.clear()
    
    def _text_processing_worker(self, worker_id: int):
        """Worker thread for processing text chunks."""
        print(f"Text processing worker {worker_id} started")
        
        while not self.stop_event.is_set():
            try:
                # Get chunk from queue
                item = self.chunk_queue.get(timeout=1)
                if item is None:  # Sentinel value
                    break
                
                chunk_index, chunk = item
                
                # Process the chunk
                if self.on_chunk_processed:
                    self.on_chunk_processed(chunk, "processing")
                
                # Convert chunk to speech
                audio_data = self._convert_chunk_to_speech(chunk)
                
                if audio_data is not None:
                    # Add to audio queue with index for ordering
                    self.audio_queue.put((chunk_index, audio_data))
                    
                    if self.on_chunk_processed:
                        self.on_chunk_processed(chunk, "completed")
                else:
                    print(f"Failed to convert chunk {chunk_index} to speech")
                    if self.on_chunk_processed:
                        self.on_chunk_processed(chunk, "failed")
                
                self.chunk_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in text processing worker {worker_id}: {e}")
                self.chunk_queue.task_done()
        
        print(f"Text processing worker {worker_id} stopped")
    
    def _audio_streaming_worker(self):
        """Worker thread for streaming audio to Audio2Face."""
        print("Audio streaming worker started")
        
        try:
            if USE_GRPC:
                self._stream_to_audio2face()
            else:
                self._stream_to_file()
                
        except Exception as e:
            print(f"Error in audio streaming worker: {e}")
        
        print("Audio streaming worker stopped")
    
    def _convert_chunk_to_speech(self, chunk: TextChunk) -> Optional[np.ndarray]:
        """Convert a text chunk to speech using ElevenLabs streaming."""
        try:
            print(f"Converting chunk to speech via ElevenLabs: '{chunk.text[:30]}...'")
            
            # Use ElevenLabs streaming TTS
            audio_stream = self.tts_processor.stream_text(chunk.text)
            
            # Collect all audio data
            audio_chunks = []
            for audio_chunk in audio_stream:
                # Ensure audio chunk is a numpy array and 1D
                if audio_chunk is not None and len(audio_chunk) > 0:
                    # Convert to numpy array if it's not already
                    if not isinstance(audio_chunk, np.ndarray):
                        audio_chunk = np.array(audio_chunk)
                    
                    # Flatten the array if it's not 1D
                    if audio_chunk.ndim > 1:
                        audio_chunk = audio_chunk.flatten()
                    
                    audio_chunks.append(audio_chunk)
                
            if audio_chunks:
                # Combine all audio chunks
                combined_audio = np.concatenate(audio_chunks)
                print(f"Chunk converted successfully: {len(combined_audio)} samples")
                return combined_audio
            else:
                print("No audio data received from ElevenLabs")
                return None
                
        except Exception as e:
            print(f"Error converting chunk to speech: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_mp3_data(self, mp3_buffer: bytearray) -> Optional[np.ndarray]:
        """Process MP3 data and convert to numpy array."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(mp3_buffer)
                temp_path = temp_file.name
            
            # Load and process audio
            audio_segment = AudioSegment.from_file(temp_path, format="mp3")
            
            # Convert to mono if needed
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            
            # Resample if needed
            if audio_segment.frame_rate != TARGET_SAMPLE_RATE:
                audio_segment = audio_segment.set_frame_rate(TARGET_SAMPLE_RATE)
            
            # Convert to numpy array
            audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            audio_array = audio_array / (2**15)  # Normalize to [-1, 1]
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return audio_array
            
        except Exception as e:
            print(f"Error processing MP3 data: {e}")
            return None
    
    def _stream_to_audio2face(self):
        """Stream audio to Audio2Face via gRPC."""
        try:
            url = "localhost:50051"
            channel = grpc.insecure_channel(url)
            stub = audio2face_pb2_grpc.Audio2FaceStub(channel)
            
            start_marker = audio2face_pb2.PushAudioRequestStart(
                samplerate=TARGET_SAMPLE_RATE,
                instance_name="/World/audio2face/PlayerStreaming",
                block_until_playback_is_finished=False,
            )
            
            def request_generator():
                yield audio2face_pb2.PushAudioStreamRequest(start_marker=start_marker)
                
                # Process audio chunks in order
                expected_index = 0
                audio_buffer = {}
                
                while True:
                    try:
                        item = self.audio_queue.get(timeout=1)
                        if item is None:  # Sentinel value
                            break
                        
                        chunk_index, audio_data = item
                        audio_buffer[chunk_index] = audio_data
                        
                        # Send chunks in order
                        while expected_index in audio_buffer:
                            audio_chunk = audio_buffer.pop(expected_index)
                            
                            # Break audio into smaller chunks for streaming
                            for i in range(0, len(audio_chunk), FRAME_BUFFER_SIZE):
                                chunk = audio_chunk[i:i+FRAME_BUFFER_SIZE]
                                yield audio2face_pb2.PushAudioStreamRequest(
                                    audio_data=chunk.tobytes()
                                )
                                time.sleep(0.001)  # Small delay to avoid overwhelming
                            
                            expected_index += 1
                            
                            # Call callback if set
                            if self.on_audio_ready:
                                self.on_audio_ready(audio_chunk)
                        
                    except queue.Empty:
                        continue
            
            # Send audio stream
            response = stub.PushAudioStream(request_generator())
            
            if response.success:
                print("Audio streaming completed successfully")
            else:
                print(f"Error in audio streaming: {response.message}")
                
        except Exception as e:
            print(f"Error streaming to Audio2Face: {e}")
        finally:
            if 'channel' in locals():
                channel.close()
    
    def _stream_to_file(self):
        """Stream audio directly to speakers (fallback when gRPC is not available)."""
        try:
            # Process audio chunks in order
            expected_index = 0
            audio_buffer = {}
            all_audio = []
            
            while True:
                try:
                    item = self.audio_queue.get(timeout=1)
                    if item is None:  # Sentinel value
                        break
                    
                    chunk_index, audio_data = item
                    audio_buffer[chunk_index] = audio_data
                    
                    # Process chunks in order
                    while expected_index in audio_buffer:
                        audio_chunk = audio_buffer.pop(expected_index)
                        all_audio.append(audio_chunk)
                        
                        if self.on_audio_ready:
                            self.on_audio_ready(audio_chunk)
                        
                        expected_index += 1
                        
                except queue.Empty:
                    continue
            
            # Combine all audio and play directly
            if all_audio:
                combined_audio = np.concatenate(all_audio)
                print(f"Combined audio: {len(combined_audio)} samples")
                
                # Record playback start time
                playback_start_time = time.time()
                print(f"▶️  Starting audio playback at: {playback_start_time:.3f}s")
                
                # Play the audio using sounddevice
                print("Starting audio playback...")
                sd.play(combined_audio, TARGET_SAMPLE_RATE)
                sd.wait()  # Wait for playback to finish
                
                playback_end_time = time.time()
                playback_duration = playback_end_time - playback_start_time
                print(f"✅ Audio playback completed. Duration: {playback_duration:.3f}s")
                
                print("Audio playback completed")
                
        except Exception as e:
            print(f"Error streaming audio: {e}")
            import traceback
            traceback.print_exc()
    
    def stop_streaming(self):
        """Stop the streaming process."""
        self.stop_event.set()
        self.is_processing = False
    
    def set_chunk_strategy(self, strategy: str):
        """Set the text chunking strategy."""
        self.text_chunker.set_chunk_strategy(strategy)
    
    def set_chunk_limits(self, min_size: int = 20, max_size: int = 200):
        """Set minimum and maximum chunk sizes."""
        self.text_chunker.set_chunk_limits(min_size, max_size)

    def process_text(self, text: str) -> None:
        """Process text through the streaming TTS pipeline."""
        print(f"Starting streaming TTS processing for text: '{text[:50]}...'")
        
        # Chunk the text
        chunks = list(self.text_chunker.chunk_text(text))
        print(f"Text chunked into {len(chunks)} chunks")
        
        # Process chunks in parallel
        self._process_chunks_parallel(chunks)
        
    def _process_chunks_parallel(self, chunks: List[TextChunk]) -> None:
        """Process text chunks in parallel."""
        print(f"Processing {len(chunks)} chunks in parallel...")
        
        # Create a queue for processed audio chunks
        audio_queue = queue.Queue()
        
        # Submit all chunks for processing
        futures = []
        for i, chunk in enumerate(chunks):
            future = self.executor.submit(self._process_chunk, i, chunk, audio_queue)
            futures.append(future)
        
        # Collect and play audio in order
        self._play_audio_in_order(audio_queue, len(chunks))
        
        # Wait for all futures to complete
        for future in futures:
            future.result()
            
        print("All chunks processed successfully")
        
        if self.on_streaming_complete:
            self.on_streaming_complete() 

    def _play_audio_in_order(self, audio_queue: queue.Queue, total_chunks: int) -> None:
        """Play audio chunks in the correct order."""
        try:
            print(f"Playing {total_chunks} audio chunks in order...")
            
            # Process audio chunks in order
            expected_index = 0
            audio_buffer = {}
            all_audio = []
            
            while expected_index < total_chunks:
                try:
                    item = audio_queue.get(timeout=5)  # 5 second timeout
                    if item is None:  # Sentinel value
                        break
                    
                    chunk_index, audio_data = item
                    audio_buffer[chunk_index] = audio_data
                    
                    # Process chunks in order
                    while expected_index in audio_buffer:
                        audio_chunk = audio_buffer.pop(expected_index)
                        all_audio.append(audio_chunk)
                        
                        if self.on_audio_ready:
                            self.on_audio_ready(audio_chunk)
                        
                        expected_index += 1
                        
                except queue.Empty:
                    print(f"Timeout waiting for chunk {expected_index}")
                    break
            
            # Combine all audio and play directly
            if all_audio:
                combined_audio = np.concatenate(all_audio)
                print(f"Combined audio: {len(combined_audio)} samples")
                
                # Play the audio using sounddevice
                print("Starting audio playback...")
                sd.play(combined_audio, TARGET_SAMPLE_RATE)
                sd.wait()  # Wait for playback to finish
                
                print("Audio playback completed")
            else:
                print("No audio data to play")
                
        except Exception as e:
            print(f"Error playing audio in order: {e}")
            import traceback
            traceback.print_exc()

    def _process_chunk(self, chunk_index: int, chunk: TextChunk, audio_queue: queue.Queue) -> None:
        """Process a single text chunk."""
        try:
            print(f"Processing chunk {chunk_index}: '{chunk.text[:30]}...'")
            
            # Convert chunk to speech using ElevenLabs streaming
            audio_data = self._convert_chunk_to_speech(chunk)
            
            if audio_data is not None:
                # Put audio data in queue with chunk index for ordering
                audio_queue.put((chunk_index, audio_data))
                print(f"Chunk {chunk_index} converted to audio: {len(audio_data)} samples")
                
                if self.on_chunk_processed:
                    self.on_chunk_processed(chunk, "success")
            else:
                print(f"Failed to convert chunk {chunk_index} to speech")
                if self.on_chunk_processed:
                    self.on_chunk_processed(chunk, "failed")
                    
        except Exception as e:
            print(f"Error processing chunk {chunk_index}: {e}")
            if self.on_chunk_processed:
                self.on_chunk_processed(chunk, f"error: {e}")
                
    def _convert_chunk_to_speech(self, chunk: TextChunk) -> Optional[np.ndarray]:
        """Convert a text chunk to speech using ElevenLabs streaming."""
        try:
            print(f"Converting chunk to speech via ElevenLabs: '{chunk.text[:30]}...'")
            
            # Use ElevenLabs streaming TTS
            audio_stream = self.tts_processor.stream_text(chunk.text)
            
            # Collect all audio data
            audio_chunks = []
            for audio_chunk in audio_stream:
                # Ensure audio chunk is a numpy array and 1D
                if audio_chunk is not None and len(audio_chunk) > 0:
                    # Convert to numpy array if it's not already
                    if not isinstance(audio_chunk, np.ndarray):
                        audio_chunk = np.array(audio_chunk)
                    
                    # Flatten the array if it's not 1D
                    if audio_chunk.ndim > 1:
                        audio_chunk = audio_chunk.flatten()
                    
                    audio_chunks.append(audio_chunk)
                
            if audio_chunks:
                # Combine all audio chunks
                combined_audio = np.concatenate(audio_chunks)
                print(f"Chunk converted successfully: {len(combined_audio)} samples")
                return combined_audio
            else:
                print("No audio data received from ElevenLabs")
                return None
                
        except Exception as e:
            print(f"Error converting chunk to speech: {e}")
            import traceback
            traceback.print_exc()
            return None 