"""
Simple Silero VAD demo script.
"""

import torch
import sounddevice as sd
import numpy as np
import time

def silero_vad_demo():
    """Demo of Silero VAD functionality."""
    print("Loading Silero VAD model...")
    
    # Load the model
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad', 
        model='silero_vad', 
        force_reload=False  # Use cached model if available
    )
    (get_speech_timestamps, _, _, *_) = utils
    
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 16000  # 1 second
    
    def callback(indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
            
        # Convert to mono
        mono = np.mean(indata, axis=1)
        tensor_audio = torch.from_numpy(mono).float()
        
        # Get speech timestamps
        segments = get_speech_timestamps(
            tensor_audio, 
            model, 
            sampling_rate=SAMPLE_RATE,
            min_speech_duration_ms=300,  # 0.3 seconds
            min_silence_duration_ms=1500  # 1.5 seconds
        )
        
        for seg in segments:
            print(f"🎤 Speech detected from {seg['start']/SAMPLE_RATE:.2f}s to {seg['end']/SAMPLE_RATE:.2f}s")
    
    print("🎤 Recording with Silero VAD...")
    print("Press Ctrl+C to stop")
    
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, 
            channels=1, 
            callback=callback, 
            blocksize=CHUNK_SIZE,
            dtype=np.float32
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n⏹️  Stopped recording")

if __name__ == "__main__":
    silero_vad_demo() 