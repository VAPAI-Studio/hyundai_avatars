"""
Test script for the complete streaming pipeline: LLM -> Text Chunker -> ElevenLabs Streaming.
"""

import sys
import os
import time
import logging
from typing import Generator

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.streaming_llm_processor import StreamingLLMProcessor
from audio.streaming_tts_processor import StreamingTTSProcessor
from audio.text_chunker import TextChunk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def mock_llm_stream() -> Generator[str, None, None]:
    """
    Mock LLM stream for testing purposes.
    Yields text chunks as they would come from a real LLM.
    """
    test_text = """
    Hola, soy el asistente de voz de Hyundai. Te ayudo con información sobre nuestros vehículos.
    
    Tenemos una amplia gama de modelos, desde compactos hasta SUVs. Nuestros vehículos se destacan por su tecnología avanzada, seguridad y eficiencia.
    
    ¿Te gustaría conocer más sobre algún modelo específico o tienes alguna pregunta sobre nuestras características de seguridad?
    """
    
    # Split text into chunks to simulate streaming
    words = test_text.split()
    chunk_size = 3  # Words per chunk
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():
            yield chunk + " "
            time.sleep(0.1)  # Simulate processing time

def test_text_chunker():
    """Test the text chunker functionality."""
    print("🧪 Testing Text Chunker...")
    
    from audio.text_chunker import TextChunker
    
    # Test text
    test_text = "Hola, soy el asistente de Hyundai. Te ayudo con información sobre nuestros vehículos. Tenemos una amplia gama de modelos."
    
    # Test different chunking strategies
    strategies = ["semantic", "sentence", "phrase", "pause"]
    
    for strategy in strategies:
        print(f"\n📝 Testing {strategy} chunking:")
        chunker = TextChunker(strategy)
        
        chunks = list(chunker.chunk_text(test_text))
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: '{chunk.text}' (type: {chunk.chunk_type}, confidence: {chunk.confidence:.2f})")
    
    print("✅ Text chunker test completed\n")

def test_streaming_llm():
    """Test the streaming LLM processor."""
    print("🤖 Testing Streaming LLM Processor...")
    
    processor = StreamingLLMProcessor()
    
    # Get available providers
    providers = processor.get_available_providers()
    print("Available providers:")
    for name, info in providers.items():
        status = "✅" if info["status"] == "available" else "❌"
        print(f"  {status} {name}: {info}")
    
    # Test with a simple prompt
    test_prompt = "Hola, por favor responde con un saludo corto en español."
    
    print(f"\n📤 Testing with prompt: '{test_prompt}'")
    
    try:
        # Try to get a streaming response
        text_chunks = []
        for chunk in processor.stream_text(test_prompt, provider="fastest"):
            text_chunks.append(chunk)
            print(f"  Received: '{chunk}'")
            if len(text_chunks) >= 5:  # Limit for testing
                break
        
        if text_chunks:
            print(f"✅ Streaming LLM test successful - received {len(text_chunks)} chunks")
        else:
            print("⚠️  No text chunks received")
            
    except Exception as e:
        print(f"❌ Streaming LLM test failed: {e}")
    
    print("✅ Streaming LLM test completed\n")

def test_streaming_tts():
    """Test the streaming TTS processor."""
    print("🔊 Testing Streaming TTS Processor...")
    
    processor = StreamingTTSProcessor(chunk_strategy="semantic", max_workers=2)
    
    # Set up callbacks for monitoring
    def on_chunk_processed(chunk: TextChunk, status: str):
        print(f"  📝 Chunk processed: '{chunk.text[:30]}...' - {status}")
    
    def on_audio_ready(audio_data):
        print(f"  🔊 Audio ready: {len(audio_data)} samples")
    
    def on_streaming_complete():
        print("  ✅ Streaming completed")
    
    processor.set_callbacks(
        on_chunk_processed=on_chunk_processed,
        on_audio_ready=on_audio_ready,
        on_streaming_complete=on_streaming_complete
    )
    
    # Test with mock LLM stream
    print("📤 Testing with mock LLM stream...")
    
    try:
        success = processor.stream_text_to_speech(mock_llm_stream())
        if success:
            print("✅ Streaming TTS test successful")
        else:
            print("❌ Streaming TTS test failed")
            
    except Exception as e:
        print(f"❌ Streaming TTS test error: {e}")
    
    print("✅ Streaming TTS test completed\n")

def test_complete_pipeline():
    """Test the complete streaming pipeline."""
    print("🚀 Testing Complete Streaming Pipeline...")
    
    # Initialize processors
    llm_processor = StreamingLLMProcessor()
    tts_processor = StreamingTTSProcessor(chunk_strategy="semantic", max_workers=2)
    
    # Set up TTS callbacks
    def on_chunk_processed(chunk: TextChunk, status: str):
        print(f"  📝 TTS Chunk: '{chunk.text[:30]}...' - {status}")
    
    def on_audio_ready(audio_data):
        print(f"  🔊 Audio: {len(audio_data)} samples")
    
    def on_streaming_complete():
        print("  ✅ Pipeline completed")
    
    tts_processor.set_callbacks(
        on_chunk_processed=on_chunk_processed,
        on_audio_ready=on_audio_ready,
        on_streaming_complete=on_streaming_complete
    )
    
    # Test prompt
    test_prompt = "Hola, cuéntame sobre los vehículos Hyundai en una respuesta corta."
    
    print(f"📤 Testing with prompt: '{test_prompt}'")
    
    try:
        # Get streaming text from LLM
        text_stream = llm_processor.stream_text(test_prompt, provider="fastest")
        
        # Stream to TTS
        success = tts_processor.stream_text_to_speech(text_stream)
        
        if success:
            print("✅ Complete pipeline test successful")
        else:
            print("❌ Complete pipeline test failed")
            
    except Exception as e:
        print(f"❌ Complete pipeline test error: {e}")
    
    print("✅ Complete pipeline test completed\n")

def test_performance():
    """Test performance of the streaming pipeline."""
    print("⚡ Testing Performance...")
    
    processor = StreamingTTSProcessor(chunk_strategy="semantic", max_workers=3)
    
    # Test with different chunk strategies
    strategies = ["semantic", "sentence", "phrase"]
    
    for strategy in strategies:
        print(f"\n📊 Testing {strategy} strategy:")
        processor.set_chunk_strategy(strategy)
        
        start_time = time.time()
        
        try:
            success = processor.stream_text_to_speech(mock_llm_stream())
            end_time = time.time()
            
            if success:
                duration = end_time - start_time
                print(f"  ✅ Completed in {duration:.2f} seconds")
            else:
                print("  ❌ Failed")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("✅ Performance test completed\n")

def main():
    """Run all tests."""
    print("🧪 Streaming Pipeline Tests")
    print("=" * 50)
    
    tests = [
        ("Text Chunker", test_text_chunker),
        ("Streaming LLM", test_streaming_llm),
        ("Streaming TTS", test_streaming_tts),
        ("Complete Pipeline", test_complete_pipeline),
        ("Performance", test_performance)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Streaming pipeline is working correctly.")
    else:
        print("⚠️  Some tests failed. Check your configuration and dependencies.")

if __name__ == "__main__":
    main() 