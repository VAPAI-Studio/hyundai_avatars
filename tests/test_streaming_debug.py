"""
Debug test for streaming LLM processor to identify why responses are blank.
"""

import sys
import os
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.streaming_llm_processor import StreamingLLMProcessor

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_streaming_debug():
    """Debug test for streaming LLM processor."""
    print("🔍 Debugging Streaming LLM Processor...")
    
    # Initialize processor
    processor = StreamingLLMProcessor()
    
    # Check available providers
    providers = processor.get_available_providers()
    print("\n📋 Available providers:")
    for name, info in providers.items():
        status = "✅" if info["status"] == "available" else "❌"
        print(f"  {status} {name}: {info}")
    
    # Test with a simple prompt
    test_prompt = "Hola, responde con un saludo corto."
    print(f"\n📤 Testing with prompt: '{test_prompt}'")
    
    try:
        # Try to get a streaming response
        print("🔄 Starting streaming...")
        text_chunks = []
        total_chars = 0
        
        for chunk in processor.stream_text(test_prompt, provider="fastest"):
            text_chunks.append(chunk)
            total_chars += len(chunk)
            print(f"  📝 Chunk {len(text_chunks)}: '{chunk}' (length: {len(chunk)})")
            
            # Limit for testing
            if len(text_chunks) >= 10:
                break
        
        print(f"\n📊 Results:")
        print(f"  Total chunks received: {len(text_chunks)}")
        print(f"  Total characters: {total_chars}")
        print(f"  Combined text: '{''.join(text_chunks)}'")
        
        if text_chunks:
            print("✅ Streaming test successful")
        else:
            print("❌ No text chunks received")
            
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()

def test_provider_specific():
    """Test specific providers."""
    print("\n🎯 Testing Specific Providers...")
    
    processor = StreamingLLMProcessor()
    providers = processor.get_available_providers()
    
    test_prompt = "Di 'hola' en español."
    
    for provider_name, info in providers.items():
        if info["status"] == "available":
            print(f"\n🧪 Testing {provider_name}...")
            try:
                chunks = []
                for chunk in processor.stream_text(test_prompt, provider=provider_name):
                    chunks.append(chunk)
                    print(f"  {provider_name}: '{chunk}'")
                    if len(chunks) >= 3:  # Limit for testing
                        break
                
                if chunks:
                    print(f"  ✅ {provider_name} working")
                else:
                    print(f"  ❌ {provider_name} returned no chunks")
                    
            except Exception as e:
                print(f"  ❌ {provider_name} error: {e}")

def main():
    """Run debug tests."""
    print("🔍 Streaming LLM Debug Tests")
    print("=" * 40)
    
    test_streaming_debug()
    test_provider_specific()
    
    print("\n" + "=" * 40)
    print("Debug tests completed")

if __name__ == "__main__":
    main() 