"""
Test script for local LLM integration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.local_llm_processor import LocalLLMProcessor
from utils.config import USE_LOCAL_LLM, OLLAMA_URL, LOCAL_LLM_MODEL

def test_local_llm_connection():
    """Test connection to Ollama."""
    print("Testing Ollama connection...")
    
    processor = LocalLLMProcessor(base_url=OLLAMA_URL)
    
    if processor.is_available():
        print("✅ Ollama connection successful")
        models = processor.get_available_models()
        print(f"Available models: {models}")
        return True
    else:
        print("❌ Ollama connection failed")
        return False

def test_model_processing():
    """Test model processing with a simple prompt."""
    print("\nTesting model processing...")
    
    processor = LocalLLMProcessor(base_url=OLLAMA_URL)
    
    if not processor.is_available():
        print("❌ Ollama not available, skipping test")
        return False
    
    # Test with a simple prompt
    test_prompt = "Hello! Please respond with a short greeting in Spanish."
    
    try:
        response = processor.process_with_model(
            text=test_prompt,
            model_name=LOCAL_LLM_MODEL,
            temperature=0.7,
            max_tokens=100
        )
        
        if response:
            print(f"✅ Model processing successful")
            print(f"Prompt: {test_prompt}")
            print(f"Response: {response}")
            return True
        else:
            print("❌ Model processing failed - no response")
            return False
            
    except Exception as e:
        print(f"❌ Model processing error: {e}")
        return False

def test_conversation_history():
    """Test conversation history handling."""
    print("\nTesting conversation history...")
    
    processor = LocalLLMProcessor(base_url=OLLAMA_URL)
    
    if not processor.is_available():
        print("❌ Ollama not available, skipping test")
        return False
    
    # Create conversation history
    conversation_history = [
        {"role": "user", "content": "What is your name?"},
        {"role": "assistant", "content": "I am an AI assistant."}
    ]
    
    test_prompt = "What did I just ask you?"
    
    try:
        response = processor.process_with_model(
            text=test_prompt,
            model_name=LOCAL_LLM_MODEL,
            conversation_history=conversation_history,
            temperature=0.7,
            max_tokens=100
        )
        
        if response:
            print(f"✅ Conversation history test successful")
            print(f"Prompt: {test_prompt}")
            print(f"Response: {response}")
            return True
        else:
            print("❌ Conversation history test failed - no response")
            return False
            
    except Exception as e:
        print(f"❌ Conversation history test error: {e}")
        return False

def main():
    """Run all tests."""
    print("🤖 Local LLM Integration Tests")
    print("=" * 40)
    
    # Check if local LLM is enabled
    if not USE_LOCAL_LLM:
        print("⚠️  Local LLM is not enabled in config")
        print("Set USE_LOCAL_LLM=true in your .env file to enable")
        return
    
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Default model: {LOCAL_LLM_MODEL}")
    
    # Run tests
    tests = [
        test_local_llm_connection,
        test_model_processing,
        test_conversation_history
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n{'='*40}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Local LLM integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check your Ollama setup.")

if __name__ == "__main__":
    main() 