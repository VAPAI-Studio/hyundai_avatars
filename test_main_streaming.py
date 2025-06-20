"""
Simple test to run the main application with streaming enabled.
"""

import sys
import os
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set environment variable to enable streaming
os.environ["USE_STREAMING_PIPELINE"] = "true"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_main_streaming():
    """Test the main application with streaming enabled."""
    print("🚀 Testing Main Application with Streaming Pipeline")
    print("=" * 50)
    
    try:
        from main import VoiceAssistant
        
        # Initialize the voice assistant
        print("📱 Initializing Voice Assistant...")
        assistant = VoiceAssistant()
        
        # Test the streaming pipeline directly
        print("\n🧪 Testing streaming pipeline directly...")
        
        test_text = "Hola, cuéntame sobre Hyundai"
        print(f"Input: {test_text}")
        
        # Test streaming pipeline
        success = assistant._process_with_streaming_pipeline(test_text)
        
        if success:
            print("✅ Streaming pipeline test successful")
        else:
            print("❌ Streaming pipeline test failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_main_streaming() 