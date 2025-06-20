"""
Utility script for setting up and managing local LLM with Ollama.
"""

import os
import sys
import subprocess
import requests
import json
from typing import List, Dict, Optional

def check_ollama_installation():
    """Check if Ollama is installed and running."""
    try:
        # Check if ollama command is available
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama is installed but not working properly")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Ollama command timed out")
        return False

def check_ollama_service():
    """Check if Ollama service is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            return True
        else:
            print(f"❌ Ollama service returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Ollama service is not running: {e}")
        return False

def get_available_models() -> List[Dict]:
    """Get list of available models from Ollama."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("models", [])
        else:
            print(f"Failed to get models: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error getting models: {e}")
        return []

def list_models():
    """List all available models."""
    models = get_available_models()
    if not models:
        print("No models found. You may need to pull a model first.")
        return
    
    print("\n=== Available Models ===")
    for model in models:
        name = model.get("name", "Unknown")
        size = model.get("size", 0)
        size_gb = size / (1024**3) if size > 0 else 0
        print(f"📦 {name} ({size_gb:.1f} GB)")
    print("=======================\n")

def pull_model(model_name: str = "mistral:7b"):
    """Pull a model from Ollama."""
    print(f"Pulling model {model_name}...")
    print("This may take several minutes depending on your internet connection.")
    
    try:
        response = requests.post(
            "http://localhost:11434/api/pull",
            json={"name": model_name},
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            print(f"✅ Successfully pulled model {model_name}")
            return True
        else:
            print(f"❌ Failed to pull model: {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print("❌ Pull operation timed out")
        return False
    except Exception as e:
        print(f"❌ Error pulling model: {e}")
        return False

def test_model(model_name: str = "mistral:7b"):
    """Test a model with a simple prompt."""
    print(f"Testing model {model_name}...")
    
    try:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "Hello! Please respond with 'Hello from local LLM!'"}
            ],
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("message", {}).get("content", "")
            print(f"✅ Model test successful!")
            print(f"Response: {content}")
            return True
        else:
            print(f"❌ Model test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def show_installation_instructions():
    """Show instructions for installing Ollama."""
    print("\n=== Ollama Installation Instructions ===")
    print("1. Visit https://ollama.ai")
    print("2. Download and install Ollama for your operating system")
    print("3. Start Ollama service:")
    print("   - Windows: Ollama should start automatically")
    print("   - macOS: Run 'ollama serve' in terminal")
    print("   - Linux: Run 'ollama serve' in terminal")
    print("4. Pull a model (e.g., 'ollama pull mistral:7b')")
    print("5. Set USE_LOCAL_LLM=true in your .env file")
    print("========================================\n")

def main():
    """Main function for the setup script."""
    print("🤖 Local LLM Setup Utility")
    print("=" * 30)
    
    # Check Ollama installation
    if not check_ollama_installation():
        show_installation_instructions()
        return
    
    # Check Ollama service
    if not check_ollama_service():
        print("\nTo start Ollama service:")
        print("- Windows: Restart Ollama application")
        print("- macOS/Linux: Run 'ollama serve' in terminal")
        return
    
    # Show available models
    list_models()
    
    # Interactive menu
    while True:
        print("\nOptions:")
        print("1. List available models")
        print("2. Pull a model (e.g., mistral:7b)")
        print("3. Test a model")
        print("4. Show installation instructions")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            list_models()
        elif choice == "2":
            model_name = input("Enter model name (e.g., mistral:7b): ").strip()
            if model_name:
                pull_model(model_name)
                list_models()  # Refresh the list
        elif choice == "3":
            model_name = input("Enter model name to test: ").strip()
            if model_name:
                test_model(model_name)
        elif choice == "4":
            show_installation_instructions()
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main() 