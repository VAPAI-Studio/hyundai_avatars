"""
Module for processing text with local LLM models using Ollama.
"""

import requests
import time
import logging
import json
from typing import Optional, List, Dict, Any
from utils.config import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class LocalLLMProcessor:
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the local LLM processor.
        
        Args:
            base_url: The URL where Ollama is running (default: http://localhost:11434)
        """
        self.base_url = base_url
        self.available_models = []
        self._check_ollama_connection()
        
    def _check_ollama_connection(self):
        """Check if Ollama is running and get available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                self.available_models = [model["name"] for model in models_data.get("models", [])]
                logger.info(f"Connected to Ollama. Available models: {self.available_models}")
            else:
                logger.warning(f"Ollama connection failed with status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to Ollama at {self.base_url}: {e}")
            logger.info("To use local LLM, please install and start Ollama: https://ollama.ai")
            
    def is_available(self) -> bool:
        """Check if Ollama is available and has models."""
        return len(self.available_models) > 0
        
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return self.available_models.copy()
        
    def process_with_model(self, text: str, model_name: str = "mistral:7b", 
                          conversation_history: Optional[List[Dict[str, str]]] = None,
                          temperature: float = 0.7, max_tokens: int = 500) -> Optional[str]:
        """
        Process text with a local LLM model.
        
        Args:
            text: The text to process
            model_name: The name of the model to use (default: mistral:7b)
            conversation_history: List of previous messages in the conversation
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The generated response text, or None if failed
        """
        if not self.is_available():
            logger.error("Ollama is not available")
            return None
            
        if model_name not in self.available_models:
            logger.error(f"Model {model_name} not found. Available models: {self.available_models}")
            return None
            
        try:
            # Prepare messages with conversation history
            messages = []
            
            # Add system message
            messages.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })
            
            # Add conversation history if available
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": text
            })
            
            # Prepare the request payload
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            # Make the request to Ollama
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "").strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Ollama: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in local LLM processing: {e}")
            return None
            
    def pull_model(self, model_name: str = "mistral:7b") -> bool:
        """
        Pull a model from Ollama's model library.
        
        Args:
            model_name: The name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Pulling model {model_name}...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=300  # 5 minutes timeout for model download
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully pulled model {model_name}")
                # Refresh available models
                self._check_ollama_connection()
                return True
            else:
                logger.error(f"Failed to pull model {model_name}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
            
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Args:
            model_name: The name of the model
            
        Returns:
            Model information dictionary or None if not found
        """
        try:
            response = requests.get(f"{self.base_url}/api/show", 
                                  json={"name": model_name}, 
                                  timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get model info for {model_name}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return None 