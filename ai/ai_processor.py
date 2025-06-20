"""
Module for sending text to AI models and receiving responses.
"""

import requests
import time
import logging
import json
import anthropic
import openai
from utils.config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY,
    AI_PROVIDER, CHATGPT_MODEL, CLAUDE_MODEL, DEEPSEEK_MODEL,
    SYSTEM_PROMPT, USE_LOCAL_LLM, OLLAMA_URL, LOCAL_LLM_MODEL,
    LOCAL_LLM_TEMPERATURE, LOCAL_LLM_MAX_TOKENS
)

logger = logging.getLogger(__name__)

class AIProcessor:
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"
        self.local_llm_processor = None
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize API clients based on available API keys."""
        if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
            self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
        if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "your_anthropic_api_key_here":
            self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            
        # Initialize local LLM processor if enabled
        if USE_LOCAL_LLM:
            try:
                from .local_llm_processor import LocalLLMProcessor
                self.local_llm_processor = LocalLLMProcessor(base_url=OLLAMA_URL)
                if self.local_llm_processor.is_available():
                    logger.info("Local LLM processor initialized successfully")
                else:
                    logger.warning("Local LLM is enabled but Ollama is not available")
            except ImportError as e:
                logger.error(f"Failed to import LocalLLMProcessor: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize local LLM processor: {e}")
    
    def process_with_all_available(self, text, conversation_history=None):
        """
        Process text with all available AI models in parallel and return the fastest response.
        Returns a tuple of (provider_name, response_text).
        
        Args:
            text: The text to process
            conversation_history: List of previous messages in the conversation
        """
        results = []
        start_time = time.time()
        
        # Define model processing functions
        model_processors = []
        
        if self.openai_client:
            model_processors.append((self._process_with_chatgpt, "chatgpt"))
            
        if self.anthropic_client:
            model_processors.append((self._process_with_claude, "claude"))
            
        if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your_deepseek_api_key_here":
            model_processors.append((self._process_with_deepseek, "deepseek"))
            
        # Add local LLM if available
        if self.local_llm_processor and self.local_llm_processor.is_available():
            model_processors.append((self._process_with_local_llm, "local_llm"))
            
        if not model_processors:
            logger.error("No AI models configured. Please add at least one API key or enable local LLM in config.py")
            return None, "Error: No AI models configured"
            
        # Use only the specified provider if set
        if AI_PROVIDER != "fastest":
            for processor, name in model_processors:
                if name == AI_PROVIDER:
                    response = processor(text, conversation_history)
                    if response:
                        processing_time = time.time() - start_time
                        logger.info(f"Got response from {name} in {processing_time:.2f} seconds")
                        print(f"AI processing finished ({name}): '{response[:100]}{'...' if len(response) > 100 else ''}'")
                        return name, response
            
            # Fallback to any available provider if specified one fails
            logger.warning(f"Specified provider {AI_PROVIDER} failed, trying alternatives")
        
        # Try all providers and use the fastest response
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=len(model_processors)) as executor:
            # Submit all tasks
            future_to_model = {
                executor.submit(processor, text, conversation_history): name
                for processor, name in model_processors
            }
            
            # Get the first completed task
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    if result:
                        processing_time = time.time() - start_time
                        logger.info(f"Got response from {model_name} in {processing_time:.2f} seconds")
                        print(f"AI processing finished ({model_name}): '{result[:100]}{'...' if len(result) > 100 else ''}'")
                        return model_name, result
                except Exception as e:
                    logger.error(f"Error with {model_name}: {e}")
        
        # If all models fail, return an error
        print("AI processing finished: All models failed")
        return None, "Error: All AI models failed to generate a response"
    
    def _process_with_chatgpt(self, text, conversation_history=None):
        """Send text to ChatGPT and get response."""
        if not self.openai_client:
            logger.error("OpenAI client not initialized")
            return None
            
        try:
            # Prepare messages with conversation history
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            # Add conversation history if available
            if conversation_history:
                messages.extend(conversation_history)
            else:
                # If no history, just add the current message
                messages.append({"role": "user", "content": text})
                
            response = self.openai_client.chat.completions.create(
                model=CHATGPT_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error with ChatGPT: {e}")
            return None
    
    def _process_with_claude(self, text, conversation_history=None):
        """Send text to Claude and get response."""
        if not self.anthropic_client:
            logger.error("Anthropic client not initialized")
            return None
            
        try:
            # Prepare messages with conversation history
            messages = []
            
            # Add conversation history if available
            if conversation_history:
                messages.extend(conversation_history)
            else:
                # If no history, just add the current message
                messages.append({"role": "user", "content": text})
                
            response = self.anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=500,
                messages=messages,
                system=SYSTEM_PROMPT
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error with Claude: {e}")
            return None
    
    def _process_with_deepseek(self, text, conversation_history=None):
        """Send text to DeepSeek and get response."""
        if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_deepseek_api_key_here":
            logger.error("DeepSeek API key not configured")
            return None
            
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
            }
            
            # Prepare messages with conversation history
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            # Add conversation history if available
            if conversation_history:
                messages.extend(conversation_history)
            else:
                # If no history, just add the current message
                messages.append({"role": "user", "content": text})
                
            payload = {
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            response = requests.post(
                self.deepseek_api_url,
                headers=headers,
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error with DeepSeek: {e}")
            return None
            
    def _process_with_local_llm(self, text, conversation_history=None):
        """Process text with local LLM using Ollama."""
        if not self.local_llm_processor:
            logger.error("Local LLM processor not initialized")
            return None
            
        try:
            response = self.local_llm_processor.process_with_model(
                text=text,
                model_name=LOCAL_LLM_MODEL,
                conversation_history=conversation_history,
                temperature=LOCAL_LLM_TEMPERATURE,
                max_tokens=LOCAL_LLM_MAX_TOKENS
            )
            return response
        except Exception as e:
            logger.error(f"Error with local LLM: {e}")
            return None
            
    def get_available_models(self):
        """Get information about available models."""
        models_info = {}
        
        if self.openai_client:
            models_info["chatgpt"] = {"status": "available", "model": CHATGPT_MODEL}
        else:
            models_info["chatgpt"] = {"status": "unavailable", "reason": "API key not configured"}
            
        if self.anthropic_client:
            models_info["claude"] = {"status": "available", "model": CLAUDE_MODEL}
        else:
            models_info["claude"] = {"status": "unavailable", "reason": "API key not configured"}
            
        if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your_deepseek_api_key_here":
            models_info["deepseek"] = {"status": "available", "model": DEEPSEEK_MODEL}
        else:
            models_info["deepseek"] = {"status": "unavailable", "reason": "API key not configured"}
            
        if self.local_llm_processor:
            if self.local_llm_processor.is_available():
                available_models = self.local_llm_processor.get_available_models()
                models_info["local_llm"] = {
                    "status": "available", 
                    "model": LOCAL_LLM_MODEL,
                    "available_models": available_models
                }
            else:
                models_info["local_llm"] = {"status": "unavailable", "reason": "Ollama not running or no models available"}
        else:
            models_info["local_llm"] = {"status": "unavailable", "reason": "Local LLM not enabled"}
            
        return models_info
