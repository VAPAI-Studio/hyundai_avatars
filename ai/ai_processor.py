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
    SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)

class AIProcessor:
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize API clients based on available API keys."""
        if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
            self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
        if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "your_anthropic_api_key_here":
            self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
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
            
        if not model_processors:
            logger.error("No API keys configured. Please add at least one API key in config.py")
            return None, "Error: No API keys configured"
            
        # Use only the specified provider if set
        if AI_PROVIDER != "fastest":
            for processor, name in model_processors:
                if name == AI_PROVIDER:
                    response = processor(text, conversation_history)
                    if response:
                        processing_time = time.time() - start_time
                        logger.info(f"Got response from {name} in {processing_time:.2f} seconds")
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
                        return model_name, result
                except Exception as e:
                    logger.error(f"Error with {model_name}: {e}")
        
        # If all models fail, return an error
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
