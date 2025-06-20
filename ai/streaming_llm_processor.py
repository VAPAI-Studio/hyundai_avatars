"""
Module for streaming text from LLM models as they generate responses.
"""

import requests
import time
import logging
import json
import anthropic
import openai
from typing import Generator, Optional, List, Dict, Any
from utils.config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY,
    AI_PROVIDER, CHATGPT_MODEL, CLAUDE_MODEL, DEEPSEEK_MODEL,
    SYSTEM_PROMPT, USE_LOCAL_LLM, OLLAMA_URL, LOCAL_LLM_MODEL,
    LOCAL_LLM_TEMPERATURE, LOCAL_LLM_MAX_TOKENS
)

logger = logging.getLogger(__name__)

class StreamingLLMProcessor:
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"
        self.local_llm_processor = None
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize API clients based on available API keys."""
        if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
            try:
                self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None
            
        if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "your_anthropic_api_key_here":
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                logger.info("Anthropic client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                self.anthropic_client = None
            
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
    
    def stream_text(self, text: str, conversation_history: Optional[List[Dict[str, str]]] = None, 
                   provider: str = "fastest") -> Generator[str, None, None]:
        """
        Stream text from LLM models as they generate responses.
        
        Args:
            text: The text to process
            conversation_history: List of previous messages in the conversation
            provider: Which provider to use ("fastest", "chatgpt", "claude", "deepseek", "local_llm")
            
        Yields:
            Generated text chunks as they become available
        """
        logger.info(f"Starting streaming with provider: {provider}")
        logger.info(f"Input text: {text[:100]}...")
        
        if provider == "fastest":
            # Try providers in order of preference
            providers = ["chatgpt", "claude", "deepseek", "local_llm"]
        else:
            providers = [provider]
            
        for provider_name in providers:
            logger.info(f"Trying provider: {provider_name}")
            try:
                if provider_name == "chatgpt" and self.openai_client:
                    logger.info("Attempting ChatGPT streaming...")
                    yield from self._stream_from_chatgpt(text, conversation_history)
                    logger.info("ChatGPT streaming completed successfully")
                    return
                elif provider_name == "claude" and self.anthropic_client:
                    logger.info("Attempting Claude streaming...")
                    yield from self._stream_from_claude(text, conversation_history)
                    logger.info("Claude streaming completed successfully")
                    return
                elif provider_name == "deepseek" and DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your_deepseek_api_key_here":
                    logger.info("Attempting DeepSeek streaming...")
                    yield from self._stream_from_deepseek(text, conversation_history)
                    logger.info("DeepSeek streaming completed successfully")
                    return
                elif provider_name == "local_llm" and self.local_llm_processor and self.local_llm_processor.is_available():
                    logger.info("Attempting Local LLM streaming...")
                    yield from self._stream_from_local_llm(text, conversation_history)
                    logger.info("Local LLM streaming completed successfully")
                    return
                else:
                    logger.warning(f"Provider {provider_name} is not available")
            except Exception as e:
                logger.error(f"Error streaming from {provider_name}: {e}")
                continue
                
        # If all providers fail, yield an error message and fallback response
        logger.error("All LLM providers failed to generate a response")
        error_msg = "Lo siento, no pude procesar tu solicitud en este momento. Por favor, intenta de nuevo."
        yield error_msg
    
    def _stream_from_chatgpt(self, text: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Generator[str, None, None]:
        """Stream text from ChatGPT."""
        try:
            # Prepare messages with conversation history
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            # Add conversation history if available
            if conversation_history:
                messages.extend(conversation_history)
            else:
                # If no history, just add the current message
                messages.append({"role": "user", "content": text})
                
            logger.info(f"Sending request to ChatGPT with {len(messages)} messages")
            
            response = self.openai_client.chat.completions.create(
                model=CHATGPT_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                stream=True
            )
            
            chunk_count = 0
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    chunk_count += 1
                    logger.debug(f"ChatGPT chunk {chunk_count}: '{content}'")
                    yield content
                    
            logger.info(f"ChatGPT streaming completed with {chunk_count} chunks")
                    
        except Exception as e:
            logger.error(f"Error streaming from ChatGPT: {e}")
            yield f"Error: {str(e)}"
    
    def _stream_from_claude(self, text: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Generator[str, None, None]:
        """Stream text from Claude."""
        try:
            # Prepare messages with conversation history
            messages = []
            
            # Add conversation history if available
            if conversation_history:
                messages.extend(conversation_history)
            else:
                # If no history, just add the current message
                messages.append({"role": "user", "content": text})
                
            logger.info(f"Sending request to Claude with {len(messages)} messages")
            
            with self.anthropic_client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=500,
                messages=messages,
                system=SYSTEM_PROMPT
            ) as stream:
                chunk_count = 0
                for text_chunk in stream.text_stream:
                    chunk_count += 1
                    logger.debug(f"Claude chunk {chunk_count}: '{text_chunk}'")
                    yield text_chunk
                    
            logger.info(f"Claude streaming completed with {chunk_count} chunks")
                    
        except Exception as e:
            logger.error(f"Error streaming from Claude: {e}")
            yield f"Error: {str(e)}"
    
    def _stream_from_deepseek(self, text: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Generator[str, None, None]:
        """Stream text from DeepSeek."""
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
                "max_tokens": 500,
                "stream": True
            }
            
            logger.info(f"Sending request to DeepSeek with {len(messages)} messages")
            
            response = requests.post(
                self.deepseek_api_url,
                headers=headers,
                json=payload,
                stream=True
            )
            
            chunk_count = 0
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    chunk_count += 1
                                    logger.debug(f"DeepSeek chunk {chunk_count}: '{content}'")
                                    yield content
                        except json.JSONDecodeError:
                            continue
                            
            logger.info(f"DeepSeek streaming completed with {chunk_count} chunks")
                            
        except Exception as e:
            logger.error(f"Error streaming from DeepSeek: {e}")
            yield f"Error: {str(e)}"
    
    def _stream_from_local_llm(self, text: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Generator[str, None, None]:
        """Stream text from local LLM using Ollama."""
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
                "model": LOCAL_LLM_MODEL,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": LOCAL_LLM_TEMPERATURE,
                    "num_predict": LOCAL_LLM_MAX_TOKENS
                }
            }
            
            logger.info(f"Sending request to Local LLM with {len(messages)} messages")
            
            # Make the streaming request to Ollama
            response = requests.post(
                f"{self.local_llm_processor.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=30
            )
            
            chunk_count = 0
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if 'message' in chunk and 'content' in chunk['message']:
                            content = chunk['message']['content']
                            chunk_count += 1
                            logger.debug(f"Local LLM chunk {chunk_count}: '{content}'")
                            yield content
                    except json.JSONDecodeError:
                        continue
                        
            logger.info(f"Local LLM streaming completed with {chunk_count} chunks")
                        
        except Exception as e:
            logger.error(f"Error streaming from local LLM: {e}")
            yield f"Error: {str(e)}"
    
    def get_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available streaming providers."""
        providers_info = {}
        
        if self.openai_client:
            providers_info["chatgpt"] = {"status": "available", "model": CHATGPT_MODEL}
        else:
            providers_info["chatgpt"] = {"status": "unavailable", "reason": "API key not configured or client failed to initialize"}
            
        if self.anthropic_client:
            providers_info["claude"] = {"status": "available", "model": CLAUDE_MODEL}
        else:
            providers_info["claude"] = {"status": "unavailable", "reason": "API key not configured or client failed to initialize"}
            
        if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your_deepseek_api_key_here":
            providers_info["deepseek"] = {"status": "available", "model": DEEPSEEK_MODEL}
        else:
            providers_info["deepseek"] = {"status": "unavailable", "reason": "API key not configured"}
            
        if self.local_llm_processor:
            if self.local_llm_processor.is_available():
                available_models = self.local_llm_processor.get_available_models()
                providers_info["local_llm"] = {
                    "status": "available", 
                    "model": LOCAL_LLM_MODEL,
                    "available_models": available_models
                }
            else:
                providers_info["local_llm"] = {"status": "unavailable", "reason": "Ollama not running or no models available"}
        else:
            providers_info["local_llm"] = {"status": "unavailable", "reason": "Local LLM not enabled"}
            
        return providers_info 

    def generate_response(self, text: str, conversation_history: Optional[List[Dict[str, str]]] = None, 
                         provider: str = "fastest") -> str:
        """
        Generate a complete response from LLM models.
        
        Args:
            text: The text to process
            conversation_history: List of previous messages in the conversation
            provider: Which provider to use ("fastest", "chatgpt", "claude", "deepseek", "local_llm")
            
        Returns:
            Complete generated text response
        """
        logger.info(f"Generating response with provider: {provider}")
        logger.info(f"Input text: {text[:100]}...")
        
        # Collect all chunks from the stream
        response_chunks = []
        try:
            for chunk in self.stream_text(text, conversation_history, provider):
                response_chunks.append(chunk)
                print(f"LLM chunk: '{chunk}'")
                
            # Combine all chunks into the complete response
            complete_response = "".join(response_chunks)
            logger.info(f"Generated complete response: {complete_response[:100]}...")
            return complete_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Lo siento, no pude procesar tu solicitud en este momento. Por favor, intenta de nuevo." 