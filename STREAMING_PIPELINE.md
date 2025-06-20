# Streaming Pipeline: LLM → Text Chunker → ElevenLabs

This document describes the streaming text pipeline implementation that enables real-time text-to-speech with minimal latency.

## Overview

The streaming pipeline consists of three main components:

1. **Streaming LLM Processor** - Streams text as it's being generated from various LLM providers
2. **Text Chunker** - Splits text into meaningful chunks based on semantic boundaries or speaker pauses
3. **Streaming TTS Processor** - Streams text chunks to ElevenLabs while continuing to send more text in parallel

## Architecture

```
LLM (Streaming) → Text Chunker → ElevenLabs (Parallel Processing) → Audio2Face
```

### Key Features

- **Real-time streaming**: Text is processed as it's generated, not waiting for complete responses
- **Parallel processing**: Multiple text chunks can be processed simultaneously
- **Semantic chunking**: Text is split at natural break points for better speech flow
- **Fallback support**: Automatically falls back to traditional pipeline if streaming fails
- **Configurable**: Multiple chunking strategies and processing parameters

## Components

### 1. Streaming LLM Processor (`ai/streaming_llm_processor.py`)

Supports streaming from multiple LLM providers:
- **OpenAI ChatGPT** (GPT-4, GPT-3.5)
- **Anthropic Claude** (Claude-3, Claude-3.5)
- **DeepSeek** (DeepSeek Chat)
- **Local LLM** (Ollama models)

**Usage:**
```python
from ai.streaming_llm_processor import StreamingLLMProcessor

processor = StreamingLLMProcessor()
text_stream = processor.stream_text("Your prompt here", provider="fastest")

for chunk in text_stream:
    print(chunk, end="", flush=True)
```

### 2. Text Chunker (`audio/text_chunker.py`)

Splits text into meaningful chunks using different strategies:

- **Semantic**: Splits at semantic boundaries (conjunctions, transitions)
- **Sentence**: Splits at sentence endings
- **Phrase**: Splits at punctuation marks
- **Pause**: Splits at natural pause patterns

**Usage:**
```python
from audio.text_chunker import TextChunker

chunker = TextChunker(strategy="semantic")
chunks = chunker.chunk_text("Your text here")

for chunk in chunks:
    print(f"Chunk: {chunk.text} (type: {chunk.chunk_type})")
```

### 3. Streaming TTS Processor (`audio/streaming_tts_processor.py`)

Handles parallel processing of text chunks to ElevenLabs:

- **Parallel workers**: Multiple threads process chunks simultaneously
- **Ordered output**: Maintains correct order of audio chunks
- **Real-time streaming**: Streams audio to Audio2Face as it's generated
- **Error handling**: Graceful fallback and error recovery

**Usage:**
```python
from audio.streaming_tts_processor import StreamingTTSProcessor

processor = StreamingTTSProcessor(chunk_strategy="semantic", max_workers=3)
success = processor.stream_text_to_speech(text_stream)
```

## Configuration

Add these environment variables to your `.env` file:

```bash
# Enable streaming pipeline
USE_STREAMING_PIPELINE=true

# Text chunking strategy
STREAMING_CHUNK_STRATEGY=semantic  # semantic, sentence, phrase, pause

# Processing parameters
STREAMING_MAX_WORKERS=3
STREAMING_MIN_CHUNK_SIZE=20
STREAMING_MAX_CHUNK_SIZE=200
```

## Integration with Main Application

The streaming pipeline is integrated into the main voice assistant with automatic fallback:

```python
# In main.py
if USE_STREAMING_PIPELINE:
    success = self._process_with_streaming_pipeline(text)
else:
    success = self._process_with_traditional_pipeline(text)
```

## Testing

Run the comprehensive test suite:

```bash
python tests/test_streaming_pipeline.py
```

This will test:
- Text chunker functionality
- Streaming LLM processors
- Streaming TTS processing
- Complete pipeline integration
- Performance benchmarks

## Performance Benefits

### Latency Reduction
- **Traditional**: Wait for complete LLM response → Process entire text → Generate audio
- **Streaming**: Start processing as text arrives → Parallel chunk processing → Real-time audio

### Typical Improvements
- **First audio chunk**: 2-3 seconds faster
- **Overall response time**: 30-50% reduction
- **Perceived responsiveness**: Immediate feedback

## Error Handling

The pipeline includes robust error handling:

1. **LLM failures**: Falls back to alternative providers
2. **TTS failures**: Falls back to traditional pipeline
3. **Network issues**: Automatic retry with exponential backoff
4. **Chunking errors**: Graceful degradation to simpler strategies

## Monitoring and Debugging

### Callbacks
The streaming TTS processor provides callbacks for monitoring:

```python
def on_chunk_processed(chunk, status):
    print(f"Chunk processed: {chunk.text[:30]}... - {status}")

def on_audio_ready(audio_data):
    print(f"Audio ready: {len(audio_data)} samples")

def on_streaming_complete():
    print("Streaming completed")

processor.set_callbacks(
    on_chunk_processed=on_chunk_processed,
    on_audio_ready=on_audio_ready,
    on_streaming_complete=on_streaming_complete
)
```

### Logging
Enable debug logging to monitor the pipeline:

```python
import logging
logging.getLogger('audio.streaming_tts_processor').setLevel(logging.DEBUG)
logging.getLogger('ai.streaming_llm_processor').setLevel(logging.DEBUG)
```

## Troubleshooting

### Common Issues

1. **No streaming providers available**
   - Check API keys in `.env` file
   - Verify network connectivity
   - Ensure Ollama is running (for local LLM)

2. **Chunks too small/large**
   - Adjust `STREAMING_MIN_CHUNK_SIZE` and `STREAMING_MAX_CHUNK_SIZE`
   - Try different chunking strategies

3. **Audio out of order**
   - Reduce `STREAMING_MAX_WORKERS`
   - Check network latency to ElevenLabs

4. **High memory usage**
   - Reduce `STREAMING_MAX_WORKERS`
   - Increase chunk sizes
   - Monitor audio buffer sizes

### Performance Tuning

1. **For low latency**: Use `semantic` chunking with 2-3 workers
2. **For high throughput**: Use `sentence` chunking with 4-5 workers
3. **For stability**: Use `phrase` chunking with 2 workers

## Future Enhancements

- **Adaptive chunking**: Dynamic chunk size based on content
- **Voice cloning**: Support for multiple voices
- **Emotion detection**: Adjust TTS parameters based on content
- **Streaming STT**: Real-time speech-to-text for even lower latency
- **WebSocket support**: Real-time communication with web clients 