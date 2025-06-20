# Hyundai Voice Assistant

A voice assistant for Hyundai that uses AI to answer questions about the brand. Features advanced speech detection using Silero VAD (Voice Activity Detection) for improved accuracy and interruption handling.

## Features

- **Silero VAD Integration**: Advanced voice activity detection using Silero VAD for precise speech detection
- **Interruption Support**: Can detect and respond to user interruptions while the avatar is speaking
- **Multi-AI Provider Support**: OpenAI, Anthropic, DeepSeek, and Local LLM integration
- **Local LLM Support**: Run models like Mistral 7B locally using Ollama
- **Real-time Audio Processing**: Streams audio directly to Audio2Face for seamless avatar interaction
- **Configurable Settings**: Easy configuration through environment variables

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yvesfogel/hyundai_avatars.git
cd hyundai_avatars
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```bash
cp .env.example .env
```
Then edit the `.env` file and add your API keys:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- DEEPSEEK_API_KEY
- ELEVENLABS_API_KEY

5. Run the application:
```bash
python main.py
```

## Local LLM Setup (Optional)

The voice assistant supports running local LLM models using Ollama. This allows you to use models like Mistral 7B without requiring API keys or internet connectivity.

### Installing Ollama

1. Visit [https://ollama.ai](https://ollama.ai) and download Ollama for your operating system
2. Install and start Ollama:
   - **Windows**: Ollama should start automatically after installation
   - **macOS/Linux**: Run `ollama serve` in terminal

### Setting up Local LLM

1. Use the setup utility to manage your local models:
```bash
python utils/local_llm_setup.py
```

2. Pull a model (e.g., Mistral 7B):
```bash
ollama pull mistral:7b
```

3. Enable local LLM in your `.env` file:
```bash
USE_LOCAL_LLM=true
LOCAL_LLM_MODEL=mistral:7b
```

### Local LLM Configuration

Add these variables to your `.env` file:

```bash
# Local LLM Settings
USE_LOCAL_LLM=true
OLLAMA_URL=http://localhost:11434
LOCAL_LLM_MODEL=mistral:7b
LOCAL_LLM_TEMPERATURE=0.7
LOCAL_LLM_MAX_TOKENS=500
```

### Available Models

Popular models you can use with Ollama:
- `mistral:7b` - Fast and efficient 7B parameter model
- `llama2:7b` - Meta's Llama 2 model
- `codellama:7b` - Code-focused model
- `phi:2.7b` - Microsoft's Phi model (very fast)
- `gemma:2b` - Google's Gemma model (lightweight)

### Testing Local LLM

Test your local LLM setup:
```bash
python utils/local_llm_setup.py
```

Choose option 3 to test a specific model.

## Silero VAD Configuration

The voice assistant now uses Silero VAD for improved speech detection. Key configuration options:

- `SILENCE_TIMEOUT`: Seconds of silence before stopping recording (default: 1.5s)
- `MIN_SPEECH_DURATION`: Minimum speech duration to consider valid (default: 0.3s)
- `SAMPLE_RATE`: Audio sample rate (default: 16000Hz for optimal VAD performance)

### Model Caching

The Silero VAD model is automatically cached after the first download. You can manage the cache using:

```bash
# Check cache status
python utils/check_cache.py

# Clear cache (forces re-download)
python utils/check_cache.py clear

# Show cache information
python utils/check_cache.py info
```

The model is typically cached in `~/.cache/torch/hub/snakers4_silero-vad_master/` on Linux/Mac or `%USERPROFILE%\.cache\torch\hub\snakers4_silero-vad_master\` on Windows.

## Testing

Test the Silero VAD implementation:
```bash
python tests/test_silero_vad.py
```

Run a simple VAD demo:
```bash
python tests/silero_vad_demo.py
```

## Configuration

All configuration settings are in the `.env` file. See `.env.example` for available options.

### AI Provider Selection

You can specify which AI provider to use by setting `AI_PROVIDER` in your `.env` file:

- `fastest` (default) - Uses the fastest responding model
- `chatgpt` - Uses OpenAI's ChatGPT
- `claude` - Uses Anthropic's Claude
- `deepseek` - Uses DeepSeek
- `local_llm` - Uses local LLM via Ollama

## Security

- Never commit your `.env` file or any files containing API keys
- The `.gitignore` file is configured to prevent committing sensitive data
- If you accidentally commit API keys, rotate them immediately

## License

[Your License]
