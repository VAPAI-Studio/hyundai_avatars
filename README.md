# Hyundai Voice Assistant

A voice assistant for Hyundai that uses AI to answer questions about the brand. Features advanced speech detection using Silero VAD (Voice Activity Detection) for improved accuracy and interruption handling.

## Features

- **Silero VAD Integration**: Advanced voice activity detection using Silero VAD for precise speech detection
- **Interruption Support**: Can detect and respond to user interruptions while the avatar is speaking
- **Multi-AI Provider Support**: OpenAI, Anthropic, and DeepSeek integration
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

## Security

- Never commit your `.env` file or any files containing API keys
- The `.gitignore` file is configured to prevent committing sensitive data
- If you accidentally commit API keys, rotate them immediately

## License

[Your License]
