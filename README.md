# Hyundai Voice Assistant

A voice assistant for Hyundai that uses AI to answer questions about the brand.

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

## Configuration

All configuration settings are in the `.env` file. See `.env.example` for available options.

## Security

- Never commit your `.env` file or any files containing API keys
- The `.gitignore` file is configured to prevent committing sensitive data
- If you accidentally commit API keys, rotate them immediately

## License

[Your License]
