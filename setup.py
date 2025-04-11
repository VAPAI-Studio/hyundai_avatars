from setuptools import setup, find_packages

setup(
    name="hyundai_voice_assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "python-dotenv==1.0.1",
        "pyaudio==0.2.14",
        "numpy==1.26.4",
        "sounddevice==0.4.6",
        "soundfile==0.12.1",
        "openai==1.12.0",
        "anthropic==0.18.1",
        "elevenlabs==0.3.0",
        "requests==2.31.0",
        "websockets==12.0",
        "tqdm==4.66.2",
    ],
) 