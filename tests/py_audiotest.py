import py_audio2face as pya2f
from py_audio2face.settings import ASSETS_DIR

import os
import io
import soundfile as sf
from media_toolkit import AudioFile

test_audio_0 = "./test_audios/response_audio.wav"
test_audio_1 = "./test_audios/temp_audio.wav"

a2f = pya2f.Audio2Face()

def test_file_methods():
    a2f.set_emotion(anger=0.9, disgust=0.5, fear=0.1, sadness=0.3, update_settings=True)
    preset_emotion_animation = a2f.audio2face_single(
        audio_file_path=test_audio_0,
        output_path="emotion_less.usd",
        fps=60,
        emotion_auto_detect=False
    )

    auto_detected_emotion = a2f.audio2face_single(
        audio_file_path=test_audio_1,
        output_path="emotion_less.usd",
        fps=60,
        emotion_auto_detect=True
    )
    

def test_streaming():
    my_audio = AudioFile().from_file("./test_audios/response_audio.wav")

    # Convertimos el buffer a BytesIO directamente
    audio_io = my_audio._content_buffer.to_bytes_io()

    # Usamos soundfile para leer el audio
    audio, sample_rate = sf.read(audio_io)
    print(f"Audio shape: {audio.shape}, Sample rate: {sample_rate}")
    a2f.stream_audio(audio_stream=audio, samplerate=sample_rate)

if __name__ == "__main__":
    test_streaming()

