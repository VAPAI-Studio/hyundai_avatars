import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from omni.audio2face.core import Audio2FaceCore

AUDIO_FILE = r"C:\Users\Joaco\Downloads\hyundai_claude_26mar_simplificado\response_audio.mp3"

# Initialize Audio2Face
a2f = Audio2FaceCore()

def update_and_play_audio(audio_path):
    print(f"Loading audio: {audio_path}")
    a2f.set_audio_source(audio_path)
    a2f.process()
    a2f.play()  # Immediately play the animation within Audio2Face

class AudioUpdateHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path == AUDIO_FILE:
            update_and_play_audio(event.src_path)

if __name__ == "__main__":
    event_handler = AudioUpdateHandler()
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(AUDIO_FILE), recursive=False)
    observer.start()

    try:
        print("Monitoring audio file changes for real-time playback...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
