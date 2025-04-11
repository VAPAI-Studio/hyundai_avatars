"""
Module for connecting with Unreal Engine 5.5 MetaHuman.
This module watches for new audio files and triggers UE5.5 to play them on a MetaHuman.
"""

import os
import time
import logging
import socket
import json
import subprocess
import threading
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils.config import RESPONSE_AUDIO_PATH, UE5_CONNECTION_METHOD, UE5_HOST, UE5_PORT, UE5_PROJECT_PATH

logger = logging.getLogger(__name__)

class AudioFileHandler(FileSystemEventHandler):
    """Watches for changes to the audio response file and notifies UE5."""
    
    def __init__(self, ue5_bridge):
        self.ue5_bridge = ue5_bridge
        self.last_modified = 0
        
    def on_modified(self, event):
        """Called when a file is modified."""
        if not event.is_directory and event.src_path.endswith(RESPONSE_AUDIO_PATH):
            # Check if this is a new modification (avoid duplicate events)
            current_time = time.time()
            if current_time - self.last_modified > 1.0:  # 1 second threshold
                self.last_modified = current_time
                logger.info(f"Audio file modified: {event.src_path}")
                
                # Get the file size and wait until it's stable (completely written)
                self._wait_for_file_ready(event.src_path)
                
                # Notify UE5
                self.ue5_bridge.trigger_metahuman_animation(event.src_path)
                
    def _wait_for_file_ready(self, file_path, timeout=5, check_interval=0.1):
        """Wait until the file size is stable (file is completely written)."""
        end_time = time.time() + timeout
        last_size = -1
        
        while time.time() < end_time:
            try:
                current_size = os.path.getsize(file_path)
                if current_size == last_size and current_size > 0:
                    return True  # File is ready
                last_size = current_size
                time.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error checking file size: {e}")
                time.sleep(check_interval)
                
        logger.warning(f"Timed out waiting for file to be ready: {file_path}")
        return False


class UE5Bridge:
    """Bridge to connect with Unreal Engine 5.5 MetaHuman."""
    
    def __init__(self):
        self.observer = None
        self.event_handler = None
        self.running = False
        
    def start_watching(self):
        """Start watching for audio file changes."""
        if self.observer:
            logger.warning("File watcher already running")
            return False
            
        try:
            self.event_handler = AudioFileHandler(self)
            self.observer = Observer()
            
            # Watch the directory containing the audio file
            watch_dir = os.path.dirname(os.path.abspath(RESPONSE_AUDIO_PATH))
            if not watch_dir:  # If RESPONSE_AUDIO_PATH is just a filename
                watch_dir = os.getcwd()
                
            self.observer.schedule(self.event_handler, watch_dir, recursive=False)
            self.observer.start()
            self.running = True
            
            logger.info(f"Started watching for changes to {RESPONSE_AUDIO_PATH}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting file watcher: {e}")
            return False
            
    def stop_watching(self):
        """Stop watching for audio file changes."""
        if self.observer and self.running:
            try:
                self.observer.stop()
                self.observer.join()
                self.observer = None
                self.running = False
                logger.info("Stopped file watcher")
                return True
            except Exception as e:
                logger.error(f"Error stopping file watcher: {e}")
                
        return False
        
    def trigger_metahuman_animation(self, audio_path):
        """
        Trigger MetaHuman animation in UE5.5 based on the connection method.
        """
        try:
            if UE5_CONNECTION_METHOD == "remote_exec":
                return self._trigger_via_remote_exec(audio_path)
            elif UE5_CONNECTION_METHOD == "remote_control":
                return self._trigger_via_remote_control(audio_path)
            elif UE5_CONNECTION_METHOD == "tcp_socket":
                return self._trigger_via_tcp_socket(audio_path)
            elif UE5_CONNECTION_METHOD == "command_line":
                return self._trigger_via_command_line(audio_path)
            else:
                logger.error(f"Unknown UE5 connection method: {UE5_CONNECTION_METHOD}")
                return False
        except Exception as e:
            logger.error(f"Error triggering MetaHuman animation: {e}")
            return False
            
    def _trigger_via_remote_exec(self, audio_path):
        """
        Trigger MetaHuman animation using UE5's Remote Exec API.
        """
        try:
            # Format the path properly for UE5
            ue_audio_path = audio_path.replace('\\', '/')
            
            # Remote Exec API endpoint
            url = f"http://{UE5_HOST}:{UE5_PORT}/remote/object/call"
            
            # Payload for the BP_MetaHuman actor or the Speech Controller
            payload = {
                "objectPath": "/Game/MetaHumans/YourMetaHuman/BP_MetaHuman.BP_MetaHuman:PersistentLevel.BP_MetaHuman_1",
                "functionName": "PlayAudioAndAnimate",
                "parameters": {
                    "AudioFilePath": ue_audio_path
                }
            }
            
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                logger.info("Successfully triggered MetaHuman animation via Remote Exec")
                return True
            else:
                logger.error(f"Failed to trigger animation: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error using Remote Exec: {e}")
            return False
            
    def _trigger_via_remote_control(self, audio_path):
        """
        Trigger MetaHuman animation using UE5's Remote Control API.
        """
        try:
            # Format the path properly for UE5
            ue_audio_path = audio_path.replace('\\', '/')
            
            # Remote Control API endpoint
            url = f"http://{UE5_HOST}:{UE5_PORT}/remote/control/preset/function"
            
            # Payload for the Remote Control preset
            payload = {
                "preset": "MetaHumanControls",
                "function": "PlayAudioOnMetaHuman",
                "parameters": {
                    "AudioFile": ue_audio_path
                }
            }
            
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                logger.info("Successfully triggered MetaHuman animation via Remote Control")
                return True
            else:
                logger.error(f"Failed to trigger animation: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error using Remote Control: {e}")
            return False
            
    def _trigger_via_tcp_socket(self, audio_path):
        """
        Trigger MetaHuman animation by sending a message over a TCP socket.
        Requires a custom TCP listener set up in UE5.
        """
        try:
            # Format the path properly for UE5
            ue_audio_path = audio_path.replace('\\', '/')
            
            # Create socket and connect
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((UE5_HOST, UE5_PORT))
            
            # Prepare message
            message = json.dumps({
                "command": "play_audio",
                "audio_path": ue_audio_path
            })
            
            # Send message
            sock.sendall(message.encode('utf-8'))
            
            # Get response
            response = sock.recv(1024).decode('utf-8')
            sock.close()
            
            if "success" in response.lower():
                logger.info("Successfully triggered MetaHuman animation via TCP socket")
                return True
            else:
                logger.error(f"Failed to trigger animation: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error using TCP socket: {e}")
            return False
            
    def _trigger_via_command_line(self, audio_path):
        """
        Trigger MetaHuman animation by launching UE5 with command-line arguments.
        This approach is less ideal for real-time interactions but can work for testing.
        """
        try:
            # Format the path properly for UE5
            ue_audio_path = audio_path.replace('\\', '/')
            
            # Construct the command
            cmd = [
                UE5_PROJECT_PATH,
                "-game",
                "-ExecCmds=PlayMetaHumanAudio",
                f"-AudioFile={ue_audio_path}"
            ]
            
            # Execute command
            subprocess.Popen(cmd)
            logger.info("Launched UE5 with audio command")
            return True
            
        except Exception as e:
            logger.error(f"Error launching UE5: {e}")
            return False