import subprocess
import json
import os

def run_a2f_command(url, data):
    """Run a curl command to control Audio2Face"""
    # Convert data dictionary to JSON
    json_data = json.dumps(data)
    
    # Construct curl command
    curl_cmd = [
        'curl', '-X', 'POST', url,
        '-d', json_data
    ]
    
    # Execute the command
    try:
        result = subprocess.run(curl_cmd, capture_output=True, text=True)
        print(f"Command response: {result.stdout}")
        
        # Check for errors
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None
        
        # Parse and return JSON response
        return json.loads(result.stdout)
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None

def set_audio_file(audio_filepath):
    """Set the audio file in Audio2Face"""
    # Extract filename from path
    file_name = os.path.basename(audio_filepath)
    
    # Set track in Audio2Face player
    url = "http://localhost:8011/A2F/Player/SetTrack"
    data = {
        "a2f_player": "/World/audio2face/Player",
        "file_name": file_name,
        "time_range": [0, -1]  # Full range
    }
    
    response = run_a2f_command(url, data)
    
    if response and response.get("status") == "OK":
        print(f"Successfully set track to {file_name}")
        return True
    else:
        print("Failed to set audio track")
        return False

def play_animation():
    """Play the animation in Audio2Face"""
    url = "http://localhost:8011/A2F/Player/Play"
    data = {
        "a2f_player": "/World/audio2face/Player"
    }
    
    response = run_a2f_command(url, data)
    
    if response and response.get("status") == "OK":
        print("Started playback successfully")
        return True
    else:
        print("Failed to start playback")
        return False

def stop_animation():
    """Stop the animation in Audio2Face"""
    url = "http://localhost:8011/A2F/Player/Stop"
    data = {
        "a2f_player": "/World/audio2face/Player"
    }
    
    response = run_a2f_command(url, data)
    
    if response and response.get("status") == "OK":
        print("Stopped playback successfully")
        return True
    else:
        print("Failed to stop playback")
        return False

def export_blendshapes(export_directory, filename="export", format="json", fps=5):
    """Export blendshapes from Audio2Face"""
    url = "http://localhost:8011/A2F/Exporter/export_blendshapes"
    data = {
        "solver_node": "/World/audio2face/BlendshapeSolve",
        "export_directory": export_directory,
        "file_name": filename,
        "format": format,
        "batch": False,
        "fps": fps
    }
    
    response = run_a2f_command(url, data)
    
    if response and response.get("status") == "OK":
        print(f"Successfully exported blendshapes to {export_directory}/{filename}.{format}")
        return True
    else:
        print("Failed to export blendshapes")
        return False

# Example usage
if __name__ == "__main__":
    # Path to your audio file
    audio_file = "./response_audio.wav"
    
    # Set the audio file
    if set_audio_file(audio_file):
        # Play the animation
        play_animation()
        
        # If you want to export blendshapes after playing
        # Uncomment the lines below
        # import time
        # time.sleep(5)  # Wait for animation to complete
        # export_blendshapes("C:/Users/Joaco/Downloads/hyundai_claude_26mar_simplificado", "animation_export")
        
        # To stop playback
        # stop_animation()