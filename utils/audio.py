import subprocess
import os

def validate_audio_duration(file_path: str) -> float:
    """
    Queries the duration of an audio file using ffprobe.
    Raises ValueError if the duration is not between 30 and 45 seconds (inclusive).
    Returns the duration in seconds.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # ffprobe command to extract format duration
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration_str = result.stdout.strip()
        if not duration_str:
            raise ValueError("Could not parse audio duration metadata.")
        
        duration = float(duration_str)
        if duration < 30.0 or duration > 45.0:
            raise ValueError(
                f"Audio duration must be between 30 and 45 seconds. "
                f"Your file is {duration:.2f} seconds."
            )
        return duration
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else str(e)
        raise ValueError(f"Failed to read audio duration: {error_msg}")
    except ValueError as e:
        # Re-raise duration constraints or float parse errors
        raise e
    except Exception as e:
        raise ValueError(f"Unexpected error validating audio: {str(e)}")

def convert_to_wav(input_path: str, output_path: str) -> str:
    """
    Converts any input audio file (WAV, MP3, M4A) to mono 16kHz PCM WAV format.
    Whisper models work optimally with single-channel 16kHz audio.
    Returns the output path of the converted file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found for conversion: {input_path}")

    # ffmpeg command to convert audio to 16kHz mono WAV format
    cmd = [
        "ffmpeg",
        "-y",               # Overwrite output files without asking
        "-i", input_path,   # Input file
        "-ar", "16000",     # Audio sampling rate 16000Hz
        "-ac", "1",         # Audio channels: 1 (mono)
        output_path         # Output file path
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else str(e)
        raise ValueError(f"Audio conversion failed: {error_msg}")
    except Exception as e:
        raise ValueError(f"Unexpected error during audio conversion: {str(e)}")

def cleanup_file(file_path: str) -> None:
    """
    Safely deletes a file from the filesystem.
    Silently ignores errors to avoid breaking the main application flow.
    """
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass
