# backend/src/utils.py

import base64
import re
import tempfile
import os
from pydub import AudioSegment

# Cache for temp video path to avoid writing video bytes twice
_temp_video_cache = {"path": None, "hash": None}

def decode_base64_to_bytes(data: str) -> bytes:
    """
    Decode base64 string (DataURL or plain base64) to bytes.
    """
    if data.startswith("data:"):
        # Remove DataURL prefix
        data = re.sub(r"^data:.*;base64,", "", data)
    return base64.b64decode(data)

def _get_temp_video_path(video_bytes: bytes) -> str:
    """
    Get cached temp video path or create new one.
    Uses hash to detect if video bytes changed.
    """
    global _temp_video_cache
    
    video_hash = hash(video_bytes[:1000]) if len(video_bytes) > 1000 else hash(video_bytes)
    
    if _temp_video_cache["path"] and _temp_video_cache["hash"] == video_hash:
        if os.path.exists(_temp_video_cache["path"]):
            return _temp_video_cache["path"]
    
    # Clean up old temp file
    if _temp_video_cache["path"] and os.path.exists(_temp_video_cache["path"]):
        try:
            os.unlink(_temp_video_cache["path"])
        except:
            pass
    
    # Create new temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video.write(video_bytes)
        _temp_video_cache["path"] = temp_video.name
        _temp_video_cache["hash"] = video_hash
    
    return _temp_video_cache["path"]

def cleanup_temp_video():
    """Clean up cached temp video file."""
    global _temp_video_cache
    if _temp_video_cache["path"] and os.path.exists(_temp_video_cache["path"]):
        try:
            os.unlink(_temp_video_cache["path"])
        except:
            pass
    _temp_video_cache = {"path": None, "hash": None}

def extract_audio_from_video(video_bytes: bytes) -> bytes:
    """
    Extract audio from video file bytes using pydub.
    Returns audio bytes in WAV format (16kHz mono, optimized for Whisper).
    """
    temp_audio_path = None

    try:
        # Use cached temp video path
        temp_video_path = _get_temp_video_path(video_bytes)

        # Load video with pydub and extract audio
        audio = AudioSegment.from_file(temp_video_path)
        
        # Convert to 16kHz mono (Whisper's expected format)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

        # Export as WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            audio.export(temp_audio_path, format="wav")

        # Read the WAV file back as bytes
        with open(temp_audio_path, "rb") as f:
            audio_bytes = f.read()

        return audio_bytes

    except Exception as e:
        print(f"[ERROR] Failed to extract audio from video: {e}")
        return b""

    finally:
        # Cleanup audio temp file only (video temp is cached)
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except:
                pass

def extract_video_frame(video_bytes: bytes, timestamp_seconds: float = None) -> bytes:
    """
    Extract a frame from video. Tries OpenCV first, then falls back to ffmpeg.
    Returns JPEG bytes of the frame.
    """
    try:
        return _extract_frame_opencv(video_bytes, timestamp_seconds)
    except Exception as opencv_error:
        print(f"[WARNING] OpenCV failed: {opencv_error}")
        print(f"[INFO] Trying ffmpeg fallback...")
        try:
            return _extract_frame_ffmpeg(video_bytes, timestamp_seconds)
        except Exception as ffmpeg_error:
            print(f"[ERROR] Both OpenCV and ffmpeg failed")
            print(f"[ERROR] OpenCV error: {opencv_error}")
            print(f"[ERROR] ffmpeg error: {ffmpeg_error}")
            return b""

def _extract_frame_opencv(video_bytes: bytes, timestamp_seconds: float = None) -> bytes:
    """Extract frame using OpenCV."""
    import cv2

    # Use cached temp video path
    temp_video_path = _get_temp_video_path(video_bytes)
    print(f"[DEBUG] Video temp path: {temp_video_path}")
    print(f"[DEBUG] Video file size: {os.path.getsize(temp_video_path)} bytes")

    # Open video with OpenCV
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file with OpenCV")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    duration = frame_count / fps if fps > 0 else 5
    
    print(f"[DEBUG] OpenCV Video properties: FPS={fps}, Frames={frame_count}, Size={width}x{height}, Duration={duration:.2f}s")

    # Try multiple timestamps to find a good frame
    timestamps = [duration * 0.5, duration * 0.3, duration * 0.7, 1.0, 0.5]
    if timestamp_seconds is not None:
        timestamps.insert(0, timestamp_seconds)

    frame = None
    for i, ts in enumerate(timestamps):
        if ts > duration:
            print(f"[DEBUG] Skipping timestamp {ts:.2f}s (exceeds duration)")
            continue
        
        print(f"[DEBUG] Trying timestamp {ts:.2f}s (attempt {i+1})")
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        print(f"[DEBUG] Frame read result: ret={ret}, frame_shape={frame.shape if frame is not None else None}")
        
        if ret and frame is not None and frame.size > 0:
            print(f"[INFO] Extracted frame at {ts:.2f}s using OpenCV")
            break
        else:
            print(f"[DEBUG] Failed to read frame at {ts:.2f}s")

    cap.release()

    if frame is None or frame.size == 0:
        raise Exception("Could not read frame from video with OpenCV")

    # Encode as JPEG
    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise Exception("Could not encode frame as JPEG")

    print(f"[DEBUG] Successfully encoded frame with OpenCV: {len(buffer.tobytes())} bytes")
    return buffer.tobytes()

def _extract_frame_ffmpeg(video_bytes: bytes, timestamp_seconds: float = None) -> bytes:
    """Extract frame using ffmpeg as fallback."""
    import subprocess
    
    temp_video_path = _get_temp_video_path(video_bytes)
    
    # First, get video duration using ffprobe
    try:
        duration_cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', temp_video_path
        ]
        duration_output = subprocess.check_output(duration_cmd, stderr=subprocess.DEVNULL)
        duration = float(duration_output.decode().strip())
        print(f"[DEBUG] ffmpeg Video duration: {duration:.2f}s")
    except:
        duration = 5.0  # fallback duration
        print(f"[DEBUG] Could not get duration, using fallback: {duration}s")
    
    # Choose timestamp (middle of video by default)
    if timestamp_seconds is None:
        timestamp_seconds = duration * 0.5
    
    # Ensure timestamp doesn't exceed duration
    timestamp_seconds = min(timestamp_seconds, duration - 0.5)
    timestamp_seconds = max(timestamp_seconds, 0.5)
    
    print(f"[DEBUG] Using timestamp: {timestamp_seconds:.2f}s")
    
    # Create temp file for the extracted frame
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
        temp_image_path = temp_image.name
    
    try:
        # Extract frame using ffmpeg
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-v', 'quiet', '-i', temp_video_path,
            '-ss', str(timestamp_seconds), '-vframes', '1',
            '-q:v', '5', temp_image_path
        ]
        
        print(f"[DEBUG] Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
        subprocess.run(ffmpeg_cmd, check=True, stderr=subprocess.DEVNULL)
        
        # Read the extracted frame
        with open(temp_image_path, 'rb') as f:
            frame_bytes = f.read()
        
        print(f"[INFO] Successfully extracted frame using ffmpeg: {len(frame_bytes)} bytes")
        return frame_bytes
        
    finally:
        # Clean up temp image file
        if os.path.exists(temp_image_path):
            try:
                os.unlink(temp_image_path)
            except:
                pass
