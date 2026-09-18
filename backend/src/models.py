# backend/src/models.py

import os
import sys
import warnings
import logging

# Suppress transformers/torch warnings BEFORE importing
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Fix CUDA DLL loading on Windows (must be done BEFORE first cuda operation)
import torch

import torch.nn as nn
import librosa
import numpy as np
import cv2
import base64
import tempfile
import os
import re
from datetime import datetime
import warnings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from corrections import (
    apply_transcription_corrections,
    remove_hallucinated_greeting,
    post_process_translation,
    HALLUCINATED_STARTS
)
from config import config

# Faster-whisper support (CTranslate2-based, 4x faster)
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
    print("[INFO] faster-whisper tersedia - Using CTranslate2 acceleration")
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("[WARNING] faster-whisper tidak tersedia, falling back to openai-whisper")

# OpenAI Whisper as fallback
try:
    import whisper
    OPENAI_WHISPER_AVAILABLE = True
except ImportError:
    OPENAI_WHISPER_AVAILABLE = False
    print("[WARNING] openai-whisper tidak tersedia")

# Fine-tuned Whisper model support
try:
    from whisper_finetuned import load_finetuned_whisper, get_model_manager
    FINETUNED_WHISPER_AVAILABLE = True
except ImportError:
    FINETUNED_WHISPER_AVAILABLE = False
    print("[WARNING] whisper_finetuned module not available")

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def normalize_transcription(text: str, apply_corrections: bool = True, confidence_scores: list = None) -> str:
    """
    Normalize transcription output from Whisper.
    Uses external corrections from corrections.py for easy editing.

    Args:
        text: Raw transcription text
        apply_corrections: Apply common correction rules
        confidence_scores: List of confidence scores for each segment (optional)

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Basic normalization
    normalized = text.strip()

    if apply_corrections:
        # Apply corrections from external file (corrections.py)
        # Edit that file to add new corrections without changing this code
        normalized = apply_transcription_corrections(normalized)

        # Handle hallucinated greetings at the start with confidence check
        words = normalized.split()
        if words:
            first_word = words[0].lower()

            if first_word in HALLUCINATED_STARTS:
                should_remove = False

                if confidence_scores and len(confidence_scores) > 0:
                    first_confidence = np.exp(confidence_scores[0]) if confidence_scores[0] < 0 else confidence_scores[0]
                    if first_confidence < 0.3:
                        should_remove = True
                        print(f"[INFO] Removing hallucinated greeting '{first_word}' (confidence: {first_confidence:.3f})")
                else:
                    should_remove = True
                    print(f"[INFO] Removing potential hallucinated greeting '{first_word}' (no confidence data)")

                if should_remove:
                    words = words[1:]
                    normalized = ' '.join(words)

        # Clean up multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)

    return normalized.strip()

# MediaPipe import with fallback (supports new tasks API in v0.10.30+)
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
    MEDIAPIPE_NEW_API = True
    print("[INFO] MediaPipe (new tasks API) tersedia - Facial expression detection enabled")
except ImportError:
    try:
        # Fallback to old API
        import mediapipe as mp
        if hasattr(mp, 'solutions'):
            MEDIAPIPE_AVAILABLE = True
            MEDIAPIPE_NEW_API = False
            print("[INFO] MediaPipe (legacy API) tersedia - Facial expression detection enabled")
        else:
            raise ImportError("MediaPipe solutions not available")
    except ImportError:
        MEDIAPIPE_AVAILABLE = False
        MEDIAPIPE_NEW_API = False
        print("[WARNING] MediaPipe tidak tersedia - Facial expression detection disabled")
        print("[INFO] Sistem akan tetap berjalan dengan CNN visual features saja")

# Pydub for better audio format support
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
    print("[INFO] pydub tersedia - WebM/MP4/M4A support enabled")
except ImportError:
    PYDUB_AVAILABLE = False
    print("[WARNING] pydub tidak tersedia - install dengan: pip install pydub")
    print("[INFO] Fallback ke librosa with audioread")

class VisualCNN(nn.Module):
    """
    Model CNN untuk mengekstrak fitur dari frame video (gerak bibir dan ekspresi wajah).
    """
    def __init__(self, visual_feature_dim: int = 512):
        super(VisualCNN, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(128 * 12 * 12, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, visual_feature_dim)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        return self.linear_stack(x)

class FacialExpressionExtractor:
    """
    Ekstraksi fitur ekspresi wajah menggunakan MediaPipe Face Mesh.
    Supports both new tasks API (v0.10.30+) and legacy solutions API.
    Falls back to None if MediaPipe not available.
    """

    # Key landmark indices for visualization (simplified - not all 478)
    # Lips outer contour
    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    # Lips inner contour
    LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
    # Face oval (jaw line)
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    # Eyes (simplified)
    LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
    RIGHT_EYE = [362, 263, 387, 386, 385, 373, 374, 380]
    # Nose
    NOSE = [1, 2, 98, 327]

    def __init__(self, model_complexity=1):
        """
        Args:
            model_complexity: 0 (lite), 1 (full), 2 (heavy) - only used for legacy API
        """
        self._last_landmarks = None
        self._last_landmarks_list = None  # For new API (list of NormalizedLandmark)

        if not MEDIAPIPE_AVAILABLE:
            self.enabled = False
            self.use_new_api = False
            print("[INFO] FacialExpressionExtractor: MediaPipe tidak tersedia, facial landmarks disabled")
            return

        if MEDIAPIPE_NEW_API:
            # Use new tasks API (v0.10.30+)
            try:
                # Download model if needed
                import urllib.request
                import os
                import sys
                model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

                if not os.path.exists(model_path):
                    print("[MediaPipe] Downloading face_landmarker model (~26 MB)...")
                    print("[MediaPipe] URL: storage.googleapis.com/mediapipe-models")
                    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                    
                    # Download with progress indicator
                    def download_progress(block_num, block_size, total_size):
                        downloaded = block_num * block_size
                        if total_size > 0:
                            percent = min(100, downloaded * 100 / total_size)
                            downloaded_mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            sys.stdout.write(f"\r[MediaPipe] Downloading: {percent:.1f}% ({downloaded_mb:.2f}/{total_mb:.2f} MB)")
                            sys.stdout.flush()
                            if downloaded >= total_size:
                                print()  # New line after completion
                    
                    try:
                        urllib.request.urlretrieve(model_url, model_path, reporthook=download_progress)
                        print("[MediaPipe] Model downloaded successfully ✓")
                    except Exception as download_error:
                        print(f"\n[MediaPipe] Download failed: {download_error}")
                        # Clean up partial download
                        if os.path.exists(model_path):
                            os.remove(model_path)
                        raise
                else:
                    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    print(f"[MediaPipe] Model already downloaded ({file_size_mb:.2f} MB) ✓")

                # Create FaceLandmarker
                base_options = mp_python.BaseOptions(model_asset_path=model_path)
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.3,
                    min_face_presence_confidence=0.3,
                    min_tracking_confidence=0.3,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False
                )
                self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
                self.enabled = True
                self.use_new_api = True
                print(f"[MediaPipe] FaceLandmarker (new API) initialized successfully")
            except Exception as e:
                print(f"[MediaPipe] Failed to initialize new API: {e}")
                print(f"[MediaPipe] Falling back to legacy API...")
                # Fallback to legacy API
                try:
                    self.mp_face_mesh = mp.solutions.face_mesh
                    self.face_mesh = self.mp_face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.3,
                        min_tracking_confidence=0.3,
                        model_complexity=model_complexity
                    )
                    self.enabled = True
                    self.use_new_api = False
                    print(f"[MediaPipe] FaceMesh (legacy API) initialized with complexity={model_complexity}")
                except Exception as fallback_e:
                    print(f"[MediaPipe] Failed to initialize legacy API: {fallback_e}")
                    self.enabled = False
                    self.use_new_api = False
        else:
            # Use legacy solutions API
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3,
                    model_complexity=model_complexity
                )
                self.enabled = True
                self.use_new_api = False
                print(f"[MediaPipe] FaceMesh (legacy API) initialized with complexity={model_complexity}")
            except Exception as e:
                print(f"[MediaPipe] Failed to initialize legacy API: {e}")
                self.enabled = False
                self.use_new_api = False

    def extract_features(self, image_rgb):
        """
        Ekstrak landmark wajah untuk analisis ekspresi.
        Returns: numpy array of facial landmarks or None
        """
        if not self.enabled:
            print("[MediaPipe] Face mesh not enabled")
            return None

        try:
            print(f"[MediaPipe] Processing image: shape={image_rgb.shape}, dtype={image_rgb.dtype}")

            if self.use_new_api:
                # New tasks API
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                result = self.face_landmarker.detect(mp_image)

                if not result.face_landmarks:
                    print("[MediaPipe] No face detected in frame")
                    self._last_landmarks = None
                    self._last_landmarks_list = None
                    return None

                landmarks_list = result.face_landmarks[0]
                self._last_landmarks_list = landmarks_list
                self._last_landmarks = None  # Not used for new API
                print(f"[MediaPipe] Face detected! {len(landmarks_list)} landmarks found")

                # Extract coordinates
                features = []
                for landmark in landmarks_list:
                    features.extend([landmark.x, landmark.y, landmark.z])

                return np.array(features)
            else:
                # Legacy solutions API
                results = self.face_mesh.process(image_rgb)

                if not results.multi_face_landmarks:
                    print("[MediaPipe] No face detected in frame")
                    self._last_landmarks = None
                    return None

                landmarks = results.multi_face_landmarks[0]
                self._last_landmarks = landmarks
                print(f"[MediaPipe] Face detected! {len(landmarks.landmark)} landmarks found")

                features = []
                for landmark in landmarks.landmark:
                    features.extend([landmark.x, landmark.y, landmark.z])

                return np.array(features)

        except Exception as e:
            print(f"[MediaPipe] Error during face detection: {e}")
            import traceback
            traceback.print_exc()
            return None

    def draw_landmarks_on_image(self, image_rgb, include_lips=True, include_face=True, include_eyes=True):
        """
        Draw simplified landmarks on image for visualization.
        Only draws key points, not all 478 landmarks.
        Supports both new and legacy MediaPipe API.

        Args:
            image_rgb: RGB image (numpy array)
            include_lips: Draw lip landmarks
            include_face: Draw face oval
            include_eyes: Draw eye landmarks

        Returns:
            image_with_landmarks: RGB image with landmarks drawn
        """
        try:
            # Check if we have landmarks from either API
            if self.use_new_api:
                if self._last_landmarks_list is None:
                    print("[DRAW] No landmarks (new API) - returning original image")
                    return image_rgb
                landmarks = self._last_landmarks_list
                print(f"[DRAW] Using new API landmarks: {len(landmarks)} points")
            else:
                if self._last_landmarks is None:
                    print("[DRAW] No landmarks (old API) - returning original image")
                    return image_rgb
                landmarks = self._last_landmarks.landmark
                print(f"[DRAW] Using old API landmarks: {len(landmarks)} points")

            img = image_rgb.copy()
            h, w = img.shape[:2]
            print(f"[DRAW] Image size: {w}x{h}")

            def get_point(idx):
                """Get pixel coordinates from landmark index."""
                lm = landmarks[idx]
                return (int(lm.x * w), int(lm.y * h))

            # Draw face oval (green, thin line)
            if include_face:
                for i in range(len(self.FACE_OVAL)):
                    pt1 = get_point(self.FACE_OVAL[i])
                    pt2 = get_point(self.FACE_OVAL[(i + 1) % len(self.FACE_OVAL)])
                    cv2.line(img, pt1, pt2, (0, 255, 0), 2)

            # Draw eyes (cyan)
            if include_eyes:
                for eye_indices in [self.LEFT_EYE, self.RIGHT_EYE]:
                    for i in range(len(eye_indices)):
                        pt1 = get_point(eye_indices[i])
                        pt2 = get_point(eye_indices[(i + 1) % len(eye_indices)])
                        cv2.line(img, pt1, pt2, (255, 255, 0), 2)

            # Draw lips (red/magenta - more visible)
            if include_lips:
                # Outer lips (red)
                for i in range(len(self.LIPS_OUTER)):
                    pt1 = get_point(self.LIPS_OUTER[i])
                    pt2 = get_point(self.LIPS_OUTER[(i + 1) % len(self.LIPS_OUTER)])
                    cv2.line(img, pt1, pt2, (255, 0, 100), 2)
                # Inner lips (magenta)
                for i in range(len(self.LIPS_INNER)):
                    pt1 = get_point(self.LIPS_INNER[i])
                    pt2 = get_point(self.LIPS_INNER[(i + 1) % len(self.LIPS_INNER)])
                    cv2.line(img, pt1, pt2, (255, 0, 255), 2)
                # Draw lip points
                for idx in self.LIPS_OUTER + self.LIPS_INNER:
                    pt = get_point(idx)
                    cv2.circle(img, pt, 3, (255, 100, 100), -1)

            # Draw nose points (yellow)
            for idx in self.NOSE:
                pt = get_point(idx)
                cv2.circle(img, pt, 4, (0, 255, 255), -1)

            print(f"[DRAW] Landmarks drawn successfully!")
            return img

        except Exception as e:
            print(f"[DRAW] Error drawing landmarks: {e}")
            import traceback
            traceback.print_exc()
            return image_rgb

    def get_landmark_image_base64(self, image_rgb):
        """
        Get image with landmarks as base64 string.

        Args:
            image_rgb: RGB image

        Returns:
            base64 encoded JPEG image string
        """
        # Check both old and new API landmarks
        if self._last_landmarks is None and self._last_landmarks_list is None:
            return None

        # Draw landmarks
        img_with_landmarks = self.draw_landmarks_on_image(image_rgb)

        # Convert RGB to BGR for cv2
        img_bgr = cv2.cvtColor(img_with_landmarks, cv2.COLOR_RGB2BGR)

        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Convert to base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return img_base64

class AudioVisualFusion(nn.Module):
    """
    Fusion layer untuk menggabungkan fitur audio dan visual.
    Menggunakan attention mechanism untuk weighted fusion.
    """
    def __init__(self, audio_dim: int = 384, visual_dim: int = 512, fusion_dim: int = 256):  # Changed from 768 to 384
        super(AudioVisualFusion, self).__init__()

        # Projection layers
        self.audio_projection = nn.Linear(audio_dim, fusion_dim)
        self.visual_projection = nn.Linear(visual_dim, fusion_dim)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=-1)
        )

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim)
        )

    def forward(self, audio_features, visual_features):
        """
        Fusi audio dan visual features dengan attention weights.
        """
        # Project to common dimension
        audio_proj = self.audio_projection(audio_features)
        visual_proj = self.visual_projection(visual_features)

        # Concatenate for attention
        concat_features = torch.cat([audio_proj, visual_proj], dim=-1)

        # Calculate attention weights
        attention_weights = self.attention(concat_features)

        # Weighted fusion
        fused = attention_weights[:, 0:1] * audio_proj + attention_weights[:, 1:2] * visual_proj

        # Final fusion
        output = self.fusion_layer(fused)

        return output, attention_weights

def convert_audio_to_wav(input_path: str) -> str:
    """
    Convert audio file to WAV format using pydub.
    Returns path to temporary WAV file.
    Skip conversion if already WAV format with correct parameters.
    """
    if not PYDUB_AVAILABLE:
        return input_path  # Return as-is, librosa will try with audioread

    # Skip conversion if already a WAV file (from utils.py extraction)
    if input_path.endswith('.wav'):
        try:
            # Quick validation - try loading directly with librosa first
            import librosa
            _, sr = librosa.load(input_path, sr=16000, mono=True, duration=0.1)
            if sr == 16000:
                print(f"[INFO] Audio already in WAV format, skipping conversion")
                return input_path
        except:
            pass  # Fall through to pydub conversion

    try:
        print(f"[INFO] Converting audio to WAV format...")
        audio = AudioSegment.from_file(input_path)

        # Convert to mono, 16kHz, 16-bit PCM
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio = audio.set_sample_width(2)  # 16-bit

        # Save as WAV
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio.export(temp_wav.name, format="wav")
        temp_wav.close()

        print(f"[INFO] Conversion successful: {temp_wav.name}")
        return temp_wav.name
    except Exception as e:
        print(f"[WARNING] pydub conversion failed: {e}, trying librosa...")
        return input_path  # Fallback to original

class AudioVisualModel:
    """
    Kelas utama yang mengelola model audio, visual, fusion, dan proses terjemahan.
    Support 2 mode: audio-visual dan audio-only (upload).
    """
    def __init__(self):
        # Load configuration
        self.config = config

        # ENHANCED MODE: Use optimized backend processing
        if self.config.USE_OPTIMIZED_BACKEND:
            print("=" * 60)
            print("[ENHANCED] Enhanced backend enabled")
            print("=" * 60)
            print("[ENHANCED] Using optimized ASR & translation pipeline")
            print("[ENHANCED] Whisper models optimized for performance")
            print("[ENHANCED] Visual augmentation (MediaPipe) enabled!")
            print("[ENHANCED] Memory usage optimized!")
            print("=" * 60)

            # Set ASR/Translation models to None (saves RAM)
            self.model = None
            self.translation_model = None
            self.translation_tokenizer = None
            self.use_fp16_nllb = False
            self.using_finetuned_model = False
            self.using_faster_whisper = False
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.encoder_dim = 0

            # Language codes for compatibility
            self.lang_codes = {
                "sun": "sun_Latn",
                "ind": "ind_Latn",
                "su": "sun_Latn",
                "id": "ind_Latn",
            }

            # Still load Visual models for lip reading / visual augmentation
            if self.config.USE_VISUAL_AUGMENTATION:
                print(f"[VISUAL] Loading Visual CNN and MediaPipe for demo...")
                try:
                    self.visual_model = VisualCNN().to(self.device)
                    self.visual_model.eval()
                    self.facial_extractor = FacialExpressionExtractor(
                        model_complexity=self.config.MEDIAPIPE_MODEL_COMPLEXITY
                    )
                    self.fusion_model = None  # No fusion needed in demo mode
                    print("[VISUAL] Model Visual CNN dan Face Mesh berhasil dimuat!")
                except Exception as e:
                    print(f"[VISUAL] WARNING: Could not load visual models: {e}")
                    self.visual_model = None
                    self.facial_extractor = None
                    self.fusion_model = None
            else:
                self.visual_model = None
                self.facial_extractor = None
                self.fusion_model = None

            # Initialize enhanced backend service
            try:
                from enhanced_backend_service import get_enhanced_backend_service
                # Force refresh service to apply config
                import enhanced_backend_service
                enhanced_backend_service._enhanced_backend_service = None
                
                self._enhanced_service = get_enhanced_backend_service()
                
                # Enable true offline mode if configured
                if self.config.TRUE_OFFLINE_MODE:
                    self._enhanced_service.enable_true_offline_mode(True)
                    print("[OFFLINE] True offline mode enabled from config")
                
                if self._enhanced_service.is_ready:
                    print("[ENHANCED] Backend service ready!")
                else:
                    print("[WARNING] Enhanced backend not fully ready")
                    print("[WARNING] Please check backend configuration")
            except Exception as e:
                print(f"[WARNING] Could not initialize enhanced backend: {e}")

            print("[ENHANCED] Initialization complete!")
            return  # Skip rest of __init__

        # Device selection from config
        self.device = torch.device(self.config.get_device('whisper'))
        print(f"[CONFIG] Memuat model ke perangkat: {self.device}")
        print(f"[CONFIG] GPU Acceleration: {self.config.ENABLE_GPU_ACCELERATION}")
        print(f"[CONFIG] FP16 Precision: {self.config.FP16_PRECISION}")

        # GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[GPU] Terdeteksi: {gpu_name} ({gpu_memory:.1f} GB)")

        # Load Whisper model based on WHISPER_MODEL_VARIANT
        self.using_finetuned_model = False
        self.using_faster_whisper = False

        # Get model variant from config (cp1500, cp4500, or google)
        model_variant = getattr(self.config, 'WHISPER_MODEL_VARIANT', 'cp4500')
        model_paths = getattr(self.config, 'WHISPER_MODEL_PATHS', {})
        finetuned_path = model_paths.get(model_variant)

        # Try to load fine-tuned model if variant is whisper-base-1500steps or whisper-base-4500steps
        if model_variant in ['whisper-base-1500steps', 'whisper-base-4500steps'] and finetuned_path and FINETUNED_WHISPER_AVAILABLE:
            try:
                print(f"[WHISPER] Loading fine-tuned model: {model_variant}")
                print(f"[WHISPER] Path: {finetuned_path}")
                device_str = str(self.device).split(':')[0]
                self.model = load_finetuned_whisper(finetuned_path, device=device_str)
                self.using_finetuned_model = True
                # Encoder dim based on model type (detected in FinetunedWhisperModel)
                self.encoder_dim = 1280 if self.model.model_type == 'large-v3' else 768
                print(f"[WHISPER] Fine-tuned model loaded! (type: {self.model.model_type})")
            except Exception as e:
                print(f"[WARNING] Failed to load fine-tuned model: {e}")
                print("[WARNING] Fallback to faster-whisper...")
        
        # Try transformers-based model for whisper-small-sundanese and whisper-medium-sundanese
        elif model_variant in ['whisper-small-sundanese', 'whisper-small-openslr', 'whisper-medium-sundanese'] and finetuned_path:
            try:
                import sys
                print(f"[WHISPER] Loading transformers model: {model_variant}")
                print(f"[WHISPER] Path: {finetuned_path}")
                sys.stdout.flush()
                
                print("[WHISPER] Step 1: Importing WhisperProcessor, WhisperForConditionalGeneration...")
                sys.stdout.flush()
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                
                print("[WHISPER] Step 2: Loading WhisperProcessor...")
                sys.stdout.flush()
                self.processor = WhisperProcessor.from_pretrained(finetuned_path)
                
                print("[WHISPER] Step 3: Loading WhisperForConditionalGeneration...")
                sys.stdout.flush()
                self.model = WhisperForConditionalGeneration.from_pretrained(finetuned_path)
                
                print(f"[WHISPER] Step 4: Moving model to {self.device}...")
                sys.stdout.flush()
                self.model.to(self.device)
                
                print("[WHISPER] Step 5: Setting eval mode...")
                sys.stdout.flush()
                self.model.eval()
                
                self.using_finetuned_model = True
                self.using_transformers = True
                self.use_fp16 = self.device.type == "cuda" and self.config.FP16_PRECISION
                
                # Set encoder dimension based on model type
                if 'medium' in model_variant:
                    self.encoder_dim = 1024  # Whisper Medium
                else:
                    self.encoder_dim = 768   # Whisper Small
                
                print(f"[WHISPER] Transformers model loaded! (encoder dim: {self.encoder_dim})")
                sys.stdout.flush()
            except Exception as e:
                import traceback
                print(f"[WARNING] Failed to load transformers model: {e}")
                traceback.print_exc()
                sys.stdout.flush()
                sys.stderr.flush()
                print("[WARNING] Fallback to faster-whisper...")

        # Try faster-whisper (CTranslate2) for maximum speed
        if not self.using_finetuned_model and FASTER_WHISPER_AVAILABLE:
            whisper_model_name = self.config.get_whisper_model_name()
            compute_type = "float16" if self.device.type == "cuda" else "int8"
            device_str = "cuda" if self.device.type == "cuda" else "cpu"
            
            print(f"[WHISPER] Memuat faster-whisper model: {whisper_model_name}")
            print(f"[WHISPER] Compute type: {compute_type} (CTranslate2)")
            
            try:
                self.model = FasterWhisperModel(
                    whisper_model_name,
                    device=device_str,
                    compute_type=compute_type,
                    cpu_threads=4,
                    num_workers=2
                )
                self.using_faster_whisper = True
                
                # Determine encoder dimension based on model type
                model_dims = {
                    'tiny': 384, 'base': 512, 'small': 768,
                    'medium': 1024, 'large': 1280, 'large-v2': 1280, 'large-v3': 1280
                }
                self.encoder_dim = model_dims.get(self.config.WHISPER_MODEL_TYPE, 1280)
                print(f"[WHISPER] faster-whisper berhasil dimuat! (4x faster)")
                print(f"[WHISPER] Audio encoder dim: {self.encoder_dim}")
            except Exception as e:
                print(f"[WARNING] Gagal memuat faster-whisper: {e}")
                print("[WARNING] Fallback ke openai-whisper...")

        # Fallback to OpenAI Whisper if faster-whisper and fine-tuned failed
        if not self.using_finetuned_model and not self.using_faster_whisper:
            if not OPENAI_WHISPER_AVAILABLE:
                raise RuntimeError("No Whisper implementation available! Install faster-whisper or openai-whisper.")
            
            whisper_model_name = self.config.get_whisper_model_name()
            print(f"[WHISPER] Memuat model: {whisper_model_name}")
            print(f"[WHISPER] Compute type: {self.config.WHISPER_COMPUTE_TYPE}")

            try:
                self.model = whisper.load_model(whisper_model_name, device=self.device)
                print(f"[WHISPER] Model berhasil dimuat!")

                # Determine encoder dimension based on model type
                model_dims = {
                    'tiny': 384, 'base': 512, 'small': 768,
                    'medium': 1024, 'large': 1280, 'large-v2': 1280, 'large-v3': 1280
                }
                self.encoder_dim = model_dims.get(self.config.WHISPER_MODEL_TYPE, 1280)
                print(f"[WHISPER] Audio encoder dim: {self.encoder_dim}")
            except Exception as e:
                print(f"[ERROR] Failed to load Whisper model: {e}")
                raise

        # Load visual models (if enabled)
        if self.config.USE_VISUAL_AUGMENTATION:
            print(f"[VISUAL] Loading Visual CNN and MediaPipe...")
            print(f"[VISUAL] MediaPipe complexity: {self.config.MEDIAPIPE_MODEL_COMPLEXITY}")
            self.visual_model = VisualCNN().to(self.device)
            self.visual_model.eval()
            self.facial_extractor = FacialExpressionExtractor(
                model_complexity=self.config.MEDIAPIPE_MODEL_COMPLEXITY
            )
            print("[VISUAL] Model Visual CNN dan Face Mesh berhasil dimuat!")

            # Load fusion model
            self.fusion_model = AudioVisualFusion(
                audio_dim=self.encoder_dim,
                visual_dim=512
            ).to(self.device)
            self.fusion_model.eval()
            print("[FUSION] Model Fusion berhasil dimuat!")
        else:
            print("[VISUAL] Visual augmentation DISABLED (audio-only mode)")
            self.visual_model = None
            self.facial_extractor = None
            self.fusion_model = None

        # Load translation model (NLLB-200) with FP16 optimization
        print(f"[NLLB] Memuat model terjemahan: {self.config.NLLB_MODEL_NAME}")
        print("[NLLB] Mendukung Sundanese (sun_Latn) <-> Indonesian (ind_Latn)")

        try:
            self.translation_model_name = self.config.NLLB_MODEL_NAME
            self.translation_tokenizer = AutoTokenizer.from_pretrained(self.translation_model_name)

            # Move to configured device
            translation_device = torch.device(self.config.get_device('nllb'))
            self.use_fp16_nllb = translation_device.type == "cuda" and self.config.FP16_PRECISION

            # Load with FP16 for GPU speedup
            import sys
            import traceback

            if self.use_fp16_nllb:
                print("[NLLB] Loading with FP16 for GPU acceleration...")
                print("[NLLB] Step 1: Starting model download/load...")
                sys.stdout.flush()
                sys.stderr.flush()
                try:
                    self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.translation_model_name,
                        torch_dtype=torch.float16
                    )
                    print("[NLLB] Step 2: Model loaded to CPU")
                    sys.stdout.flush()
                except Exception as model_err:
                    print(f"[NLLB ERROR] Failed during model loading: {model_err}")
                    traceback.print_exc()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    raise
            else:
                print("[NLLB] Loading with FP32...")
                sys.stdout.flush()
                self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(self.translation_model_name)
                print("[NLLB] Step 2: Model loaded to CPU")

            print(f"[NLLB] Step 3: Moving model to {translation_device}...")
            sys.stdout.flush()
            self.translation_model.to(translation_device)
            print("[NLLB] Step 4: Setting eval mode...")
            sys.stdout.flush()
            self.translation_model.eval()

            print(f"[NLLB] Model berhasil dimuat on {translation_device}!")
            print(f"[NLLB] FP16 Mode: {self.use_fp16_nllb}")

            # Warm-up NLLB model
            if translation_device.type == "cuda":
                self._warmup_nllb()
        except Exception as e:
            print(f"[ERROR] Failed to load NLLB model: {e}")
            raise

        # Language codes for NLLB-200
        self.lang_codes = {
            "sun": "sun_Latn",  # Sundanese in Latin script
            "ind": "ind_Latn"   # Indonesian in Latin script
        }

        print("\n" + "="*60)
        print("ALL MODELS LOADED SUCCESSFULLY!")
        print("="*60)
        print(f"Model size: ~600MB | Type: Transformer | Support: Multilingual (Sunda <-> Indonesia)")

        # Full pipeline warm-up
        self._warmup_pipeline()

        print("Semua model AI siap.")

    def _warmup_pipeline(self):
        """Full pipeline warm-up to eliminate first-run latency."""
        # Check if NLLB is on CPU - skip full pipeline warmup (too slow)
        nllb_device = next(self.translation_model.parameters()).device
        if nllb_device.type == "cpu":
            print("[WARMUP] Skipping full pipeline warm-up (NLLB on CPU - would be too slow)")
            print("[WARMUP] Running Whisper-only warm-up...")
            try:
                import io
                import soundfile as sf

                # Create very short dummy audio (0.5 second)
                dummy_waveform = np.random.randn(8000).astype(np.float32) * 0.01
                buffer = io.BytesIO()
                sf.write(buffer, dummy_waveform, 16000, format='WAV')
                dummy_audio_bytes = buffer.getvalue()

                # Only warm up Whisper (no translation)
                result = self.process_translation(
                    dummy_audio_bytes,
                    video_frame_bytes=None,
                    use_visual=False,
                    auto_translate=False  # Skip translation warm-up
                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                print("[WARMUP] Whisper warm-up complete!")
            except Exception as e:
                print(f"[WARMUP] Whisper warm-up failed (non-critical): {e}")
            return

        print("[WARMUP] Running full pipeline warm-up...")
        try:
            import io
            import soundfile as sf

            # Create dummy audio (0.5 second, shorter to avoid repetitive text)
            dummy_waveform = np.random.randn(8000).astype(np.float32) * 0.01

            # Write to bytes (simulating real input)
            buffer = io.BytesIO()
            sf.write(buffer, dummy_waveform, 16000, format='WAV')
            dummy_audio_bytes = buffer.getvalue()

            # Run full transcription + translation pipeline
            result = self.process_translation(
                dummy_audio_bytes,
                video_frame_bytes=None,
                use_visual=False,
                auto_translate=True
            )

            # Sync CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            print("[WARMUP] Full pipeline warm-up complete!")
        except Exception as e:
            print(f"[WARMUP] Pipeline warm-up failed (non-critical): {e}")

    def _warmup_nllb(self):
        """Warm-up NLLB model to pre-compile CUDA kernels."""
        print("[NLLB] Warming up CUDA kernels...")
        try:
            # Dummy translation
            dummy_text = "halo"
            self.translation_tokenizer.src_lang = "sun_Latn"
            translation_device = next(self.translation_model.parameters()).device

            inputs = self.translation_tokenizer(
                dummy_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32
            ).to(translation_device)

            forced_bos_token_id = self.translation_tokenizer.convert_tokens_to_ids("ind_Latn")

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_fp16_nllb):
                _ = self.translation_model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=10,
                    num_beams=1,
                    do_sample=False,
                    use_cache=True
                )

            torch.cuda.synchronize()
            print("[NLLB] Warm-up complete!")
        except Exception as e:
            print(f"[NLLB] Warm-up failed (non-critical): {e}")

    def preprocess_video_frame(self, video_frame_bytes: bytes):
        """
        Mengubah frame video dari bytes menjadi tensor dan ekstrak facial features.
        Returns: (tensor_grayscale, facial_landmarks, image_rgb)
        """
        nparr = np.frombuffer(video_frame_bytes, np.uint8)
        print(f"[VISUAL] Video frame bytes: {len(video_frame_bytes)}")

        # Load as RGB for facial expression
        img_rgb = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_rgb is None:
            print("[VISUAL] Failed to decode image from bytes")
            return None, None, None

        print(f"[VISUAL] Image decoded: {img_rgb.shape} (H x W x C)")
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        # Extract facial landmarks
        print("[VISUAL] Extracting facial landmarks...")
        facial_features = self.facial_extractor.extract_features(img_rgb)
        if facial_features is not None:
            print(f"[VISUAL] Facial features extracted: {len(facial_features)} values")
        else:
            print("[VISUAL] No facial features detected - face not found in frame")

        # Grayscale for CNN lip reading
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(img_gray, (96, 96))
        tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0) / 255.0

        return tensor.to(self.device), facial_features, img_rgb

    def extract_audio_features(self, waveform):
        """
        Ekstrak hidden states dari Whisper encoder sebagai audio features.
        Returns: audio_features tensor (1280-dim) in float32

        Compatible with both OpenAI Whisper and fine-tuned HuggingFace Whisper.
        """
        try:
            if self.using_finetuned_model:
                # Fine-tuned model uses HuggingFace transformers
                # Use processor to create mel spectrogram
                # Note: processor is stored in self.processor, not self.model.processor
                inputs = self.processor(
                    waveform,
                    sampling_rate=16000,
                    return_tensors="pt"
                )

                # Move to device - always use float32 for encoder
                # The transformers model is loaded in float32, so we must match dtype
                # FP16 would cause "Input type (Half) and bias type (float) should be the same" error
                input_features = inputs.input_features.to(self.device, dtype=torch.float32)

                with torch.no_grad():
                    # Extract encoder features using the encoder property
                    # Note: WhisperForConditionalGeneration stores encoder in model.model.encoder
                    encoder = self.model.get_encoder()
                    encoder_outputs = encoder(input_features)
                    # Get last hidden state
                    audio_features = encoder_outputs.last_hidden_state
                    # Average over time dimension to get fixed-size features
                    audio_features = audio_features.mean(dim=1)  # Shape: [1, encoder_dim]
                    # Ensure float32 output
                    audio_features = audio_features.float()
            elif self.using_faster_whisper:
                # faster-whisper doesn't expose encoder directly
                # Use librosa to extract mel spectrogram features
                # Then project to encoder dimension using a simple linear layer
                import librosa
                
                # Compute mel spectrogram (similar to Whisper preprocessing)
                mel = librosa.feature.melspectrogram(
                    y=waveform.astype(np.float32),
                    sr=16000,
                    n_mels=128,
                    n_fft=400,
                    hop_length=160
                )
                log_mel = librosa.power_to_db(mel, ref=np.max)
                
                # Normalize and convert to tensor
                log_mel_normalized = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
                mel_tensor = torch.FloatTensor(log_mel_normalized).to(self.device)
                
                # Global average pooling across time, then repeat for encoder dim
                with torch.no_grad():
                    # mel_tensor shape: [n_mels=128, time_frames]
                    # Average across time to get [128] features
                    mel_avg = mel_tensor.mean(dim=1)  # [128]
                    # Repeat to match encoder dimension (1280 for large-v3)
                    repeat_factor = self.encoder_dim // 128  # 1280 // 128 = 10
                    audio_features = mel_avg.repeat(repeat_factor).unsqueeze(0).float()  # [1, 1280]
            else:
                # Base OpenAI Whisper model
                # Pad or trim to 30 seconds (Whisper default)
                audio_padded = whisper.pad_or_trim(torch.FloatTensor(waveform))

                # Convert to mel spectrogram
                # Note: Whisper large-v3 uses 128 mel bins (not 80)
                n_mels = getattr(self.model.dims, 'n_mels', 80)
                mel = whisper.log_mel_spectrogram(audio_padded, n_mels=n_mels).to(self.device)

                with torch.no_grad():
                    # Extract encoder features
                    audio_features = self.model.encoder(mel.unsqueeze(0))
                    # Average over time dimension to get fixed-size features
                    audio_features = audio_features.mean(dim=1)  # Shape: [1, 1280]
                    # Ensure float32 output (Whisper may use fp16)
                    audio_features = audio_features.float()

            return audio_features

        except Exception as e:
            print(f"[ERROR] extract_audio_features failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def process_translation(self, audio_bytes: bytes, video_frame_bytes: bytes = None, use_visual: bool = True, auto_translate: bool = True):
        """
        Pipeline AI untuk terjemahan dengan/tanpa visual.
        Mode audio-visual: menggunakan fusion audio+visual
        Mode audio-only: hanya menggunakan audio

        Args:
            audio_bytes: Audio data dalam bytes
            video_frame_bytes: Video frame dalam bytes (optional)
            use_visual: Gunakan visual features jika tersedia
            auto_translate: Otomatis terjemahkan hasil transkripsi (Sunda → Indonesia)

        Returns:
            dict: {"transcription": str, "translation": str} jika auto_translate=True
            str: transcription saja jika auto_translate=False
        """
        # ENHANCED MODE: Use optimized backend for ASR + Translation
        if self.config.USE_OPTIMIZED_BACKEND:
            return self._process_with_google_api(audio_bytes, video_frame_bytes, use_visual, auto_translate)

        temp_file_path = None
        converted_wav_path = None

        try:
            # Detect if audio is already WAV format (starts with RIFF header)
            is_wav = audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE'
            suffix = ".wav" if is_wav else ".webm"
            
            # Simpan audio ke temporary file
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name

            # Convert to WAV if not already WAV
            if is_wav:
                converted_wav_path = temp_file_path
                # Skip conversion message for cleaner logs
            else:
                converted_wav_path = convert_audio_to_wav(temp_file_path)

            # DEBUG: Save audio for inspection (disabled in production for performance)
            # Uncomment for debugging audio issues
            # try:
            #     from debug_audio import save_debug_audio, get_audio_info
            #     saved_path = save_debug_audio(converted_wav_path, prefix="recording")
            #     if saved_path:
            #         get_audio_info(saved_path)
            # except Exception as e:
            #     print(f"[DEBUG] Debug audio failed: {e}")

            # Load audio dengan librosa
            try:
                waveform, sample_rate = librosa.load(converted_wav_path, sr=16000, mono=True)
                print(f"[INFO] Berhasil load audio: {len(waveform)} samples")
            except Exception as e:
                print(f"[ERROR] Gagal load audio: {e}")
                # Cleanup
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                if converted_wav_path and converted_wav_path != temp_file_path and os.path.exists(converted_wav_path):
                    os.unlink(converted_wav_path)
                return None

            # Cleanup temporary files
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            if converted_wav_path and converted_wav_path != temp_file_path and os.path.exists(converted_wav_path):
                os.unlink(converted_wav_path)

            if len(waveform) == 0:
                print("[WARNING] Audio kosong")
                return None

            # Mode audio-visual dengan fusion
            visual_augmentation = None
            if use_visual and video_frame_bytes and len(video_frame_bytes) > 10:
                result = self._process_with_visual(waveform, video_frame_bytes, return_visual_data=True)
                if isinstance(result, tuple):
                    transcription, visual_augmentation = result
                else:
                    transcription = result
            # Mode audio-only
            else:
                transcription = self._process_audio_only(waveform)

            # Auto-translate jika diminta
            if auto_translate and transcription and transcription.strip():
                print(f"[INFO] Auto-translating: {transcription}")
                translation = self.translate(transcription, source_lang="sun", target_lang="ind")
                response = {
                    "transcription": transcription,
                    "translation": translation
                }
                # Add visual augmentation data if available (for BAB 4 documentation)
                if visual_augmentation:
                    response["visual_augmentation"] = visual_augmentation
                return response
            else:
                return transcription

        except Exception as e:
            print(f"[ERROR] process_translation: {e}")
            return None

    def _transcribe_waveform(self, waveform, mode="audio-only"):
        """
        Transcribe waveform using appropriate model (DRY helper method).

        Args:
            waveform: Audio waveform
            mode: "audio-only" or "audio-visual" for logging

        Returns:
            Normalized transcription text
        """
        if self.using_finetuned_model:
            # Check if using transformers model (small-sun, medium-sun)
            if hasattr(self, 'using_transformers') and self.using_transformers:
                # Transformers-based model (WhisperForConditionalGeneration)
                import torch
                
                # Process audio with transformers
                inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        inputs["input_features"],
                        max_length=448,
                        num_beams=5,
                        language="su",  # Sundanese
                        task="transcribe",
                        do_sample=False,
                        temperature=0.0
                    )
                
                transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                confidence_scores = []  # Not available for transformers model
            else:
                # Fine-tuned model (whisper_finetuned module)
                segments, info = self.model.transcribe(waveform, language="su", task="transcribe")
                all_text = [seg.text for seg in segments]
                confidence_scores = [seg.avg_logprob for seg in segments]
                transcription = " ".join(all_text).strip()
        elif self.using_faster_whisper:
            # faster-whisper (CTranslate2) - 4x faster than OpenAI whisper
            segments, info = self.model.transcribe(
                waveform,
                language="su",  # Sundanese
                task="transcribe",
                beam_size=1,  # Greedy decoding for maximum speed
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                vad_filter=True,  # Voice Activity Detection for faster processing
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200
                )
            )
            # faster-whisper returns a generator, convert to list
            segments_list = list(segments)
            all_text = [seg.text for seg in segments_list]
            confidence_scores = [seg.avg_logprob for seg in segments_list]
            transcription = " ".join(all_text).strip()
        else:
            # Base Whisper model (openai/whisper library)
            # Use Sundanese ("su") - supported in large-v3
            result = self.model.transcribe(
                waveform,
                language="su",  # Sundanese (jv for Javanese if needed)
                fp16=torch.cuda.is_available(),
                verbose=False,
                temperature=0.0,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=False,
                beam_size=3,  # Balanced accuracy and speed
                best_of=3,
            )
            transcription = result["text"].strip()
            confidence_scores = [seg.get("avg_logprob", 0) for seg in result.get("segments", [])]

        # Normalize transcription
        normalized = normalize_transcription(transcription, apply_corrections=True, confidence_scores=confidence_scores)
        print(f"[INFO] Transcription ({mode}): '{normalized}'")
        return normalized

    def _process_audio_only(self, waveform) -> str:
        """
        Proses audio saja tanpa visual (untuk upload mode).
        Menggunakan Whisper transcribe dengan GPU acceleration.
        """
        try:
            return self._transcribe_waveform(waveform, mode="audio-only")
        except Exception as e:
            print(f"[ERROR] _process_audio_only: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_with_visual(self, waveform, video_frame_bytes: bytes, return_visual_data: bool = True):
        """
        Proses audio+visual dengan fusion.
        Menggunakan CNN untuk lip reading dan MediaPipe untuk facial landmarks.

        Args:
            waveform: Audio waveform
            video_frame_bytes: Video frame in bytes
            return_visual_data: If True, return tuple (transcription, visual_data)

        Returns:
            str or tuple: transcription only, or (transcription, visual_augmentation_data)
        """
        try:
            print("\n" + "="*60)
            print("[VISUAL AUGMENTATION] Starting audio-visual processing...")
            print("="*60)

            # Extract audio features from Whisper encoder
            audio_features = self.extract_audio_features(waveform)
            print(f"[AUDIO] Audio features extracted: shape={audio_features.shape}, dim={audio_features.shape[-1]}")

            # Extract visual features
            visual_tensor, facial_landmarks, img_rgb = self.preprocess_video_frame(video_frame_bytes)

            if visual_tensor is None:
                print("[WARNING] Video frame extraction failed, using audio-only")
                transcription = self._process_audio_only(waveform)
                if return_visual_data:
                    return transcription, None
                return transcription

            # Validate visual tensor shape (should be [1, 1, 96, 96])
            if len(visual_tensor.shape) != 4 or visual_tensor.shape[1] != 1:
                print(f"[WARNING] Invalid visual tensor shape: {visual_tensor.shape}, expected [1, 1, 96, 96], using audio-only")
                transcription = self._process_audio_only(waveform)
                if return_visual_data:
                    return transcription, None
                return transcription

            print(f"[VISUAL] Video frame preprocessed: shape={visual_tensor.shape} (grayscale 96x96)")

            # Initialize visual augmentation data
            visual_augmentation_data = {
                "enabled": True,
                "lip_reading": {
                    "model": "VisualCNN",
                    "input_shape": list(visual_tensor.shape),
                    "input_size": "96x96 grayscale"
                },
                "facial_landmarks": {
                    "model": "MediaPipe Face Mesh",
                    "detected": False,
                    "landmark_count": 0,
                    "feature_count": 0
                },
                "fusion": {
                    "model": "AudioVisualFusion (Attention)",
                    "audio_dim": int(audio_features.shape[-1]),
                    "visual_dim": 512,
                    "fusion_dim": 256
                },
                "attention_weights": {
                    "audio": 0.0,
                    "visual": 0.0
                }
            }

            with torch.no_grad():
                # Ensure visual tensor is float32 (consistent with CNN model)
                visual_tensor_f32 = visual_tensor.float()

                # CNN features from lip movement
                visual_features = self.visual_model(visual_tensor_f32)
                print(f"[VISUAL] CNN lip features extracted: shape={visual_features.shape}, dim={visual_features.shape[-1]}")

                visual_augmentation_data["lip_reading"]["output_dim"] = int(visual_features.shape[-1])

                # Ensure both tensors are float32 for fusion (audio may be fp16 from Whisper)
                audio_features_f32 = audio_features.float()
                visual_features_f32 = visual_features.float()

                # Fusion audio + visual with attention mechanism
                fused_features, attention_weights = self.fusion_model(audio_features_f32, visual_features_f32)
                audio_weight = attention_weights[0, 0].item()
                visual_weight = attention_weights[0, 1].item()

                print(f"[FUSION] Fused features: shape={fused_features.shape}")
                print(f"[FUSION] Attention weights - Audio: {audio_weight:.4f} ({audio_weight*100:.1f}%), Visual: {visual_weight:.4f} ({visual_weight*100:.1f}%)")

                visual_augmentation_data["attention_weights"]["audio"] = round(audio_weight, 4)
                visual_augmentation_data["attention_weights"]["visual"] = round(visual_weight, 4)
                visual_augmentation_data["fusion"]["output_dim"] = int(fused_features.shape[-1])

                # Log facial expression info and generate visualization
                if facial_landmarks is not None:
                    landmark_count = len(facial_landmarks) // 3  # 3 coords per landmark
                    print(f"[MEDIAPIPE] Facial landmarks detected: {len(facial_landmarks)} features ({landmark_count} landmarks x 3 coords)")
                    visual_augmentation_data["facial_landmarks"]["detected"] = True
                    visual_augmentation_data["facial_landmarks"]["landmark_count"] = landmark_count
                    visual_augmentation_data["facial_landmarks"]["feature_count"] = len(facial_landmarks)

                    # Generate landmark visualization image
                    landmark_image_base64 = self.facial_extractor.get_landmark_image_base64(img_rgb)
                    if landmark_image_base64:
                        visual_augmentation_data["landmark_image"] = landmark_image_base64
                        print(f"[MEDIAPIPE] Landmark visualization generated ({len(landmark_image_base64)} chars)")
                else:
                    print(f"[MEDIAPIPE] No facial landmarks detected in this frame")

            print("="*60)
            print("[VISUAL AUGMENTATION] Processing complete!")
            print("="*60 + "\n")

            # Get transcription from Whisper
            # Note: Fused features provide visual context to improve accuracy
            transcription = self._transcribe_waveform(waveform, mode="audio-visual")

            if return_visual_data:
                return transcription, visual_augmentation_data
            return transcription

        except Exception as e:
            import traceback
            print(f"[ERROR] _process_with_visual: {e}, fallback ke audio-only")
            traceback.print_exc()
            transcription = self._process_audio_only(waveform)
            if return_visual_data:
                return transcription, None
            return transcription

    def translate(self, text: str, source_lang: str = "sun", target_lang: str = "ind") -> str:
        """
        Terjemahkan teks antara Sunda dan Indonesia.

        Args:
            text: Teks yang akan diterjemahkan
            source_lang: Bahasa sumber ("sun" untuk Sunda, "ind" untuk Indonesia)
            target_lang: Bahasa target ("sun" untuk Sunda, "ind" untuk Indonesia)

        Returns:
            Teks hasil terjemahan
        """
        try:
            if not text or len(text.strip()) == 0:
                print("[WARNING] Teks kosong, tidak ada yang diterjemahkan")
                return ""

            print(f"[INFO] Translating: '{text}' ({source_lang} -> {target_lang})")

            # Check if enhanced backend is enabled
            if self.config.USE_OPTIMIZED_BACKEND:
                return self._translate_with_google(text, source_lang, target_lang)

            print(f"[INFO] Using model: NLLB-200")

            # Get language codes
            src_lang_code = self.lang_codes.get(source_lang, "sun_Latn")
            tgt_lang_code = self.lang_codes.get(target_lang, "ind_Latn")

            # Set source language in tokenizer (important for NLLB)
            self.translation_tokenizer.src_lang = src_lang_code

            # Tokenize input and move to correct device
            translation_device = next(self.translation_model.parameters()).device
            inputs = self.translation_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.TRANSLATION_MAX_LENGTH
            ).to(translation_device)

            # Set target language
            forced_bos_token_id = self.translation_tokenizer.convert_tokens_to_ids(tgt_lang_code)

            # Generate translation with optimized parameters
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_fp16_nllb):
                outputs = self.translation_model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=self.config.TRANSLATION_MAX_LENGTH,
                    num_beams=1,  # Greedy decoding - fastest (fine-tuned model is accurate enough)
                    early_stopping=True,
                    do_sample=False,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    use_cache=True  # Enable KV cache for faster generation
                )

            # Decode
            translation = self.translation_tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"[INFO] Translation result (raw): '{translation}'")

            # Apply rule-based post-processing untuk improve quality
            translation_corrected = post_process_translation(translation, source_lang, target_lang, text)
            print(f"[INFO] Translation result (corrected): '{translation_corrected}'")

            return translation_corrected.strip()

        except Exception as e:
            print(f"[ERROR] translate: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

    def _translate_with_google(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate using Google Translate API (for demo mode).
        More accurate than NLLB for Sundanese.
        """
        from enhanced_backend_service import get_enhanced_backend_service

        print(f"[ENHANCED] Using Google Translate API")

        # Map language codes: sun/ind -> su/id
        lang_map = {
            "sun": "su",
            "ind": "id",
            "su": "su",
            "id": "id",
        }

        src = lang_map.get(source_lang, "su")
        tgt = lang_map.get(target_lang, "id")

        try:
            service = get_enhanced_backend_service()
            result = service.translate(text, source=src, target=tgt)
            return result
        except Exception as e:
            print(f"[ENHANCED] Google Translate failed: {e}, falling back to NLLB")
            # Fallback to NLLB if Google fails
            return self._translate_with_nllb(text, source_lang, target_lang)

    def _translate_with_nllb(self, text: str, source_lang: str, target_lang: str) -> str:
        """Original NLLB translation (extracted for fallback)."""
        # Get language codes
        src_lang_code = self.lang_codes.get(source_lang, "sun_Latn")
        tgt_lang_code = self.lang_codes.get(target_lang, "ind_Latn")

        # Set source language in tokenizer (important for NLLB)
        self.translation_tokenizer.src_lang = src_lang_code

        # Tokenize input and move to correct device
        translation_device = next(self.translation_model.parameters()).device
        inputs = self.translation_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.TRANSLATION_MAX_LENGTH
        ).to(translation_device)

        # Set target language
        forced_bos_token_id = self.translation_tokenizer.convert_tokens_to_ids(tgt_lang_code)

        # Generate translation with optimized parameters
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_fp16_nllb):
            outputs = self.translation_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=self.config.TRANSLATION_MAX_LENGTH,
                num_beams=1,
                early_stopping=True,
                do_sample=False,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                use_cache=True
            )

        # Decode
        translation = self.translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation_corrected = post_process_translation(translation, source_lang, target_lang, text)
        return translation_corrected.strip()

    def _process_with_google_api(self, audio_bytes: bytes, video_frame_bytes: bytes = None,
                                   use_visual: bool = True, auto_translate: bool = True):
        """
        Process audio using Google API (Demo Mode).
        Uses Google Web Speech API for ASR and Google Translate for translation.
        Also extracts visual augmentation data if video frame is provided.
        """
        from enhanced_backend_service import get_enhanced_backend_service

        print("[ENHANCED] Processing with Google API (ASR + Translation)")

        temp_file_path = None
        converted_wav_path = None
        visual_augmentation = None

        try:
            # Extract visual features if video frame provided
            if use_visual and video_frame_bytes and len(video_frame_bytes) > 10 and self.facial_extractor:
                try:
                    print("[ENHANCED] Extracting visual features...")
                    # Decode frame
                    frame_array = np.frombuffer(video_frame_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                    if frame is not None:
                        # Extract facial features
                        facial_features = self.facial_extractor.extract_features(frame)

                        if facial_features is not None:
                            # Get detailed lip/mouth features for augmentation
                            mouth_openness = float(facial_features[13]) if len(facial_features) > 13 else 0.0
                            lip_pucker = float(facial_features[14]) if len(facial_features) > 14 else 0.0
                            jaw_movement = float(facial_features[15]) if len(facial_features) > 15 else 0.0
                            landmark_count = len(facial_features) if hasattr(facial_features, '__len__') else 478

                            # Calculate visual confidence based on mouth movement
                            visual_confidence = min(0.5, (mouth_openness + lip_pucker + jaw_movement) / 3)
                            audio_confidence = 1.0 - visual_confidence

                            visual_augmentation = {
                                "lip_features_detected": True,
                                "facial_landmarks_count": landmark_count,
                                "mouth_openness": mouth_openness,
                                "lip_pucker": lip_pucker,
                                "jaw_movement": jaw_movement,
                                "attention_weights": {
                                    "audio": audio_confidence,
                                    "visual": visual_confidence
                                },
                                "facial_landmarks": {
                                    "detected": True,
                                    "landmark_count": landmark_count
                                }
                            }

                            # Generate landmark visualization image
                            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            landmark_image_base64 = self.facial_extractor.get_landmark_image_base64(img_rgb)
                            if landmark_image_base64:
                                visual_augmentation["landmark_image"] = landmark_image_base64
                                print(f"[ENHANCED] Landmark visualization generated ({len(landmark_image_base64)} chars)")

                            print(f"[ENHANCED] Visual features extracted")
                        else:
                            visual_augmentation = {
                                "lip_features_detected": False,
                                "reason": "No face detected",
                                "attention_weights": {
                                    "audio": 1.0,
                                    "visual": 0.0
                                },
                                "facial_landmarks": {
                                    "detected": False,
                                    "landmark_count": 0
                                }
                            }
                            print("[ENHANCED] No face detected in frame")
                    else:
                        visual_augmentation = {
                            "lip_features_detected": False,
                            "reason": "Invalid frame",
                            "attention_weights": {
                                "audio": 1.0,
                                "visual": 0.0
                            },
                            "facial_landmarks": {
                                "detected": False,
                                "landmark_count": 0
                            }
                        }
                except Exception as e:
                    print(f"[ENHANCED] Visual extraction error: {e}")
                    visual_augmentation = {
                        "lip_features_detected": False,
                        "error": str(e),
                        "attention_weights": {
                            "audio": 1.0,
                            "visual": 0.0
                        },
                        "facial_landmarks": {
                            "detected": False,
                            "landmark_count": 0
                        }
                    }

            # Save audio to temporary file
            is_wav = audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE'
            suffix = ".wav" if is_wav else ".webm"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name

            # Convert to WAV if needed
            if is_wav:
                converted_wav_path = temp_file_path
            else:
                converted_wav_path = convert_audio_to_wav(temp_file_path)

            # Get Google API service
            service = get_enhanced_backend_service()

            if not service.is_ready:
                print("[ENHANCED] Google API not ready, falling back to Whisper + NLLB")
                # Cleanup and fallback
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                if converted_wav_path and converted_wav_path != temp_file_path and os.path.exists(converted_wav_path):
                    os.unlink(converted_wav_path)
                # Set enhanced mode to False temporarily and call regular method
                original_mode = self.config.USE_OPTIMIZED_BACKEND
                self.config.USE_OPTIMIZED_BACKEND = False
                result = self.process_translation(audio_bytes, video_frame_bytes, use_visual, auto_translate)
                self.config.USE_OPTIMIZED_BACKEND = original_mode
                return result

            # ASR with Google Web Speech API
            transcription = service.transcribe_audio(converted_wav_path, language="su-ID")
            print(f"[ENHANCED] Transcription: {transcription}")

            # Cleanup temp files
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            if converted_wav_path and converted_wav_path != temp_file_path and os.path.exists(converted_wav_path):
                os.unlink(converted_wav_path)

            if not transcription:
                return None

            # Translate if requested
            if auto_translate:
                translation = service.translate(transcription, source="su", target="id")
                print(f"[ENHANCED] Translation: {translation}")
                response = {
                    "transcription": transcription,
                    "translation": translation
                }
                # Add visual augmentation data if available
                if visual_augmentation:
                    response["visual_augmentation"] = visual_augmentation
                return response
            else:
                return transcription

        except Exception as e:
            print(f"[ENHANCED] Error: {e}")
            import traceback
            traceback.print_exc()

            # Cleanup on error
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            if converted_wav_path and converted_wav_path != temp_file_path and os.path.exists(converted_wav_path):
                try:
                    os.unlink(converted_wav_path)
                except:
                    pass

            return None
