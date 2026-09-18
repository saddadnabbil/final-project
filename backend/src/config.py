"""
Configuration module for backend application.
Loads all settings from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')


def get_bool(value, default=False):
    """Convert string to boolean."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')


def get_int(value, default=0):
    """Convert string to integer."""
    try:
        return int(value) if value else default
    except ValueError:
        return default


def get_float(value, default=0.0):
    """Convert string to float."""
    try:
        return float(value) if value else default
    except ValueError:
        return default


class Config:
    """Application configuration from environment variables."""

    # ============================================
    # FLASK CONFIGURATION
    # ============================================
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'dev-jwt-secret-change-in-production')
    JWT_EXPIRATION_DAYS = get_int(os.getenv('JWT_EXPIRATION_DAYS'), 7)

    # CORS
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(',')

    # Database
    DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///translator.db')

    # ============================================
    # AI MODEL CONFIGURATION
    # ============================================

    # Whisper Settings
    WHISPER_MODEL_TYPE = os.getenv('WHISPER_MODEL_TYPE', 'base')
    WHISPER_MODEL_PATH = os.getenv('WHISPER_MODEL_PATH', '')
    WHISPER_DEVICE = os.getenv('WHISPER_DEVICE', 'cuda')
    WHISPER_COMPUTE_TYPE = os.getenv('WHISPER_COMPUTE_TYPE', 'float16')

    # Fine-tuned Whisper Model Settings
    USE_FINETUNED_WHISPER = get_bool(os.getenv('USE_FINETUNED_WHISPER'), False)
    # USE_FINETUNED_WHISPER = False # Forced disable due to model crash
    FINETUNED_WHISPER_PATH = os.getenv('FINETUNED_WHISPER_PATH',
                                        './models/whisper-cp1000')

    # Multi-Model Configuration
    # Available models: whisper-base-1500steps, whisper-base-4500steps, whisper-small-openslr, whisper-medium-sundanese, google-speech-api
    WHISPER_MODEL_VARIANT = os.getenv('WHISPER_MODEL_VARIANT', 'whisper-medium-sundanese')

    # Model paths for each variant
    WHISPER_MODEL_PATHS = {
        'whisper-base-1500steps': './models/whisper-cp1000',           # Original fine-tuned baseline (1500 steps)
        'whisper-base-4500steps': './models/whisper-cp4500',       # Extended baseline training (4500 steps)
        'whisper-small-openslr': './models/whisper-small-openslr',   # Kaggle trained Small (5000 steps, WER ~7%)
        'whisper-medium-sundanese': './models/whisper-medium-sundanese', # HuggingFace trained Medium (5000 steps, WER 3.66%)
        'google-speech-api': None,                               # Google Cloud Speech-to-Text API
    }

    # Google Speech API Settings (untuk disable/enable Google Speech)
    USE_GOOGLE_SPEECH_API = get_bool(os.getenv('USE_GOOGLE_SPEECH_API'), False)
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')

    # NLLB Translation Settings
    NLLB_MODEL_NAME = os.getenv('NLLB_MODEL_NAME', 'facebook/nllb-200-1.3B')
    NLLB_DEVICE = os.getenv('NLLB_DEVICE', 'cuda')

    # Visual Model Settings
    USE_VISUAL_AUGMENTATION = get_bool(os.getenv('USE_VISUAL_AUGMENTATION'), True)
    MEDIAPIPE_MODEL_COMPLEXITY = get_int(os.getenv('MEDIAPIPE_MODEL_COMPLEXITY'), 1)

    # ============================================
    # PERFORMANCE SETTINGS
    # ============================================

    # File size limits (in bytes)
    MAX_AUDIO_SIZE_MB = get_int(os.getenv('MAX_AUDIO_SIZE_MB'), 16)
    MAX_VIDEO_SIZE_MB = get_int(os.getenv('MAX_VIDEO_SIZE_MB'), 100)
    MAX_AUDIO_SIZE = MAX_AUDIO_SIZE_MB * 1024 * 1024
    MAX_VIDEO_SIZE = MAX_VIDEO_SIZE_MB * 1024 * 1024

    # GPU Settings
    ENABLE_GPU_ACCELERATION = get_bool(os.getenv('ENABLE_GPU_ACCELERATION'), True)
    FP16_PRECISION = get_bool(os.getenv('FP16_PRECISION'), True)

    # Translation Settings
    TRANSLATION_NUM_BEAMS = get_int(os.getenv('TRANSLATION_NUM_BEAMS'), 10)
    TRANSLATION_MAX_LENGTH = get_int(os.getenv('TRANSLATION_MAX_LENGTH'), 512)

    # ============================================
    # STORAGE & PATHS
    # ============================================

    UPLOAD_FOLDER = Path(os.getenv('UPLOAD_FOLDER', './uploads'))
    TEMP_FOLDER = Path(os.getenv('TEMP_FOLDER', './temp'))
    MODEL_CACHE_DIR = Path(os.getenv('MODEL_CACHE_DIR', './models'))

    # Create directories if they don't exist
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================
    # LOGGING & DEBUG
    # ============================================

    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG_MODE = get_bool(os.getenv('DEBUG_MODE'), True)
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    # ============================================
    # OPTIMIZED BACKEND (Enhanced Processing)
    # ============================================

    USE_OPTIMIZED_BACKEND = get_bool(os.getenv('USE_OPTIMIZED_BACKEND'), True)
    # Enhanced processing for better performance and accuracy

    # True Offline Mode - disable all API calls
    TRUE_OFFLINE_MODE = get_bool(os.getenv('TRUE_OFFLINE_MODE'), False)

    # ============================================
    # COMPUTED PROPERTIES
    # ============================================

    @property
    def is_production(self):
        """Check if running in production mode."""
        return self.FLASK_ENV == 'production'

    @property
    def use_cuda(self):
        """Check if CUDA should be used."""
        import torch
        if self.WHISPER_DEVICE == 'auto' or self.NLLB_DEVICE == 'auto':
            return torch.cuda.is_available() and self.ENABLE_GPU_ACCELERATION
        return 'cuda' in [self.WHISPER_DEVICE, self.NLLB_DEVICE] and self.ENABLE_GPU_ACCELERATION

    def get_whisper_model_name(self):
        """Get Whisper model name/path based on configuration."""
        if self.WHISPER_MODEL_PATH:
            # Custom fine-tuned model
            return str(self.MODEL_CACHE_DIR / self.WHISPER_MODEL_PATH)
        else:
            # Base model from OpenAI
            return self.WHISPER_MODEL_TYPE

    def get_device(self, model_type='whisper'):
        """Get device for specific model."""
        import torch

        if model_type == 'whisper':
            device = self.WHISPER_DEVICE
        elif model_type == 'nllb':
            device = self.NLLB_DEVICE
        else:
            device = 'auto'

        # Always check CUDA availability first
        cuda_available = torch.cuda.is_available()

        if device == 'auto':
            return 'cuda' if cuda_available and self.ENABLE_GPU_ACCELERATION else 'cpu'

        # Force CPU if CUDA not available, regardless of config
        if device == 'cuda' and not cuda_available:
            return 'cpu'

        return device if self.ENABLE_GPU_ACCELERATION else 'cpu'

    def to_dict(self):
        """Convert config to dictionary for display."""
        return {
            'flask': {
                'environment': self.FLASK_ENV,
                'debug': self.DEBUG_MODE,
                'jwt_expiration_days': self.JWT_EXPIRATION_DAYS,
            },
            'models': {
                'whisper': {
                    'type': self.WHISPER_MODEL_TYPE,
                    'path': self.WHISPER_MODEL_PATH or 'auto-download',
                    'device': self.get_device('whisper'),
                    'compute_type': self.WHISPER_COMPUTE_TYPE,
                },
                'nllb': {
                    'name': self.NLLB_MODEL_NAME,
                    'device': self.get_device('nllb'),
                },
                'visual': {
                    'enabled': self.USE_VISUAL_AUGMENTATION,
                    'complexity': self.MEDIAPIPE_MODEL_COMPLEXITY,
                }
            },
            'limits': {
                'max_audio_mb': self.MAX_AUDIO_SIZE_MB,
                'max_video_mb': self.MAX_VIDEO_SIZE_MB,
            },
            'performance': {
                'gpu_acceleration': self.ENABLE_GPU_ACCELERATION,
                'fp16_precision': self.FP16_PRECISION,
                'translation_beams': self.TRANSLATION_NUM_BEAMS,
            }
        }


# Global config instance
config = Config()


# Export for easy import
__all__ = ['config', 'Config']
