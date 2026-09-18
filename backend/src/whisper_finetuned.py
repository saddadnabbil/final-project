"""
Wrapper for fine-tuned Whisper model (checkpoint-1000)
This module provides integration of the fine-tuned Whisper Large v3 model
trained on Sundanese dataset.

Optimized for speed with:
- FP16 inference on GPU
- Optimized generation parameters
- Efficient memory management
"""

import os
import warnings

# Suppress transformers warnings BEFORE importing transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig
import logging

# Set transformers logging to error only
logging.getLogger("transformers").setLevel(logging.ERROR)


class FinetunedWhisperModel:
    """
    Fine-tuned Whisper model for Sundanese transcription.
    Supports both Whisper Large v3 and Whisper Small base models.

    Optimized for RTX 3070 with FP16 inference.
    """

    def __init__(self, model_path: str, device: str = "cuda", use_fp16: bool = True):
        """
        Initialize fine-tuned Whisper model with GPU optimization.

        Args:
            model_path: Path to fine-tuned model checkpoint
            device: Device to load model on ("cuda" or "cpu")
            use_fp16: Use FP16 for faster inference (GPU only)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device.type == "cuda"

        print(f"[FINETUNED-WHISPER] Loading from {model_path}...")
        print(f"[FINETUNED-WHISPER] Device: {self.device}")
        print(f"[FINETUNED-WHISPER] FP16 Mode: {self.use_fp16}")

        # Detect base model from config
        import json
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)

        d_model = config.get("d_model", 1280)
        if d_model == 1280:
            base_model = "openai/whisper-large-v3"
            self.model_type = "large-v3"
        elif d_model == 768:
            base_model = "openai/whisper-small"
            self.model_type = "small"
        elif d_model == 512:
            base_model = "openai/whisper-base"
            self.model_type = "base"
        else:
            base_model = "openai/whisper-small"
            self.model_type = "unknown"

        print(f"[FINETUNED-WHISPER] Detected base model: {base_model} (d_model={d_model})")

        # Load processor from base model (tokenizer files not in checkpoint)
        self.processor = WhisperProcessor.from_pretrained(
            base_model,
            language="su",  # Sundanese
            task="transcribe"
        )

        # Load fine-tuned model weights with optimization
        if self.use_fp16:
            # Load in FP16 for faster inference
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16
            )
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()

        # Configure generation settings to avoid warnings
        self.generation_config = GenerationConfig(
            max_length=128,
            num_beams=2,
            do_sample=False,
            use_cache=True,
            return_timestamps=False,
            task="transcribe",
            language="su",  # Sundanese
            suppress_tokens=[],
            begin_suppress_tokens=[220, 50257],
        )

        # NOTE: torch.compile disabled - causes ~2s warm-up overhead on first run
        # The compilation overhead is larger than the speedup for short audio
        # For production with many requests, consider enabling with mode="max-autotune"
        # if hasattr(torch, 'compile') and self.device.type == "cuda":
        #     self.model = torch.compile(self.model, mode="reduce-overhead")

        print(f"[FINETUNED-WHISPER] Model loaded successfully!")
        print(f"[FINETUNED-WHISPER] Model type: Whisper {self.model_type} (fine-tuned)")
        print(f"[FINETUNED-WHISPER] Parameters: {self.model.num_parameters():,}")
        print(f"[FINETUNED-WHISPER] Dataset: Sundanese speech samples")

        # Warm-up: Run dummy inference to pre-compile CUDA kernels
        if self.device.type == "cuda":
            self._warmup()

    def _warmup(self):
        """
        Warm-up inference to pre-compile CUDA kernels.
        This eliminates the first-run overhead during actual inference.
        """
        print("[FINETUNED-WHISPER] Warming up CUDA kernels...")
        try:
            # Create 3 seconds of dummy audio for more complete warm-up
            dummy_audio = np.random.randn(48000).astype(np.float32) * 0.01

            # Run full inference pipeline to warm up all kernels
            inputs = self.processor(
                dummy_audio,
                sampling_rate=16000,
                return_tensors="pt"
            )

            if self.use_fp16:
                inputs = inputs.input_features.to(self.device, dtype=torch.float16)
            else:
                inputs = inputs.input_features.to(self.device)

            # Run with same parameters as actual inference
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_fp16):
                _ = self.model.generate(
                    inputs,
                    max_length=50,  # Longer to warm up decoding loop
                    num_beams=1,
                    do_sample=False,
                    use_cache=True
                )

            # Also warm up encoder separately (used for feature extraction)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_fp16):
                _ = self.model.model.encoder(inputs)

            # Synchronize CUDA to ensure kernels are compiled
            torch.cuda.synchronize()
            print("[FINETUNED-WHISPER] Warm-up complete!")
        except Exception as e:
            print(f"[FINETUNED-WHISPER] Warm-up failed (non-critical): {e}")

    @property
    def encoder(self):
        """Expose encoder for audio feature extraction (used by visual fusion)."""
        return self.model.model.encoder

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "su",
        task: str = "transcribe",
        **kwargs
    ):
        """
        Transcribe audio using fine-tuned model.

        Args:
            audio: Audio waveform (numpy array, 16kHz mono)
            language: Language code ("su" for Sundanese)
            task: Task type ("transcribe")
            **kwargs: Additional generation parameters

        Returns:
            Generator of segments with transcription text
        """
        try:
            # Process in chunks if audio is too long
            max_chunk_duration = 30  # seconds
            chunk_samples = max_chunk_duration * 16000

            segments = []

            if len(audio) > chunk_samples:
                print(f"[FINETUNED-WHISPER] Audio longer than {max_chunk_duration}s, processing in chunks...")
                num_chunks = int(np.ceil(len(audio) / chunk_samples))

                for i in range(num_chunks):
                    start_sample = i * chunk_samples
                    end_sample = min((i + 1) * chunk_samples, len(audio))
                    chunk = audio[start_sample:end_sample]

                    # Process chunk
                    text = self._transcribe_chunk(chunk)

                    # Create segment object compatible with faster-whisper
                    segment = type('Segment', (), {
                        'text': text,
                        'start': start_sample / 16000,
                        'end': end_sample / 16000,
                        'avg_logprob': -0.1  # Mock confidence score
                    })()

                    segments.append(segment)
            else:
                # Process entire audio at once
                text = self._transcribe_chunk(audio)

                segment = type('Segment', (), {
                    'text': text,
                    'start': 0.0,
                    'end': len(audio) / 16000,
                    'avg_logprob': -0.1
                })()

                segments.append(segment)

            # Mock info object compatible with faster-whisper
            info = type('TranscriptionInfo', (), {
                'language': language,
                'language_probability': 0.95
            })()

            return (s for s in segments), info

        except Exception as e:
            print(f"[FINETUNED-WHISPER] Error during transcription: {e}")
            raise

    def _transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """
        Transcribe a single audio chunk with optimized inference.

        Args:
            audio_chunk: Audio waveform chunk

        Returns:
            Transcription text
        """
        # Prepare input
        inputs = self.processor(
            audio_chunk,
            sampling_rate=16000,
            return_tensors="pt"
        )

        # Move to device with correct dtype
        if self.use_fp16:
            inputs = inputs.input_features.to(self.device, dtype=torch.float16)
        else:
            inputs = inputs.input_features.to(self.device)

        # Generate transcription with optimized parameters
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_fp16):
            predicted_ids = self.model.generate(
                inputs,
                generation_config=self.generation_config,
            )

        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return transcription.strip()


def load_finetuned_whisper(model_path: str, device: str = "cuda"):
    """
    Load fine-tuned Whisper model.

    Args:
        model_path: Path to model checkpoint directory
        device: Device to load on

    Returns:
        FinetunedWhisperModel instance
    """
    # Convert relative path to absolute
    if not os.path.isabs(model_path):
        # Assume relative to backend directory
        backend_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(backend_dir, model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fine-tuned model not found at: {model_path}")

    return FinetunedWhisperModel(model_path, device=device)


class MultiModelManager:
    """
    Manager for multiple Whisper model variants.
    Supports: cp1500 (early fine-tuned), cp4500 (extended training), google (API)
    """

    def __init__(self):
        self._models = {}  # Cache loaded models
        self._current_variant = None

    def get_available_models(self):
        """Get list of available model variants."""
        from config import config
        import json

        available = []
        for variant, path in config.WHISPER_MODEL_PATHS.items():
            if variant == 'google':
                # Google API always available (if internet connection)
                available.append({
                    'id': 'google',
                    'name': 'Google Speech API',
                    'description': 'Enhanced Google Speech Recognition API for Sundanese',
                    'type': 'api',
                    'base_model': 'Google Cloud',
                    'status': 'available'
                })
            else:
                # Check if local model exists
                if path:
                    backend_dir = os.path.dirname(os.path.dirname(__file__))
                    full_path = os.path.join(backend_dir, path)
                    exists = os.path.exists(full_path)

                    # Detect base model type from config
                    base_model = "unknown"
                    if exists:
                        try:
                            config_path = os.path.join(full_path, "config.json")
                            with open(config_path, 'r') as f:
                                model_config = json.load(f)
                            d_model = model_config.get("d_model", 0)
                            if d_model == 1280:
                                base_model = "Whisper Large v3"
                            elif d_model == 768:
                                base_model = "Whisper Small"
                            elif d_model == 512:
                                base_model = "Whisper Base"
                        except:
                            pass

                    steps = '1500' if variant == 'cp1500' else '4500'
                    available.append({
                        'id': variant,
                        'name': f'Fine-tuned ({steps} steps)',
                        'description': f'{base_model} trained for {steps} steps on Sundanese',
                        'type': 'local',
                        'base_model': base_model,
                        'path': full_path,
                        'status': 'available' if exists else 'not_found'
                    })

        return available

    def load_model(self, variant: str, device: str = "cuda"):
        """
        Load a specific model variant.

        Args:
            variant: Model variant (cp1500, cp4500, google)
            device: Device to load on

        Returns:
            Model instance or None for google (API-based)
        """
        from config import config

        if variant == 'google':
            # Google API doesn't need a loaded model
            self._current_variant = 'google'
            print(f"[MULTI-MODEL] Selected Google Speech API")
            return None

        if variant in self._models:
            # Return cached model
            self._current_variant = variant
            print(f"[MULTI-MODEL] Using cached model: {variant}")
            return self._models[variant]

        # Load new model
        path = config.WHISPER_MODEL_PATHS.get(variant)
        if not path:
            raise ValueError(f"Unknown model variant: {variant}")

        print(f"[MULTI-MODEL] Loading model variant: {variant}")
        model = load_finetuned_whisper(path, device=device)
        self._models[variant] = model
        self._current_variant = variant

        return model

    def get_current_variant(self):
        """Get currently selected model variant."""
        return self._current_variant

    def transcribe(self, audio, variant: str = None, language: str = "su", **kwargs):
        """
        Transcribe audio using specified or current model variant.

        Args:
            audio: Audio data (numpy array or file path)
            variant: Model variant to use (optional, uses current if not specified)
            language: Language code
            **kwargs: Additional parameters

        Returns:
            Transcription result
        """
        from config import config

        variant = variant or self._current_variant or config.WHISPER_MODEL_VARIANT

        if variant == 'google':
            # Use Google Speech API via enhanced_backend_service
            from enhanced_backend_service import get_enhanced_backend_service
            service = get_enhanced_backend_service()

            # If audio is numpy array, need to save temporarily
            if isinstance(audio, np.ndarray):
                import tempfile
                import scipy.io.wavfile as wav
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    wav.write(f.name, 16000, (audio * 32767).astype(np.int16))
                    audio_path = f.name
            else:
                audio_path = audio

            lang_code = "su-ID" if language == "su" else "id-ID"
            result = service.transcribe_audio(audio_path, language=lang_code)

            # Return in compatible format
            segment = type('Segment', (), {
                'text': result,
                'start': 0.0,
                'end': 0.0,
                'avg_logprob': -0.1
            })()
            info = type('TranscriptionInfo', (), {
                'language': language,
                'language_probability': 0.95
            })()

            return iter([segment]), info
        else:
            # Use local fine-tuned model
            model = self.load_model(variant)
            return model.transcribe(audio, language=language, **kwargs)


# Singleton instance
_model_manager = None


def get_model_manager() -> MultiModelManager:
    """Get or create singleton MultiModelManager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = MultiModelManager()
    return _model_manager
