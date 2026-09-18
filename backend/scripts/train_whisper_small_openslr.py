#!/usr/bin/env python3
"""
Train Whisper Small on OpenSLR 36 Sundanese Dataset
====================================================
Fine-tunes Whisper Small for Sundanese ASR following Raharjo & Zahra (2025).

Features:
    - Auto-resume from checkpoint: continues training if interrupted
    - Robust error handling: saves progress on any failure
    - Memory optimized: gradient checkpointing + FP16 for RTX 3070 8GB
    - Best model tracking: automatically saves best model based on WER

Hardware Requirements:
    - GPU: RTX 3070 8GB (or similar)
    - RAM: 16GB+
    - Disk: ~10GB for model checkpoints

Training Configuration:
    - Model: openai/whisper-small (244M params)
    - Dataset: OpenSLR 36 (~220K utterances)
    - Steps: 5000 (like Raharjo paper)
    - Expected WER: ~2-5%

Usage:
    python scripts/train_whisper_small_openslr.py

    # Resume from checkpoint:
    python scripts/train_whisper_small_openslr.py --resume

Reference:
    Raharjo, A., & Zahra, A. (2025). Javanese and Sundanese speech recognition
    using Whisper. Computer Science and Information Technologies, 6(3), 253–261.
"""

import os
import sys
import json
import torch
import logging
import argparse
import gc
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union, Optional
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

# ============================================
# CONFIGURATION
# ============================================

@dataclass
class TrainingConfig:
    """Training configuration matching Raharjo paper."""

    # Model
    model_name: str = "openai/whisper-small"  # 244M params
    language: str = "sun"  # Sundanese ISO code
    task: str = "transcribe"

    # Paths
    dataset_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "backend" / "dataset" / "openslr36_prepared")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "backend" / "models" / "whisper-small-openslr")

    # Training hyperparameters (Raharjo-style)
    max_steps: int = 5000
    per_device_train_batch_size: int = 4  # Safe for RTX 3070 8GB
    per_device_eval_batch_size: int = 2   # Reduced for evaluation stability
    gradient_accumulation_steps: int = 8  # Effective batch = 32
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 25
    save_total_limit: int = 3  # Keep last 3 checkpoints

    # Performance
    fp16: bool = True  # Mixed precision
    gradient_checkpointing: bool = True  # Save VRAM
    dataloader_num_workers: int = 2  # Reduced for stability
    optim: str = "adamw_torch"

    # Generation
    generation_max_length: int = 225
    predict_with_generate: bool = True

    # Misc
    seed: int = 42
    push_to_hub: bool = False

    def __post_init__(self):
        # Convert to Path if string
        if isinstance(self.dataset_dir, str):
            self.dataset_dir = Path(self.dataset_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


# ============================================
# LOGGING SETUP
# ============================================

def setup_logging(output_dir: Path):
    """Setup logging with file and console handlers."""
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'training.log'

    # Create formatters and handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')  # Append mode for resume
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ============================================
# DATASET LOADING
# ============================================

def load_dataset_from_manifest(manifest_path: Path, logger, max_samples: int = None):
    """Load dataset from JSON manifest."""
    from datasets import Dataset, Audio

    logger.info(f"Loading dataset from {manifest_path}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    # Limit samples if specified
    if max_samples and len(entries) > max_samples:
        logger.info(f"Limiting to {max_samples} samples (from {len(entries)})")
        entries = entries[:max_samples]

    # Convert to HuggingFace Dataset format (skip validation for speed)
    data = {
        'audio': [e['audio_path'] for e in entries],
        'transcription': [e['transcription'] for e in entries],
    }

    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))

    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def prepare_dataset(batch, processor):
    """Prepare single example for training."""
    import numpy as np

    audio = batch["audio"]
    audio_array = audio["array"]

    # Convert to float32 and ensure correct shape
    if isinstance(audio_array, np.ndarray):
        audio_array = audio_array.astype(np.float32)

    # Compute input features
    batch["input_features"] = processor.feature_extractor(
        audio_array,
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Encode target text
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids

    return batch


# ============================================
# METRICS
# ============================================

def compute_metrics(pred, processor, metric):
    """Compute WER metric."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# ============================================
# DATA COLLATOR
# ============================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for Whisper training."""

    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 for loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# ============================================
# CHECKPOINT UTILITIES
# ============================================

def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in output directory."""
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None

    # Sort by step number
    def get_step(cp):
        try:
            return int(cp.name.split("-")[1])
        except:
            return 0

    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints[0]


def save_training_state(output_dir: Path, state: dict):
    """Save training state for manual recovery."""
    state_file = output_dir / "training_state.json"
    state["saved_at"] = datetime.now().isoformat()

    try:
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save training state: {e}")


# ============================================
# GPU UTILITIES
# ============================================

def check_gpu():
    """Check GPU availability and memory."""
    if not torch.cuda.is_available():
        return None, None, None

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_memory_free = (torch.cuda.get_device_properties(0).total_memory -
                       torch.cuda.memory_allocated(0)) / 1e9

    return gpu_name, gpu_memory_total, gpu_memory_free


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


# ============================================
# MAIN TRAINING FUNCTION
# ============================================

def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Whisper on OpenSLR 36")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint to resume from")
    args = parser.parse_args()

    # Initialize config
    config = TrainingConfig()

    # Setup logging
    logger = setup_logging(config.output_dir)

    print("=" * 70)
    print("Whisper Small Fine-tuning for Sundanese (OpenSLR 36)")
    print("=" * 70)
    logger.info("Starting training script")

    # Check GPU
    gpu_name, gpu_total, gpu_free = check_gpu()
    if gpu_name is None:
        logger.error("CUDA not available! Training requires GPU.")
        print("\n[ERROR] No GPU detected. Training requires CUDA-capable GPU.")
        sys.exit(1)

    logger.info(f"GPU: {gpu_name} ({gpu_total:.1f} GB total, {gpu_free:.1f} GB free)")

    # Check if we have enough VRAM
    if gpu_total < 6:
        logger.warning(f"Low VRAM detected ({gpu_total:.1f} GB). May encounter OOM errors.")
        # Reduce batch size automatically
        config.per_device_train_batch_size = 2
        config.per_device_eval_batch_size = 1
        logger.info("Automatically reduced batch sizes for low VRAM")

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Language: {config.language}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Batch size: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  FP16: {config.fp16}")
    print(f"  Gradient checkpointing: {config.gradient_checkpointing}")
    print(f"  Save every: {config.save_steps} steps")
    print()

    # Check dataset
    if not config.dataset_dir.exists():
        logger.error(f"Dataset not found: {config.dataset_dir}")
        print("\n[ERROR] Dataset directory not found!")
        print("Please run: python scripts/prepare_openslr36_dataset.py")
        sys.exit(1)

    # Check for required files
    required_files = ['train.json', 'validation.json']
    for rf in required_files:
        if not (config.dataset_dir / rf).exists():
            logger.error(f"Missing required file: {rf}")
            print(f"\n[ERROR] Missing {rf} in dataset directory!")
            sys.exit(1)

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Check for resume
    resume_checkpoint = None
    if args.checkpoint:
        resume_checkpoint = Path(args.checkpoint)
        if not resume_checkpoint.exists():
            logger.error(f"Checkpoint not found: {resume_checkpoint}")
            sys.exit(1)
    elif args.resume:
        resume_checkpoint = find_latest_checkpoint(config.output_dir)
        if resume_checkpoint:
            logger.info(f"Found checkpoint to resume: {resume_checkpoint}")
        else:
            logger.info("No checkpoint found, starting from scratch")

    # Import dependencies (after GPU check)
    logger.info("Loading dependencies...")
    try:
        from transformers import (
            WhisperProcessor,
            WhisperForConditionalGeneration,
            Seq2SeqTrainingArguments,
            Seq2SeqTrainer,
        )
        import evaluate
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        print("\n[ERROR] Missing required packages. Please install:")
        print("  pip install transformers datasets evaluate")
        sys.exit(1)

    # Load processor
    logger.info(f"Loading processor from {config.model_name}...")
    try:
        processor = WhisperProcessor.from_pretrained(
            config.model_name,
            language=config.language,
            task=config.task
        )
    except Exception as e:
        logger.error(f"Failed to load processor: {e}")
        print("\n[ERROR] Could not load Whisper processor. Check internet connection.")
        sys.exit(1)

    # Load model
    logger.info(f"Loading model from {config.model_name}...")
    try:
        model = WhisperForConditionalGeneration.from_pretrained(config.model_name)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        print("\n[ERROR] Could not load Whisper model. Check internet connection.")
        sys.exit(1)

    # Configure model
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # Required for gradient checkpointing

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Clear memory before loading datasets
    clear_gpu_memory()

    # Load datasets
    logger.info("Loading datasets...")

    # MEMORY OPTIMIZATION: Limit dataset size for machines with limited RAM
    # For full dataset (164K samples), you need ~24GB+ RAM
    # For subset (20K samples), you need ~8GB RAM
    MAX_TRAIN_SAMPLES = 20000  # Set to None for full dataset
    MAX_EVAL_SAMPLES = 2000    # Set to None for full dataset

    if MAX_TRAIN_SAMPLES:
        logger.info(f"[MEMORY MODE] Limiting training to {MAX_TRAIN_SAMPLES:,} samples")
        logger.info("For full dataset, run on machine with 24GB+ RAM or use cloud GPU")

    try:
        train_dataset = load_dataset_from_manifest(
            config.dataset_dir / "train.json", logger, max_samples=MAX_TRAIN_SAMPLES
        )
        eval_dataset = load_dataset_from_manifest(
            config.dataset_dir / "validation.json", logger, max_samples=MAX_EVAL_SAMPLES
        )
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        print(f"\n[ERROR] Could not load datasets: {e}")
        sys.exit(1)

    # Prepare datasets with caching
    logger.info("Preparing datasets (this may take a while)...")
    logger.info("Using disk-based caching to save RAM")

    cache_dir = config.output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load from cache if exists, otherwise process and cache
        # Use different cache for subset vs full dataset
        cache_suffix = f"_{len(train_dataset)}" if MAX_TRAIN_SAMPLES else "_full"
        train_cache = cache_dir / f"train_processed{cache_suffix}"
        eval_cache = cache_dir / f"eval_processed{cache_suffix}"

        if train_cache.exists():
            logger.info("Loading train dataset from cache...")
            from datasets import load_from_disk
            train_dataset = load_from_disk(str(train_cache))
        else:
            logger.info("Processing train dataset (will be cached for future use)...")
            train_dataset = train_dataset.map(
                lambda x: prepare_dataset(x, processor),
                remove_columns=train_dataset.column_names,
                num_proc=1,
                writer_batch_size=100,  # Small batches to reduce memory
                keep_in_memory=False,  # Write to disk immediately
                desc="Preparing train dataset",
            )
            logger.info("Saving processed train dataset to cache...")
            train_dataset.save_to_disk(str(train_cache))

        if eval_cache.exists():
            logger.info("Loading eval dataset from cache...")
            from datasets import load_from_disk
            eval_dataset = load_from_disk(str(eval_cache))
        else:
            logger.info("Processing eval dataset (will be cached for future use)...")
            eval_dataset = eval_dataset.map(
                lambda x: prepare_dataset(x, processor),
                remove_columns=eval_dataset.column_names,
                num_proc=1,
                writer_batch_size=100,
                keep_in_memory=False,
                desc="Preparing eval dataset",
            )
            logger.info("Saving processed eval dataset to cache...")
            eval_dataset.save_to_disk(str(eval_cache))

    except Exception as e:
        logger.error(f"Failed to prepare datasets: {e}")
        print(f"\n[ERROR] Dataset preparation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Metric
    try:
        metric = evaluate.load("wer")
    except Exception as e:
        logger.warning(f"Could not load WER metric: {e}")
        metric = None

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(config.output_dir),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=config.predict_with_generate,
        generation_max_length=config.generation_max_length,
        dataloader_num_workers=config.dataloader_num_workers,
        optim=config.optim,
        seed=config.seed,
        report_to=["tensorboard"],
        push_to_hub=config.push_to_hub,
        # Resume settings
        ignore_data_skip=True,  # Speed up resume
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, metric) if metric else {},
        processing_class=processor.feature_extractor,
    )

    # Save initial state
    save_training_state(config.output_dir, {
        "status": "starting",
        "config": {
            "model_name": config.model_name,
            "max_steps": config.max_steps,
            "batch_size": config.per_device_train_batch_size,
            "learning_rate": config.learning_rate,
        },
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
    })

    # Train!
    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)
    print("\n[INFO] Training started. Checkpoints will be saved every", config.save_steps, "steps.")
    print("[INFO] Press Ctrl+C to safely stop and save progress.\n")

    try:
        # Resume from checkpoint if specified
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

        # Save metrics
        metrics = train_result.metrics
        logger.info(f"Training complete! Final metrics: {metrics}")

        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        processor.save_pretrained(config.output_dir)

        # Save training state
        save_training_state(config.output_dir, {
            "status": "completed",
            "metrics": metrics,
        })

        # Evaluate on test set if available
        test_manifest = config.dataset_dir / "test.json"
        if test_manifest.exists():
            logger.info("Evaluating on test set...")
            try:
                test_dataset = load_dataset_from_manifest(test_manifest, logger)
                test_dataset = test_dataset.map(
                    lambda x: prepare_dataset(x, processor),
                    remove_columns=test_dataset.column_names,
                    num_proc=1,
                )

                test_results = trainer.evaluate(test_dataset)
                logger.info(f"Test results: {test_results}")

                # Save results
                with open(config.output_dir / "test_results.json", 'w') as f:
                    json.dump(test_results, f, indent=2)

                print(f"\nTest WER: {test_results.get('eval_wer', 'N/A'):.2f}%")

            except Exception as e:
                logger.warning(f"Could not evaluate on test set: {e}")

        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"\nModel saved to: {config.output_dir}")
        print(f"\nTo use this model in the application:")
        print(f"  1. Set USE_FINETUNED_WHISPER=true in backend/.env")
        print(f"  2. Set FINETUNED_WHISPER_PATH={config.output_dir}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C)")
        print("\n\n[INFO] Training interrupted. Saving checkpoint...")

        # Save current state
        try:
            trainer.save_model(config.output_dir / "checkpoint-interrupted")
            save_training_state(config.output_dir, {"status": "interrupted"})
            print("[OK] Checkpoint saved to 'checkpoint-interrupted'")
            print("\nTo resume training, run:")
            print(f"  python {sys.argv[0]} --resume")
        except Exception as e:
            logger.error(f"Failed to save interrupted checkpoint: {e}")
            print(f"[WARN] Could not save checkpoint: {e}")

    except torch.cuda.OutOfMemoryError:
        logger.error("GPU out of memory!")
        clear_gpu_memory()
        print("\n[ERROR] GPU ran out of memory!")
        print("Try reducing batch size in the script:")
        print("  per_device_train_batch_size: int = 2")
        print("\nProgress has been saved. Resume with:")
        print(f"  python {sys.argv[0]} --resume")
        save_training_state(config.output_dir, {"status": "oom_error"})
        sys.exit(1)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

        # Try to save checkpoint
        try:
            trainer.save_model(config.output_dir / "checkpoint-error")
            save_training_state(config.output_dir, {
                "status": "error",
                "error": str(e)
            })
            print(f"\n[INFO] Error checkpoint saved. Resume with:")
            print(f"  python {sys.argv[0]} --resume")
        except:
            pass

        sys.exit(1)


if __name__ == "__main__":
    main()
