#!/usr/bin/env python3
"""
Whisper Medium Training for Kaggle Environment
===============================================
Optimized for Kaggle P100 GPU (16GB VRAM)

IMPORTANT: Run this in Cell 1 first, then restart kernel:
    !pip uninstall -y datasets
    !pip install datasets==2.21.0 soundfile librosa transformers evaluate jiwer

Then run this script in Cell 2.
"""

import os
import sys
import json
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

# Kaggle paths
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

# ============================================
# CONFIGURATION FOR KAGGLE P100
# ============================================

@dataclass
class KaggleTrainingConfig:
    """Training config optimized for Kaggle P100 with Whisper Medium."""

    # Model - Using Medium for better accuracy
    model_name: str = "openai/whisper-medium"
    language: str = "sun"
    task: str = "transcribe"

    # Output path
    output_dir: Path = KAGGLE_WORKING / "whisper-medium-sundanese"

    # Training (optimized for Medium on P100 16GB)
    max_steps: int = 5000
    per_device_train_batch_size: int = 8   # Reduced for Medium model
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2   # Effective batch = 8*2 = 16
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 3

    # Performance
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    optim: str = "adamw_torch"

    # Generation
    generation_max_length: int = 225
    predict_with_generate: bool = True

    # Misc
    seed: int = 42
    push_to_hub: bool = False


config = KaggleTrainingConfig()


# ============================================
# DATASET LOADING
# ============================================

def find_dataset_path():
    """Auto-find dataset in Kaggle input."""
    print("Searching for dataset in /kaggle/input...")

    # Check specific paths first (your dataset structure)
    specific_paths = [
        KAGGLE_INPUT / "openslr36-sundanese-asr-prepared" / "kaggle_dataset" / "manifests",
        KAGGLE_INPUT / "openslr36-sundanese-asr-prepared" / "manifests",
        KAGGLE_INPUT / "openslr36-sundanese-asr-prepared",
    ]

    for path in specific_paths:
        print(f"  Checking: {path}")
        train_file = path / "train.json"
        val_file = path / "validation.json"

        if train_file.exists() and val_file.exists():
            print(f"  ✓ Found dataset at: {path}")
            return path

    # Fallback: scan all datasets
    print("  Scanning all datasets...")
    for dataset_dir in KAGGLE_INPUT.iterdir():
        if dataset_dir.is_dir():
            print(f"  Found: {dataset_dir}")

            # Check multiple possible locations
            check_paths = [
                dataset_dir / "kaggle_dataset" / "manifests",
                dataset_dir / "manifests",
                dataset_dir,
            ]

            for check_path in check_paths:
                train_file = check_path / "train.json"
                val_file = check_path / "validation.json"

                if train_file.exists() and val_file.exists():
                    print(f"  ✓ Using dataset: {check_path}")
                    return check_path

    raise FileNotFoundError(
        "Dataset not found! Please add dataset to notebook:\n"
        "1. Click 'Add Data' in Kaggle\n"
        "2. Search for your uploaded dataset\n"
        "3. Add to notebook\n\n"
        "Expected structure:\n"
        "  /kaggle/input/your-dataset/kaggle_dataset/manifests/train.json"
    )


def load_dataset_from_manifest(manifest_path: Path, audio_base_path: Path = None):
    """Load dataset from JSON manifest."""
    from datasets import Dataset, Audio

    print(f"Loading {manifest_path.name}...")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    # Fix audio paths if needed
    audio_paths = []
    for e in entries:
        audio_path = e['audio_path']

        # If audio_base_path provided, remap paths
        if audio_base_path:
            # Extract relative path from original path
            # Original: /kaggle/input/openslr36-audio/asr_sundanese_a/asr_sundanese/data/aa/file.flac
            # We need: /kaggle/input/openslr36-sundanese-asr-prepared/kaggle_dataset/audio/asr_sundanese_a/...
            if '/openslr36-audio/' in audio_path:
                rel_path = audio_path.split('/openslr36-audio/')[-1]
                audio_path = str(audio_base_path / rel_path)
            elif '\\openslr36-audio\\' in audio_path:
                rel_path = audio_path.split('\\openslr36-audio\\')[-1]
                audio_path = str(audio_base_path / rel_path)

        # Fix Windows backslashes to forward slashes for Linux/Kaggle
        audio_path = audio_path.replace('\\', '/')
        audio_paths.append(audio_path)

    # Convert to HuggingFace Dataset
    data = {
        'audio': audio_paths,
        'transcription': [e['transcription'] for e in entries],
    }

    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))

    print(f"  Loaded {len(dataset):,} samples")
    if audio_paths:
        print(f"  Sample audio path: {audio_paths[0]}")
    return dataset


def prepare_dataset(batch, processor):
    """Prepare batch for training."""
    audio = batch["audio"]

    # Compute input features
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Encode target text
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids

    return batch


# ============================================
# DATA COLLATOR
# ============================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for Whisper."""

    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ============================================
# METRICS
# ============================================

def compute_metrics(pred, processor, metric):
    """Compute WER."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 70)
    print("Whisper Medium Training on Kaggle P100")
    print("=" * 70)

    # Check GPU
    if not torch.cuda.is_available():
        print("\n⚠️ WARNING: No GPU detected!")
        print("Enable GPU: Settings → Accelerator → GPU P100")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n✓ GPU: {gpu_name} ({gpu_memory:.1f} GB)")

    # Find dataset
    dataset_dir = find_dataset_path()

    # Create output dir
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Import dependencies
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    import evaluate

    # Load processor and model
    print(f"\nLoading {config.model_name}...")
    processor = WhisperProcessor.from_pretrained(
        config.model_name,
        language=config.language,
        task=config.task
    )

    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Load datasets
    print("\nLoading datasets...")

    # Determine audio base path (sibling to manifests folder)
    # Structure: .../kaggle_dataset/manifests/train.json
    #            .../kaggle_dataset/audio/...
    if dataset_dir.name == "manifests":
        audio_base = dataset_dir.parent / "audio"
    else:
        audio_base = dataset_dir / "audio"

    if audio_base.exists():
        print(f"  Audio base path: {audio_base}")
    else:
        print(f"  Warning: Audio path not found at {audio_base}")
        audio_base = None  # Let it use paths from manifest as-is

    train_dataset = load_dataset_from_manifest(dataset_dir / "train.json", audio_base)
    eval_dataset = load_dataset_from_manifest(dataset_dir / "validation.json", audio_base)

    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = train_dataset.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=train_dataset.column_names,
        num_proc=2,
        desc="Preparing train",
    )
    eval_dataset = eval_dataset.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=eval_dataset.column_names,
        num_proc=2,
        desc="Preparing eval",
    )

    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Eval: {len(eval_dataset):,} samples")

    # Data collator and metric
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")

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
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, metric),
        processing_class=processor.feature_extractor,
    )

    # Train
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    print(f"Config:")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Save every: {config.save_steps} steps")
    print()

    try:
        trainer.train()

        # Save final model
        print("\nSaving final model...")
        trainer.save_model()
        processor.save_pretrained(config.output_dir)

        # Evaluate on test set if available
        test_file = dataset_dir / "test.json"
        if test_file.exists():
            print("\nEvaluating on test set...")
            test_dataset = load_dataset_from_manifest(test_file, audio_base)
            test_dataset = test_dataset.map(
                lambda x: prepare_dataset(x, processor),
                remove_columns=test_dataset.column_names,
                num_proc=2,
            )

            test_results = trainer.evaluate(test_dataset)
            print(f"\n✓ Test WER: {test_results['eval_wer']:.2f}%")

            with open(config.output_dir / "test_results.json", 'w') as f:
                json.dump(test_results, f, indent=2)

        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"\nModel saved to: {config.output_dir}")
        print("\nTo download:")
        print("1. Click 'Save Version' in Kaggle")
        print("2. Download output from 'Data' tab")

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()

        # Try to save checkpoint
        try:
            trainer.save_model(config.output_dir / "checkpoint-error")
            print("\n[INFO] Error checkpoint saved")
        except:
            pass

        raise


if __name__ == "__main__":
    main()
