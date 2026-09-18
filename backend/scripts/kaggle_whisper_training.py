#!/usr/bin/env python3
"""
Whisper Medium Training for Kaggle - Clean Import Version
==========================================================

Usage in Kaggle Notebook:

Cell 1 - Install dependencies:
    !pip uninstall -y datasets
    !pip install datasets==2.21.0 soundfile librosa transformers evaluate jiwer
    # Then click "Restart Session"

Cell 2 - Run training:
    !wget -q https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/scripts/kaggle_whisper_training.py
    from kaggle_whisper_training import run_training
    run_training()

Or if you upload this file to Kaggle dataset:
    import sys
    sys.path.append('/kaggle/input/your-scripts-dataset')
    from kaggle_whisper_training import run_training
    run_training()
"""

import os
import sys
import json
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Kaggle paths
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


@dataclass
class TrainingConfig:
    """Training config for Whisper on Kaggle P100."""

    # Model
    model_name: str = "openai/whisper-medium"
    language: str = "sun"
    task: str = "transcribe"

    # Output
    output_dir: Path = KAGGLE_WORKING / "whisper-medium-sundanese"

    # Training params (optimized for P100 16GB)
    max_steps: int = 5000
    batch_size: int = 8
    eval_batch_size: int = 4
    gradient_accumulation: int = 2  # Effective batch = 16
    learning_rate: float = 1e-5
    warmup_steps: int = 500

    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50

    # Performance
    fp16: bool = True
    gradient_checkpointing: bool = True


def find_dataset():
    """Find dataset in Kaggle input."""
    search_paths = [
        KAGGLE_INPUT / "openslr36-sundanese-asr-prepared" / "kaggle_dataset" / "manifests",
        KAGGLE_INPUT / "openslr36-sundanese-asr-prepared" / "manifests",
        KAGGLE_INPUT / "openslr36-sundanese-asr-prepared",
    ]

    for path in search_paths:
        if (path / "train.json").exists():
            print(f"Found dataset: {path}")
            return path

    # Fallback: scan all
    for d in KAGGLE_INPUT.iterdir():
        if d.is_dir():
            for sub in ["kaggle_dataset/manifests", "manifests", ""]:
                check = d / sub if sub else d
                if (check / "train.json").exists():
                    print(f"Found dataset: {check}")
                    return check

    raise FileNotFoundError("Dataset not found! Add it to your Kaggle notebook.")


def load_manifest(path: Path, audio_base: Path = None):
    """Load dataset from manifest JSON."""
    from datasets import Dataset, Audio

    print(f"Loading {path.name}...")

    with open(path, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    # Fix audio paths
    audio_paths = []
    for e in entries:
        ap = e['audio_path']

        if audio_base and '/openslr36-audio/' in ap:
            rel = ap.split('/openslr36-audio/')[-1]
            ap = str(audio_base / rel)

        audio_paths.append(ap.replace('\\', '/'))

    dataset = Dataset.from_dict({
        'audio': audio_paths,
        'transcription': [e['transcription'] for e in entries],
    })
    dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))

    print(f"  Loaded {len(dataset):,} samples")
    return dataset


def prepare_batch(batch, processor):
    """Prepare single batch for training."""
    audio = batch["audio"]

    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids

    return batch


@dataclass
class DataCollator:
    """Data collator for Whisper."""
    processor: Any

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def compute_wer(pred, processor, metric):
    """Compute WER metric."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    return {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}


def run_training(config: TrainingConfig = None):
    """Main training function."""

    if config is None:
        config = TrainingConfig()

    # Ensure dependencies are installed
    try:
        import evaluate
    except ImportError:
        print("Installing missing dependencies...")
        os.system("pip install -q evaluate jiwer")
        import evaluate

    print("=" * 60)
    print("Whisper Medium Training on Kaggle")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU! Enable GPU in Settings -> Accelerator")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Find dataset
    dataset_dir = find_dataset()
    audio_base = dataset_dir.parent / "audio" if dataset_dir.name == "manifests" else dataset_dir / "audio"

    if not audio_base.exists():
        print(f"Warning: Audio not found at {audio_base}")
        audio_base = None

    # Create output dir
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Imports
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    import evaluate

    # Load model
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
    train_data = load_manifest(dataset_dir / "train.json", audio_base)
    eval_data = load_manifest(dataset_dir / "validation.json", audio_base)

    # Prepare datasets
    print("\nPreparing datasets...")
    train_data = train_data.map(
        lambda x: prepare_batch(x, processor),
        remove_columns=train_data.column_names,
        num_proc=2,
        desc="Preparing train",
    )
    eval_data = eval_data.map(
        lambda x: prepare_batch(x, processor),
        remove_columns=eval_data.column_names,
        num_proc=2,
        desc="Preparing eval",
    )

    print(f"  Train: {len(train_data):,}")
    print(f"  Eval: {len(eval_data):,}")

    # Setup training
    data_collator = DataCollator(processor=processor)
    metric = evaluate.load("wer")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(config.output_dir),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        fp16=config.fp16,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        dataloader_num_workers=2,
        report_to=["tensorboard"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_wer(pred, processor, metric),
        processing_class=processor.feature_extractor,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting Training...")
    print(f"  Batch: {config.batch_size} x {config.gradient_accumulation} = {config.batch_size * config.gradient_accumulation}")
    print(f"  Steps: {config.max_steps}")
    print(f"  LR: {config.learning_rate}")
    print("=" * 60 + "\n")

    trainer.train()

    # Save
    print("\nSaving model...")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)

    # Test evaluation
    test_file = dataset_dir / "test.json"
    if test_file.exists():
        print("\nEvaluating on test set...")
        test_data = load_manifest(test_file, audio_base)
        test_data = test_data.map(
            lambda x: prepare_batch(x, processor),
            remove_columns=test_data.column_names,
            num_proc=2,
        )
        results = trainer.evaluate(test_data)
        print(f"Test WER: {results['eval_wer']:.2f}%")

        with open(config.output_dir / "test_results.json", 'w') as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Model saved to: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    run_training()
