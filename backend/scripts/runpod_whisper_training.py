#!/usr/bin/env python3
"""
Whisper Medium Training for RunPod
===================================
Optimized for RTX 4090/3090/A100 GPUs

SETUP DI RUNPOD:
1. Pilih template: RunPod Pytorch 2.1 atau pytorch/pytorch
2. GPU: RTX 4090 ($0.74/hr) atau RTX 3090 ($0.44/hr)
3. Disk: 50GB (cukup untuk model + dataset)
4. Cloud Type: Secure Cloud (lebih stabil)

STEP 1 - Upload dataset ke HuggingFace Hub dulu (run lokal):
    python scripts/upload_dataset_to_hub.py

STEP 2 - Di RunPod terminal, run:
    pip install -q datasets transformers evaluate jiwer accelerate soundfile librosa
    wget https://raw.githubusercontent.com/YOUR_REPO/main/scripts/runpod_whisper_training.py
    python runpod_whisper_training.py

Atau copy-paste code ini langsung ke terminal RunPod.
"""

import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

# ============================================
# CONFIGURATION
# ============================================

@dataclass
class RunPodConfig:
    """Training config optimized for RunPod GPUs."""

    # Model
    model_name: str = "openai/whisper-medium"  # 769M params
    language: str = "sun"  # Sundanese
    task: str = "transcribe"

    # Dataset
    dataset_name: str = "saddadnabbil/openslr36-sundanese-asr"

    # Output
    output_dir: Path = Path("./whisper-medium-sundanese")
    hub_model_id: str = "saddadnabbil/whisper-medium-sundanese"  # For push to hub

    # Training (optimized for 24GB VRAM - RTX 4090/3090)
    max_steps: int = 5000
    per_device_train_batch_size: int = 16  # Higher for 24GB
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1   # Effective batch = 16
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01

    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 25
    save_total_limit: int = 2

    # Performance
    fp16: bool = True
    bf16: bool = False  # Set True for A100
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4

    # Hub
    push_to_hub: bool = True
    hub_strategy: str = "checkpoint"


config = RunPodConfig()


# ============================================
# HELPER FUNCTIONS
# ============================================

def prepare_batch(batch, processor):
    """Prepare batch for training."""
    audio = batch["audio"]

    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids

    return batch


@dataclass
class DataCollatorSpeechSeq2Seq:
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


# ============================================
# MAIN TRAINING
# ============================================

def main():
    print("=" * 70)
    print("Whisper Medium Training on RunPod")
    print("=" * 70)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected!")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n✓ GPU: {gpu_name} ({gpu_memory:.1f} GB)")

    # Auto-adjust for A100
    if "A100" in gpu_name:
        config.bf16 = True
        config.fp16 = False
        config.per_device_train_batch_size = 32
        print("  -> A100 detected: Using BF16 and batch_size=32")

    # Imports
    from datasets import load_dataset, Audio
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    import evaluate

    # Load dataset from HuggingFace Hub
    print(f"\nLoading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name)

    print(f"  Train: {len(dataset['train']):,} samples")
    print(f"  Validation: {len(dataset['validation']):,} samples")

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

    print(f"  Parameters: {model.num_parameters():,}")

    # Prepare datasets
    print("\nPreparing datasets...")

    # Ensure audio is loaded correctly
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    train_dataset = dataset["train"].map(
        lambda x: prepare_batch(x, processor),
        remove_columns=dataset["train"].column_names,
        num_proc=4,
        desc="Preparing train",
    )

    eval_dataset = dataset["validation"].map(
        lambda x: prepare_batch(x, processor),
        remove_columns=dataset["validation"].column_names,
        num_proc=4,
        desc="Preparing eval",
    )

    print(f"  Train prepared: {len(train_dataset):,}")
    print(f"  Eval prepared: {len(eval_dataset):,}")

    # Setup training
    config.output_dir.mkdir(parents=True, exist_ok=True)

    data_collator = DataCollatorSpeechSeq2Seq(processor=processor)
    metric = evaluate.load("wer")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(config.output_dir),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        bf16=config.bf16,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        dataloader_num_workers=config.dataloader_num_workers,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id if config.push_to_hub else None,
        hub_strategy=config.hub_strategy,
        report_to=["tensorboard"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_wer(pred, processor, metric),
        processing_class=processor.feature_extractor,
    )

    # Train
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    print(f"  Model: {config.model_name}")
    print(f"  Batch: {config.per_device_train_batch_size}")
    print(f"  Steps: {config.max_steps}")
    print(f"  LR: {config.learning_rate}")
    print(f"  FP16: {config.fp16}, BF16: {config.bf16}")
    print()

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)

    # Evaluate on test set if available
    if "test" in dataset:
        print("\nEvaluating on test set...")
        test_dataset = dataset["test"].map(
            lambda x: prepare_batch(x, processor),
            remove_columns=dataset["test"].column_names,
            num_proc=4,
        )

        test_results = trainer.evaluate(test_dataset)
        print(f"\n✓ Test WER: {test_results['eval_wer']:.2f}%")

        with open(config.output_dir / "test_results.json", 'w') as f:
            json.dump(test_results, f, indent=2)

    # Push to hub
    if config.push_to_hub:
        print(f"\nPushing to Hub: {config.hub_model_id}")
        trainer.push_to_hub()

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nModel saved to: {config.output_dir}")
    if config.push_to_hub:
        print(f"Model on Hub: https://huggingface.co/{config.hub_model_id}")


if __name__ == "__main__":
    main()
