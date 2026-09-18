#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_real_data.py - Evaluasi model dengan dataset audio Sunda yang sebenarnya

Menggunakan dataset dari:
- backend/dataset/su_id_female/ (2,401 sampel)
- backend/dataset/su_id_male/ (1,812 sampel)
- backend/dataset/audio-visual-dataset-ready/ (video MP4)

Usage:
    python scripts/evaluate_real_data.py --samples 20
    python scripts/evaluate_real_data.py --samples 50 --include-video
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "src"))

# Add NVIDIA DLL paths for CUDA 12 (ctranslate2/faster-whisper)
# Check both root .venv and backend/.venv
VENV_PATHS = [
    PROJECT_ROOT / "backend" / ".venv",  # Backend venv (primary)
    PROJECT_ROOT / ".venv",               # Root venv (fallback)
]
for venv_path in VENV_PATHS:
    nvidia_base = venv_path / "Lib" / "site-packages" / "nvidia"
    if nvidia_base.exists():
        for subdir in ["cublas", "cudnn", "cuda_runtime"]:
            dll_path = nvidia_base / subdir / "bin"
            if dll_path.exists():
                try:
                    os.add_dll_directory(str(dll_path))
                except Exception:
                    pass  # Ignore if already added or not supported

# Dataset paths
DATASET_DIR = PROJECT_ROOT / "backend" / "dataset"
FEMALE_DIR = DATASET_DIR / "su_id_female"
MALE_DIR = DATASET_DIR / "su_id_male"
VIDEO_DIR = DATASET_DIR / "audio-visual-dataset-ready"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "tests" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_tsv_dataset(tsv_path: Path, wavs_dir: Path) -> List[Dict]:
    """
    Load dataset from TSV file.

    Format TSV: filename<TAB><empty><TAB>transcription

    Returns:
        List of {audio_path, transcription, sample_id}
    """
    samples = []

    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                filename = parts[0]
                transcription = parts[2]  # Sundanese text

                audio_path = wavs_dir / f"{filename}.wav"
                if audio_path.exists():
                    samples.append({
                        "sample_id": filename,
                        "audio_path": str(audio_path),
                        "reference_transcription": transcription,
                        "source": "tsv"
                    })

    return samples


def load_all_datasets() -> Dict[str, List[Dict]]:
    """Load all available datasets."""
    datasets = {
        "female": [],
        "male": [],
        "video": []
    }

    # Load female dataset
    female_tsv = FEMALE_DIR / "line_index.tsv"
    female_wavs = FEMALE_DIR / "wavs"
    if female_tsv.exists():
        datasets["female"] = load_tsv_dataset(female_tsv, female_wavs)
        print(f"[INFO] Loaded {len(datasets['female'])} female samples")

    # Load male dataset
    male_tsv = MALE_DIR / "line_index.tsv"
    male_wavs = MALE_DIR / "wavs"
    if male_tsv.exists():
        datasets["male"] = load_tsv_dataset(male_tsv, male_wavs)
        print(f"[INFO] Loaded {len(datasets['male'])} male samples")

    # Load video dataset
    if VIDEO_DIR.exists():
        for video_file in VIDEO_DIR.glob("*.mp4"):
            datasets["video"].append({
                "sample_id": video_file.stem,
                "video_path": str(video_file),
                "source": "video"
            })
        print(f"[INFO] Loaded {len(datasets['video'])} video samples")

    return datasets


def select_samples(datasets: Dict, num_samples: int, include_video: bool = False) -> List[Dict]:
    """
    Select random samples from datasets.

    Args:
        datasets: Dictionary of loaded datasets
        num_samples: Number of audio samples to select
        include_video: Include video samples

    Returns:
        List of selected samples
    """
    # Combine male and female
    all_audio = datasets["female"] + datasets["male"]

    # Random select
    if len(all_audio) > num_samples:
        selected = random.sample(all_audio, num_samples)
    else:
        selected = all_audio

    # Add video if requested
    if include_video and datasets["video"]:
        selected.extend(datasets["video"])

    return selected


class RealDataEvaluator:
    """Evaluator using real audio data."""

    def __init__(self):
        self.model = None
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "dataset_source": "backend/dataset/su_id_*",
                "evaluation_type": "real_audio"
            },
            "transcription": {
                "samples": [],
                "aggregate": {}
            },
            "translation": {
                "samples": [],
                "aggregate": {}
            },
            "processing_times": []
        }

    def load_models(self):
        """Load AI models."""
        print("\n[INFO] Loading AI models...")
        print("[INFO] This may take 1-2 minutes and use ~6GB VRAM...")

        try:
            from models import AudioVisualModel
            self.model = AudioVisualModel()
            print("[INFO] Models loaded successfully!")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            return False

    def evaluate_audio_sample(self, sample: Dict) -> Dict:
        """
        Evaluate a single audio sample.

        Args:
            sample: {audio_path, reference_transcription, sample_id}

        Returns:
            Evaluation result
        """
        from metrics import calculate_wer, calculate_cer, calculate_bleu

        audio_path = sample["audio_path"]
        reference = sample["reference_transcription"]
        sample_id = sample["sample_id"]

        # Read audio
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()

        # Get model prediction
        start_time = time.time()
        result = self.model.process_translation(
            audio_bytes,
            video_frame_bytes=None,
            use_visual=False,
            auto_translate=True
        )
        processing_time = time.time() - start_time

        if result is None:
            return {
                "sample_id": sample_id,
                "error": "Model failed to process",
                "processing_time": processing_time
            }

        hypothesis_transcription = result.get("transcription", "")
        hypothesis_translation = result.get("translation", "")

        # Calculate metrics
        wer_result = calculate_wer(reference, hypothesis_transcription)
        cer_result = calculate_cer(reference, hypothesis_transcription)

        eval_result = {
            "sample_id": sample_id,
            "reference": reference,
            "hypothesis_transcription": hypothesis_transcription,
            "hypothesis_translation": hypothesis_translation,
            "wer": wer_result,
            "cer": cer_result,
            "processing_time": round(processing_time, 2)
        }

        # Store results
        self.results["transcription"]["samples"].append(eval_result)
        self.results["processing_times"].append(processing_time)

        return eval_result

    def evaluate_video_sample(self, sample: Dict) -> Dict:
        """
        Evaluate a video sample (audio-visual mode).

        Args:
            sample: {video_path, sample_id}

        Returns:
            Evaluation result
        """
        from utils import extract_audio_from_video, extract_video_frame

        video_path = sample["video_path"]
        sample_id = sample["sample_id"]

        # Read video
        with open(video_path, 'rb') as f:
            video_bytes = f.read()

        # Extract audio and frame
        audio_bytes = extract_audio_from_video(video_bytes)
        video_frame_bytes = extract_video_frame(video_bytes)

        if audio_bytes is None:
            return {
                "sample_id": sample_id,
                "error": "Failed to extract audio from video"
            }

        # Get model prediction (with visual if available)
        start_time = time.time()
        use_visual = video_frame_bytes is not None and len(video_frame_bytes) > 100

        result = self.model.process_translation(
            audio_bytes,
            video_frame_bytes=video_frame_bytes if use_visual else None,
            use_visual=use_visual,
            auto_translate=True
        )
        processing_time = time.time() - start_time

        if result is None:
            return {
                "sample_id": sample_id,
                "error": "Model failed to process video"
            }

        eval_result = {
            "sample_id": sample_id,
            "mode": "audio-visual" if use_visual else "audio-only",
            "hypothesis_transcription": result.get("transcription", ""),
            "hypothesis_translation": result.get("translation", ""),
            "visual_augmentation": result.get("visual_augmentation"),
            "processing_time": round(processing_time, 2)
        }

        return eval_result

    def compute_aggregate(self):
        """Compute aggregate metrics."""
        samples = self.results["transcription"]["samples"]

        if not samples:
            return

        # Filter out errors
        valid_samples = [s for s in samples if "error" not in s]

        if not valid_samples:
            return

        wer_values = [s["wer"]["wer"] for s in valid_samples]
        cer_values = [s["cer"]["cer"] for s in valid_samples]

        self.results["transcription"]["aggregate"] = {
            "total_samples": len(valid_samples),
            "errors": len(samples) - len(valid_samples),
            "average_wer": round(sum(wer_values) / len(wer_values), 2),
            "min_wer": round(min(wer_values), 2),
            "max_wer": round(max(wer_values), 2),
            "average_cer": round(sum(cer_values) / len(cer_values), 2),
            "min_cer": round(min(cer_values), 2),
            "max_cer": round(max(cer_values), 2),
            "average_processing_time": round(sum(self.results["processing_times"]) / len(self.results["processing_times"]), 2)
        }

    def save_results(self, output_path: str):
        """Save results to JSON."""
        self.compute_aggregate()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"\n[INFO] Results saved to: {output_path}")

    def print_summary(self):
        """Print evaluation summary."""
        self.compute_aggregate()

        print("\n" + "="*60)
        print("REAL DATA EVALUATION SUMMARY")
        print("="*60)

        agg = self.results["transcription"]["aggregate"]
        if agg:
            print(f"\n[DATASET] Source: backend/dataset/su_id_*")
            print("-"*40)
            print(f"  Total Samples Evaluated: {agg['total_samples']}")
            print(f"  Errors/Failures:         {agg.get('errors', 0)}")
            print(f"\n[TRANSCRIPTION METRICS]")
            print("-"*40)
            print(f"  Average WER:   {agg['average_wer']}%")
            print(f"  Average CER:   {agg['average_cer']}%")
            print(f"  WER Range:     {agg['min_wer']}% - {agg['max_wer']}%")
            print(f"  CER Range:     {agg['min_cer']}% - {agg['max_cer']}%")
            print(f"\n[PERFORMANCE]")
            print("-"*40)
            print(f"  Avg Processing Time: {agg['average_processing_time']}s per sample")

        print("\n" + "="*60)


def run_evaluation(num_samples: int = 10, include_video: bool = False):
    """
    Run real data evaluation.

    Args:
        num_samples: Number of audio samples to evaluate
        include_video: Include video samples
    """
    print("\n" + "="*70)
    print("REAL DATA EVALUATION - Sundanese Speech Recognition")
    print("="*70)
    print(f"Samples to evaluate: {num_samples}")
    print(f"Include video: {include_video}")

    # Load datasets
    print("\n[STEP 1/4] Loading datasets...")
    datasets = load_all_datasets()

    total_available = len(datasets["female"]) + len(datasets["male"])
    print(f"[INFO] Total audio samples available: {total_available}")

    # Select samples
    print(f"\n[STEP 2/4] Selecting {num_samples} random samples...")
    selected = select_samples(datasets, num_samples, include_video)
    print(f"[INFO] Selected {len(selected)} samples for evaluation")

    # Initialize evaluator
    evaluator = RealDataEvaluator()

    # Load models
    print("\n[STEP 3/4] Loading AI models...")
    if not evaluator.load_models():
        print("[ERROR] Cannot proceed without models")
        return None

    # Run evaluation
    print(f"\n[STEP 4/4] Evaluating {len(selected)} samples...")
    print("-"*50)

    for i, sample in enumerate(selected):
        print(f"\n[{i+1}/{len(selected)}] Evaluating: {sample['sample_id'][:30]}...")

        if sample.get("source") == "video":
            result = evaluator.evaluate_video_sample(sample)
        else:
            result = evaluator.evaluate_audio_sample(sample)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            wer = result.get("wer", {}).get("wer", "N/A")
            cer = result.get("cer", {}).get("cer", "N/A")
            pt = result.get("processing_time", "N/A")
            print(f"  WER: {wer}% | CER: {cer}% | Time: {pt}s")
            print(f"  Ref: {result.get('reference', '')[:50]}...")
            print(f"  Hyp: {result.get('hypothesis_transcription', '')[:50]}...")

    # Print summary
    evaluator.print_summary()

    # Save results
    output_path = OUTPUT_DIR / f"evaluation_real_{num_samples}samples.json"
    evaluator.save_results(str(output_path))

    return evaluator.results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model with real Sundanese audio dataset"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=20,
        help="Number of audio samples to evaluate (default: 20)"
    )
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="Include video samples in evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)

    # Run evaluation
    run_evaluation(
        num_samples=args.samples,
        include_video=args.include_video
    )


if __name__ == "__main__":
    main()
