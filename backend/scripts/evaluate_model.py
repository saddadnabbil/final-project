#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_model.py - Script untuk evaluasi model Whisper dan NLLB

Menghitung metrik evaluasi (WER, CER, BLEU) dengan data test yang sebenarnya.
Hasil evaluasi ini akan digunakan untuk menggantikan data dummy di BAB 4 skripsi.

Usage:
    python scripts/evaluate_model.py --test-data tests/data/test_dataset.json
    python scripts/evaluate_model.py --quick  # Quick test dengan sample data
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "src"))

# Import metrics functions
from metrics import (
    calculate_wer,
    calculate_cer,
    calculate_bleu,
    calculate_all_metrics,
    normalize_text
)


class ModelEvaluator:
    """
    Kelas untuk mengevaluasi performa model speech-to-text dan translation.
    """

    def __init__(self, load_models: bool = True):
        """
        Initialize evaluator.

        Args:
            load_models: If True, load AI models for live transcription.
                        If False, only use pre-computed transcriptions.
        """
        self.model = None
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "transcription": {
                "samples": [],
                "aggregate": {}
            },
            "translation": {
                "samples": [],
                "aggregate": {}
            }
        }

        if load_models:
            self._load_models()

    def _load_models(self):
        """Load AI models for live evaluation."""
        try:
            print("[INFO] Loading AI models for evaluation...")
            from models import AudioVisualModel
            self.model = AudioVisualModel()
            print("[INFO] Models loaded successfully!")
        except Exception as e:
            print(f"[WARNING] Could not load models: {e}")
            print("[INFO] Will use pre-computed transcriptions only")
            self.model = None

    def evaluate_transcription(
        self,
        reference: str,
        hypothesis: str,
        sample_id: str = None
    ) -> Dict:
        """
        Evaluate a single transcription against reference.

        Args:
            reference: Ground truth transcription (Sundanese)
            hypothesis: Model output transcription
            sample_id: Optional sample identifier

        Returns:
            Dictionary with WER, CER, and details
        """
        wer_result = calculate_wer(reference, hypothesis)
        cer_result = calculate_cer(reference, hypothesis)

        result = {
            "sample_id": sample_id,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": wer_result,
            "cer": cer_result
        }

        self.results["transcription"]["samples"].append(result)
        return result

    def evaluate_translation(
        self,
        reference: str,
        hypothesis: str,
        source_text: str = None,
        sample_id: str = None
    ) -> Dict:
        """
        Evaluate a single translation against reference.

        Args:
            reference: Ground truth translation (Indonesian)
            hypothesis: Model output translation
            source_text: Original Sundanese text
            sample_id: Optional sample identifier

        Returns:
            Dictionary with BLEU and details
        """
        bleu_result = calculate_bleu(reference, hypothesis)

        result = {
            "sample_id": sample_id,
            "source": source_text,
            "reference": reference,
            "hypothesis": hypothesis,
            "bleu": bleu_result
        }

        self.results["translation"]["samples"].append(result)
        return result

    def evaluate_from_audio(
        self,
        audio_path: str,
        reference_transcription: str,
        reference_translation: str = None,
        sample_id: str = None
    ) -> Dict:
        """
        Evaluate model on audio file with live transcription.

        Args:
            audio_path: Path to audio file
            reference_transcription: Ground truth Sundanese transcription
            reference_translation: Ground truth Indonesian translation (optional)
            sample_id: Optional sample identifier

        Returns:
            Dictionary with all metrics
        """
        if self.model is None:
            raise RuntimeError("Models not loaded. Initialize with load_models=True")

        # Read audio file
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
            return {"error": "Model failed to process audio"}

        hypothesis_transcription = result.get("transcription", "")
        hypothesis_translation = result.get("translation", "")

        # Evaluate transcription
        trans_result = self.evaluate_transcription(
            reference_transcription,
            hypothesis_transcription,
            sample_id
        )
        trans_result["processing_time"] = processing_time

        # Evaluate translation if reference provided
        transl_result = None
        if reference_translation:
            transl_result = self.evaluate_translation(
                reference_translation,
                hypothesis_translation,
                hypothesis_transcription,
                sample_id
            )

        return {
            "transcription": trans_result,
            "translation": transl_result,
            "processing_time": processing_time
        }

    def compute_aggregate_metrics(self) -> Dict:
        """
        Compute aggregate metrics across all samples.

        Returns:
            Dictionary with average WER, CER, BLEU, etc.
        """
        # Aggregate transcription metrics
        trans_samples = self.results["transcription"]["samples"]
        if trans_samples:
            wer_values = [s["wer"]["wer"] for s in trans_samples]
            cer_values = [s["cer"]["cer"] for s in trans_samples]

            self.results["transcription"]["aggregate"] = {
                "total_samples": len(trans_samples),
                "average_wer": round(sum(wer_values) / len(wer_values), 2),
                "min_wer": round(min(wer_values), 2),
                "max_wer": round(max(wer_values), 2),
                "average_cer": round(sum(cer_values) / len(cer_values), 2),
                "min_cer": round(min(cer_values), 2),
                "max_cer": round(max(cer_values), 2),
                "wer_accuracy": round(100 - sum(wer_values) / len(wer_values), 2),
                "cer_accuracy": round(100 - sum(cer_values) / len(cer_values), 2)
            }

        # Aggregate translation metrics
        transl_samples = self.results["translation"]["samples"]
        if transl_samples:
            bleu_values = [s["bleu"]["bleu"] for s in transl_samples]

            self.results["translation"]["aggregate"] = {
                "total_samples": len(transl_samples),
                "average_bleu": round(sum(bleu_values) / len(bleu_values), 2),
                "min_bleu": round(min(bleu_values), 2),
                "max_bleu": round(max(bleu_values), 2)
            }

        return self.results

    def get_results(self) -> Dict:
        """Get all evaluation results."""
        self.compute_aggregate_metrics()
        return self.results

    def save_results(self, output_path: str):
        """Save results to JSON file."""
        self.compute_aggregate_metrics()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Results saved to: {output_path}")

    def print_summary(self):
        """Print evaluation summary to console."""
        self.compute_aggregate_metrics()

        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)

        # Transcription metrics
        trans_agg = self.results["transcription"]["aggregate"]
        if trans_agg:
            print("\n[TRANSCRIPTION] TRANSCRIPTION METRICS (Sundanese ASR)")
            print("-"*40)
            print(f"  Total Samples: {trans_agg['total_samples']}")
            print(f"  Average WER:   {trans_agg['average_wer']}%")
            print(f"  Average CER:   {trans_agg['average_cer']}%")
            print(f"  WER Range:     {trans_agg['min_wer']}% - {trans_agg['max_wer']}%")
            print(f"  CER Range:     {trans_agg['min_cer']}% - {trans_agg['max_cer']}%")
            print(f"  Word Accuracy: {trans_agg['wer_accuracy']}%")
            print(f"  Char Accuracy: {trans_agg['cer_accuracy']}%")

        # Translation metrics
        transl_agg = self.results["translation"]["aggregate"]
        if transl_agg:
            print("\n[TRANSLATION] TRANSLATION METRICS (Sundanese -> Indonesian)")
            print("-"*40)
            print(f"  Total Samples: {transl_agg['total_samples']}")
            print(f"  Average BLEU:  {transl_agg['average_bleu']}")
            print(f"  BLEU Range:    {transl_agg['min_bleu']} - {transl_agg['max_bleu']}")

        print("\n" + "="*60)


def run_quick_test():
    """Run quick test with sample data (no audio files needed)."""
    print("[INFO] Running quick evaluation with sample data...")

    evaluator = ModelEvaluator(load_models=False)

    # Sample transcription pairs (Sundanese)
    # Reference vs Hypothesis (simulated model output)
    transcription_samples = [
        {
            "id": "sample_001",
            "reference": "wilujeng enjing sadayana",
            "hypothesis": "wilujeng enjing sadayana"
        },
        {
            "id": "sample_002",
            "reference": "kumaha damang bapa",
            "hypothesis": "kumaha damang bapa"
        },
        {
            "id": "sample_003",
            "reference": "abdi teh ti bandung",
            "hypothesis": "abdi ti bandung"
        },
        {
            "id": "sample_004",
            "reference": "hatur nuhun pisan",
            "hypothesis": "hatur nuhun pisan"
        },
        {
            "id": "sample_005",
            "reference": "punten bade tumaros",
            "hypothesis": "punten bade tumaros"
        },
        {
            "id": "sample_006",
            "reference": "mangga atuh geura calik",
            "hypothesis": "mangga atuh geura calik"
        },
        {
            "id": "sample_007",
            "reference": "abdi mah teu terang",
            "hypothesis": "abdi mah teu terang"
        },
        {
            "id": "sample_008",
            "reference": "hayu urang ka pasar",
            "hypothesis": "hayu urang ka pasar"
        },
        {
            "id": "sample_009",
            "reference": "nuhun kana bantosanana",
            "hypothesis": "nuhun kana bantosanana"
        },
        {
            "id": "sample_010",
            "reference": "abdi bade ka sakola",
            "hypothesis": "abdi bade ka sakola"
        }
    ]

    # Sample translation pairs (Sundanese → Indonesian)
    translation_samples = [
        {
            "id": "sample_001",
            "source": "wilujeng enjing sadayana",
            "reference": "selamat pagi semuanya",
            "hypothesis": "selamat pagi semuanya"
        },
        {
            "id": "sample_002",
            "source": "kumaha damang bapa",
            "reference": "bagaimana kabar bapak",
            "hypothesis": "bagaimana kabar bapak"
        },
        {
            "id": "sample_003",
            "source": "abdi teh ti bandung",
            "reference": "saya dari bandung",
            "hypothesis": "saya dari bandung"
        },
        {
            "id": "sample_004",
            "source": "hatur nuhun pisan",
            "reference": "terima kasih banyak",
            "hypothesis": "terima kasih banyak"
        },
        {
            "id": "sample_005",
            "source": "punten bade tumaros",
            "reference": "permisi mau bertanya",
            "hypothesis": "permisi mau bertanya"
        },
        {
            "id": "sample_006",
            "source": "mangga atuh geura calik",
            "reference": "silakan segera duduk",
            "hypothesis": "silakan segera duduk"
        },
        {
            "id": "sample_007",
            "source": "abdi mah teu terang",
            "reference": "saya tidak tahu",
            "hypothesis": "saya tidak tahu"
        },
        {
            "id": "sample_008",
            "source": "hayu urang ka pasar",
            "reference": "ayo kita ke pasar",
            "hypothesis": "ayo kita ke pasar"
        },
        {
            "id": "sample_009",
            "source": "nuhun kana bantosanana",
            "reference": "terima kasih atas bantuannya",
            "hypothesis": "terima kasih atas bantuannya"
        },
        {
            "id": "sample_010",
            "source": "abdi bade ka sakola",
            "reference": "saya mau ke sekolah",
            "hypothesis": "saya mau ke sekolah"
        }
    ]

    # Run transcription evaluation
    print("\n[INFO] Evaluating transcription samples...")
    for sample in transcription_samples:
        result = evaluator.evaluate_transcription(
            sample["reference"],
            sample["hypothesis"],
            sample["id"]
        )
        print(f"  {sample['id']}: WER={result['wer']['wer']}%, CER={result['cer']['cer']}%")

    # Run translation evaluation
    print("\n[INFO] Evaluating translation samples...")
    for sample in translation_samples:
        result = evaluator.evaluate_translation(
            sample["reference"],
            sample["hypothesis"],
            sample["source"],
            sample["id"]
        )
        print(f"  {sample['id']}: BLEU={result['bleu']['bleu']}")

    # Print summary
    evaluator.print_summary()

    # Save results
    output_path = PROJECT_ROOT / "tests" / "data" / "evaluation_results_quick.json"
    evaluator.save_results(str(output_path))

    return evaluator.get_results()


def run_full_evaluation(test_data_path: str):
    """Run full evaluation with test dataset."""
    print(f"[INFO] Running full evaluation with: {test_data_path}")

    # Load test dataset
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    evaluator = ModelEvaluator(load_models=True)

    # Evaluate each sample
    for i, sample in enumerate(test_data["samples"]):
        print(f"\n[INFO] Evaluating sample {i+1}/{len(test_data['samples'])}: {sample['id']}")

        if "audio_path" in sample and os.path.exists(sample["audio_path"]):
            # Live evaluation with audio
            result = evaluator.evaluate_from_audio(
                sample["audio_path"],
                sample["reference_transcription"],
                sample.get("reference_translation"),
                sample["id"]
            )
        else:
            # Evaluation with pre-computed transcription
            if "hypothesis_transcription" in sample:
                evaluator.evaluate_transcription(
                    sample["reference_transcription"],
                    sample["hypothesis_transcription"],
                    sample["id"]
                )

            if "hypothesis_translation" in sample and "reference_translation" in sample:
                evaluator.evaluate_translation(
                    sample["reference_translation"],
                    sample["hypothesis_translation"],
                    sample.get("hypothesis_transcription", ""),
                    sample["id"]
                )

    # Print summary
    evaluator.print_summary()

    # Save results
    output_path = Path(test_data_path).parent / "evaluation_results_full.json"
    evaluator.save_results(str(output_path))

    return evaluator.get_results()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate speech-to-text and translation model performance"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test dataset JSON file"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with sample data (no audio files needed)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON"
    )

    args = parser.parse_args()

    if args.quick:
        run_quick_test()
    elif args.test_data:
        run_full_evaluation(args.test_data)
    else:
        print("Usage: python evaluate_model.py --quick")
        print("       python evaluate_model.py --test-data path/to/test_dataset.json")
        print("\nRunning quick test by default...")
        run_quick_test()


if __name__ == "__main__":
    main()
