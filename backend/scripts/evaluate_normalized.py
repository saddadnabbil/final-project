#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Normalized Evaluation - Fair WER/CER calculation

Normalizes reference and hypothesis for fair comparison:
- Lowercase
- Remove punctuation
- Normalize spacing
- Handle number format differences
"""

import json
import re
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def normalize_text(text: str) -> str:
    """Normalize text for fair WER comparison."""
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[,.\-!?;:]', '', text)

    # Normalize spacing (multiple spaces → single)
    text = re.sub(r'\s+', ' ', text)

    # Strip
    text = text.strip()

    return text


def calculate_normalized_wer(reference: str, hypothesis: str) -> Dict:
    """Calculate WER with normalized text."""
    from jiwer import wer, cer

    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    wer_score = wer(ref_norm, hyp_norm) * 100
    cer_score = cer(ref_norm, hyp_norm) * 100

    return {
        "wer": round(wer_score, 2),
        "cer": round(cer_score, 2),
        "ref_normalized": ref_norm,
        "hyp_normalized": hyp_norm
    }


def main():
    # Load original results
    results_file = PROJECT_ROOT / "tests" / "data" / "evaluation_real_50samples.json"

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print("="*60)
    print("NORMALIZED EVALUATION")
    print("="*60)
    print("\nRe-calculating WER/CER with normalization...")

    samples = results["transcription"]["samples"]

    wer_scores = []
    cer_scores = []

    print(f"\nProcessing {len(samples)} samples...")

    for i, sample in enumerate(samples, 1):
        ref = sample["reference"]
        hyp = sample["hypothesis_transcription"]

        # Calculate normalized WER
        norm_result = calculate_normalized_wer(ref, hyp)

        wer_scores.append(norm_result["wer"])
        cer_scores.append(norm_result["cer"])

        # Update sample
        sample["normalized_wer"] = norm_result["wer"]
        sample["normalized_cer"] = norm_result["cer"]

        if i % 10 == 0:
            print(f"  Processed {i}/{len(samples)} samples...")

    # Calculate averages
    avg_wer = sum(wer_scores) / len(wer_scores)
    avg_cer = sum(cer_scores) / len(cer_scores)

    print("\n" + "="*60)
    print("NORMALIZED RESULTS")
    print("="*60)
    print(f"\n[ORIGINAL]")
    print(f"  Average WER: {results['transcription']['aggregate']['average_wer']:.2f}%")
    print(f"  Average CER: {results['transcription']['aggregate']['average_cer']:.2f}%")

    print(f"\n[NORMALIZED]")
    print(f"  Average WER: {avg_wer:.2f}%")
    print(f"  Average CER: {avg_cer:.2f}%")

    print(f"\n[IMPROVEMENT]")
    print(f"  WER Reduction: {results['transcription']['aggregate']['average_wer'] - avg_wer:.2f} points")
    print(f"  CER Reduction: {results['transcription']['aggregate']['average_cer'] - avg_cer:.2f} points")

    # Save normalized results
    results["transcription"]["normalized_aggregate"] = {
        "average_wer": round(avg_wer, 2),
        "average_cer": round(avg_cer, 2),
        "min_wer": round(min(wer_scores), 2),
        "max_wer": round(max(wer_scores), 2),
        "min_cer": round(min(cer_scores), 2),
        "max_cer": round(max(cer_scores), 2),
    }

    output_file = PROJECT_ROOT / "tests" / "data" / "evaluation_normalized_50samples.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Normalized results saved to: {output_file}")
    print("="*60)

    return avg_wer, avg_cer


if __name__ == "__main__":
    main()
