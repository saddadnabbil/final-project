# backend/src/metrics.py
"""
Metrics calculation module for speech translation evaluation.
Implements WER, CER, BLEU, and processing time metrics.
"""

import re
import time
import numpy as np
from typing import Dict, List, Optional, Tuple


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    - Lowercase
    - Remove extra whitespace
    - Remove punctuation
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text


def levenshtein_distance(s1: List[str], s2: List[str]) -> Tuple[int, int, int, int]:
    """
    Calculate Levenshtein distance between two sequences.

    Returns:
        (distance, substitutions, deletions, insertions)
    """
    m, n = len(s1), len(s2)

    # DP table: (distance, S, D, I)
    dp = [[(0, 0, 0, 0) for _ in range(n + 1)] for _ in range(m + 1)]

    # Initialize
    for i in range(m + 1):
        dp[i][0] = (i, 0, i, 0)  # All deletions
    for j in range(n + 1):
        dp[0][j] = (j, 0, 0, j)  # All insertions

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                # Substitution
                sub = (dp[i-1][j-1][0] + 1, dp[i-1][j-1][1] + 1, dp[i-1][j-1][2], dp[i-1][j-1][3])
                # Deletion
                dele = (dp[i-1][j][0] + 1, dp[i-1][j][1], dp[i-1][j][2] + 1, dp[i-1][j][3])
                # Insertion
                ins = (dp[i][j-1][0] + 1, dp[i][j-1][1], dp[i][j-1][2], dp[i][j-1][3] + 1)

                # Choose minimum distance
                dp[i][j] = min([sub, dele, ins], key=lambda x: x[0])

    return dp[m][n]


def calculate_wer(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Calculate Word Error Rate (WER).

    WER = (S + D + I) / N * 100%

    Where:
        S = Substitutions
        D = Deletions
        I = Insertions
        N = Number of words in reference

    Args:
        reference: Ground truth text
        hypothesis: Predicted text

    Returns:
        Dictionary with WER and component metrics
    """
    if not reference:
        return {"wer": 100.0, "substitutions": 0, "deletions": 0, "insertions": 0, "total_words": 0, "accuracy": 0.0}

    # Normalize and tokenize
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split() if hypothesis else []

    if len(ref_words) == 0:
        return {"wer": 100.0, "substitutions": 0, "deletions": 0, "insertions": 0, "total_words": 0, "accuracy": 0.0}

    # Calculate Levenshtein distance
    distance, subs, dels, ins = levenshtein_distance(ref_words, hyp_words)

    # Calculate WER
    wer = (distance / len(ref_words)) * 100
    wer = min(wer, 100.0)  # Cap at 100%

    return {
        "wer": round(wer, 2),
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "total_words": len(ref_words),
        "accuracy": round(max(0, 100 - wer), 2)
    }


def calculate_cer(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Calculate Character Error Rate (CER).

    CER = (S + D + I) / N * 100%

    Where:
        S = Character substitutions
        D = Character deletions
        I = Character insertions
        N = Number of characters in reference

    Args:
        reference: Ground truth text
        hypothesis: Predicted text

    Returns:
        Dictionary with CER and component metrics
    """
    if not reference:
        return {"cer": 100.0, "substitutions": 0, "deletions": 0, "insertions": 0, "total_chars": 0, "accuracy": 0.0}

    # Normalize and convert to character list (remove spaces for CER)
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", "")) if hypothesis else []

    if len(ref_chars) == 0:
        return {"cer": 100.0, "substitutions": 0, "deletions": 0, "insertions": 0, "total_chars": 0, "accuracy": 0.0}

    # Calculate Levenshtein distance
    distance, subs, dels, ins = levenshtein_distance(ref_chars, hyp_chars)

    # Calculate CER
    cer = (distance / len(ref_chars)) * 100
    cer = min(cer, 100.0)  # Cap at 100%

    return {
        "cer": round(cer, 2),
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "total_chars": len(ref_chars),
        "accuracy": round(max(0, 100 - cer), 2)
    }


def calculate_bleu(reference: str, hypothesis: str, max_n: int = 4) -> Dict[str, float]:
    """
    Calculate BLEU score (Bilingual Evaluation Understudy).

    BLEU measures n-gram precision with brevity penalty.

    Args:
        reference: Ground truth translation
        hypothesis: Predicted translation
        max_n: Maximum n-gram size (default 4)

    Returns:
        Dictionary with BLEU score and component metrics
    """
    if not reference or not hypothesis:
        return {"bleu": 0.0, "brevity_penalty": 0.0, "precisions": []}

    # Normalize and tokenize
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    if len(hyp_words) == 0 or len(ref_words) == 0:
        return {"bleu": 0.0, "brevity_penalty": 0.0, "precisions": []}

    # Calculate n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        # Get n-grams
        ref_ngrams = {}
        for i in range(len(ref_words) - n + 1):
            ngram = tuple(ref_words[i:i+n])
            ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1

        hyp_ngrams = {}
        for i in range(len(hyp_words) - n + 1):
            ngram = tuple(hyp_words[i:i+n])
            hyp_ngrams[ngram] = hyp_ngrams.get(ngram, 0) + 1

        # Calculate clipped count
        clipped_count = 0
        total_count = 0
        for ngram, count in hyp_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))
            total_count += count

        if total_count > 0:
            precision = clipped_count / total_count
        else:
            precision = 0

        precisions.append(precision)

    # Calculate brevity penalty
    if len(hyp_words) >= len(ref_words):
        bp = 1.0
    else:
        bp = np.exp(1 - len(ref_words) / len(hyp_words))

    # Calculate BLEU score (geometric mean of precisions)
    if all(p > 0 for p in precisions):
        log_precision = sum(np.log(p) for p in precisions) / len(precisions)
        bleu = bp * np.exp(log_precision) * 100
    else:
        bleu = 0.0

    return {
        "bleu": round(bleu, 2),
        "brevity_penalty": round(bp, 4),
        "precisions": [round(p * 100, 2) for p in precisions]
    }


def get_audio_duration(audio_bytes: bytes) -> Optional[float]:
    """
    Get duration of audio in seconds using pydub.

    Returns:
        float: Duration in seconds, or None if cannot be determined
    """
    try:
        import io
        from pydub import AudioSegment

        # Create audio segment from bytes
        audio_io = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(audio_io)

        # Get duration in seconds
        duration = len(audio) / 1000.0  # pydub returns milliseconds
        return max(0.1, min(duration, 3600))  # Cap between 0.1s and 1 hour
    except Exception as e:
        print(f"[WARNING] Could not determine audio duration: {e}")
        return None


def calculate_metrics(
    transcription: str,
    translation: str,
    audio_bytes: Optional[bytes] = None,
    processing_start_time: Optional[float] = None,
    reference_transcription: Optional[str] = None,
    reference_translation: Optional[str] = None
) -> Dict:
    """
    Calculate all metrics for audio/video processing.

    Args:
        transcription: Transcribed text (hypothesis)
        translation: Translated text (hypothesis)
        audio_bytes: Audio data (for duration calculation)
        processing_start_time: Start time of processing (for timing)
        reference_transcription: Ground truth transcription (for WER/CER)
        reference_translation: Ground truth translation (for BLEU)

    Returns:
        dict: Dictionary with all calculated metrics
    """
    metrics = {}

    # Calculate processing time
    if processing_start_time:
        metrics['processing_time'] = round(time.time() - processing_start_time, 2)

    # Get audio duration
    if audio_bytes:
        duration = get_audio_duration(audio_bytes)
        if duration:
            metrics['audio_duration'] = round(duration, 2)

    # Calculate WER and CER if reference transcription provided
    if reference_transcription:
        metrics['wer'] = calculate_wer(reference_transcription, transcription)
        metrics['cer'] = calculate_cer(reference_transcription, transcription)

    # Calculate BLEU if reference translation provided
    if reference_translation:
        metrics['bleu'] = calculate_bleu(reference_translation, translation)

    return metrics


def calculate_all_metrics(
    transcription_ref: str,
    transcription_hyp: str,
    translation_ref: Optional[str] = None,
    translation_hyp: Optional[str] = None
) -> Dict:
    """
    Calculate all metrics for a transcription/translation pair.

    Args:
        transcription_ref: Reference transcription (Sundanese)
        transcription_hyp: Predicted transcription
        translation_ref: Reference translation (Indonesian) - optional
        translation_hyp: Predicted translation - optional

    Returns:
        Dictionary with all metrics
    """
    result = {
        "transcription": {
            "reference": transcription_ref,
            "hypothesis": transcription_hyp,
            "wer": calculate_wer(transcription_ref, transcription_hyp),
            "cer": calculate_cer(transcription_ref, transcription_hyp)
        }
    }

    if translation_ref and translation_hyp:
        result["translation"] = {
            "reference": translation_ref,
            "hypothesis": translation_hyp,
            "bleu": calculate_bleu(translation_ref, translation_hyp)
        }

    return result
