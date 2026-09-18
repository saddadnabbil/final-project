#!/usr/bin/env python3
"""
Prepare OpenSLR 36 Dataset for Whisper Training
================================================
Parses the manually extracted dataset and creates HuggingFace Dataset format.

Expected structure (after manual extraction):
    openslr36/
      asr_sundanese_0/
        asr_sundanese/
          utt_spk_text.tsv
          data/
            00/
              000018c074.flac
              ...

Features:
    - Handles manually extracted folder structure
    - FLAC audio support
    - Checkpoint support: resume from where it stopped
    - Missing audio tolerance: continues even if some files missing (zip 7)

Usage:
    python scripts/prepare_openslr36_dataset.py
"""

import os
import sys
import json
import random
from pathlib import Path
from collections import Counter
from datetime import datetime

# Configuration
DATASET_DIR = Path(__file__).parent.parent / "backend" / "dataset" / "openslr36"
OUTPUT_DIR = Path(__file__).parent.parent / "backend" / "dataset" / "openslr36_prepared"
CHECKPOINT_FILE = OUTPUT_DIR / ".preparation_checkpoint.json"

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Filter settings
MIN_DURATION_SEC = 0.5   # Minimum audio duration
MAX_DURATION_SEC = 30.0  # Maximum audio duration (Whisper limit)
MIN_TEXT_LENGTH = 2      # Minimum transcription length

# Processing settings
CHECKPOINT_INTERVAL = 10000  # Save checkpoint every N entries
RANDOM_SEED = 42


def load_checkpoint():
    """Load preparation checkpoint."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    return {
        "stage": "start",
        "processed_entries": 0,
        "valid_entries": [],
        "stats": {},
        "last_updated": None
    }


def save_checkpoint(checkpoint):
    """Save preparation checkpoint."""
    checkpoint["last_updated"] = datetime.now().isoformat()
    try:
        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        temp_file = CHECKPOINT_FILE.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        temp_file.replace(CHECKPOINT_FILE)
        return True
    except Exception as e:
        print(f"Warning: Could not save checkpoint: {e}")
        return False


def get_audio_duration_flac(flac_path):
    """Get audio duration in seconds for FLAC files."""
    try:
        import soundfile as sf
        info = sf.info(str(flac_path))
        return info.duration
    except ImportError:
        # Fallback: try mutagen
        try:
            from mutagen.flac import FLAC
            audio = FLAC(str(flac_path))
            return audio.info.length
        except ImportError:
            # Fallback: estimate from file size (rough)
            # FLAC ~700kbps average, 16kHz mono = ~88KB/sec
            file_size = flac_path.stat().st_size
            return file_size / 88000  # Rough estimate
        except Exception:
            return None
    except Exception:
        return None


def get_audio_duration(audio_path):
    """Get audio duration in seconds."""
    suffix = audio_path.suffix.lower()

    if suffix == '.flac':
        return get_audio_duration_flac(audio_path)
    elif suffix == '.wav':
        try:
            import wave
            with wave.open(str(audio_path), 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate == 0:
                    return None
                return frames / float(rate)
        except Exception:
            return None
    else:
        # Try soundfile for other formats
        try:
            import soundfile as sf
            info = sf.info(str(audio_path))
            return info.duration
        except Exception:
            return None


def find_all_tsv_files(dataset_dir):
    """Find all utt_spk_text.tsv files in the dataset directory."""
    tsv_files = []

    # Pattern: asr_sundanese_X/asr_sundanese/utt_spk_text.tsv
    for subdir in sorted(dataset_dir.iterdir()):
        if subdir.is_dir() and subdir.name.startswith("asr_sundanese_"):
            tsv_path = subdir / "asr_sundanese" / "utt_spk_text.tsv"
            if tsv_path.exists():
                tsv_files.append(tsv_path)

    return tsv_files


def find_audio_file(utt_id, data_dirs):
    """Find audio file for utterance ID across all data directories."""
    # utt_id format: 000018c074
    # File location: data/00/000018c074.flac (first 2 chars are subfolder)

    prefix = utt_id[:2]  # First 2 characters

    for data_dir in data_dirs:
        # Check in subfolder
        audio_path = data_dir / prefix / f"{utt_id}.flac"
        if audio_path.exists():
            return audio_path

        # Also try direct in data folder
        audio_path = data_dir / f"{utt_id}.flac"
        if audio_path.exists():
            return audio_path

        # Try wav format
        audio_path = data_dir / prefix / f"{utt_id}.wav"
        if audio_path.exists():
            return audio_path

    return None


def parse_all_metadata(tsv_files):
    """Parse all utt_spk_text.tsv files."""
    entries = []

    for tsv_path in tsv_files:
        folder_name = tsv_path.parent.parent.name  # e.g., asr_sundanese_0
        data_dir = tsv_path.parent / "data"

        try:
            with open(tsv_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    # Format: utterance_id\tspeaker_id\ttranscription
                    parts = line.split('\t')

                    if len(parts) >= 3:
                        utt_id = parts[0].strip()
                        speaker_id = parts[1].strip()
                        transcription = parts[2].strip()

                        entries.append({
                            'utt_id': utt_id,
                            'speaker_id': speaker_id,
                            'transcription': transcription,
                            'data_dir': str(data_dir),
                            'source': folder_name
                        })
                    elif len(parts) == 2:
                        # Fallback: utt_id\ttranscription
                        utt_id = parts[0].strip()
                        transcription = parts[1].strip()

                        entries.append({
                            'utt_id': utt_id,
                            'speaker_id': 'unknown',
                            'transcription': transcription,
                            'data_dir': str(data_dir),
                            'source': folder_name
                        })

        except Exception as e:
            print(f"  Warning: Error reading {tsv_path}: {e}")
            continue

        print(f"  Parsed {tsv_path.parent.parent.name}: {line_num:,} entries")

    return entries


def process_entries(entries, start_idx=0, existing_valid=None):
    """Process metadata entries and match with audio files."""
    valid_entries = existing_valid if existing_valid else []
    stats = Counter()

    total = len(entries)

    # Group data_dirs for faster lookup
    data_dirs_cache = {}

    for i, entry in enumerate(entries[start_idx:], start_idx):
        utt_id = entry['utt_id']
        transcription = entry['transcription']
        data_dir = Path(entry['data_dir'])

        # Check text length
        if len(transcription) < MIN_TEXT_LENGTH:
            stats['short_text'] += 1
            continue

        # Find audio file
        if str(data_dir) not in data_dirs_cache:
            data_dirs_cache[str(data_dir)] = data_dir

        audio_path = find_audio_file(utt_id, [data_dir])
        if audio_path is None:
            stats['missing_audio'] += 1
            continue

        # Check duration (optional - skip if no soundfile/mutagen)
        duration = get_audio_duration(audio_path)
        if duration is None:
            # If we can't get duration, assume it's valid
            duration = 5.0  # Default estimate
            stats['duration_estimated'] += 1
        elif duration < MIN_DURATION_SEC:
            stats['too_short'] += 1
            continue
        elif duration > MAX_DURATION_SEC:
            stats['too_long'] += 1
            continue

        # Valid entry
        valid_entries.append({
            'audio_path': str(audio_path.absolute()),
            'transcription': transcription,
            'duration': duration,
            'utt_id': utt_id,
            'speaker_id': entry.get('speaker_id', 'unknown')
        })
        stats['valid'] += 1

        # Progress and checkpoint
        processed = i + 1
        if processed % 20000 == 0:
            pct = (processed / total) * 100
            print(f"  Processed {processed:,}/{total:,} ({pct:.1f}%) - Valid: {stats['valid']:,}")

        if processed % CHECKPOINT_INTERVAL == 0:
            checkpoint = {
                "stage": "processing",
                "processed_entries": processed,
                "valid_entries": valid_entries,
                "stats": dict(stats),
            }
            save_checkpoint(checkpoint)

    return valid_entries, dict(stats)


def create_manifest(entries, split_name, output_dir):
    """Create manifest JSON file for a split."""
    manifest_path = output_dir / f"{split_name}.json"

    try:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error creating {split_name}.json: {e}")
        return False


def create_csv(entries, split_name, output_dir):
    """Create CSV file for a split."""
    csv_path = output_dir / f"{split_name}.csv"

    try:
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("audio_path,transcription,duration\n")
            for entry in entries:
                text = entry['transcription'].replace('"', '""')
                f.write(f"\"{entry['audio_path']}\",\"{text}\",{entry['duration']:.2f}\n")
        return True
    except Exception as e:
        print(f"Error creating {split_name}.csv: {e}")
        return False


def main():
    print("=" * 60)
    print("OpenSLR 36 Dataset Preparation")
    print("For manually extracted folders")
    print("=" * 60)
    print(f"\nSource: {DATASET_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Check directory exists
    if not DATASET_DIR.exists():
        print(f"Error: Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)

    # Find all TSV files
    print("Scanning for metadata files...")
    tsv_files = find_all_tsv_files(DATASET_DIR)

    if not tsv_files:
        print("Error: No utt_spk_text.tsv files found!")
        print("Expected structure: openslr36/asr_sundanese_X/asr_sundanese/utt_spk_text.tsv")
        sys.exit(1)

    print(f"Found {len(tsv_files)} metadata files:")
    for tsv in tsv_files:
        print(f"  - {tsv.parent.parent.name}")

    # Check for missing folders (like asr_sundanese_7)
    expected = set(f"asr_sundanese_{i:x}" for i in range(16))
    found = set(tsv.parent.parent.name for tsv in tsv_files)
    missing = expected - found
    if missing:
        print(f"\n[INFO] Missing folders (can be added later): {sorted(missing)}")

    # Load checkpoint
    checkpoint = load_checkpoint()
    start_idx = checkpoint.get("processed_entries", 0)
    existing_valid = checkpoint.get("valid_entries", [])

    if start_idx > 0 and checkpoint.get("stage") != "complete":
        print(f"\nResuming from checkpoint: {start_idx:,} entries already processed")
        print(f"Valid entries so far: {len(existing_valid):,}")

    # Parse all metadata
    print("\nParsing metadata files...")
    entries = parse_all_metadata(tsv_files)

    if not entries:
        print("Error: No entries found in metadata files!")
        sys.exit(1)

    print(f"\nTotal entries in metadata: {len(entries):,}")

    # Count FLAC files
    flac_count = 0
    for tsv in tsv_files:
        data_dir = tsv.parent / "data"
        if data_dir.exists():
            flac_count += sum(1 for _ in data_dir.rglob("*.flac"))
    print(f"Total FLAC files found: {flac_count:,}")

    # Process entries
    print("\nProcessing entries and matching audio files...")
    if start_idx > 0 and checkpoint.get("stage") != "complete":
        print(f"(Resuming from entry {start_idx:,})")
        valid_entries, stats = process_entries(
            entries,
            start_idx=start_idx,
            existing_valid=existing_valid
        )
        # Merge stats
        old_stats = checkpoint.get("stats", {})
        for key, value in old_stats.items():
            stats[key] = stats.get(key, 0) + value
    else:
        valid_entries, stats = process_entries(entries)

    print(f"\nFiltering statistics:")
    print(f"  Valid entries: {stats.get('valid', 0):,}")
    print(f"  Missing audio: {stats.get('missing_audio', 0):,}")
    print(f"  Duration estimated: {stats.get('duration_estimated', 0):,}")
    print(f"  Too short (<{MIN_DURATION_SEC}s): {stats.get('too_short', 0):,}")
    print(f"  Too long (>{MAX_DURATION_SEC}s): {stats.get('too_long', 0):,}")
    print(f"  Short text: {stats.get('short_text', 0):,}")

    if not valid_entries:
        print("\nError: No valid entries found!")
        sys.exit(1)

    # Calculate total duration
    total_duration = sum(e['duration'] for e in valid_entries)
    print(f"\nTotal audio duration: {total_duration/3600:.2f} hours (estimated)")

    # Note about missing files
    missing_pct = (stats.get('missing_audio', 0) / len(entries)) * 100 if entries else 0
    if missing_pct > 5:
        print(f"\n[INFO] {missing_pct:.1f}% of audio files are missing.")
        print("This is expected if some folders (like asr_sundanese_7) are not yet extracted.")

    # Shuffle and split
    print("\nShuffling and splitting dataset...")
    random.seed(RANDOM_SEED)
    random.shuffle(valid_entries)

    n_total = len(valid_entries)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_entries = valid_entries[:n_train]
    val_entries = valid_entries[n_train:n_train + n_val]
    test_entries = valid_entries[n_train + n_val:]

    print(f"  Train: {len(train_entries):,} samples ({len(train_entries)/n_total*100:.1f}%)")
    print(f"  Validation: {len(val_entries):,} samples ({len(val_entries)/n_total*100:.1f}%)")
    print(f"  Test: {len(test_entries):,} samples ({len(test_entries)/n_total*100:.1f}%)")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save manifests
    print("\nSaving manifests...")

    create_manifest(train_entries, 'train', OUTPUT_DIR)
    create_manifest(val_entries, 'validation', OUTPUT_DIR)
    create_manifest(test_entries, 'test', OUTPUT_DIR)
    print("  [OK] JSON manifests created")

    create_csv(train_entries, 'train', OUTPUT_DIR)
    create_csv(val_entries, 'validation', OUTPUT_DIR)
    create_csv(test_entries, 'test', OUTPUT_DIR)
    print("  [OK] CSV manifests created")

    # Save metadata
    metadata = {
        'total_samples': n_total,
        'train_samples': len(train_entries),
        'val_samples': len(val_entries),
        'test_samples': len(test_entries),
        'total_duration_hours': total_duration / 3600,
        'audio_format': 'flac',
        'source': 'OpenSLR 36 - Large Sundanese ASR Dataset',
        'missing_folders': list(missing) if missing else [],
        'created_at': datetime.now().isoformat()
    }

    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("  [OK] Metadata saved")

    # Mark complete
    save_checkpoint({
        "stage": "complete",
        "processed_entries": len(entries),
        "valid_count": len(valid_entries),
        "stats": stats,
    })

    # Summary
    print("\n" + "=" * 60)
    print("Preparation Complete")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Files created:")
    print(f"  - train.json ({len(train_entries):,} samples)")
    print(f"  - validation.json ({len(val_entries):,} samples)")
    print(f"  - test.json ({len(test_entries):,} samples)")
    print(f"\nNext step: Start training")
    print(f"  python scripts/train_whisper_small_openslr.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.")
        print("Run the script again to resume.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
