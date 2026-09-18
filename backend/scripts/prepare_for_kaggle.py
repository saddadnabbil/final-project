#!/usr/bin/env python3
"""
Prepare Dataset for Kaggle Upload
==================================
Creates Kaggle-ready dataset package with audio files + manifests.

Output: kaggle_dataset/ folder ready to upload to Kaggle
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
MANIFESTS_DIR = Path(__file__).parent.parent / "backend" / "dataset" / "openslr36_prepared"
AUDIO_BASE = Path(__file__).parent.parent / "backend" / "dataset" / "openslr36"
OUTPUT_DIR = Path(__file__).parent.parent / "kaggle_dataset"

def create_kaggle_manifest(manifest_path, split_name, output_dir):
    """Create Kaggle-compatible manifest with relative paths."""

    print(f"\nProcessing {split_name}...")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    # Update paths to relative and collect audio files
    audio_files_to_copy = []
    kaggle_entries = []

    for entry in tqdm(entries, desc=f"  Updating {split_name}"):
        audio_path = Path(entry['audio_path'])

        # Skip if file doesn't exist
        if not audio_path.exists():
            print(f"  Warning: {audio_path} not found, skipping")
            continue

        # Get relative path from openslr36 root
        try:
            # Extract path after 'openslr36/'
            parts = audio_path.parts
            openslr_idx = parts.index('openslr36')
            rel_parts = parts[openslr_idx + 1:]  # everything after openslr36/
            rel_path = Path(*rel_parts)

            # New entry with Kaggle path
            kaggle_entry = entry.copy()
            kaggle_entry['audio_path'] = f"/kaggle/input/openslr36-audio/{rel_path}"
            kaggle_entries.append(kaggle_entry)

            # Track audio file to copy
            audio_files_to_copy.append((audio_path, rel_path))

        except (ValueError, IndexError) as e:
            print(f"  Warning: Could not parse path {audio_path}: {e}")
            continue

    # Save updated manifest
    manifest_out = output_dir / "manifests" / f"{split_name}.json"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_out, 'w', encoding='utf-8') as f:
        json.dump(kaggle_entries, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Saved {len(kaggle_entries):,} entries to {manifest_out}")

    return audio_files_to_copy


def copy_audio_files(audio_files, output_dir):
    """Copy audio files to output directory."""

    audio_out = output_dir / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    print(f"\nCopying {len(audio_files):,} audio files...")

    copied = 0
    for src_path, rel_path in tqdm(audio_files, desc="  Copying"):
        dest_path = audio_out / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if not dest_path.exists():
            shutil.copy2(src_path, dest_path)
            copied += 1

    print(f"  ✓ Copied {copied:,} files")


def create_dataset_metadata(output_dir, stats):
    """Create dataset-metadata.json for Kaggle."""

    metadata = {
        "title": "OpenSLR36 Sundanese ASR Dataset (Prepared for Whisper)",
        "id": "your-username/openslr36-sundanese-prepared",  # User must update
        "licenses": [{"name": "CC-BY-SA-4.0"}],
        "keywords": ["audio", "speech recognition", "sundanese", "whisper", "asr"],
        "description": (
            "OpenSLR 36 Sundanese ASR dataset prepared for Whisper fine-tuning.\n\n"
            f"- Total samples: {stats['total']:,}\n"
            f"- Train: {stats['train']:,}\n"
            f"- Validation: {stats.get('validation', 0):,}\n"
            f"- Test: {stats['test']:,}\n\n"
            "Format: FLAC audio files + JSON manifests"
        )
    }

    metadata_path = output_dir / "dataset-metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Created {metadata_path}")
    print("  ⚠️ REMEMBER: Update 'id' field with your Kaggle username!")


def main():
    print("=" * 60)
    print("Prepare Dataset for Kaggle Upload")
    print("=" * 60)

    # Check source
    if not MANIFESTS_DIR.exists():
        print(f"\nError: Manifests not found at {MANIFESTS_DIR}")
        print("Run: python scripts/prepare_openslr36_dataset.py first")
        return

    # Clear output dir
    if OUTPUT_DIR.exists():
        print(f"\nCleaning output directory...")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True)

    # Process each split
    all_audio_files = []
    stats = {}

    for split_name in ['train', 'validation', 'test']:
        manifest_file = MANIFESTS_DIR / f"{split_name}.json"

        if not manifest_file.exists():
            print(f"Warning: {manifest_file} not found, skipping")
            continue

        audio_files = create_kaggle_manifest(manifest_file, split_name, OUTPUT_DIR)
        all_audio_files.extend(audio_files)
        stats[split_name] = len(audio_files)

    stats['total'] = sum(stats.values())

    # Remove duplicates
    unique_audio = list(set(all_audio_files))
    print(f"\nUnique audio files: {len(unique_audio):,}")

    # Copy audio files
    copy_audio_files(unique_audio, OUTPUT_DIR)

    # Create metadata
    create_dataset_metadata(OUTPUT_DIR, stats)

    # Create README
    readme = OUTPUT_DIR / "README.md"
    with open(readme, 'w', encoding='utf-8') as f:
        f.write(f"""# OpenSLR36 Sundanese ASR Dataset

Prepared for Whisper fine-tuning.

## Contents

- `manifests/` - JSON manifests (train, validation, test)
- `audio/` - FLAC audio files organized by folder

## Statistics

- Total samples: {stats['total']:,}
- Train: {stats.get('train', 0):,}
- Validation: {stats.get('val', 0):,}
- Test: {stats.get('test', 0):,}

## Usage

```python
import json

# Load manifest
with open('manifests/train.json') as f:
    data = json.load(f)

# Each entry has:
# - audio_path: path to FLAC file
# - transcription: Sundanese text
# - duration: audio duration in seconds
```

## Source

OpenSLR 36: http://www.openslr.org/resources/36/
""")

    print("\n" + "=" * 60)
    print("Dataset Ready for Kaggle!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print(f"1. Edit {OUTPUT_DIR}/dataset-metadata.json")
    print(f"   - Change 'id' to: your-username/openslr36-sundanese-prepared")
    print(f"2. Zip entire 'kaggle_dataset' folder")
    print(f"3. Upload to Kaggle Datasets")
    print(f"\nEstimated size: ~{len(unique_audio) * 75 / 1024:.1f} MB")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
