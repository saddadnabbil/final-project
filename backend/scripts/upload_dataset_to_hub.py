#!/usr/bin/env python3
"""
Upload OpenSLR36 Sundanese Dataset to HuggingFace Hub
======================================================

Ini akan upload dataset ke HuggingFace Hub supaya bisa diakses dari RunPod.

BEFORE RUNNING:
1. Create HuggingFace account: https://huggingface.co/join
2. Create access token: https://huggingface.co/settings/tokens
   - Click "New token"
   - Name: "runpod-training"
   - Role: "Write"
   - Copy the token

3. Login via CLI:
   pip install huggingface_hub
   huggingface-cli login

4. Run this script:
   python scripts/upload_dataset_to_hub.py
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

# ============================================
# CONFIGURATION - EDIT THIS!
# ============================================

# Your HuggingFace username
HF_USERNAME = "YOUR_USERNAME"  # <-- CHANGE THIS!

# Dataset name on Hub
DATASET_NAME = "openslr36-sundanese-asr"

# Local paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MANIFESTS_DIR = PROJECT_ROOT / "backend" / "dataset" / "openslr36_prepared"
AUDIO_BASE = PROJECT_ROOT / "backend" / "dataset" / "openslr36"


def check_huggingface_login():
    """Check if user is logged in to HuggingFace."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Logged in as: {user_info['name']}")
        return user_info['name']
    except Exception as e:
        print("✗ Not logged in to HuggingFace!")
        print("\nTo login:")
        print("  1. pip install huggingface_hub")
        print("  2. huggingface-cli login")
        print("  3. Paste your token from https://huggingface.co/settings/tokens")
        return None


def load_manifest_with_audio(manifest_path: Path, split_name: str):
    """Load manifest and verify audio files exist."""
    print(f"\nLoading {split_name}...")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    valid_entries = []
    missing_count = 0

    for entry in tqdm(entries, desc=f"  Checking {split_name}"):
        audio_path = Path(entry['audio_path'])

        if audio_path.exists():
            valid_entries.append({
                'audio_path': str(audio_path),
                'transcription': entry['transcription'],
                'duration': entry.get('duration', 0),
                'utt_id': entry.get('utt_id', ''),
                'speaker_id': entry.get('speaker_id', ''),
            })
        else:
            missing_count += 1

    print(f"  Valid: {len(valid_entries):,}, Missing: {missing_count}")
    return valid_entries


def create_and_upload_dataset(username: str):
    """Create HuggingFace dataset and upload."""
    from datasets import Dataset, DatasetDict, Audio, Features, Value

    full_dataset_name = f"{username}/{DATASET_NAME}"
    print(f"\n{'='*60}")
    print(f"Creating dataset: {full_dataset_name}")
    print(f"{'='*60}")

    # Load all splits
    splits_data = {}

    for split_name in ['train', 'validation', 'test']:
        manifest_file = MANIFESTS_DIR / f"{split_name}.json"

        if manifest_file.exists():
            entries = load_manifest_with_audio(manifest_file, split_name)
            splits_data[split_name] = entries
        else:
            print(f"  Warning: {split_name}.json not found")

    if not splits_data:
        print("ERROR: No data found!")
        return

    # Create DatasetDict
    print("\nCreating Dataset objects...")

    dataset_dict = {}
    for split_name, entries in splits_data.items():
        print(f"  Creating {split_name}...")

        # Create dataset from dict
        ds = Dataset.from_dict({
            'audio': [e['audio_path'] for e in entries],
            'transcription': [e['transcription'] for e in entries],
            'duration': [e['duration'] for e in entries],
            'utt_id': [e['utt_id'] for e in entries],
            'speaker_id': [e['speaker_id'] for e in entries],
        })

        # Cast audio column
        ds = ds.cast_column('audio', Audio(sampling_rate=16000))

        dataset_dict[split_name] = ds

    dataset = DatasetDict(dataset_dict)

    print(f"\nDataset summary:")
    for split_name, ds in dataset.items():
        print(f"  {split_name}: {len(ds):,} samples")

    # Upload to Hub
    print(f"\n{'='*60}")
    print(f"Uploading to HuggingFace Hub...")
    print(f"{'='*60}")
    print(f"  Repository: {full_dataset_name}")
    print(f"  This may take 1-3 hours for ~15GB of audio...")
    print()

    dataset.push_to_hub(
        full_dataset_name,
        private=False,  # Set True if you want private
        max_shard_size="500MB",
    )

    print(f"\n✓ Dataset uploaded successfully!")
    print(f"\nDataset URL: https://huggingface.co/datasets/{full_dataset_name}")

    return full_dataset_name


def main():
    print("=" * 60)
    print("Upload OpenSLR36 Sundanese to HuggingFace Hub")
    print("=" * 60)

    # Check paths
    if not MANIFESTS_DIR.exists():
        print(f"\nERROR: Manifests not found at {MANIFESTS_DIR}")
        print("Run: python scripts/prepare_openslr36_dataset.py first")
        return

    if not AUDIO_BASE.exists():
        print(f"\nERROR: Audio not found at {AUDIO_BASE}")
        return

    # Check login
    username = check_huggingface_login()
    if not username:
        return

    # Update username if placeholder
    global HF_USERNAME
    if HF_USERNAME == "YOUR_USERNAME":
        HF_USERNAME = username
        print(f"  Using detected username: {HF_USERNAME}")

    # Confirm
    print(f"\nWill upload to: {HF_USERNAME}/{DATASET_NAME}")
    print("This will upload ~15GB of audio files.")

    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Upload
    dataset_name = create_and_upload_dataset(HF_USERNAME)

    if dataset_name:
        print(f"\n{'='*60}")
        print("NEXT STEPS FOR RUNPOD:")
        print(f"{'='*60}")
        print(f"\n1. Edit scripts/runpod_whisper_training.py:")
        print(f"   dataset_name = \"{dataset_name}\"")
        print(f"   hub_model_id = \"{HF_USERNAME}/whisper-medium-sundanese\"")
        print(f"\n2. In RunPod, run:")
        print(f"   pip install datasets transformers evaluate jiwer accelerate soundfile librosa")
        print(f"   # Copy training script")
        print(f"   python runpod_whisper_training.py")


if __name__ == "__main__":
    main()
