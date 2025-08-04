"""
Fix dataset metadata by removing references to missing files.
"""

import json
from pathlib import Path
import argparse

def fix_dataset_json(data_root: str, json_file: str):
    """Fix dataset JSON by removing missing files."""
    data_root = Path(data_root)
    json_path = data_root / json_file
    
    if not json_path.exists():
        print(f"JSON file not found: {json_path}")
        return
    
    # Load existing metadata
    with open(json_path, 'r') as f:
        samples = json.load(f)
    
    print(f"Original dataset has {len(samples)} samples")
    
    # Filter valid samples
    valid_samples = []
    
    for sample in samples:
        # Check image path
        image_path = data_root / sample["image_path"]
        condition_path = data_root / sample["condition_path"]
        
        if image_path.exists() and condition_path.exists():
            valid_samples.append(sample)
        else:
            print(f"Removing missing sample: {sample['image_path']}")
    
    print(f"Fixed dataset has {len(valid_samples)} valid samples")
    
    # Save fixed dataset
    backup_path = json_path.with_suffix('.json.backup')
    json_path.rename(backup_path)
    print(f"Backed up original to: {backup_path}")
    
    with open(json_path, 'w') as f:
        json.dump(valid_samples, f, indent=2)
    
    print(f"Fixed dataset saved to: {json_path}")

def main():
    parser = argparse.ArgumentParser(description="Fix dataset JSON files")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--json-files", nargs='+', default=["train.json", "val.json"], help="JSON files to fix")
    
    args = parser.parse_args()
    
    for json_file in args.json_files:
        print(f"\nFixing {json_file}...")
        fix_dataset_json(args.data_root, json_file)

if __name__ == "__main__":
    main()