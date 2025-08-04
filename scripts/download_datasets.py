"""
Download and setup popular datasets for ControlNet training.

This script helps download and prepare common datasets used for ControlNet training.
"""

import requests
import zipfile
import tarfile
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from huggingface_hub import hf_hub_download, snapshot_download


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_fill50k():
    """Download the Fill50K dataset (commonly used for ControlNet)."""
    print("Downloading Fill50K dataset...")
    
    # This is a placeholder - you'd need to implement actual download logic
    # The original Fill50K dataset links may not be publicly available
    
    dataset_info = {
        "name": "Fill50K",
        "description": "Dataset with 50K images and various conditioning types",
        "size": "~2GB",
        "conditions": ["canny", "depth", "pose", "normal", "seg"],
        "note": "You may need to create this dataset yourself or find alternative sources"
    }
    
    return dataset_info


def download_coco_with_conditions(output_dir: Path):
    """Download COCO validation set and generate conditions."""
    print("Setting up COCO dataset with synthetic conditions...")
    
    coco_dir = output_dir / "coco_controlnet"
    coco_dir.mkdir(parents=True, exist_ok=True)
    
    # Download COCO validation images (smaller subset)
    coco_url = "http://images.cocodataset.org/zips/val2017.zip"
    coco_zip = coco_dir / "val2017.zip"
    
    if not coco_zip.exists():
        print("Downloading COCO validation images...")
        download_file(coco_url, coco_zip)
        
        # Extract
        with zipfile.ZipFile(coco_zip, 'r') as zip_ref:
            zip_ref.extractall(coco_dir)
    
    # Download COCO annotations
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" 
    ann_zip = coco_dir / "annotations.zip"
    
    if not ann_zip.exists():
        print("Downloading COCO annotations...")
        download_file(ann_url, ann_zip)
        
        with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
            zip_ref.extractall(coco_dir)
    
    return coco_dir


def create_dataset_from_huggingface(dataset_name: str, output_dir: Path):
    """Download dataset from Hugging Face Hub."""
    try:
        print(f"Downloading {dataset_name} from Hugging Face...")
        
        # Popular ControlNet datasets on HuggingFace
        datasets = {
            "fusing/fill50k": "Fill50K dataset with multiple conditions",
            "lllyasviel/ControlNet": "Original ControlNet training data",
            "multimodalart/controlnet-canny": "Canny edge dataset",
            "multimodalart/controlnet-depth": "Depth map dataset"
        }
        
        if dataset_name in datasets:
            snapshot_download(
                repo_id=dataset_name,
                local_dir=output_dir / dataset_name.replace("/", "_"),
                repo_type="dataset"
            )
            return True
        else:
            print(f"Unknown dataset: {dataset_name}")
            print("Available datasets:")
            for name, desc in datasets.items():
                print(f"  {name}: {desc}")
            return False
            
    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")
        return False


def setup_custom_dataset_from_images(image_dir: Path, output_dir: Path, condition_type: str = "canny"):
    """Create a ControlNet dataset from a directory of images."""
    print(f"Creating {condition_type} dataset from {image_dir}...")
    
    # This would use the synthetic dataset creator
    import subprocess
    import sys
    
    script_path = Path(__file__).parent / "create_synthetic_dataset.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--input-dir", str(image_dir),
        "--output-dir", str(output_dir),
        "--condition-type", condition_type,
        "--generate-captions"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Dataset created successfully!")
        return True
    else:
        print(f"Error creating dataset: {result.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download and setup ControlNet datasets")
    parser.add_argument("--output-dir", type=str, default="./datasets", help="Output directory")
    parser.add_argument("--dataset", type=str, choices=[
        "fill50k", "coco", "custom", "huggingface"
    ], default="coco", help="Dataset to download")
    parser.add_argument("--huggingface-name", type=str, help="Hugging Face dataset name")
    parser.add_argument("--custom-images", type=str, help="Directory of images for custom dataset")
    parser.add_argument("--condition-type", type=str, default="canny", help="Condition type for custom dataset")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == "fill50k":
        info = download_fill50k()
        print(json.dumps(info, indent=2))
        
    elif args.dataset == "coco":
        coco_dir = download_coco_with_conditions(output_dir)
        print(f"COCO dataset downloaded to: {coco_dir}")
        print("Next steps:")
        print(f"  1. Run: python scripts/create_synthetic_dataset.py --input-dir {coco_dir}/val2017 --output-dir {output_dir}/coco_controlnet --condition-type canny")
        print("  2. This will create Canny edge conditions for COCO images")
        
    elif args.dataset == "huggingface":
        if not args.huggingface_name:
            print("Please specify --huggingface-name")
            return
        success = create_dataset_from_huggingface(args.huggingface_name, output_dir)
        if success:
            print(f"Dataset downloaded to: {output_dir}")
            
    elif args.dataset == "custom":
        if not args.custom_images:
            print("Please specify --custom-images directory")
            return
        custom_dir = Path(args.custom_images)
        if not custom_dir.exists():
            print(f"Directory does not exist: {custom_dir}")
            return
        setup_custom_dataset_from_images(custom_dir, output_dir / "custom_dataset", args.condition_type)
    
    print("\nDataset setup complete!")
    print("\nNext steps:")
    print("1. Verify the dataset structure matches the expected format")
    print("2. Start training with:")
    print(f"   python examples/train_controlnet.py --data-root {output_dir}/your_dataset")


if __name__ == "__main__":
    main()