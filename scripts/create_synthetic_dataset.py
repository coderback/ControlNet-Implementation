"""
Create a synthetic ControlNet dataset from regular images.

This script processes a directory of images to create conditioning data
(Canny edges, depth maps, etc.) automatically.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from transformers import BlipProcessor, BlipForConditionalGeneration
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.condition_encoder import preprocess_canny, preprocess_depth


def generate_canny_edges(image_path: Path, output_path: Path, low_threshold: int = 100, high_threshold: int = 200):
    """Generate Canny edges from an image."""
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Save as PNG for better quality
    cv2.imwrite(str(output_path), edges)
    return True


def generate_depth_map(image_path: Path, output_path: Path):
    """Generate depth map using MiDaS (requires torch hub)."""
    try:
        import torch
        
        # Load MiDaS model
        model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        model.eval()
        
        transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        transform = transforms.small_transform
        
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        input_tensor = transform(image_rgb).unsqueeze(0)
        
        # Generate depth
        with torch.no_grad():
            prediction = model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy and normalize
        depth = prediction.cpu().numpy()
        depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        
        # Save depth map
        cv2.imwrite(str(output_path), depth_normalized)
        return True
        
    except Exception as e:
        print(f"Error generating depth for {image_path}: {e}")
        return False


def generate_captions(image_paths: list, batch_size: int = 8):
    """Generate captions using BLIP model."""
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        captions = {}
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Generating captions"):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load and process images
            images = []
            valid_paths = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    images.append(image)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
            
            if not images:
                continue
            
            # Generate captions
            inputs = processor(images, return_tensors="pt", padding=True)
            out = model.generate(**inputs, max_length=50, num_beams=5)
            
            batch_captions = processor.batch_decode(out, skip_special_tokens=True)
            
            # Store captions
            for path, caption in zip(valid_paths, batch_captions):
                captions[path.stem] = caption
        
        return captions
        
    except Exception as e:
        print(f"Error generating captions: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Create synthetic ControlNet dataset")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for dataset")
    parser.add_argument("--condition-type", type=str, default="canny", choices=["canny", "depth"], help="Type of conditioning to generate")
    parser.add_argument("--generate-captions", action="store_true", help="Generate captions using BLIP")
    parser.add_argument("--max-images", type=int, help="Maximum number of images to process")
    parser.add_argument("--image-size", type=int, default=512, help="Resize images to this size")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "conditions" / args.condition_type).mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(input_dir.glob(f"*{ext}")))
        image_paths.extend(list(input_dir.glob(f"*{ext.upper()}")))
    
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process images
    processed_samples = []
    
    for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
        try:
            # Load and resize image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((args.image_size, args.image_size), Image.LANCZOS)
            
            # Save processed image
            output_image_path = output_dir / "images" / f"img_{i:06d}.jpg"
            image.save(output_image_path, quality=95)
            
            # Generate conditioning
            condition_path = output_dir / "conditions" / args.condition_type / f"img_{i:06d}.png"
            
            if args.condition_type == "canny":
                # Convert PIL to OpenCV format
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                cv2.imwrite(str(condition_path), edges)
                success = True
                
            elif args.condition_type == "depth":
                success = generate_depth_map(output_image_path, condition_path)
            
            else:
                success = False
            
            if success:
                processed_samples.append({
                    "image_path": str(output_image_path.relative_to(output_dir)),
                    "condition_path": str(condition_path.relative_to(output_dir)),
                    "prompt": "",
                    "original_path": str(image_path)
                })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    print(f"Successfully processed {len(processed_samples)} images")
    
    # Generate captions if requested
    if args.generate_captions and processed_samples:
        print("Generating captions...")
        image_paths_for_captions = [output_dir / sample["image_path"] for sample in processed_samples]
        captions = generate_captions(image_paths_for_captions)
        
        # Update samples with captions
        for sample in processed_samples:
            image_name = Path(sample["image_path"]).stem
            if image_name in captions:
                sample["prompt"] = captions[image_name]
    
    # Save dataset metadata
    train_split = int(len(processed_samples) * 0.9)
    train_samples = processed_samples[:train_split]
    val_samples = processed_samples[train_split:]
    
    with open(output_dir / "train.json", 'w') as f:
        json.dump(train_samples, f, indent=2)
    
    with open(output_dir / "val.json", 'w') as f:
        json.dump(val_samples, f, indent=2)
    
    print(f"Dataset created with {len(train_samples)} training and {len(val_samples)} validation samples")
    print(f"Dataset saved to: {output_dir}")
    
    # Create dataset info
    info = {
        "condition_type": args.condition_type,
        "image_size": args.image_size,
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "has_captions": args.generate_captions
    }
    
    with open(output_dir / "dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"  ├── images/           # {len(processed_samples)} processed images")
    print(f"  ├── conditions/")
    print(f"  │   └── {args.condition_type}/    # {len(processed_samples)} condition maps")
    print(f"  ├── train.json        # {len(train_samples)} training samples")
    print(f"  ├── val.json          # {len(val_samples)} validation samples")
    print(f"  └── dataset_info.json # Dataset metadata")


if __name__ == "__main__":
    main()