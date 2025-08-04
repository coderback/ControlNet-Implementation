"""
Example inference script for ControlNet.

This demonstrates how to generate images using a trained ControlNet model.
"""

import sys
from pathlib import Path
import argparse
from PIL import Image
import numpy as np
import cv2
import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.inference.generate import ControlNetInference


def create_canny_condition(image_path: str, low_threshold: int = 100, high_threshold: int = 200) -> np.ndarray:
    """Create Canny edge condition from an image."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges


def main():
    parser = argparse.ArgumentParser(description="Generate images with ControlNet")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--condition", type=str, required=True, help="Path to conditioning image")
    parser.add_argument("--output", type=str, default="generated.png", help="Output image path")
    parser.add_argument("--controlnet-path", type=str, help="Path to trained ControlNet weights")
    parser.add_argument("--condition-type", type=str, default="canny", help="Type of conditioning")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="CFG guidance scale")
    parser.add_argument("--controlnet-scale", type=float, default=1.0, help="ControlNet conditioning scale")  
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    
    args = parser.parse_args()
    
    print("Initializing ControlNet inference...")
    
    # Initialize inference
    inference = ControlNetInference(
        controlnet_path=args.controlnet_path,
        condition_type=args.condition_type,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"Generating image with prompt: '{args.prompt}'")
    print(f"Using condition: {args.condition}")
    
    # Generate image
    try:
        image = inference.generate(
            prompt=args.prompt,
            condition_input=args.condition,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_scale,
            image_resolution=args.resolution,
            seed=args.seed
        )
        
        # Save result
        image.save(args.output)
        print(f"Generated image saved to {args.output}")
        
        # Also save a comparison if we have the condition image
        if Path(args.condition).exists():
            condition_img = Image.open(args.condition).resize((args.resolution, args.resolution))
            
            # Create comparison image
            comparison = Image.new('RGB', (args.resolution * 2, args.resolution))
            comparison.paste(condition_img, (0, 0))
            comparison.paste(image, (args.resolution, 0))
            
            comparison_path = Path(args.output).with_name(f"comparison_{Path(args.output).name}")
            comparison.save(comparison_path)
            print(f"Comparison image saved to {comparison_path}")
            
    except Exception as e:
        print(f"Error during generation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import torch
    exit(main())