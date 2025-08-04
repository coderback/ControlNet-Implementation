"""
Analyze and validate your dataset for ControlNet training.

This script helps you understand what types of images you have
and whether they're suitable for your conditioning type.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel


class DatasetAnalyzer:
    """Analyze dataset suitability for ControlNet training."""
    
    def __init__(self):
        self.stats = defaultdict(list)
        self.image_info = []
        
        # Load CLIP for content analysis
        try:
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.has_clip = True
        except:
            print("CLIP not available - skipping content analysis")
            self.has_clip = False
    
    def analyze_image_properties(self, image_path: Path):
        """Analyze basic image properties."""
        try:
            # Load with PIL for basic info
            pil_image = Image.open(image_path)
            width, height = pil_image.size
            
            # Load with OpenCV for advanced analysis
            cv_image = cv2.imread(str(image_path))
            if cv_image is None:
                return None
                
            # Basic properties
            info = {
                "path": str(image_path),
                "width": width,
                "height": height,
                "aspect_ratio": width / height,
                "file_size": image_path.stat().st_size / (1024 * 1024),  # MB
            }
            
            # Color analysis
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            info["brightness"] = np.mean(gray)
            info["contrast"] = np.std(gray)
            
            # Edge density (important for Canny conditioning)
            edges = cv2.Canny(gray, 50, 150)
            info["edge_density"] = np.sum(edges > 0) / (width * height)
            
            # Color saturation
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            info["saturation"] = np.mean(hsv[:, :, 1])
            
            # Blur detection (Laplacian variance)
            info["sharpness"] = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            return info
            
        except Exception as e:
            print(f"Error analyzing {image_path}: {e}")
            return None
    
    def analyze_content_diversity(self, image_paths: list, batch_size: int = 32):
        """Analyze content diversity using CLIP."""
        if not self.has_clip:
            return {}
        
        # Categories to check for
        categories = [
            "person", "people", "human", "face", "portrait",
            "building", "architecture", "house", "room", "interior",
            "landscape", "nature", "tree", "mountain", "sky",
            "vehicle", "car", "truck", "airplane", "boat",
            "animal", "dog", "cat", "bird", "wildlife",
            "food", "restaurant", "kitchen", "cooking",
            "art", "painting", "drawing", "abstract",
            "technology", "computer", "phone", "electronics"
        ]
        
        category_scores = defaultdict(list)
        
        print("Analyzing content diversity with CLIP...")
        
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load images
            images = []
            valid_paths = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    images.append(image)
                    valid_paths.append(path)
                except:
                    continue
            
            if not images:
                continue
            
            # Process with CLIP
            inputs = self.clip_processor(
                text=categories,
                images=images,
                return_tensors="pt",
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Store results
            for img_idx, path in enumerate(valid_paths):
                for cat_idx, category in enumerate(categories):
                    score = probs[img_idx, cat_idx].item()
                    if score > 0.1:  # Only store significant matches
                        category_scores[category].append((str(path), score))
        
        return category_scores
    
    def check_conditioning_suitability(self, image_info: list, condition_type: str):
        """Check if images are suitable for specific conditioning type."""
        recommendations = []
        
        if condition_type == "canny":
            # Check edge density
            low_edge_images = [img for img in image_info if img["edge_density"] < 0.05]
            if low_edge_images:
                recommendations.append(f"⚠️  {len(low_edge_images)} images have very low edge density (< 5%)")
                recommendations.append("   Consider adding more images with clear object boundaries")
            
            # Check sharpness
            blurry_images = [img for img in image_info if img["sharpness"] < 100]
            if blurry_images:
                recommendations.append(f"⚠️  {len(blurry_images)} images appear blurry")
                recommendations.append("   Blurry images don't work well with Canny edge detection")
        
        elif condition_type == "depth":
            # Check for flat/2D images (low contrast might indicate flat scenes)
            flat_images = [img for img in image_info if img["contrast"] < 30]
            if flat_images:
                recommendations.append(f"⚠️  {len(flat_images)} images have low contrast")
                recommendations.append("   Low contrast might indicate flat scenes unsuitable for depth")
        
        elif condition_type == "pose":
            recommendations.append("💡 For pose conditioning, ensure images contain:")
            recommendations.append("   - Clear, unobstructed human figures")
            recommendations.append("   - Variety of poses and viewpoints")
            recommendations.append("   - Different demographics and clothing styles")
        
        return recommendations
    
    def generate_report(self, image_dir: Path, condition_type: str = "canny", max_images: int = None):
        """Generate comprehensive dataset analysis report."""
        print(f"Analyzing dataset: {image_dir}")
        print(f"Condition type: {condition_type}")
        
        # Find images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(image_dir.glob(f"**/*{ext}")))
            image_paths.extend(list(image_dir.glob(f"**/*{ext.upper()}")))
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"Found {len(image_paths)} images")
        
        # Analyze image properties
        print("Analyzing image properties...")
        valid_images = []
        
        for image_path in tqdm(image_paths):
            info = self.analyze_image_properties(image_path)
            if info:
                valid_images.append(info)
                self.image_info.append(info)
        
        print(f"Successfully analyzed {len(valid_images)} images")
        
        # Content diversity analysis
        content_analysis = self.analyze_content_diversity(image_paths[:min(1000, len(image_paths))])
        
        # Generate statistics
        report = self.create_report(valid_images, content_analysis, condition_type)
        
        return report
    
    def create_report(self, image_info: list, content_analysis: dict, condition_type: str):
        """Create detailed analysis report."""
        if not image_info:
            return {"error": "No valid images found"}
        
        # Basic statistics
        widths = [img["width"] for img in image_info]
        heights = [img["height"] for img in image_info]
        aspect_ratios = [img["aspect_ratio"] for img in image_info]
        file_sizes = [img["file_size"] for img in image_info]
        
        report = {
            "dataset_summary": {
                "total_images": len(image_info),
                "avg_resolution": f"{np.mean(widths):.0f}x{np.mean(heights):.0f}",
                "resolution_range": f"{min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}",
                "avg_file_size_mb": f"{np.mean(file_sizes):.2f}",
                "total_size_gb": f"{sum(file_sizes) / 1024:.2f}"
            },
            
            "image_quality": {
                "avg_brightness": f"{np.mean([img['brightness'] for img in image_info]):.1f}",
                "avg_contrast": f"{np.mean([img['contrast'] for img in image_info]):.1f}",
                "avg_saturation": f"{np.mean([img['saturation'] for img in image_info]):.1f}",
                "avg_sharpness": f"{np.mean([img['sharpness'] for img in image_info]):.1f}",
            },
            
            "conditioning_analysis": {
                "condition_type": condition_type,
                "avg_edge_density": f"{np.mean([img['edge_density'] for img in image_info]):.3f}",
                "suitability_recommendations": self.check_conditioning_suitability(image_info, condition_type)
            }
        }
        
        # Add content diversity if available
        if content_analysis:
            top_categories = sorted(content_analysis.items(), key=lambda x: len(x[1]), reverse=True)[:10]
            report["content_diversity"] = {
                "top_categories": [(cat, len(matches)) for cat, matches in top_categories],
                "total_categories_detected": len(content_analysis)
            }
        
        return report
    
    def save_report(self, report: dict, output_path: Path):
        """Save report to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze dataset for ControlNet training")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--condition-type", type=str, default="canny", help="Type of conditioning")
    parser.add_argument("--output", type=str, default="dataset_analysis.json", help="Output report file")
    parser.add_argument("--max-images", type=int, help="Maximum number of images to analyze")
    
    args = parser.parse_args()
    
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Directory does not exist: {image_dir}")
        return
    
    # Run analysis
    analyzer = DatasetAnalyzer()
    report = analyzer.generate_report(image_dir, args.condition_type, args.max_images)
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET ANALYSIS SUMMARY")
    print("="*50)
    
    if "error" in report:
        print(f"❌ {report['error']}")
        return
    
    summary = report["dataset_summary"]
    quality = report["image_quality"]
    conditioning = report["conditioning_analysis"]
    
    print(f"📊 Total Images: {summary['total_images']}")
    print(f"📐 Average Resolution: {summary['avg_resolution']}")
    print(f"💾 Total Size: {summary['total_size_gb']} GB")
    print(f"✨ Average Sharpness: {quality['avg_sharpness']}")
    print(f"🎨 Edge Density: {conditioning['avg_edge_density']}")
    
    if "content_diversity" in report:
        print(f"🏷️  Content Categories: {report['content_diversity']['total_categories_detected']}")
        print("📂 Top Categories:")
        for cat, count in report["content_diversity"]["top_categories"][:5]:
            print(f"   - {cat}: {count} images")
    
    print("\n💡 Recommendations:")
    for rec in conditioning["suitability_recommendations"]:
        print(f"   {rec}")
    
    # Save detailed report
    analyzer.save_report(report, Path(args.output))


if __name__ == "__main__":
    main()