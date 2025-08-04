# ControlNet Implementation

A PyTorch implementation of ControlNet from the paper "Adding Conditional Control to Text-to-Image Diffusion Models" by Zhang et al.

## Overview

ControlNet allows precise control over image generation in text-to-image diffusion models by adding spatial conditioning inputs like:
- Canny edges
- Depth maps  
- Human poses
- Segmentation maps
- Surface normals
- User scribbles

## Features

- **Zero Convolution Architecture**: Implements the key innovation from the paper
- **Multiple Conditioning Types**: Support for Canny, depth, pose, and more
- **Training from Scratch**: Complete training pipeline with proper hyperparameters
- **Easy Inference**: Simple API for generating controlled images
- **Monitoring**: Built-in W&B logging and checkpointing
- **Modular Design**: Easy to extend for new conditioning types

## Installation

```bash
git clone https://github.com/coderback/ControlNet-Implementation.git
cd ControlNet-Implementation
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
python examples/train_controlnet.py \
    --data-root /path/to/your/dataset \
    --condition-type canny \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --num-epochs 100 \
    --output-dir ./outputs
```

### Inference

```bash
python examples/generate_images.py \
    --prompt "a beautiful landscape with mountains" \
    --condition /path/to/canny_edges.png \
    --controlnet-path ./outputs/controlnet-10000.pt \
    --output generated_image.png
```

### Programmatic Usage

```python
from src.inference.generate import ControlNetInference
from PIL import Image

# Initialize inference
inference = ControlNetInference(
    controlnet_path="path/to/controlnet.pt",
    condition_type="canny"
)

# Generate image
image = inference.generate(
    prompt="a futuristic city",
    condition_input="canny_edges.png",
    num_inference_steps=20,
    guidance_scale=7.5
)

# Save result
image.save("output.png")
```

## Dataset Preparation

ControlNet expects datasets in the following structure:

```
dataset/
├── images/           # Original images
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── conditions/       # Conditioning inputs
│   └── canny/       # or depth, pose, etc.
│       ├── img001.png
│       ├── img002.png
│       └── ...
└── prompts.json     # Optional text prompts
```

### Synthetic Dataset

For quick experimentation, you can use the synthetic Canny dataset:

```python
from src.data.dataset import SyntheticCannyDataset

dataset = SyntheticCannyDataset(
    image_dir="/path/to/images",
    image_size=512
)
```

## Architecture Details

### Zero Convolutions

The key innovation in ControlNet is the use of zero-initialized convolution layers:

```python
class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        # Zero initialization is crucial
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
```

### ControlNet Block

Each ControlNet block implements:
```
yc = F(x; Θ) + Z(F(x + Z(c; Θz1); Θc); Θz2)
```

Where:
- `F(x; Θ)` is the frozen original block
- `F(x + Z(c; Θz1); Θc)` is the trainable copy with conditioning
- `Z(·; ·)` are zero convolution layers

### Training Process

1. **Lock Original U-Net**: All parameters frozen
2. **Trainable Copy**: Create copy of encoder blocks + middle block
3. **Zero Convolutions**: Connect with zero-initialized layers
4. **Gradual Learning**: Model learns controls without affecting base model initially
5. **Sudden Convergence**: Typically occurs around 6-10K steps

## Supported Conditioning Types

| Type | Description | Input Format |
|------|-------------|--------------|
| `canny` | Canny edge detection | Grayscale edge maps |
| `depth` | Depth maps | Single-channel depth |
| `pose` | Human pose keypoints | RGB pose visualizations |
| `segmentation` | Semantic segmentation | RGB segmentation maps |
| `normal` | Surface normals | RGB normal maps |
| `scribble` | User scribbles | Binary/grayscale sketches |

## Training Tips

### Hyperparameters (from paper)

- **Learning Rate**: 1e-4
- **Batch Size**: 4-8 (depends on GPU memory)
- **Prompt Dropout**: 50% (replace prompts with empty strings)
- **Optimizer**: AdamW with cosine scheduling
- **Mixed Precision**: Recommended for faster training

### Hardware Requirements

- **Minimum**: 8GB VRAM (batch size 1-2)
- **Recommended**: 16GB+ VRAM (batch size 4-8)
- **Training Time**: ~2-5 days on RTX 3090 for good results

### Monitoring Training

Watch for the "sudden convergence phenomenon":
- Loss drops suddenly after 6-10K steps
- Model starts following conditioning inputs
- Quality improves dramatically

## Configuration

Training configuration can be customized via JSON:

```json
{
  "learning_rate": 1e-4,
  "batch_size": 4,
  "num_epochs": 100,
  "condition_type": "canny",
  "mixed_precision": true,
  "gradient_accumulation_steps": 1,
  "max_grad_norm": 1.0
}
```

## Evaluation

### Quantitative Metrics
- **FID Score**: Fréchet Inception Distance
- **CLIP Score**: Text-image alignment  
- **Condition Fidelity**: How well conditions are followed

### Qualitative Assessment
- Visual inspection of generated images
- Comparison with original ControlNet results
- User studies for preference ranking

## Limitations

- **Memory Usage**: Requires significant GPU memory
- **Training Time**: Long training times for good results
- **Dataset Dependency**: Quality depends heavily on training data
- **Conditioning Quality**: Better conditions → better results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Submit a pull request

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{zhang2023controlnet,
  title={Adding Conditional Control to Text-to-Image Diffusion Models},
  author={Zhang, Lvmin and Rao, Anyi and Agrawala, Maneesh},
  journal={arXiv preprint arXiv:2302.05543},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original ControlNet paper by Zhang et al.
- Hugging Face Diffusers library
- Stable Diffusion by Stability AI
- CLIP by OpenAI

## Troubleshooting

### Common Issues

**Out of Memory Errors**
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision training

**Slow Convergence**
- Check learning rate (try 1e-4 to 1e-5)
- Ensure prompt dropout is enabled
- Verify conditioning data quality

**Poor Results**
- Increase training steps (>50K recommended)
- Improve conditioning data quality
- Try different conditioning scales during inference