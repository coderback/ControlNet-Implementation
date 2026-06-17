# RunPod training image for ControlNet-Implementation.
# Base ships with CUDA 12.1 + torch, so we skip the cu128 torch pins from requirements.txt
# and install only the rest. Build:  docker build -f docker/Dockerfile -t controlnet-train .
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface

WORKDIR /workspace

# System deps for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

# Install python deps (torch already present in base image)
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir \
    diffusers>=0.34.0 transformers>=4.54.1 accelerate>=1.9.0 \
    opencv-python-headless "numpy>=2.0" Pillow tqdm omegaconf einops \
    wandb matplotlib "albumentations<1.4.0" controlnet-aux \
    torchmetrics clean-fid scikit-image

COPY . .
RUN pip install --no-cache-dir -e . || true

ENTRYPOINT ["bash", "docker/runpod_entrypoint.sh"]
