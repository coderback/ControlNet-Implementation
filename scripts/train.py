"""
Training entrypoint. Launch with accelerate:

    accelerate launch scripts/train.py --config configs/smoke_local.yaml

Loads a YAML config (OmegaConf) and hands off to src.training.trainer.train.
"""

import argparse

from omegaconf import OmegaConf

from src.training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train ControlNet")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config in configs/")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. training.max_train_steps=500")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))

    train(config)


if __name__ == "__main__":
    main()
