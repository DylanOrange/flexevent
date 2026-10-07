import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.detach().cpu().contiguous() for key, value in state.items()}, args.output)


if __name__ == "__main__":
    main()
