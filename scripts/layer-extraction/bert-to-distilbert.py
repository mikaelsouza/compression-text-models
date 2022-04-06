import argparse
from typing import List
import torch
import transformers
import json
import logging

if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Script to extract layers from BERTimbau."
    )
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    logging.info(f"Loading script configuration from: {args.config_path}")
    with open(args.config_path, "r") as f:
        extraction_config = json.load(f)

    logging.info(f"Loaded configuration: {extraction_config}")

    model_type: str = extraction_config["model_type"]
    model_name: str = extraction_config["model_name"]
    vocab_transform: bool = extraction_config["vocab_transform"]
    extraction_idx: List[int] = extraction_config["extraction_idx"]

    
