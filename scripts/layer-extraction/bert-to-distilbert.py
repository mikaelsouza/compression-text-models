# Adapted from https://github.com/huggingface/transformers/blob/main/examples/research_projects/distillation/scripts/extract_distilbert.py


import argparse
import json
import logging
from pathlib import Path
from typing import List

import torch
from transformers import BertForMaskedLM


def main():
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Script to extract layers from BERTimbau."
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    config_path = args.config_path

    logging.info(f"Loading script configuration from: {config_path}")
    with open(config_path, "r") as f:
        extraction_config = json.load(f)

    logging.info(f"Loaded configuration: {extraction_config}")

    model_type: str = extraction_config["model_type"]
    model_name: str = extraction_config["model_name"]
    vocab_transform: bool = extraction_config["vocab_transform"]
    extraction_idx: List[int] = extraction_config["extraction_idx"]
    dump_path: Path = Path(extraction_config["dump_path"])

    logging.info(f"Loading {model_type} model: {model_name}")
    prefix = model_type
    model = BertForMaskedLM.from_pretrained(model_name)

    logging.info(f"Model Loaded\n" "Extracting model layers...")
    state_dict = model.state_dict()
    compressed_sd = dict()

    for w in ["word_embeddings", "position_embeddings"]:
        compressed_sd[f"distilbert.embeddings.{w}.weight"] = state_dict[
            f"{prefix}.embeddings.{w}.weight"
        ]
    for w in ["weight", "bias"]:
        compressed_sd[f"distilbert.embeddings.LayerNorm.{w}"] = state_dict[
            f"{prefix}.embeddings.LayerNorm.{w}"
        ]

    std_idx = 0
    for teacher_idx in extraction_idx:
        for w in ["weight", "bias"]:
            compressed_sd[
                f"distilbert.transformer.layer.{std_idx}.attention.q_lin.{w}"
            ] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.self.query.{w}"
            ]
            compressed_sd[
                f"distilbert.transformer.layer.{std_idx}.attention.k_lin.{w}"
            ] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.self.key.{w}"
            ]
            compressed_sd[
                f"distilbert.transformer.layer.{std_idx}.attention.v_lin.{w}"
            ] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.self.value.{w}"
            ]
            compressed_sd[
                f"distilbert.transformer.layer.{std_idx}.attention.out_lin.{w}"
            ] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.output.dense.{w}"
            ]
            compressed_sd[
                f"distilbert.transformer.layer.{std_idx}.sa_layer_norm.{w}"
            ] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.attention.output.LayerNorm.{w}"
            ]
            compressed_sd[
                f"distilbert.transformer.layer.{std_idx}.ffn.lin1.{w}"
            ] = state_dict[
                f"{prefix}.encoder.layer.{teacher_idx}.intermediate.dense.{w}"
            ]
            compressed_sd[
                f"distilbert.transformer.layer.{std_idx}.ffn.lin2.{w}"
            ] = state_dict[f"{prefix}.encoder.layer.{teacher_idx}.output.dense.{w}"]
            compressed_sd[
                f"distilbert.transformer.layer.{std_idx}.output_layer_norm.{w}"
            ] = state_dict[f"{prefix}.encoder.layer.{teacher_idx}.output.LayerNorm.{w}"]
        std_idx += 1

    compressed_sd["vocab_projector.weight"] = state_dict[
        "cls.predictions.decoder.weight"
    ]
    compressed_sd["vocab_projector.bias"] = state_dict["cls.predictions.bias"]

    if vocab_transform:
        for w in ["weight", "bias"]:
            compressed_sd[f"vocab_transform.{w}"] = state_dict[
                f"cls.predictions.transform.dense.{w}"
            ]
            compressed_sd[f"vocab_layer_norm.{w}"] = state_dict[
                f"cls.predictions.transform.LayerNorm.{w}"
            ]
    logging.info(
        f"Number of layers selectec for distillation: {std_idx}\n"
        f"Number of params transferred for distillation: {len(compressed_sd.keys())}"
    )

    dump_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving extracted model layers to {dump_path}")
    torch.save(compressed_sd, dump_path)


if __name__ == "__main__":
    main()
