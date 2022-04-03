import logging
from datasets import load_dataset
from typing import List, Dict
from itertools import chain
from transformers import AutoTokenizer


def join_document_paragraphs(doc_paragraphs: List[List[str]]) -> Dict[str, str]:
    paragraphs = " ".join(chain(*doc_paragraphs["text"]["paragraphs"]))
    return {"processed_text": paragraphs}


def main():

    num_processes = 4
    dataset_path = "data/processed/brwac-000"

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    logging.info(
        f"Converting RAW brwac dataset list of paragraphs into single strings for each document."
    )

    logging.info(f"Loading brwac dataset.")
    brwac_ds = load_dataset(
        path="brwac",
        data_dir="data/raw/brwac",
    )

    logging.info(f"Applying transformation function using {num_processes} threads.")
    brwac_ds["train"] = brwac_ds["train"].map(
        join_document_paragraphs, num_proc=num_processes
    )

    logging.info(f"Dropping old list of paragraphs.")
    brwac_ds["train"] = brwac_ds["train"].remove_columns(["text"])

    logging.info(
        f"Tokenizing dataset with BERT Tokenizer using {num_processes} threads."
    )
    tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    brwac_ds["train"] = brwac_ds["train"].map(
        lambda example: tokenizer(example["processed_text"]),
        num_proc=num_processes,
    )

    logging.info(f"Saving new preprocessed dataset to disk. Save path: {dataset_path}")
    brwac_ds.save_to_disk(dataset_path)


if __name__ == "__main__":
    main()
