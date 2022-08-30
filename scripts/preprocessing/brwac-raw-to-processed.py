import logging
from itertools import chain
from typing import Dict, List

import psutil
from datasets import load_dataset
from transformers import AutoTokenizer


def join_document_paragraphs(doc_paragraphs: List[List[str]]) -> Dict[str, str]:
    paragraphs = " ".join(chain(*doc_paragraphs["text"]["paragraphs"]))
    return {"text": paragraphs}


def separate_sentences(l):
    def add_bos_token(l, cls_id=101, sep_id=102):
        return [cls_id, *l, sep_id]

    def divide_chunks(l, n=510):
        return [l[i : i + n] for i in range(0, len(l), n)]

    sentence = l["token_ids"][1:-1]
    sentences = divide_chunks(sentence)
    sentences = list(map(add_bos_token, sentences))
    return {"token_ids": sentences}


def main():
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    num_processes = psutil.cpu_count()
    dataset_path = "data/raw/brwac"
    processed_dataset_path = "data/processed/brwac"
    logging.info(f'Using {num_processes} threads to process dataset "{dataset_path}".')

    # Part 1
    logging.info(f"Loading dataset.")
    brwac_ds = load_dataset(
        path="brwac",
        data_dir=dataset_path,
        split="train",
    )

    # Part 2
    unused_columns = ["title", "doc_id", "uri"]
    logging.info(f"Dropping unused columns: {unused_columns}")
    brwac_ds = brwac_ds.remove_columns(unused_columns)

    # Part 3
    logging.info(f"Joining sentences.")
    brwac_ds = brwac_ds.map(join_document_paragraphs, num_proc=num_processes)

    # Part 4
    logging.info(f"Tokenizing dataset with Neuralmind's BERT Tokenizer.")
    model_name = "neuralmind/bert-base-portuguese-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenize_sentence = lambda sentence: {
        "token_ids": tokenizer.encode(sentence["text"])
    }
    brwac_ds = brwac_ds.map(
        tokenize_sentence,
        num_proc=num_processes,
    )

    # Part 5
    logging.info(f"Removing text sentences from dataset.")
    brwac_ds = brwac_ds.remove_columns(["text"])

    # Part 6
    logging.info(f"Breaking input_ids into 512 sized lists.")
    brwac_ds = brwac_ds.map(
        separate_sentences,
        num_proc=num_processes,
    )

    # Part 7
    logging.info(
        f"Saving new preprocessed dataset to disk. Save path: {processed_dataset_path}"
    )
    brwac_ds.save_to_disk(processed_dataset_path)


if __name__ == "__main__":
    main()
