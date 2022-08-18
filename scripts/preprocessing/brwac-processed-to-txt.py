import logging

import tqdm
from datasets.load import load_from_disk


def main():

    dataset_path = "data/processed/brwac-000"
    saved_path = "data/clean/brwac-joined-sentences.txt"

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    logging.info(f"Converting processed brwac paragraphs into a txt file.")

    logging.info(f"Loading brwac dataset.")
    brwac_ds = load_from_disk(
        dataset_path,
    )

    logging.info(f"Recording the data into a file at {saved_path}")
    with open(saved_path, "w") as f:
        for sample in tqdm.tqdm(brwac_ds["train"]):
            sentence = sample["processed_text"] + "\n"
            f.write(sentence)


if __name__ == "__main__":
    main()
