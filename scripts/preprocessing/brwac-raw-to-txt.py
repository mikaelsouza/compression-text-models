import logging

import tqdm
from datasets.load import load_dataset


def main():

    dataset_path = "data/raw/brwac"
    saved_path = "data/clean/brwac-separated-sentences.txt"

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    logging.info(
        f"Converting RAW brwac dataset list of paragraphs into single strings for each document."
    )

    logging.info(f"Loading brwac dataset.")
    brwac_ds = load_dataset(
        path="brwac",
        data_dir=dataset_path,
    )

    logging.info(f"Recording the data into a file at {saved_path}")
    with open(saved_path, "w") as f:
        for sample in tqdm.tqdm(brwac_ds["train"]):
            paragraphs = sample["text"]["paragraphs"]
            for sentence in paragraphs:
                joined_sentence = " ".join(sentence) + "\n"
                f.write(joined_sentence)


if __name__ == "__main__":
    main()
