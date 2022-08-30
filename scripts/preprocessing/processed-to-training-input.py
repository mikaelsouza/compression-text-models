import torch
import tqdm
from datasets.load import load_from_disk


def main():
    dataset_path = "data/processed/brwac"
    clean_tokens = "data/bin/tokenized-tensor.pt"
    ds = load_from_disk(dataset_path)

    tokenized_sentences = list()
    tensors = list()

    for idx, data in enumerate(tqdm.tqdm(ds)):
        sentences = data["token_ids"]
        for tokens in sentences:

            num_tokens = len(tokens)

            if num_tokens < 11:
                continue
            if tokens.count(100) >= num_tokens // 2:
                continue
            if num_tokens < 512:
                tokens.extend([0] * (512 - num_tokens))
            tokenized_sentences.append(tokens)
        if idx % 10_000 == 0:
            tensors.append(torch.LongTensor(tokenized_sentences))
            tokenized_sentences = list()
    tensors.append(torch.LongTensor(tokenized_sentences))
    tokenized_tensor = torch.cat(tensors)
    torch.save(tokenized_tensor, clean_tokens)


if __name__ == "__main__":
    main()
