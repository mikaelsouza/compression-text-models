# Data organization

The data is organized as follows:


```bash
clean/                                          # Sentences in .txt format
    brwac-joined-sentences.txt                  # Each dataset example is one line
    brwac-separated-sentences.txt               # Each dataset example is composed by many lines
external/
    brwac-000/brwac.vert.gz                     # Original compressed BRWAC dataset
processed/
    brwac-000/train/
    brwac-tokenized/brwac-joined-paragraphs/    # Data preprocessed by distilbert scripts
        token-counts.pickle                     # List counting all tokens in the text
        tokenized-sentences.pickle              # List of all tokenized sentences
raw/
    brwac/brwac.vert                            # Original uncompressed BRWAC dataset
```
