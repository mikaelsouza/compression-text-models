.PHONY: download-brwac
download-brwac:
	mkdir -p data/external/
	curl http://nlpserver2.inf.ufrgs.br/brwac-download/brwac.vert.gz --output data/external/brwac.vert.gz

.PHONY: extract-brwac
extract-brwac:
	mkdir -p data/raw/brwac/
	gzip --keep -d --stdout data/external/brwac.vert.gz > data/raw/brwac/brwac.vert

.PHONY: raw-to-processed-brwac
raw-to-processed-brwac:
	python scripts/preprocessing/brwac-raw-to-processed.py

.PHONY: raw-to-txt-brwac
raw-to-txt-brwac:
	python scripts/preprocessing/brwac-raw-to-txt.py

.PHONY: processed-to-txt-brwac
processed-to-text-brwac:
	python scripts/preprocessing/brwac-processed-to-txt.py

.PHONY: train-distilbert
train-distilbert:
	python src/distillation/train.py \
    --student_type distilbert \
    --student_config configs/distillation/distilbert-base-cased.json \
    --teacher_type bert \
    --teacher_name neuralmind/bert-base-portuguese-cased \
    --alpha_ce 0.33 --alpha_mlm 0.33 --alpha_cos 0.33 --alpha_clm 0.0 --mlm \
    --freeze_pos_embs \
    --dump_path models/training-results/baseline/$(shell date +%s)/joined \
    --data_file data/processed/tokenized/joined-sentences.pickle \
    --token_counts data/processed/tokenized/joined-token-counts.pickle \
    --force \
    --n_epoch 1
