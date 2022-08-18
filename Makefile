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
