.PHONY: download-brwac
download-brwac:
	mkdir -p data/external/
	curl http://nlpserver2.inf.ufrgs.br/brwac-download/brwac.vert.gz --output data/external/brwac.gz

.PHONY: extract-brwac
extract-brwac:
	mkdir -p data/raw/
	gzip --keep --stdout data/external/brwac.gz > data/raw/brwac.txt
