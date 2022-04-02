.PHONY: download-brwac
download-brwac:
	mkdir -p data/external/
	curl http://nlpserver2.inf.ufrgs.br/brwac-download/brwac.vert.gz --output data/external/brwac.vert.gz

.PHONY: extract-brwac
extract-brwac:
	mkdir -p data/raw/brwac/
	gzip --keep -d --stdout data/external/brwac.vert.gz > data/raw/brwac/brwac.vert
