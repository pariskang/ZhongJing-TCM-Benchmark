# ZhongJing-TCM benchmark — pipeline orchestration.
# Offline demo:  make demo   (uses the mock LLM provider, no API key needed)

PY ?= python
RUN := $(PY) run.py

.PHONY: help install test lint \
        ingest quality topics label generate dtqf assemble evaluate stats \
        pipeline demo clean

help:
	@echo "Targets:"
	@echo "  install   - pip install -r requirements.txt"
	@echo "  test      - run the pytest suite"
	@echo "  ingest    - M1+M2: ingest/clean/de-dup + quality gate"
	@echo "  topics    - M3: chunk + BERTopic"
	@echo "  label     - M4: nine-category mapping + topic cards"
	@echo "  generate  - M5: LLM question generation"
	@echo "  dtqf      - M6: dynamic question filtering"
	@echo "  assemble  - M7: token counting + dataset packaging"
	@echo "  evaluate  - M8: model evaluation (MODEL=gpt-4o)"
	@echo "  stats     - M9: ANOVA + regression + DP segmentation"
	@echo "  pipeline  - M1->M7 end-to-end"
	@echo "  demo      - pipeline with the offline mock provider"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest

lint:
	-ruff check src tests
	-black --check src tests

ingest:
	$(RUN) ingest
quality:
	$(RUN) quality
topics:
	$(RUN) topics
label:
	$(RUN) label
generate:
	$(RUN) generate
dtqf:
	$(RUN) dtqf
assemble:
	$(RUN) assemble
evaluate:
	$(RUN) evaluate --model $(or $(MODEL),gpt-4o)
stats:
	$(RUN) stats

pipeline:
	$(RUN) pipeline

demo:
	ZHONGJING_LLM_PROVIDER=mock $(RUN) pipeline

clean:
	rm -rf data/interim/* data/final/* results/* .cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
