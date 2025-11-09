.PHONY: help setup run run-normalize review export clean-stems clean-midi clean-manifests clean-all process-file

# Default target: show help
help:
	@echo "AI MIDI Pipeline - Available targets:"
	@echo ""
	@echo "  make setup              - Create uv environment and install dependencies"
	@echo "  make run                - Process all files in data/raw/*.wav (no key normalization)"
	@echo "  make run-normalize      - Process all files with key normalization to C/Am"
	@echo "  make review             - Open review UI for low-confidence items"
	@echo "  make export             - Export all final MIDIs to out_midis/"
	@echo "  make export OUT=<dir>   - Export all final MIDIs to specified directory"
	@echo ""
	@echo "  make process-file FILE=<path>           - Process a single audio file"
	@echo "  make process-file-norm FILE=<path>      - Process a single file with key normalization"
	@echo ""
	@echo "  make clean-stems        - Remove all stem separation outputs"
	@echo "  make clean-midi         - Remove all MIDI outputs"
	@echo "  make clean-manifests    - Remove all manifest files"
	@echo "  make clean-all          - Remove stems, MIDI, and manifests"
	@echo ""

# Setup environment
setup:
	@echo "==> Setting up environment with uv..."
	./setup.bash

# Python interpreter from venv
PYTHON := .venv/bin/python

# Run pipeline on all raw files (default: no key normalization)
run:
	@echo "==> Processing all files in data/raw/*.wav..."
	$(PYTHON) pipeline.py run-batch "data/raw/*.wav"

# Run pipeline with key normalization
run-normalize:
	@echo "==> Processing all files with key normalization..."
	$(PYTHON) pipeline.py run-batch "data/raw/*.wav" --normalize-key

# Review pending items
review:
	@echo "==> Opening review UI..."
	$(PYTHON) pipeline.py review-pending

# Export MIDI files (default output directory)
OUT ?= out_midis
export:
	@echo "==> Exporting MIDI files to $(OUT)/..."
	$(PYTHON) pipeline.py export-midi --out $(OUT)

# Process a single file (requires FILE=path/to/file.wav)
process-file:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: FILE not specified. Usage: make process-file FILE=path/to/file.wav"; \
		exit 1; \
	fi
	@echo "==> Processing $(FILE)..."
	$(PYTHON) pipeline.py run-batch "$(FILE)"

# Process a single file with key normalization
process-file-norm:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: FILE not specified. Usage: make process-file-norm FILE=path/to/file.wav"; \
		exit 1; \
	fi
	@echo "==> Processing $(FILE) with key normalization..."
	$(PYTHON) pipeline.py run-batch "$(FILE)" --normalize-key

# Clean targets
clean-stems:
	@echo "==> Removing stem outputs..."
	rm -rf data/stems/*
	@echo "Stems cleaned."

clean-midi:
	@echo "==> Removing MIDI outputs..."
	rm -rf data/midi/*
	@echo "MIDI outputs cleaned."

clean-manifests:
	@echo "==> Removing manifest files..."
	rm -f manifests/*.json
	@echo "Manifests cleaned."

clean-all: clean-stems clean-midi clean-manifests
	@echo "==> All outputs cleaned."
