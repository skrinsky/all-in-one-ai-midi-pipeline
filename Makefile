.PHONY: help setup run run-normalize stems drum-stems review export clean-stems clean-midi clean-manifests clean-all process-file stem drum-stem midi-filter midi-filter-bass midi-histogram

# Default target: show help
help:
	@echo "AI MIDI Pipeline - Available targets:"
	@echo ""
	@echo "  make setup              - Create uv environment and install dependencies"
	@echo ""
	@echo "FULL PIPELINE (separation + transcription + assembly):"
	@echo "  make run                - Process all files in data/raw/*.wav (no key normalization)"
	@echo "  make run-normalize      - Process all files with key normalization to C/Am"
	@echo "  make process-file FILE=<path>           - Process a single audio file"
	@echo "  make process-file-norm FILE=<path>      - Process a single file with key normalization"
	@echo ""
	@echo "STEM-TO-MIDI (direct conversion, no separation):"
	@echo "  make stems PATTERN=<glob>      - Convert stems to MIDI with Basic Pitch"
	@echo "  make drum-stems PATTERN=<glob> - Convert drum stems to MIDI with ADTOF"
	@echo "  make stem FILE=<path>          - Convert single stem with Basic Pitch"
	@echo "  make drum-stem FILE=<path>     - Convert single drum stem with ADTOF"
	@echo ""
	@echo "MIDI CLEANUP (velocity/pitch filtering):"
	@echo "  make midi-filter NAME=Song [SUFFIX=Clean1] [T=40]  - Filter all tracks"
	@echo "  make midi-filter-bass NAME=Song [SUFFIX=Clean1]    - Filter bass only (pitch<=C4)"
	@echo "  make midi-histogram NAME=Song                      - Show histograms (no output)"
	@echo "  (Paths expand: NAME -> data/midi/NAME/NAME.mid, SUFFIX -> NAMESuffix.mid)"
	@echo ""
	@echo "OTHER:"
	@echo "  make review             - Open review UI for low-confidence items"
	@echo "  make export             - Export all final MIDIs to out_midis/"
	@echo "  make export OUT=<dir>   - Export all final MIDIs to specified directory"
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

# Stem-to-MIDI conversion (bypasses separation)
PATTERN ?= data/stems/*.wav

stems:
	@echo "==> Converting stems to MIDI with Basic Pitch ($(PATTERN))..."
	$(PYTHON) pipeline.py run-batch "$(PATTERN)" --assume-stems

drum-stems:
	@echo "==> Converting drum stems to MIDI with ADTOF ($(PATTERN))..."
	$(PYTHON) pipeline.py run-batch "$(PATTERN)" --assume-drum-stems

stem:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: FILE not specified. Usage: make stem FILE=path/to/stem.wav"; \
		exit 1; \
	fi
	@echo "==> Converting $(FILE) to MIDI with Basic Pitch..."
	$(PYTHON) pipeline.py run-batch "$(FILE)" --assume-stems

drum-stem:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: FILE not specified. Usage: make drum-stem FILE=path/to/drums.wav"; \
		exit 1; \
	fi
	@echo "==> Converting $(FILE) to MIDI with ADTOF..."
	$(PYTHON) pipeline.py run-batch "$(FILE)" --assume-drum-stems

# MIDI cleanup targets
T ?=
SUFFIX ?= Cleaned

midi-filter:
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME required. Usage: make midi-filter NAME=Song [SUFFIX=Clean1] [T=threshold]"; \
		exit 1; \
	fi
	@echo "==> Filtering MIDI $(NAME)..."
	$(PYTHON) clean-midi.py "$(NAME)" "$(SUFFIX)" $(if $(T),--threshold $(T),)

midi-filter-bass:
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME required. Usage: make midi-filter-bass NAME=Song [SUFFIX=Clean1] [T=threshold]"; \
		exit 1; \
	fi
	@echo "==> Filtering bass track $(NAME)..."
	$(PYTHON) clean-midi.py "$(NAME)" "$(SUFFIX)" --bass $(if $(T),--threshold $(T),)

midi-histogram:
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME required. Usage: make midi-histogram NAME=Song"; \
		exit 1; \
	fi
	@echo "==> Showing histograms for $(NAME)..."
	$(PYTHON) clean-midi.py "$(NAME)" --histogram-only

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
