# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AI MIDI Pipeline**: End-to-end pipeline for converting stereo audio masters into aligned, labeled multi-track MIDI suitable for model training. The pipeline processes audio through stem separation, tempo/meter detection, transcription (pitched and drums), track assignment, optional key normalization, cleanup, and final MIDI export.

## Development Environment

**Python Version**: Python 3.10 (required)

Virtual environment setup:
```bash
python3.10 -m venv .venv-ai-midi
source .venv-ai-midi/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline

Main entry point: `pipeline.py`

Process audio files (default, no key normalization):
```bash
python pipeline.py run-batch "data/raw/*.wav"
```

Process with key normalization to C major / A minor:
```bash
python pipeline.py run-batch "data/raw/*.wav" --normalize-key
```

Review items flagged for human review:
```bash
python pipeline.py review-pending
```

Export all final MIDIs to a flat directory:
```bash
python pipeline.py export-midi --out out_midis/
```

## Pipeline Architecture

The pipeline is a sequential 8-step process coordinated by `pipeline.py`:

1. **Stem Separation** (`steps/separate.py`): Uses HTDemucs 6-stem model, exposes 5 stems (vocals, drums, bass, guitar, other). The `other` stem merges Demucs "other" and "piano" outputs.

2. **Tempo/Meter Detection** (`steps/beats_meter.py`): Uses librosa (optionally madmom) to estimate tempo, downbeat positions, and rough time signature.

3. **Transcription**:
   - **Pitched tracks** (`steps/transcribe_melodic.py`): Uses Basic Pitch 0.3.0 on vocals, bass, guitar, other stems. Vocals are split into `voxlead` (highest active line) and `voxbg` (background/harmonies).
   - **Drums** (`steps/transcribe_drums.py`): Uses ADTOF on drums stem with velocities derived from stem RMS.

4. **Track Assignment** (`steps/assign_parts.py`): Maps detected parts into 7 canonical classes: drums, voxlead, voxbg, bass, guitar, keys, other. Only non-empty tracks are kept.

5. **Key Normalization** (`steps/key_normalize.py`): Optional (off by default). Detects global key using music21, transposes major-ish to C major and minor-ish to A minor when enabled via `--normalize-key`.

6. **Time Signature Injection** (`steps/meter_apply.py`): Optionally injects time signature meta events when meter estimation is confident.

7. **Cleanup & Quantization** (`steps/clean_quantize.py`): Removes junk events, applies gentle timing/length cleanup while preserving groove.

8. **MIDI Export** (`steps/write_midi.py`): Assembles multi-track MIDI file at `data/midi/<Song>/<Song>.mid` with tempo from meter detection, one track per canonical class.

## Data Flow & State Management

**Manifests**: Each song has a JSON manifest at `manifests/<SongID>.json` that tracks pipeline state and metadata throughout processing. The manifest is read/written after each pipeline step.

**Key Utilities**:
- `utils/manifest.py`: Manifest read/write, config loading, song ID extraction
- `utils/audio_utils.py`: Audio processing helpers
- `utils/midi_utils.py`: MIDI manipulation helpers

Each pipeline step:
1. Receives stems/data + config + manifest
2. Updates manifest with step-specific metadata
3. Returns processed data for next step
4. Manifest is persisted via `write_manifest()` after each step

## Configuration

**config.yaml** controls:
- Separation model (`htdemucs`)
- Confidence thresholds for meter/key detection (triggers manual review)
- Transcription parameters (Basic Pitch threshold, drum quantize strength)
- Cleanup parameters (quantization limits, minimum note length)
- Canonical track classes

## Directory Structure

```
data/
  raw/              # Input audio files (.wav)
  stems/            # HTDemucs outputs (5-stem layout)
  midi/             # Final multi-track MIDI outputs
manifests/          # Per-song JSON state files
steps/              # Pipeline step implementations
utils/              # Shared utilities (manifest, audio, MIDI)
```

## Key Dependencies

- **Separation**: demucs 4.0.1 (HTDemucs 6-stem model)
- **Transcription**: basic-pitch 0.3.0, adtof_pytorch (from GitHub)
- **Audio**: librosa 0.10.1, soundfile, scipy
- **MIDI**: pretty_midi, mido
- **Key detection**: music21 8.3.0
- **ML backend**: torch 2.1.0, torchaudio 2.1.0
- **Optional**: madmom for enhanced beat/downbeat detection, gradio for QC UI

## Important Implementation Notes

- The separation step uses Demucs 6-stem internally but exposes a 5-stem layout by merging "piano" into "other" (see `steps/separate.py:_merge_audio`)
- Vocal splitting logic (voxlead vs voxbg) is in `steps/transcribe_melodic.py`
- Key normalization is OFF by default; users must explicitly enable with `--normalize-key`
- Manifest updates happen synchronously after each step to enable resume/debugging
- No test suite currently exists (`tests/` directory not present)
- No `doit` script exists yet for this project
