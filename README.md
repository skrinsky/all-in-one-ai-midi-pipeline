# ALL-IN-ONE: AI MIDI Pipeline

End-to-end, mostly-automatic pipeline for turning your own stereo masters into aligned, labeled multi-track MIDI suitable for model training.

Stereo in → stems → tempo/meter → transcription → canonical tracks → (optional) key normalization → cleaned multi-track MIDI.

---

## Features (What It Does)

**1. Stem Separation (HTDemucs 5-stem)**

- Splits each track into:
  - `vocals`, `drums`, `bass`, `guitar`, `other`
- Writes paths/metadata to:
  - `data/stems/<Song>/...`
  - `manifests/<Song>.json`

**2. Tempo, Downbeats, Meter**

- Uses `librosa` (and optionally `madmom` if installed) to estimate:
  - `meter_key.tempo`
  - downbeat positions
  - rough time signature
- Estimated tempo is reused downstream (e.g. Basic Pitch `midi_tempo`).

**3. Transcription**

**Pitched (Basic Pitch 0.2.6):**

- Run on stems with tempo-aware settings and cleanup.

Vocals (`vocals` stem):
- Basic Pitch → note events
- Vocal-specific tweaks:
  - higher onset/frame thresholds
  - minimum note length
  - merge same-pitch segments (reduce double hits)
  - vibrato / micro-slide squashing
- Split into:
  - `voxlead` — highest active line
  - `voxbg` — background / harmonies

Bass:
- Basic Pitch on `bass`.
- Filters to reduce garbage (optional high-note / silence-aware cleanup).

Guitar:
- Basic Pitch on `guitar`.
- Exported as guitar-like GM program.

Other:
- Basic Pitch on `other`.
- Treated as pads/synths/etc. with a pad-like GM program.

Status is recorded in `manifest["transcription"]["pitched"]`.

**Drums (ADTOF):**

- `steps/transcribe_drums.py` uses `adtof_pytorch` on `drums` stem.
- Merges outputs into one `drums` kit.
- Hit velocities derived from stem RMS → dynamic, not all-100.
- Status in `manifest["transcription"]["drums"]`.

**4. Canonical Track Assignment**

`steps/assign_parts.py` maps everything into consistent labels:

- `drums`, `voxlead`, `voxbg`, `bass`, `guitar`, `keys` (optional), `other`

Only non-empty tracks are kept; names are stable for training.

Recorded under: `manifest["assignment"]["tracks"]`.

**5. Key Detection & Optional Normalization**

`steps/key_normalize.py`:

- Detects global key from pitched notes (ignoring drums) using `music21`.
- If enabled:
  - major-ish keys → transposed to **C major**
  - minor-ish keys → transposed to **A minor**

Writes:

```json
"key": {
  "detected_tonic": "...",
  "detected_mode": "...",
  "normalized": true/false,
  "transpose_semitones": <int>,
  "target": "C major" / "A minor" / null,
  "reason": "key normalization disabled via CLI" // when off
}
Key normalization is OFF by default; can be turned on per run via CLI.

6. Time Signature (Optional)

steps.meter_apply.insert_time_signatures injects basic TS meta when meter is confident.

7. Cleanup & Quantization

steps.clean_quantize.py applies gentle timing/length cleanup:

remove obvious junk

avoid killing feel

8. Multi-track MIDI Export

steps.write_midi.py builds:

one MIDI file per song: data/midi/<Song>/<Song>.mid

tempo from meter_key.tempo

one track per canonical class

is_drum=True for drums

track names = your labels

9. Human-in-the-loop

pipeline.py review-pending + steps/qc_render.py hooks for quick listening / inspection.

Install
Use Python 3.10 (this repo is tuned for it).

bash
Copy code
# 1) Create & activate venv
python3.10 -m venv .venv-ai-midi
source .venv-ai-midi/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
requirements.txt (summary):

Core: numpy==1.24.3, typing-extensions==4.5.0, librosa, soundfile, scipy==1.10.1, pretty_midi, mido

Separation: demucs>=4.0.0

Key detection: music21==8.3.0

Transcription: basic-pitch==0.2.6 (+ tensorflow-macos==2.13.0 installed separately/pinned)

Drums: adtof_pytorch

UI/CLI: gradio, tqdm, pyyaml

Optional: madmom (if you want its beat/downbeat features; needs extra build tooling)

You already have the exact pinned versions in your actual requirements.txt; keep that as the source of truth.

Usage
1. Add Audio
bash
Copy code
mkdir -p data/raw
cp /path/to/YourSong.wav data/raw/
2. Run the Pipeline
Default (no key normalization):

bash
Copy code
python pipeline.py run-batch "data/raw/*.wav"
With key normalization to Cmaj/Amin:

bash
Copy code
python pipeline.py run-batch "data/raw/*.wav" --normalize-key
3. Inspect Outputs
For a song YourSong.wav:

Stems: data/stems/YourSong/...

Manifest: manifests/YourSong.json

MIDI: data/midi/YourSong/YourSong.mid

4. Extra Commands
bash
Copy code
# Review items that want human ears/eyes
python pipeline.py review-pending

# Export all final MIDIs to a flat output folder
python pipeline.py export-midi --out out_midis/
