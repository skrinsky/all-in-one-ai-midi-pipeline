ALL-IN-ONE: AI MIDI Pipeline

End-to-end, mostly automatic pipeline for turning your own stereo masters into aligned, labeled multi-track MIDI suitable for model training.

Stereo in → stems → tempo/meter → transcription → canonical tracks → (optional) key normalization → cleaned multi-track MIDI

Features (What It Does)
1. Stem Separation (HTDemucs 5-stem)

Splits each track into:

vocals, drums, bass, guitar, other

Outputs:

data/stems/<Song>/...

manifests/<Song>.json

2. Tempo, Downbeats, Meter

Uses librosa (and optionally madmom if installed) to estimate:

meter_key.tempo

downbeat positions

rough time signature

Estimated tempo is reused downstream (e.g. as Basic Pitch midi_tempo).

3. Transcription
Pitched (Basic Pitch 0.2.6)

Run on stems with tempo-aware settings and cleanup.

Vocals (vocals stem):

Basic Pitch → note events

Vocal-specific tweaks:

higher onset/frame thresholds

minimum note length

merge same-pitch segments

squash tiny vibrato / slides

Split into:

voxlead — highest active line

voxbg — backgrounds / harmonies

Bass:

Basic Pitch on bass

Optional filtering to reduce obvious noise / octave errors

Guitar:

Basic Pitch on guitar

Exported with a guitar-like GM program

Other:

Basic Pitch on other

Treated as pads/synths/etc. with a pad-like GM program

Pitched transcription status is stored under:

"transcription": {
  "pitched": { ... }
}

Drums (ADTOF)

steps/transcribe_drums.py uses adtof_pytorch on the drums stem

Merges hits into a single drums kit

Velocities derived from stem RMS (dynamic, not all-100)

Drum transcription status is stored under:

"transcription": {
  "drums": { ... }
}

4. Canonical Track Assignment

steps/assign_parts.py maps detected parts into stable labels:

drums, voxlead, voxbg, bass, guitar, keys (optional), other

Only non-empty tracks are kept.

Recorded under:

"assignment": {
  "tracks": { ... }
}

5. Key Detection & Optional Normalization

steps/key_normalize.py:

Detects global key (excluding drums) using music21.

If enabled:

major-ish → C major

minor-ish → A minor

When enabled:

"key": {
  "detected_tonic": "...",
  "detected_mode": "...",
  "normalized": true,
  "transpose_semitones": <int>,
  "target": "C major" | "A minor"
}


When disabled:

"key": {
  "detected_tonic": "...",
  "detected_mode": "...",
  "normalized": false,
  "transpose_semitones": 0,
  "target": null,
  "reason": "key normalization disabled via CLI"
}


Key normalization is OFF by default. Enable per run with --normalize-key.

6. Time Signature (Optional)

steps/meter_apply.py can inject simple time signature meta events when meter estimation is confident.

7. Cleanup & Quantization

steps/clean_quantize.py:

Removes obvious garbage events

Applies gentle timing/length cleanup

Tries not to destroy the feel

8. Multi-track MIDI Export

steps/write_midi.py builds, for each song:

One MIDI file:

data/midi/<Song>/<Song>.mid

Uses:

tempo from meter_key.tempo

one track per canonical class

is_drum = True for drums

track names = canonical labels

9. Human-in-the-loop

python pipeline.py review-pending to surface items flagged for human review

steps/qc_render.py hooks for quick listening / inspection

Install

Use Python 3.10 (this repo is tuned for it).

# 1) Create & activate venv
python3.10 -m venv .venv-ai-midi
source .venv-ai-midi/bin/activate

# 2) Install dependencies
pip install -r requirements.txt


Key deps (see requirements.txt for exact pins):

Core: numpy, typing-extensions, librosa, soundfile, scipy, pretty_midi, mido

Separation: demucs>=4.0.0

Key detection: music21

Transcription: basic-pitch==0.2.6 (+ tensorflow-macos / appropriate TF for your platform)

Drums: adtof_pytorch

CLI / misc: gradio, tqdm, pyyaml

Optional: madmom for extra beat/downbeat features

Usage
1. Add Audio
mkdir -p data/raw
cp /path/to/YourSong.wav data/raw/

2. Run the Pipeline

Default (no key normalization):

python pipeline.py run-batch "data/raw/*.wav"


With key normalization (C major / A minor):

python pipeline.py run-batch "data/raw/*.wav" --normalize-key

3. Inspect Outputs

For YourSong.wav:

Stems: data/stems/YourSong/...

Manifest: manifests/YourSong.json

MIDI: data/midi/YourSong/YourSong.mid

4. Extra Commands
# See items flagged for human review
python pipeline.py review-pending

# Export all final MIDIs to a flat folder
python pipeline.py export-midi --out out_midis/
