import os
import tempfile
import numpy as np
import pretty_midi

from utils.audio_utils import load_audio_mono
from adtof_pytorch import transcribe_to_midi as adtof_to_midi


def _merge_adtof_output(mid_path: str) -> pretty_midi.Instrument:
    """
    Merge ADTOF's multi-track MIDI output into a single drum kit Instrument.

    We don't change pitches here, just:
      - collect all drum notes
      - ensure is_drum=True, name="drums"
    """
    pm = pretty_midi.PrettyMIDI(mid_path)
    kit = pretty_midi.Instrument(program=0, is_drum=True, name="drums")

    # ADTOF may output multiple instruments; we just merge them.
    for inst in pm.instruments:
        for n in inst.notes:
            # copy as-is; velocity will be refined later
            kit.notes.append(
                pretty_midi.Note(
                    start=float(n.start),
                    end=float(n.end),
                    pitch=int(n.pitch),
                    velocity=int(n.velocity) if n.velocity else 100,
                )
            )
    return kit


def _drum_hit_velocity(
    y: np.ndarray,
    sr: int,
    onset_time: float,
    win_before: float = 0.005,
    win_after: float = 0.020,
    db_floor: float = -50.0,
    db_ceil: float = -5.0,
) -> int:
    """
    Estimate MIDI velocity (1-127) for a drum hit based on local RMS around onset_time.
    """
    if y is None or sr <= 0 or y.size == 0:
        return 100  # fallback if audio missing

    start = int(max(0.0, onset_time - win_before) * sr)
    end = int(min(len(y), onset_time + win_after) * sr)
    if end <= start:
        return 100

    seg = y[start:end]
    if seg.size == 0:
        return 100

    rms = float(np.sqrt(np.mean(seg * seg) + 1e-12))
    db = 20.0 * np.log10(rms + 1e-12)

    # Guard
    if db_ceil <= db_floor:
        return 100

    # Map [db_floor, db_ceil] -> [1, 127]
    x = (db - db_floor) / (db_ceil - db_floor)
    x = max(0.0, min(1.0, x))
    vel = int(round(1 + x * 126))
    return max(1, min(127, vel))


def transcribe_drums_to_midi(drum_stem_or_path, CFG, manifest):
    """
    Called from pipeline.py:

        drums = transcribe_drums_to_midi(stems.get("drums"), CFG, manifest)

    Accepts either:
      - a direct path to the demucs drums stem, or
      - a stems dict (we'll pull ['drums']).

    Returns:
      - pretty_midi.Instrument(is_drum=True, name="drums") with velocities
      - or None if no stem / no notes.
    """
    # Normalize input to a path
    if isinstance(drum_stem_or_path, dict):
        drum_path = drum_stem_or_path.get("drums")
    else:
        drum_path = drum_stem_or_path

    if not drum_path or not os.path.exists(drum_path):
        manifest.setdefault("transcription", {})["drums"] = "missing_stem"
        return None

    # 1) Run ADTOF to get a raw drum MIDI
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_mid = os.path.join(tmpdir, "drums_adtof.mid")

        try:
            adtof_to_midi(drum_path, tmp_mid)
        except Exception as e:
            manifest.setdefault("transcription", {})["drums"] = f"error:adtof:{e}"
            return None

        if not os.path.exists(tmp_mid):
            manifest.setdefault("transcription", {})["drums"] = "error:no_mid_created"
            return None

        kit = _merge_adtof_output(tmp_mid)

    # 2) If merge produced no notes, bail
    if not getattr(kit, "notes", None):
        manifest.setdefault("transcription", {})["drums"] = "no_notes"
        return None

    # 3) Load the drum stem once for velocity estimation
    y, sr = load_audio_mono(drum_path)

    # 4) Apply per-hit velocity. If y/sr is bad, helper will just return 100.
    for n in kit.notes:
        n.velocity = _drum_hit_velocity(y, sr, n.start)

    # 5) Ensure drum flags
    kit.is_drum = True
    kit.name = "drums"

    manifest.setdefault("transcription", {})["drums"] = True
    return kit
