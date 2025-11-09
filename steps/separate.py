import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from utils.manifest import song_id_from_path


def _merge_audio(a_path, b_path, out_path):
    """
    Sum two mono/stereo wavs, normalize if needed, write to out_path.
    Returns str(out_path).
    """
    if not a_path and not b_path:
        return None
    if a_path and not b_path:
        return a_path
    if b_path and not a_path:
        return b_path

    a, sr_a = sf.read(a_path)
    b, sr_b = sf.read(b_path)
    if sr_a != sr_b:
        raise RuntimeError(f"Sample rate mismatch: {sr_a} vs {sr_b}")

    # to mono-compatible shapes
    if a.ndim > 1:
        a = a.mean(axis=1)
    if b.ndim > 1:
        b = b.mean(axis=1)

    L = max(len(a), len(b))
    a = np.pad(a, (0, L - len(a)))
    b = np.pad(b, (0, L - len(b)))

    mix = a + b
    maxv = float(np.max(np.abs(mix))) if mix.size else 0.0
    if maxv > 1.0:
        mix = mix / maxv

    sf.write(out_path, mix, sr_a)
    return str(out_path)


def separate_track(audio_path: str, CFG: dict, manifest: dict):
    """
    Use Demucs 6-stem under the hood, but expose a 5-stem layout:

        vocals, drums, bass, guitar, other

    where:
      - guitar = Demucs guitar
      - other  = (Demucs other + Demucs piano), i.e. all non-core pitched stuff
    """
    sid = song_id_from_path(audio_path)

    base_out_dir = Path("data/stems")
    base_out_dir.mkdir(parents=True, exist_ok=True)

    sep_cfg = CFG.get("separation", {})
    model_name = sep_cfg.get("demucs_model", "htdemucs_6s")

    # Demucs writes: data/stems/<model_name>/<sid>/*.wav
    song_out_dir = base_out_dir / model_name / sid

    if not (song_out_dir.exists() and any(song_out_dir.glob("*.wav"))):
        cmd = [
            "python",
            "-m",
            "demucs.separate",
            "-n",
            model_name,
            "-o",
            str(base_out_dir),
            audio_path,
        ]
        print(f"[separate] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    if not song_out_dir.exists():
        raise RuntimeError(f"[separate] Expected stems in {song_out_dir}, but folder is missing.")

    def pick(name: str):
        p = song_out_dir / f"{name}.wav"
        return str(p) if p.exists() else None

    vocals = pick("vocals")
    drums = pick("drums")
    bass = pick("bass")
    guitar = pick("guitar")
    piano = pick("piano")
    other_raw = pick("other")

    # Merge piano into other so we don't treat Demucs "piano" as a separate synth stem.
    merged_other_path = song_out_dir / "other_merged.wav"
    other = _merge_audio(other_raw, piano, merged_other_path)

    stems = {
        "vocals": vocals,
        "drums": drums,
        "bass": bass,
        "guitar": guitar,
        "other": other,
    }

    # Write to manifest
    manifest.setdefault("separation", {})
    manifest["separation"]["model"] = model_name
    manifest["separation"]["path"] = str(song_out_dir)
    manifest["separation"]["stems"] = {k: v for k, v in stems.items() if v}

    print(f"[separate] 5-stem view for {sid}: {manifest['separation']['stems']}")

    return stems
