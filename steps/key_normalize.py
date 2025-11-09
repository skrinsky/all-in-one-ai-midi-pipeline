import math

import pretty_midi
from music21 import stream, note, chord, analysis

MAJOR_LIKE = {"major", "ionian", "maj"}
MINOR_LIKE = {"minor", "aeolian", "min"}


def _collect_pitches(instruments):
    """
    Collect MIDI pitches from a dict[name -> pretty_midi.Instrument],
    skipping drums and obviously invalid notes.
    """
    pitches = []
    for name, inst in (instruments or {}).items():
        # pretty_midi.Instrument has is_drum flag
        if getattr(inst, "is_drum", False):
            continue
        for n in inst.notes:
            if 0 < n.pitch < 128:
                pitches.append(n.pitch)
    return pitches


def _detect_key_music21(pitches):
    """
    Use music21's key analyzer on a synthetic stream built
    from our MIDI pitches. Returns (tonic_str, mode_str) or (None, None).
    """
    if not pitches:
        return None, None

    s = stream.Stream()
    # Use dummy quarter notes at pitch classes; we only care about distribution.
    for p in pitches:
        try:
            s.append(note.Note(p % 128, quarterLength=1.0))
        except Exception:
            continue

    if len(s.notes) < 4:
        return None, None

    try:
        k = s.analyze("KrumhanslSchmuckler")  # standard profile-based key finder
    except Exception:
        return None, None

    tonic = (k.tonic.name if hasattr(k, "tonic") else None)
    mode = (k.mode.lower() if hasattr(k, "mode") and k.mode else None)

    return tonic, mode


def _compute_transpose_semitones(tonic, mode):
    """
    Decide how many semitones to transpose so that:
      - any major-ish key -> C major
      - any minor-ish key -> A minor
    Returns (semitones, target_key_str) or (0, None) if unknown.
    """
    if not tonic or not mode:
        return 0, None

    tonic = tonic.upper()

    # Map note name -> semitone in C-based scale (C=0..B=11)
    name_to_pc = {
        "C": 0, "B#": 0,
        "C#": 1, "DB": 1,
        "D": 2,
        "D#": 3, "EB": 3,
        "E": 4, "FB": 4,
        "F": 5, "E#": 5,
        "F#": 6, "GB": 6,
        "G": 7,
        "G#": 8, "AB": 8,
        "A": 9,
        "A#": 10, "BB": 10,
        "B": 11, "CB": 11,
    }

    pc = name_to_pc.get(tonic)
    if pc is None:
        return 0, None

    mode_norm = mode.lower()

    if mode_norm in MAJOR_LIKE:
        # shift tonic -> C (0)
        shift = (0 - pc) % 12
        target = "C major"
    elif mode_norm in MINOR_LIKE:
        # shift tonic -> A (9)
        shift = (9 - pc) % 12
        target = "A minor"
    else:
        # unknown/mode: don't mess with it
        return 0, None

    # Normalize to [-6, +6] range just to avoid huge jumps
    if shift > 6:
        shift -= 12

    return int(shift), target


def _transpose_instruments(instruments, semitones):
    """
    Return a new dict[name -> pretty_midi.Instrument] with pitches shifted.
    Drums are not touched.
    """
    if not instruments or semitones == 0:
        return instruments

    out = {}
    for name, inst in instruments.items():
        if getattr(inst, "is_drum", False):
            out[name] = inst
            continue

        new_inst = pretty_midi.Instrument(
            program=inst.program,
            is_drum=inst.is_drum,
            name=inst.name or name,
        )

        for n in inst.notes:
            new_pitch = n.pitch + semitones
            if 0 < new_pitch < 128:
                new_inst.notes.append(
                    pretty_midi.Note(
                        start=n.start,
                        end=n.end,
                        pitch=int(new_pitch),
                        velocity=n.velocity,
                    )
                )

        # Preserve control changes etc. if present
        for cc in getattr(inst, "control_changes", []):
            new_inst.control_changes.append(cc)
        for pb in getattr(inst, "pitch_bends", []):
            new_inst.pitch_bends.append(pb)

        out[name] = new_inst

    return out


def detect_and_normalize_key(assigned_instruments, CFG, manifest):
    """
    1. Detect global key from current assigned instruments (ignoring drums).
    2. If confident enough, transpose all pitched tracks so that:
         - major-ish -> C major
         - minor-ish -> A minor
    3. Update manifest['key'] with detection + transpose info.
    4. Return the (possibly) transposed instruments dict.
    """
    pitches = _collect_pitches(assigned_instruments)
    tonic, mode = _detect_key_music21(pitches)

    key_info = manifest.setdefault("key", {})
    key_info["detected_tonic"] = tonic
    key_info["detected_mode"] = mode

    semitones, target = _compute_transpose_semitones(tonic, mode)

    if target is None or semitones == 0:
        # No reliable detection or already C/A
        key_info["normalized"] = False
        key_info["transpose_semitones"] = 0
        key_info["target"] = None
        return assigned_instruments

    normalized = _transpose_instruments(assigned_instruments, semitones)

    key_info["normalized"] = True
    key_info["transpose_semitones"] = semitones
    key_info["target"] = target

    return normalized
