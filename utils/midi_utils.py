import os
import numpy as np
import pretty_midi
from mido import MidiFile, MidiTrack


def new_midi(tempo=120.0):
    return pretty_midi.PrettyMIDI(initial_tempo=tempo)


def add_tempo_changes(pm: pretty_midi.PrettyMIDI, times, bpms):
    pm._PrettyMIDI__tick_scales = None
    pm.adjust_times([0.0], [0.0])
    pm.remove_tempi(0, 1e9)
    pm._PrettyMIDI__tick_scales = None

    if len(bpms) > 0:
        pm._PrettyMIDI__tempi = np.array([bpms[0]])
        pm._PrettyMIDI__tempo_changes = np.array([times[0]])


def add_time_signature_meta(mf, numerator=4, denominator=4):
    # pretty_midi has limited TS support; for exact meta events, assemble with mido at write time.
    pass


def _track_name(track):
    for msg in track:
        if msg.is_meta and msg.type == "track_name":
            return msg.name
    return None


def _safe(name: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    return name.strip().replace(" ", "_")


def split_midi_tracks(midi_path: str, out_dir: str):
    """
    Split a type-1 multi-track MIDI into multiple MIDI files,
    one per named track (drums/bass/etc). Keeps track 0 (global meta) if present.
    """
    m = MidiFile(midi_path)
    os.makedirs(out_dir, exist_ok=True)

    global_track = m.tracks[0] if len(m.tracks) > 0 else None
    base = os.path.splitext(os.path.basename(midi_path))[0]

    for i, tr in enumerate(m.tracks):
        name = _track_name(tr)

        if i == 0 and (name is None or name == ""):
            continue

        stem = name or f"track{i}"
        out_path = os.path.join(out_dir, f"{base}__{_safe(stem)}.mid")

        out = MidiFile(type=1, ticks_per_beat=m.ticks_per_beat)

        if global_track is not None:
            t0 = MidiTrack()
            t0.extend(global_track)
            out.tracks.append(t0)

        t1 = MidiTrack()
        t1.extend(tr)
        out.tracks.append(t1)

        out.save(out_path)
        # optional:
        # print(f"Wrote: {out_path}")
