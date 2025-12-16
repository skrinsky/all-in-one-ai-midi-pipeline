#!/usr/bin/env python3
import argparse
import os
from mido import MidiFile, MidiTrack


def track_name(track):
    for msg in track:
        if msg.is_meta and msg.type == "track_name":
            return msg.name
    return None


def safe(name: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    name = name.strip().replace(" ", "_").lower()
    while "__" in name:
        name = name.replace("__", "_")
    return name


def has_notes(track) -> bool:
    # Avoid writing junk files for tracks that contain only meta events
    for msg in track:
        if not msg.is_meta and msg.type in ("note_on", "note_off"):
            return True
    return False


def split(midi_path: str, out_dir: str):
    m = MidiFile(midi_path)
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(midi_path))[0]

    # Track 0 is often global meta; keep it if it looks like global meta
    global_track = m.tracks[0] if len(m.tracks) > 0 else None
    global_name = track_name(global_track) if global_track is not None else None

    for i, tr in enumerate(m.tracks):
        name = track_name(tr)

        # Skip unnamed track 0 (typically global meta only)
        if i == 0 and (name is None or name == ""):
            continue

        # Skip tracks that contain no notes
        if not has_notes(tr):
            continue

        stem = name or f"track{i}"
        out_path = os.path.join(out_dir, f"{base}__{safe(stem)}.mid")

        out = MidiFile(type=1, ticks_per_beat=m.ticks_per_beat)

        # Add global track as track 0 (but don't duplicate if we're exporting track0 itself)
        if global_track is not None and i != 0:
            t0 = MidiTrack()
            t0.extend(global_track)
            out.tracks.append(t0)

        t1 = MidiTrack()
        t1.extend(tr)
        out.tracks.append(t1)

        out.save(out_path)
        print(f"Wrote: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("midi", help="Combined MIDI (type-1 with multiple tracks)")
    ap.add_argument("--out", required=True, help="Output directory for split MIDIs")
    args = ap.parse_args()
    split(args.midi, args.out)


if __name__ == "__main__":
    main()

