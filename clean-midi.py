#!/usr/bin/env python3
"""
clean-midi.py - Filter MIDI tracks using adaptive velocity thresholding.

Usage:
    python clean-midi.py Song Clean1                       # data/midi/Song/Song.mid -> SongClean1.mid
    python clean-midi.py Song.mid SongClean1.mid           # Same, explicit .mid
    python clean-midi.py path/to/in.mid path/to/out.mid    # Explicit paths
    python clean-midi.py Song Clean1 --bass                # Bass-specific filtering
    python clean-midi.py Song Clean1 --track guitar        # Single track

If no threshold is given, computes an adaptive threshold using Otsu's method
on the velocity histogram for each track.
"""

import argparse
import os
import sys
import numpy as np
import pretty_midi


# Bass typically ranges E1 (28) to around G3 (55), maybe up to C4 (60)
DEFAULT_BASS_PITCH_MAX = 60  # C4
DEFAULT_MIDI_DIR = "data/midi"


def resolve_paths(input_arg: str, output_arg: str, midi_dir: str) -> tuple[str, str]:
    """
    Resolve input/output arguments to full paths.

    If input_arg is a simple name (no path separator, optionally with .mid),
    expand to: {midi_dir}/{name}/{name}.mid

    If output_arg is a simple suffix (no path separator, no .mid),
    create output as: {midi_dir}/{name}/{name}{suffix}.mid
    """
    # Normalize input
    if os.path.sep in input_arg or input_arg.startswith("."):
        # Explicit path provided
        input_path = input_arg
        # Extract base name for output resolution
        base_name = os.path.splitext(os.path.basename(input_arg))[0]
        base_dir = os.path.dirname(input_arg)
    else:
        # Simple name - expand to data/midi/Name/Name.mid
        base_name = input_arg.replace(".mid", "")
        base_dir = os.path.join(midi_dir, base_name)
        input_path = os.path.join(base_dir, f"{base_name}.mid")

    # Normalize output
    if os.path.sep in output_arg or output_arg.startswith("."):
        # Explicit path provided
        output_path = output_arg
    elif output_arg.endswith(".mid"):
        # Filename with .mid extension - put in same dir as input
        output_path = os.path.join(base_dir, output_arg)
    else:
        # Simple suffix - append to base name
        output_path = os.path.join(base_dir, f"{base_name}{output_arg}.mid")

    return input_path, output_path


def velocity_histogram(notes: list, bins: int = 13) -> tuple[np.ndarray, np.ndarray]:
    """Return histogram counts and bin edges for note velocities."""
    velocities = np.array([n.velocity for n in notes])
    return np.histogram(velocities, bins=np.linspace(0, 130, bins + 1))


def print_histogram(notes: list, title: str = "Velocity histogram") -> None:
    """Print ASCII velocity histogram."""
    if not notes:
        print(f"\n{title}: no notes")
        return
    hist, edges = velocity_histogram(notes, bins=13)
    print(f"\n{title} ({len(notes)} notes):")
    for i, count in enumerate(hist):
        low, high = int(edges[i]), int(edges[i + 1])
        bar = "#" * (count // 5) if count > 0 else ""
        print(f"  {low:3d}-{high:3d}: {count:4d} {bar}")


def otsu_threshold(velocities: np.ndarray) -> int:
    """
    Compute optimal threshold using Otsu's method.

    Finds the threshold that minimizes intra-class variance (equivalently,
    maximizes inter-class variance) for a bimodal distribution.
    """
    if len(velocities) == 0:
        return 64

    v_min, v_max = int(velocities.min()), int(velocities.max())

    # Build histogram with 1-velocity bins
    hist, _ = np.histogram(velocities, bins=range(v_min, v_max + 2))
    hist = hist.astype(float)

    total = hist.sum()
    if total == 0:
        return (v_min + v_max) // 2

    # Compute cumulative sums
    sum_total = np.dot(np.arange(len(hist)), hist)

    best_thresh = v_min
    best_variance = 0.0

    sum_bg = 0.0
    weight_bg = 0.0

    for t in range(len(hist)):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue

        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]

        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        # Inter-class variance
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance > best_variance:
            best_variance = variance
            best_thresh = t + v_min

    return best_thresh


def filter_by_velocity(notes: list, threshold: int, keep_above: bool = True) -> list:
    """Filter notes by velocity threshold."""
    if keep_above:
        return [n for n in notes if n.velocity >= threshold]
    else:
        return [n for n in notes if n.velocity < threshold]


def filter_by_pitch(notes: list, pitch_min: int = 0, pitch_max: int = 127) -> list:
    """Filter notes by pitch range (inclusive)."""
    return [n for n in notes if pitch_min <= n.pitch <= pitch_max]


def quantize_monophonic(notes: list, grid_sec: float) -> list:
    """
    Keep only the loudest note per quantization bin.

    Args:
        notes: List of pretty_midi.Note objects
        grid_sec: Quantization grid size in seconds (e.g., 0.0625 for 16th at 240 BPM)

    Returns:
        Filtered list with at most one note per grid bin (the loudest).
    """
    if not notes or grid_sec <= 0:
        return notes

    # Group notes by their grid bin
    bins: dict[int, list] = {}
    for n in notes:
        bin_idx = int(n.start / grid_sec)
        if bin_idx not in bins:
            bins[bin_idx] = []
        bins[bin_idx].append(n)

    # Keep only the loudest note in each bin
    result = []
    for bin_notes in bins.values():
        loudest = max(bin_notes, key=lambda n: n.velocity)
        result.append(loudest)

    # Sort by start time
    result.sort(key=lambda n: n.start)
    return result


def count_polyphonic_bins(notes: list, grid_sec: float) -> tuple[int, int, int]:
    """
    Count bins with 0, 1, and 2+ notes.

    Returns:
        (empty_bins, mono_bins, poly_bins)
    """
    if not notes or grid_sec <= 0:
        return 0, 0, 0

    starts = np.array([n.start for n in notes])
    if len(starts) == 0:
        return 0, 0, 0

    max_bin = int(starts.max() / grid_sec) + 1
    bin_indices = (starts / grid_sec).astype(int)
    bin_counts = np.bincount(bin_indices, minlength=max_bin)

    empty = (bin_counts == 0).sum()
    mono = (bin_counts == 1).sum()
    poly = (bin_counts >= 2).sum()

    return int(empty), int(mono), int(poly)


def find_monophonic_threshold(
    notes: list,
    grid_sec: float,
    target_poly_pct: float = 1.0,
) -> int:
    """
    Find velocity threshold that achieves target polyphony percentage.

    Args:
        notes: List of notes to analyze
        grid_sec: Quantization grid in seconds
        target_poly_pct: Target percentage of bins with 2+ notes (default 1%)

    Returns:
        Velocity threshold that achieves near-monophonic output
    """
    if not notes:
        return 64

    velocities = sorted(set(n.velocity for n in notes))
    if len(velocities) <= 1:
        return velocities[0] if velocities else 64

    best_thresh = velocities[0]
    best_poly_pct = 100.0

    for thresh in velocities:
        filtered = [n for n in notes if n.velocity >= thresh]
        if not filtered:
            break

        _, mono, poly = count_polyphonic_bins(filtered, grid_sec)
        total_active = mono + poly
        if total_active == 0:
            break

        poly_pct = 100.0 * poly / total_active

        # Track the threshold closest to target without going under
        if poly_pct <= target_poly_pct:
            return thresh

        if poly_pct < best_poly_pct:
            best_poly_pct = poly_pct
            best_thresh = thresh

    return best_thresh


def grid_from_tempo_and_quantize(tempo: float, quantize: int) -> float:
    """
    Compute grid size in seconds from tempo and quantization.

    Args:
        tempo: Beats per minute
        quantize: Note value denominator (4=quarter, 8=eighth, 16=sixteenth)

    Returns:
        Grid size in seconds
    """
    beat_sec = 60.0 / tempo
    return beat_sec * 4.0 / quantize  # 4 because quarter note = 1 beat


def print_pitch_histogram(notes: list, title: str = "Pitch histogram") -> None:
    """Print ASCII pitch histogram grouped by octave."""
    if not notes:
        print(f"\n{title}: no notes")
        return

    pitches = np.array([n.pitch for n in notes])
    p_min, p_max = pitches.min(), pitches.max()

    print(f"\n{title} ({len(notes)} notes, range {p_min}-{p_max}):")
    print(f"  {pretty_midi.note_number_to_name(p_min)} to {pretty_midi.note_number_to_name(p_max)}")

    # Group by octave (C-B)
    octave_counts = {}
    for p in pitches:
        octave = p // 12 - 1  # MIDI octave convention
        octave_counts[octave] = octave_counts.get(octave, 0) + 1

    for octave in sorted(octave_counts.keys()):
        count = octave_counts[octave]
        bar = "#" * (count // 10) if count > 0 else ""
        print(f"  Octave {octave:2d}: {count:4d} {bar}")


def process_track(
    inst: pretty_midi.Instrument,
    threshold: int | None,
    pitch_min: int,
    pitch_max: int,
    histogram_only: bool,
    grid_sec: float | None = None,
    auto_mono: bool = False,
    verbose: bool = True,
) -> tuple[int, int]:
    """
    Process a single track, applying velocity, pitch, and quantize filtering.

    Args:
        inst: The instrument/track to process
        threshold: Manual velocity threshold, or None for auto
        pitch_min/max: Pitch range filter
        histogram_only: If True, don't modify notes
        grid_sec: Quantization grid in seconds (None = no quantize filter)
        auto_mono: If True, find threshold for monophonic output instead of Otsu
        verbose: Print progress

    Returns (original_count, final_count).
    """
    if not inst.notes:
        if verbose:
            print(f"\n=== {inst.name}: no notes, skipping ===")
        return 0, 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"=== {inst.name} ===")
        print_histogram(inst.notes, f"BEFORE velocity: {inst.name}")
        print_pitch_histogram(inst.notes, f"BEFORE pitch: {inst.name}")

    velocities = np.array([n.velocity for n in inst.notes])
    if verbose and len(velocities) > 0:
        v_min, v_max = velocities.min(), velocities.max()
        v_mean, v_std = velocities.mean(), velocities.std()
        print(f"\n  Velocity range: {v_min}-{v_max}, Mean: {v_mean:.1f}, Std: {v_std:.1f}")

    # Show polyphony stats if quantizing
    if grid_sec is not None and verbose:
        empty, mono, poly = count_polyphonic_bins(inst.notes, grid_sec)
        total_active = mono + poly
        if total_active > 0:
            poly_pct = 100.0 * poly / total_active
            print(f"  Polyphony: {poly} of {total_active} active bins ({poly_pct:.1f}%) have 2+ notes")

    # Compute velocity threshold
    if threshold is not None:
        thresh = threshold
        if verbose:
            print(f"  Using manual velocity threshold: {thresh}")
    elif auto_mono and grid_sec is not None:
        thresh = find_monophonic_threshold(inst.notes, grid_sec, target_poly_pct=1.0)
        if verbose:
            print(f"  Auto-monophonic velocity threshold: {thresh}")
    else:
        thresh = otsu_threshold(velocities)
        if verbose:
            print(f"  Otsu's adaptive velocity threshold: {thresh}")

    original_count = len(inst.notes)

    if histogram_only:
        kept = filter_by_velocity(inst.notes, thresh, keep_above=True)
        kept = filter_by_pitch(kept, pitch_min, pitch_max)
        if grid_sec is not None:
            kept = quantize_monophonic(kept, grid_sec)
        dropped = original_count - len(kept)
        if verbose:
            print(f"\n  Would keep {len(kept)} notes, drop {dropped}")
            desc = f"velocity >= {thresh}, pitch {pitch_min}-{pitch_max}"
            if grid_sec is not None:
                print(f"    ({desc}, quantized to {grid_sec*1000:.1f}ms grid)")
            else:
                print(f"    ({desc})")
        return original_count, len(kept)

    # Filter notes
    inst.notes = filter_by_velocity(inst.notes, thresh, keep_above=True)
    after_velocity = len(inst.notes)
    inst.notes = filter_by_pitch(inst.notes, pitch_min, pitch_max)
    after_pitch = len(inst.notes)

    # Quantize: keep only loudest note per grid bin
    if grid_sec is not None:
        inst.notes = quantize_monophonic(inst.notes, grid_sec)
        after_quantize = len(inst.notes)
    else:
        after_quantize = after_pitch

    if verbose:
        if grid_sec is not None:
            print(f"\n  Filtered: {original_count} -> {after_velocity} (velocity) -> {after_pitch} (pitch) -> {after_quantize} (quantize)")
        else:
            print(f"\n  Filtered: {original_count} -> {after_velocity} (velocity) -> {after_pitch} (pitch)")
        print_histogram(inst.notes, f"AFTER velocity: {inst.name}")
        print_pitch_histogram(inst.notes, f"AFTER pitch: {inst.name}")

        # Show final polyphony stats
        if grid_sec is not None and inst.notes:
            _, mono, poly = count_polyphonic_bins(inst.notes, grid_sec)
            total = mono + poly
            if total > 0:
                print(f"  Final polyphony: {poly} of {total} bins ({100*poly/total:.1f}%)")

    return original_count, after_quantize


def main():
    parser = argparse.ArgumentParser(
        description="Filter MIDI tracks using adaptive velocity thresholding.",
        epilog="""
Examples:
  %(prog)s Song Clean1                  # data/midi/Song/Song.mid -> SongClean1.mid
  %(prog)s Song Clean1 --bass -t 40     # Bass only, threshold 40
  %(prog)s Song Clean1 --bass -q 16     # Bass, keep loudest per 16th note
  %(prog)s Song Clean1 --bass -q 16 --auto-mono  # Auto-find monophonic threshold
  %(prog)s Song -H                      # Histogram only (output ignored)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Input: MIDI path OR base name (expands to data/midi/NAME/NAME.mid)"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="Cleaned",
        help="Output: MIDI path OR suffix (default: 'Cleaned' -> NAMECleaned.mid)"
    )
    parser.add_argument(
        "--midi-dir", "-d", default=DEFAULT_MIDI_DIR,
        help=f"Base directory for MIDI files (default: {DEFAULT_MIDI_DIR})"
    )
    parser.add_argument(
        "--threshold", "-t", type=int, default=None,
        help="Velocity threshold (default: auto via Otsu's method per track)"
    )
    parser.add_argument(
        "--bass", "-b", action="store_true",
        help=f"Bass-only mode: filter only bass track with pitch max {DEFAULT_BASS_PITCH_MAX} (C4)"
    )
    parser.add_argument(
        "--track", default=None,
        help="Filter only this track name (default: all tracks)"
    )
    parser.add_argument(
        "--histogram-only", "-H", action="store_true",
        help="Only show histograms, don't write output"
    )
    parser.add_argument(
        "--pitch-max", type=int, default=None,
        help="Maximum pitch (MIDI note number)"
    )
    parser.add_argument(
        "--pitch-min", type=int, default=None,
        help="Minimum pitch (MIDI note number). Default: 0"
    )
    parser.add_argument(
        "--quantize", "-q", type=int, default=None,
        help="Keep only loudest note per N-th note grid (e.g., 16 for 16th notes)"
    )
    parser.add_argument(
        "--auto-mono", "-m", action="store_true",
        help="Auto-find velocity threshold for monophonic output (requires --quantize)"
    )

    args = parser.parse_args()

    # Resolve paths
    input_path, output_path = resolve_paths(args.input, args.output, args.midi_dir)

    # Load MIDI
    try:
        pm = pretty_midi.PrettyMIDI(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Tracks: {[inst.name for inst in pm.instruments]}")

    # Compute quantization grid if requested
    grid_sec = None
    if args.quantize is not None:
        tempo = pm.estimate_tempo()
        grid_sec = grid_from_tempo_and_quantize(tempo, args.quantize)
        print(f"\nTempo: {tempo:.1f} BPM")
        print(f"Quantize grid: 1/{args.quantize} note = {grid_sec*1000:.1f} ms")
        if args.auto_mono:
            print("Auto-mono: will find threshold for monophonic output")

    if args.auto_mono and args.quantize is None:
        print("Warning: --auto-mono requires --quantize, ignoring", file=sys.stderr)

    # Determine which tracks to process
    if args.bass:
        track_names = ["bass"]
        default_pitch_max = DEFAULT_BASS_PITCH_MAX
        print(f"\nBass mode: filtering only 'bass' track, pitch max {default_pitch_max} (C4)")
    elif args.track:
        track_names = [args.track]
        default_pitch_max = 127
    else:
        track_names = None  # All tracks
        default_pitch_max = 127

    pitch_min = args.pitch_min if args.pitch_min is not None else 0
    pitch_max = args.pitch_max if args.pitch_max is not None else default_pitch_max

    total_before = 0
    total_after = 0

    for inst in pm.instruments:
        # Skip drums for velocity filtering (they use different semantics)
        if inst.is_drum:
            print(f"\n=== {inst.name}: drum track, skipping ===")
            continue

        # Check if we should process this track
        if track_names is not None and inst.name not in track_names:
            continue

        # Use bass-specific pitch range if in bass mode
        if args.bass and inst.name == "bass":
            p_max = pitch_max
        elif track_names is None:
            # All-track mode: no pitch filtering by default
            p_max = 127
        else:
            p_max = pitch_max

        before, after = process_track(
            inst,
            threshold=args.threshold,
            pitch_min=pitch_min,
            pitch_max=p_max,
            histogram_only=args.histogram_only,
            grid_sec=grid_sec,
            auto_mono=args.auto_mono,
        )
        total_before += before
        total_after += after

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_before} -> {total_after} notes")

    if args.histogram_only:
        print("\n(histogram-only mode, no file written)")
        sys.exit(0)

    # Write output
    pm.write(output_path)
    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()
