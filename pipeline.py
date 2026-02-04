#!/usr/bin/env python3
import argparse
import glob
import os
from tqdm import tqdm

from steps.separate import separate_track
from steps.beats_meter import estimate_tempo_downbeats_meter
from steps.transcribe_melodic import transcribe_pitched_tracks
from steps.transcribe_drums import transcribe_drums_to_midi
from steps.assign_parts import assign_seven_classes
from steps.key_normalize import detect_and_normalize_key
from steps.meter_apply import insert_time_signatures
from steps.clean_quantize import gentle_cleanup
from steps.write_midi import assemble_and_write_midi
from steps.qc_render import review_pending_items
from utils.manifest import load_config, read_manifest, write_manifest, song_id_from_path
import pretty_midi

CFG = load_config("config.yaml")


def process_one(audio_path: str, normalize_key: bool = False, no_clean: bool = False):
    # Make a per-run config so we can safely inject flags without mutating global CFG
    cfg = dict(CFG)
    cfg["no_clean"] = bool(no_clean)

    sid = song_id_from_path(audio_path)
    os.makedirs(f"data/midi/{sid}", exist_ok=True)

    manifest_path = f"manifests/{sid}.json"
    manifest = read_manifest(manifest_path)
    manifest.setdefault("song_id", sid)
    manifest.setdefault("source_audio", audio_path)

    # record CLI intent (useful for debugging)
    manifest.setdefault("pipeline_flags", {})
    manifest["pipeline_flags"]["normalize_key"] = bool(normalize_key)
    manifest["pipeline_flags"]["no_clean"] = bool(no_clean)

    # 1) separation
    stems = separate_track(audio_path, cfg, manifest)
    write_manifest(manifest_path, manifest)

    # 2) tempo/downbeats/meter
    meter_info = estimate_tempo_downbeats_meter(stems, cfg, manifest)
    write_manifest(manifest_path, manifest)

    # 3) transcription
    pitched = transcribe_pitched_tracks(stems, cfg, manifest)
    drums = transcribe_drums_to_midi(stems.get("drums"), cfg, manifest)
    write_manifest(manifest_path, manifest)

    # 4) assign 7 classes
    assigned = assign_seven_classes(pitched, drums, stems, cfg, manifest)
    write_manifest(manifest_path, manifest)

    # 5) key normalize (optional)
    if normalize_key:
        normalized = detect_and_normalize_key(assigned, cfg, manifest)
    else:
        key_info = manifest.setdefault("key", {})
        key_info.setdefault("normalized", False)
        key_info.setdefault("transpose_semitones", 0)
        key_info.setdefault("target", None)
        key_info["reason"] = "key normalization disabled via CLI"
        normalized = assigned
    write_manifest(manifest_path, manifest)

    # 6) meter insertion (optional, based on meter_info)
    with_meter = insert_time_signatures(normalized, meter_info, cfg, manifest)
    write_manifest(manifest_path, manifest)

    # 7) cleanup (optional)
    cleanup_info = manifest.setdefault("cleanup", {})
    if no_clean:
        cleanup_info["enabled"] = False
        cleanup_info["reason"] = "cleanup disabled via CLI (--no-clean)"
        cleaned = with_meter
    else:
        cleanup_info["enabled"] = True
        cleaned = gentle_cleanup(with_meter, cfg, manifest)

    write_manifest(manifest_path, manifest)

    # 8) write MIDI
    out_mid = f"data/midi/{sid}/{sid}.mid"
    assemble_and_write_midi(cleaned, meter_info, out_mid, cfg, manifest)
    write_manifest(manifest_path, manifest)

    return out_mid, manifest_path


def process_one_stem(audio_path: str, use_drums: bool = False):
    """
    Simplified processing: convert a single audio file (stem) to MIDI.
    Bypasses separation, tempo detection, multi-track assembly.

    Args:
        audio_path: Path to the audio file (already a stem)
        use_drums: If True, use ADTOF (drums transcription), else Basic Pitch

    Returns:
        (output_midi_path, manifest_path)
    """
    sid = song_id_from_path(audio_path)
    os.makedirs(f"data/midi/{sid}", exist_ok=True)
    manifest_path = f"manifests/{sid}.json"
    manifest = read_manifest(manifest_path)
    manifest.setdefault("song_id", sid)
    manifest.setdefault("source_audio", audio_path)
    manifest.setdefault("processing_mode", "stem_only")

    # For Basic Pitch, we need a minimal tempo in manifest
    if not use_drums:
        manifest.setdefault("meter_key", {}).setdefault("tempo", 120.0)

    write_manifest(manifest_path, manifest)

    # Transcribe the single stem
    if use_drums:
        # Use ADTOF for drum transcription
        instrument = transcribe_drums_to_midi(audio_path, CFG, manifest)
        if instrument is None:
            raise ValueError(f"Drum transcription failed for {audio_path}")
    else:
        # Use Basic Pitch for pitched transcription
        from steps.transcribe_melodic import _bp_predict_events, _events_to_instrument

        try:
            events = _bp_predict_events(audio_path, manifest)
            if not events:
                raise ValueError(f"No notes detected in {audio_path}")
            instrument = _events_to_instrument(events, program=0, name=os.path.basename(audio_path))
        except Exception as e:
            manifest.setdefault("transcription", {})["error"] = str(e)
            write_manifest(manifest_path, manifest)
            raise

    write_manifest(manifest_path, manifest)

    # Create MIDI file with single track
    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(instrument)

    out_mid = f"data/midi/{sid}/{sid}.mid"
    pm.write(out_mid)

    manifest["output_midi"] = out_mid
    write_manifest(manifest_path, manifest)

    return out_mid, manifest_path


def cmd_run_batch(pattern: str, normalize_key: bool = False, no_clean: bool = False,
                  assume_stems: bool = False, assume_drum_stems: bool = False):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files match: {pattern}")
        return 1

    # Determine processing mode
    if assume_stems or assume_drum_stems:
        # Simplified stem-to-MIDI processing
        use_drums = assume_drum_stems
        for f in tqdm(files, desc="Processing stems"):
            try:
                out_mid, mani = process_one_stem(f, use_drums=use_drums)
                print(f"[OK] {f} -> {out_mid}  (manifest: {mani})")
            except Exception as e:
                print(f"[ERR] {f}: {e}")
    else:
        # Full pipeline processing
        for f in tqdm(files, desc="Processing files"):
            try:
                out_mid, mani = process_one(f, normalize_key=normalize_key, no_clean=no_clean)
                print(f"[OK] {f} -> {out_mid}  (manifest: {mani})")
            except Exception as e:
                print(f"[ERR] {f}: {e}")
    return 0


def cmd_review_pending():
    review_pending_items()


def cmd_export_midi(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    import shutil
    for mid in glob.glob("data/midi/*/*.mid"):
        base = os.path.basename(mid)
        sid = os.path.basename(os.path.dirname(mid))
        dst = os.path.join(out_dir, f"{sid}__{base}")
        shutil.copy2(mid, dst)
        print(f"Exported: {dst}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    # run-batch
    r = sub.add_parser("run-batch", help="Run the full pipeline over a glob")
    r.add_argument("pattern", help='e.g., "data/raw/*.wav"')
    r.add_argument(
        "--normalize-key",
        action="store_true",
        help="Normalize pitched tracks to Cmaj/Amin",
    )
    r.add_argument(
        "--no-clean",
        action="store_true",
        help="Disable MIDI cleaning/post-processing",
    )
    r.add_argument(
        "--assume-stems",
        action="store_true",
        help="Skip separation; convert each .wav directly to MIDI using Basic Pitch",
    )
    r.add_argument(
        "--assume-drum-stems",
        action="store_true",
        help="Skip separation; convert each .wav directly to MIDI using ADTOF (drums)",
    )

    # review-pending
    sub.add_parser(
        "review-pending",
        help="Open quick-review UI for low-confidence items",
    )

    # export-midi
    e = sub.add_parser(
        "export-midi",
        help="Copy final MIDIs to an output dir",
    )
    e.add_argument("--out", required=True)

    args = ap.parse_args()

    if args.cmd == "run-batch":
        return cmd_run_batch(
            args.pattern,
            normalize_key=args.normalize_key,
            no_clean=args.no_clean,
            assume_stems=args.assume_stems,
            assume_drum_stems=args.assume_drum_stems,
        )
    elif args.cmd == "review-pending":
        return cmd_review_pending()
    elif args.cmd == "export-midi":
        return cmd_export_midi(args.out)
    else:
        ap.print_help()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
