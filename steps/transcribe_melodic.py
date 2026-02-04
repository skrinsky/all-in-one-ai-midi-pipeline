import os
import pretty_midi
import numpy as np
from utils.audio_utils import load_audio_mono

# NEW: TorchCREPE for F0 tracking
try:
    import torchcrepe
    import torch
    HAS_TORCHCREPE = True
except ImportError:
    torchcrepe = None
    torch = None
    HAS_TORCHCREPE = False

import librosa
from .key_normalize import detect_key_only


from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH

# One shared Basic Pitch model
_MODEL = Model(ICASSP_2022_MODEL_PATH)


def _get_midi_tempo(manifest: dict) -> float:
    """
    Use tempo estimated from audio (beats_meter). Fallback: 120.
    """
    mk = manifest.get("meter_key") or {}
    t = mk.get("tempo")
    try:
        t = float(t)
    except (TypeError, ValueError):
        t = None
    return t if t and t > 0 else 120.0

# =========================
# KEY PRIOR HELPERS
# =========================

_NOTE_PC = {
    "C": 0, "B#": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "E#": 5, "F": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}

_MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
_MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}  # natural minor (good enough for a prior)

def _parse_key_mode(manifest: dict):
    """
    Attempts to extract tonic + mode from:
      1) manifest["meter_key"] (your old path)
      2) manifest["key"] (from key_normalize.py)
      3) a combined string like "C minor" from either place

    Returns (tonic_pc, mode_str) or (None, None).
    """
    mk = manifest.get("meter_key") or {}
    k  = manifest.get("key") or {}

    # 1) Primary: meter_key-style fields
    tonic = mk.get("key") or mk.get("tonic") or mk.get("key_name")
    mode  = mk.get("mode") or mk.get("scale") or mk.get("key_mode")

    # 2) Fallback: key_normalize.py fields
    if tonic is None:
        tonic = k.get("detected_tonic")
        mode  = k.get("detected_mode")

    # 3) Fallback: parse combined strings from either dict
    if tonic is None:
        ks = (
            mk.get("key_signature") or mk.get("key_str")
            or k.get("detected_key") or k.get("key_str") or k.get("key_signature")
        )
        if isinstance(ks, str) and ks.strip():
            parts = ks.strip().split()
            tonic = parts[0]
            mode = parts[1] if len(parts) > 1 else mode

    if not isinstance(tonic, str):
        return None, None

    tonic = tonic.strip()
    if tonic not in _NOTE_PC:
        tonic = tonic.replace("♭", "b").replace("♯", "#").strip().capitalize()
        if tonic not in _NOTE_PC:
            return None, None

    tonic_pc = _NOTE_PC[tonic]

    mode_str = "major"
    if isinstance(mode, str):
        m = mode.strip().lower()
        if "min" in m:
            mode_str = "minor"
        elif "maj" in m:
            mode_str = "major"
    return tonic_pc, mode_str


def _scale_pitch_classes(tonic_pc: int, mode: str):
    base = _MINOR_SCALE if (mode == "minor") else _MAJOR_SCALE
    return {(tonic_pc + x) % 12 for x in base}


def _pc_distance_to_set(pc: int, pc_set: set[int]) -> int:
    """
    Returns minimum semitone distance on pitch-class circle to any pc in pc_set.
    Range: 0..6
    """
    dmin = 12
    for tgt in pc_set:
        d = (pc - tgt) % 12
        d = min(d, 12 - d)
        dmin = min(dmin, d)
    return int(dmin)


# =========================
# KEY-PRIOR VERSION OF CREPE GATE
# =========================
def _snap_to_key_scale_near(
    midi_center: float,
    key_pcs: set[int],
    max_snap_semitones: int = 2,
) -> int:
    """
    Return an integer MIDI pitch near midi_center whose pitch-class is in key_pcs.
    Searches +/- max_snap_semitones around round(midi_center).
    Falls back to round(midi_center) if nothing found.
    """
    c = int(round(float(midi_center)))
    best = None
    best_dist = 999

    for d in range(-int(max_snap_semitones), int(max_snap_semitones) + 1):
        cand = c + d
        if (cand % 12) in key_pcs:
            dist = abs(d)
            if dist < best_dist:
                best = cand
                best_dist = dist

    return best if best is not None else c

def _gate_with_crepe_then_optional_keysnap(
    events,
    f0_times, f0_hz, f0_conf,
    tonic_pc: int,
    mode: str,
    # same base thresholds as before
    conf_thresh=0.30,
    voiced_ratio_thresh=0.15,
    lookahead_s=0.01,
    win_s=0.15,
    min_conf_mean=0.18,
    min_note_dur_s=0.06,
    max_semitone_diff=2.5,
    vel_bypass=90,
    tag="[voxlead gate+keysnap]",
    # mild keysnap knobs (ONLY affects pitch of kept notes)
    keysnap_enable=True,
    keysnap_max_semitones=2,
    keysnap_only_if_weak_vel=65,   # set None to allow all velocities
    outkey_dist_apply=1,
    keysnap_require_f0_agreement=True,  # recommended
    # NEW: hard-snap tier for "completely out of key"
    hard_snap_enable=True,
    hard_outkey_dist=4,            # start safer at 4; try 3 once stable
    hard_snap_max_semitones=6,     # search wider for diatonic target
    hard_only_if_weak_vel=70,      # None to allow all velocities
    hard_only_if_short_dur_s=0.35, # None to allow long notes too
    hard_require_f0_agreement=True # keep True: only snap if close to CREPE
):
    if not events:
        return events
    if f0_times is None or f0_hz is None or f0_conf is None:
        return events
    if tonic_pc is None:
        return events

    key_pcs = _scale_pitch_classes(int(tonic_pc), mode if mode in ("major", "minor") else "major")
    events = sorted(events, key=lambda x: float(x[0]))

    kept = []
    dropped = 0
    bypassed = 0
    changed = 0
    hard_changed = 0
    mild_changed = 0

    for (s, e, p_bp, v) in events:
        s = float(s); e = float(e)
        p_bp = int(p_bp); v = int(v)
        dur = e - s
        if dur <= 0:
            continue

        # Bypass (kept as-is; no gate and no snap)
        if vel_bypass is not None and v >= int(vel_bypass):
            kept.append((s, e, p_bp, v))
            bypassed += 1
            continue

        # CREPE window
        t0 = s + float(lookahead_s)
        t1 = min(e, t0 + float(win_s))
        if t1 <= t0:
            if dur < float(min_note_dur_s):
                dropped += 1
                continue
            kept.append((s, e, p_bp, v))
            continue

        i0 = int(np.searchsorted(f0_times, t0, side="left"))
        i1 = int(np.searchsorted(f0_times, t1, side="right"))
        i0 = max(0, min(i0, len(f0_times)))
        i1 = max(i0, min(i1, len(f0_times)))

        if i1 - i0 < 2:
            if dur < float(min_note_dur_s):
                dropped += 1
                continue
            kept.append((s, e, p_bp, v))
            continue

        seg_hz = f0_hz[i0:i1]
        seg_c  = f0_conf[i0:i1]

        conf_mean = float(np.nanmean(seg_c)) if seg_c.size else 0.0
        good = (seg_c >= float(conf_thresh)) & np.isfinite(seg_hz) & (seg_hz > 0)
        voiced_ratio = float(np.mean(good)) if seg_c.size else 0.0

        f0_med = None
        if np.sum(good) >= 2:
            f0_med = float(np.nanmedian(_hz_to_midi(seg_hz[good])))

        crepe_confident = (
            (f0_med is not None)
            and (conf_mean >= float(min_conf_mean))
            and (voiced_ratio >= float(voiced_ratio_thresh))
        )

        # ---- PHASE 1: KEEP/DROP (UNCHANGED METRIC) ----
        if not crepe_confident:
            if dur < float(min_note_dur_s):
                dropped += 1
                continue
            kept.append((s, e, p_bp, v))
            continue

        # CREPE confident: require BP pitch agreement to keep
        if abs(p_bp - f0_med) > float(max_semitone_diff):
            dropped += 1
            continue

        # ---- PHASE 2: OPTIONAL KEY SNAP (ONLY AFTER KEEP) ----
        p_out = p_bp

        if keysnap_enable:
            pc_dist = _pc_distance_to_set(p_bp % 12, key_pcs)

            # Decide if hard snap applies
            hard_ok = True
            if hard_only_if_weak_vel is not None:
                hard_ok = hard_ok and (v < int(hard_only_if_weak_vel))
            if hard_only_if_short_dur_s is not None:
                hard_ok = hard_ok and (dur <= float(hard_only_if_short_dur_s))

            is_hard_out = (pc_dist >= int(hard_outkey_dist))

            # Decide if mild snap applies
            mild_ok = True
            if keysnap_only_if_weak_vel is not None:
                mild_ok = (v < int(keysnap_only_if_weak_vel))

            is_mild_out = (pc_dist >= int(outkey_dist_apply))

            # Tier 1: HARD snap for "completely out-of-key"
            if hard_snap_enable and is_hard_out and hard_ok:
                p_cand = _snap_to_key_scale_near(
                    midi_center=f0_med,
                    key_pcs=key_pcs,
                    max_snap_semitones=int(hard_snap_max_semitones),
                )
                if (not hard_require_f0_agreement) or (abs(p_cand - f0_med) <= float(max_semitone_diff)):
                    p_out = int(p_cand)
                    if p_out != p_bp:
                        hard_changed += 1

            # Tier 2: MILD snap for "questionable" (only if hard didn't change it)
            elif is_mild_out and mild_ok:
                p_cand = _snap_to_key_scale_near(
                    midi_center=f0_med,
                    key_pcs=key_pcs,
                    max_snap_semitones=int(keysnap_max_semitones),
                )
                if (not keysnap_require_f0_agreement) or (abs(p_cand - f0_med) <= float(max_semitone_diff)):
                    p_out = int(p_cand)
                    if p_out != p_bp:
                        mild_changed += 1

        if p_out != p_bp:
            changed += 1

        kept.append((s, e, p_out, v))

    kept.sort(key=lambda x: x[0])
    print(
        f"{tag} in={len(events)} kept={len(kept)} dropped={dropped} "
        f"changed_pitch={changed} (hard={hard_changed} mild={mild_changed}) "
        f"bypassed_vel>={vel_bypass}={bypassed}",
        flush=True,
    )

    if len(kept) == 0:
        print(f"{tag} kept=0 -> returning original events", flush=True)
        return events
    return kept



def _filter_bass_silence(bass_path, events,
                         rms_thresh_db=-45.0,
                         min_active_ratio=0.2):
    """
    Remove bass notes that occur where the bass stem is effectively silent.

    - rms_thresh_db: below this (dBFS-ish) we treat as silence.
    - min_active_ratio: fraction of frames in the note window that must be "loud"
                        to keep the note.
    """
    if not events or not bass_path or not os.path.exists(bass_path):
        return events

    y, sr = load_audio_mono(bass_path)
    if y is None or y.size == 0:
        return events

    # Short-time RMS
    frame_hop = int(0.01 * sr)   # 10 ms hop
    frame_len = int(0.03 * sr)   # 30 ms window
    if frame_len <= 0 or frame_hop <= 0 or frame_len >= len(y):
        return events

    rms = []
    for i in range(0, len(y) - frame_len, frame_hop):
        win = y[i:i + frame_len]
        val = np.sqrt(np.mean(win * win) + 1e-12)
        rms.append(val)
    if not rms:
        return events

    rms = np.asarray(rms)
    rms_db = 20.0 * np.log10(rms + 1e-12)

    # Time per RMS frame
    step_t = frame_hop / float(sr)

    def is_note_active(s, e):
        if e <= s:
            return False
        start_idx = int(s / step_t)
        end_idx = int(e / step_t)
        start_idx = max(0, min(start_idx, len(rms_db) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(rms_db)))
        seg = rms_db[start_idx:end_idx]
        if seg.size == 0:
            return False
        active = (seg > rms_thresh_db).mean()
        return active >= min_active_ratio

    kept = []
    for (s, e, p, v) in events:
        if is_note_active(s, e):
            kept.append((s, e, p, v))

    return kept
    
def _filter_stem_silence(
    stem_path,
    events,
    rms_thresh_db=-55.0,
    min_active_ratio=0.15,
    hop_s=0.01,
    win_s=0.03,
    tag="[silence_filter]",
):
    """
    Drop notes that occur where the stem is effectively silent (Demucs dropouts, etc.)

    - rms_thresh_db: below this (dB) treated as silence
    - min_active_ratio: fraction of frames in note window that must be above thresh
    """
    if not events or not stem_path or not os.path.exists(stem_path):
        return events

    y, sr = load_audio_mono(stem_path)
    if y is None or y.size == 0:
        return events

    frame_hop = int(max(1, round(hop_s * sr)))
    frame_len = int(max(frame_hop + 1, round(win_s * sr)))
    if frame_len >= len(y):
        return events

    rms = []
    for i in range(0, len(y) - frame_len, frame_hop):
        win = y[i:i + frame_len]
        val = np.sqrt(np.mean(win * win) + 1e-12)
        rms.append(val)

    if not rms:
        return events

    rms = np.asarray(rms)
    rms_db = 20.0 * np.log10(rms + 1e-12)
    step_t = frame_hop / float(sr)

    def active_ratio(s, e):
        if e <= s:
            return 0.0
        i0 = int(s / step_t)
        i1 = int(e / step_t)
        i0 = max(0, min(i0, len(rms_db) - 1))
        i1 = max(i0 + 1, min(i1, len(rms_db)))
        seg = rms_db[i0:i1]
        if seg.size == 0:
            return 0.0
        return float((seg > rms_thresh_db).mean())

    kept = []
    dropped = 0
    for s, e, p, v in events:
        ar = active_ratio(float(s), float(e))
        if ar >= float(min_active_ratio):
            kept.append((s, e, p, v))
        else:
            dropped += 1

    print(f"{tag} in={len(events)} kept={len(kept)} dropped={dropped} thr={rms_thresh_db} ratio>={min_active_ratio}", flush=True)
    return kept

    
def _drop_short_quiet(events, vel_max=30, max_dur_s=0.06, tag="[short+quiet]"):
    """
    Drop notes that are BOTH short and quiet.
    Good for polyphonic confetti cleanup (guitar/other).
    """
    if not events:
        return events
    kept = []
    dropped = 0
    for s, e, p, v in events:
        dur = float(e) - float(s)
        if dur <= float(max_dur_s) and int(v) <= int(vel_max):
            dropped += 1
            continue
        kept.append((float(s), float(e), int(p), int(v)))
    print(f"{tag} in={len(events)} kept={len(kept)} dropped={dropped}", flush=True)
    return kept


def _cap_density_drop_quiet_only(
    events,
    bin_s=0.05,
    max_notes_per_bin=10,
    never_drop_vel=40,     # notes >= this are immune to density dropping
    tag="[cap_density_quiet_only]",
):
    """
    Density cap that ONLY drops "quiet" notes. Loud notes (>= never_drop_vel)
    are always kept, even if that means exceeding max_notes_per_bin.

    This prevents legit notes (e.g., vel 47/60/90) from disappearing just because
    a chord landed in the same time bin.
    """
    if not events:
        return events

    events = sorted(events, key=lambda x: float(x[0]))

    # group indices by onset bin
    bins = {}
    for idx, (s, e, p, v) in enumerate(events):
        b = int(float(s) / float(bin_s))
        bins.setdefault(b, []).append(idx)

    keep_mask = [True] * len(events)
    dropped = 0

    for b, idxs in bins.items():
        if len(idxs) <= int(max_notes_per_bin):
            continue

        loud = [i for i in idxs if int(events[i][3]) >= int(never_drop_vel)]
        quiet = [i for i in idxs if int(events[i][3]) < int(never_drop_vel)]

        # Loud notes are always kept
        for i in loud:
            keep_mask[i] = True

        # Only quiet notes can be dropped
        # If we already exceed the cap with loud notes alone, we still keep them all.
        remaining = int(max_notes_per_bin) - len(loud)
        if remaining <= 0:
            # drop all quiet notes in this bin
            for i in quiet:
                if keep_mask[i]:
                    keep_mask[i] = False
                    dropped += 1
            continue

        # Keep the strongest quiet notes to fill remaining slots
        quiet_sorted = sorted(quiet, key=lambda i: int(events[i][3]), reverse=True)
        quiet_keep = set(quiet_sorted[:remaining])

        for i in quiet:
            if i not in quiet_keep and keep_mask[i]:
                keep_mask[i] = False
                dropped += 1

    kept = [ev for ev, keep in zip(events, keep_mask) if keep]
    print(f"{tag} in={len(events)} kept={len(kept)} dropped={dropped} never_drop_vel>={never_drop_vel}", flush=True)
    return kept



def _merge_same_pitch(events, max_gap=0.05):
    """
    Merge consecutive same-pitch notes separated by tiny gaps.
    Helps remove double-hits on sustained notes.
    """
    if not events:
        return events

    events = sorted(events, key=lambda x: x[0])
    merged = []
    cur_s, cur_e, cur_p, cur_v = events[0]

    for s, e, p, v in events[1:]:
        if p == cur_p and s - cur_e <= max_gap:
            # extend current note
            cur_e = max(cur_e, e)
            cur_v = max(cur_v, v)
        else:
            merged.append((cur_s, cur_e, cur_p, cur_v))
            cur_s, cur_e, cur_p, cur_v = s, e, p, v

    merged.append((cur_s, cur_e, cur_p, cur_v))
    return merged


def _squash_vibrato(events, semitone_tol=1, max_span=0.25):
    """
    Collapse very short ±1 semitone wiggles into the main note.
    If a brief note sits between two similar pitches, snap it.
    """
    if not events:
        return events

    events = sorted(events, key=lambda x: x[0])
    cleaned = []

    for i, (s, e, p, v) in enumerate(events):
        dur = e - s
        if dur < max_span and 0 < i < len(events) - 1:
            _, _, p_prev, _ = events[i - 1]
            _, _, p_next, _ = events[i + 1]
            if abs(p - p_prev) <= semitone_tol and abs(p - p_next) <= semitone_tol:
                main_p = round((p_prev + p_next) / 2)
                cleaned.append((s, e, main_p, v))
                continue
        cleaned.append((s, e, p, v))

    return cleaned

def _drop_low_vel_short(events, vel_floor=20, max_dur_s=0.10, tag="[drop_low_vel_short]"):
    """
    Drop notes that are both very quiet and very short.
    Great for polyphonic stems where BP creates tiny low-vel specks.
    """
    if not events:
        return events
    kept = []
    dropped = 0
    for s, e, p, v in events:
        dur = float(e) - float(s)
        if int(v) < int(vel_floor) and dur <= float(max_dur_s):
            dropped += 1
            continue
        kept.append((float(s), float(e), int(p), int(v)))
    print(f"{tag} in={len(events)} kept={len(kept)} dropped={dropped} (vel<{vel_floor} & dur<={max_dur_s})", flush=True)
    return kept


def _bp_predict_events(
    audio_path: str,
    manifest: dict,
    onset_threshold=0.5,
    frame_threshold=0.3,
    min_note_len=0.03,
):
    """
    Run Basic Pitch with midi_tempo from manifest and normalize output into:
        [(start, end, pitch, velocity), ...]
    Supports both:
      - dict {"notes": ...}
      - tuple (model_output, midi_data, note_events)
    """
    midi_tempo = _get_midi_tempo(manifest)

    out = predict(
        audio_path,
        _MODEL,
        midi_tempo=midi_tempo,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_len,
    )

    events = []

    # Case 1: dict-style {"notes": ...}
    if isinstance(out, dict) and "notes" in out:
        for row in out["notes"]:
            if len(row) < 4:
                continue
            onset, offset, midi, vel = row[:4]
            onset = float(onset)
            offset = float(offset)
            if offset <= onset:
                continue

            vel = float(vel)
            if 0.0 <= vel <= 1.0:
                vel *= 127.0
            vel = int(round(max(1, min(127, vel))))

            events.append((onset, offset, int(midi), vel))
        return events

    # Case 2: tuple-style (model_output, midi_data, note_events)
    if isinstance(out, (tuple, list)) and len(out) == 3:
        _, _, note_events = out
        for ev in note_events:
            if isinstance(ev, dict):
                onset = float(
                    ev.get("start_time")
                    or ev.get("onset_time")
                    or ev.get("start")
                    or 0.0
                )
                offset = float(
                    ev.get("end_time")
                    or ev.get("offset_time")
                    or ev.get("end")
                    or (onset + 0.02)
                )
                pitch = int(ev.get("pitch") or ev.get("midi_note_number") or 0)
                vel = ev.get("velocity") or ev.get("amplitude") or 80
            elif isinstance(ev, (tuple, list)) and len(ev) >= 4:
                onset, offset, pitch, vel = ev[:4]
            else:
                continue

            onset = float(onset)
            offset = float(offset)
            pitch = int(pitch)
            vel = float(vel)

            if offset <= onset or pitch <= 0:
                continue

            if 0.0 <= vel <= 1.0:
                vel *= 127.0
            vel = int(round(max(1, min(127, vel))))

            events.append((onset, offset, pitch, vel))
        return events

    raise ValueError(f"Unexpected basic_pitch.predict() output type: {type(out)}")


def _events_to_instrument(events, program=0, name=""):
    """
    Convert events list into a single pretty_midi.Instrument.
    """
    inst = pretty_midi.Instrument(program=program, is_drum=False, name=name)
    for s, e, p, v in events:
        if e <= s or p <= 0:
            continue
        inst.notes.append(
            pretty_midi.Note(
                start=float(s),
                end=float(e),
                pitch=int(p),
                velocity=int(v),
            )
        )
    return inst


def _split_lead_harmony(events):
    """
    Split vocal events into lead vs harmony:
      - for each note, look at its midpoint
      - if it's the highest active pitch at that time -> lead
      - otherwise -> harmony
    """
    if not events:
        return [], []

    lead = []
    harm = []

    for (s, e, p, v) in events:
        mid = 0.5 * (s + e)
        active_pitches = [pp for (ss, ee, pp, vv) in events if ss <= mid <= ee]
        if not active_pitches or p >= max(active_pitches):
            lead.append((s, e, p, v))
        else:
            harm.append((s, e, p, v))

    return lead, harm

# =========================
# CREPE F0 + F0-guided cleanup
# =========================

def _hz_to_midi(f_hz: float) -> float:
    return 69.0 + 12.0 * np.log2(f_hz / 440.0)


def _run_crepe_f0(
    y: np.ndarray,
    sr: int,
    step_size_ms: float = 10.0,
    device: str = "cpu",
    fmin: float = 30.0,
    fmax: float = 600.0,
    model: str = "tiny",
    batch_size: int = 1024,
):
    """
    Run TorchCREPE and return frame-wise F0: (times, freqs Hz, confidences).
    Resamples to 16k if needed.
    """
    if not HAS_TORCHCREPE:
        return None, None, None

    target_sr = 16000
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    audio = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)
    if device != "cpu":
        audio = audio.to(device)

    hop_length = int(round((step_size_ms / 1000.0) * sr))
    hop_length = max(1, hop_length)

    with torch.inference_mode():
        f0_hz, periodicity = torchcrepe.predict(
            audio, sr,
            hop_length=hop_length,
            fmin=float(fmin), fmax=float(fmax),
            model=model,
            batch_size=int(batch_size),
            device=device,
            return_periodicity=True,
        )

    f0_hz = f0_hz[0].detach().cpu().numpy()
    conf = periodicity[0].detach().cpu().numpy()
    times = (np.arange(len(f0_hz)) * hop_length) / float(sr)

    return times.astype(float), f0_hz.astype(float), conf.astype(float)


def _f0_stats_in_window(f0_times, f0_hz, f0_conf, t0, t1, conf_thresh=0.35):
    """
    Returns (midi_median, voiced_ratio, conf_mean) in [t0, t1].
    voiced_ratio is fraction of frames with conf >= conf_thresh and finite f0.
    """
    if f0_times is None or f0_hz is None or f0_conf is None:
        return None, 0.0, 0.0

    t0 = float(t0); t1 = float(t1)
    if t1 <= t0:
        return None, 0.0, 0.0

    i0 = int(np.searchsorted(f0_times, t0, side="left"))
    i1 = int(np.searchsorted(f0_times, t1, side="right"))
    i0 = max(0, min(i0, len(f0_times)))
    i1 = max(i0, min(i1, len(f0_times)))

    if i1 - i0 < 2:
        return None, 0.0, 0.0

    seg_hz = f0_hz[i0:i1]
    seg_c  = f0_conf[i0:i1]

    good = (seg_c >= conf_thresh) & np.isfinite(seg_hz) & (seg_hz > 0)
    voiced_ratio = float(good.mean()) if seg_c.size else 0.0
    conf_mean = float(np.nanmean(seg_c[good])) if good.any() else float(np.nanmean(seg_c))

    if good.sum() < 1:
        return None, voiced_ratio, conf_mean

    midi = _hz_to_midi(seg_hz[good])
    midi_med = float(np.nanmedian(midi))
    return midi_med, voiced_ratio, conf_mean


def _snap_to_nearest_octave(p_bp: int, f0_midi: float) -> int:
    """
    Snap BP pitch to nearest octave-equivalent of f0_midi.
    Useful if BP is octave-wrong but pitch class matches.
    """
    if f0_midi is None or np.isnan(f0_midi):
        return p_bp
    f0_round = int(round(f0_midi))
    candidates = [f0_round - 24, f0_round - 12, f0_round, f0_round + 12, f0_round + 24]
    return min(candidates, key=lambda c: abs(c - p_bp))

def _filter_high_pitch_low_vel(
    events,
    pitch_hi=55,            # above this is suspicious for bass (C4=60)
    vel_max=50,             # only kill if velocity is this low or lower
    max_dur_s=0.18,         # optional: only kill if the note is shortish
    keep_if_long=False,      # safety: keep long notes even if quiet/high
):
    """
    Drop bass events that look like spray: very high pitch + very low velocity.
    Optionally restrict to short durations too.

    This targets the classic "octave jump / harmonic / noise" garbage BP emits.
    """
    if not events:
        return events

    kept = []
    dropped = 0
    for s, e, p, v in events:
        dur = float(e) - float(s)
        p = int(p)
        v = int(v)

        suspicious = (p >= pitch_hi) and (v <= vel_max)

        if suspicious:
            if keep_if_long and dur > max_dur_s:
                kept.append((s, e, p, v))
                continue
            if dur <= max_dur_s:
                dropped += 1
                continue
            # if keep_if_long is False, drop regardless of dur
            if not keep_if_long:
                dropped += 1
                continue

        kept.append((s, e, p, v))

    print(f"[bass] highpitch+lowvel filter: in={len(events)} kept={len(kept)} dropped={dropped}", flush=True)
    return kept

def _gate_with_crepe(
    events,
    f0_times, f0_hz, f0_conf,
    conf_thresh=0.25,
    voiced_ratio_thresh=0.10,
    lookahead_s=0.03,
    win_s=0.20,
    min_conf_mean=0.12,
    min_note_dur_s=0.10,
    max_semitone_diff=4.0,
    do_pitch_snap=True,
    allow_pitch_change=True,
    vel_bypass=90,          # NEW: if velocity >= this, skip gating (always keep)
    debug_n=0,
    tag="[gate]",
):
    if (not events) or (f0_times is None) or (f0_hz is None) or (f0_conf is None):
        return events

    events = sorted(events, key=lambda x: x[0])

    kept = []
    dropped = 0
    changed = 0
    bypassed = 0  # NEW

    for i, (s, e, p_bp, v) in enumerate(events):
        s = float(s); e = float(e)
        p_bp = int(p_bp); v = int(v)
        dur = e - s
        if dur <= 0:
            continue

        # NEW: bypass gating for strong BP notes
        if vel_bypass is not None and v >= int(vel_bypass):
            kept.append((s, e, p_bp, v))
            bypassed += 1
            if debug_n and i < debug_n:
                print(f"{tag} i={i} vel={v} >= {vel_bypass} -> BYPASS KEEP p={p_bp}", flush=True)
            continue

        t0 = s + float(lookahead_s)
        t1 = min(e, t0 + float(win_s))
        if t1 <= t0:
            if dur < min_note_dur_s:
                dropped += 1
                continue
            kept.append((s, e, p_bp, v))
            continue

        i0 = int(np.searchsorted(f0_times, t0, side="left"))
        i1 = int(np.searchsorted(f0_times, t1, side="right"))
        i0 = max(0, min(i0, len(f0_times)))
        i1 = max(i0, min(i1, len(f0_times)))

        if i1 - i0 < 2:
            if dur < min_note_dur_s:
                dropped += 1
                continue
            kept.append((s, e, p_bp, v))
            continue

        seg_hz = f0_hz[i0:i1]
        seg_c  = f0_conf[i0:i1]

        conf_mean = float(np.nanmean(seg_c)) if seg_c.size else 0.0
        good = (seg_c >= conf_thresh) & np.isfinite(seg_hz) & (seg_hz > 0)
        voiced_ratio = float(np.mean(good)) if seg_c.size else 0.0

        f0_med = None
        if np.sum(good) >= 2:
            f0_med = float(np.nanmedian(_hz_to_midi(seg_hz[good])))

        if debug_n and i < debug_n:
            print(
                f"{tag} i={i} s={s:.3f} e={e:.3f} dur={dur:.3f} "
                f"t0={t0:.3f} t1={t1:.3f} f0_med={None if f0_med is None else round(f0_med,2)} "
                f"vr={voiced_ratio:.2f} conf_mean={conf_mean:.3f} p_bp={p_bp} v={v}",
                flush=True,
            )

        crepe_confident = (
            (f0_med is not None)
            and (conf_mean >= min_conf_mean)
            and (voiced_ratio >= voiced_ratio_thresh)
        )

        if not crepe_confident:
            if dur < min_note_dur_s:
                dropped += 1
                continue
            kept.append((s, e, p_bp, v))
            continue

        p_out = p_bp
        if do_pitch_snap:
            p_out = _snap_to_nearest_octave(p_bp, f0_med)

        if abs(p_out - f0_med) <= max_semitone_diff:
            if allow_pitch_change:
                if p_out != p_bp:
                    changed += 1
                kept.append((s, e, p_out, v))
            else:
                kept.append((s, e, p_bp, v))
        else:
            dropped += 1

    kept.sort(key=lambda x: x[0])
    print(
        f"{tag} in={len(events)} kept={len(kept)} dropped={dropped} "
        f"changed_pitch={changed} bypassed_vel>={vel_bypass}={bypassed}",
        flush=True,
    )

    if len(kept) == 0:
        print(f"{tag} kept=0 -> skipping gate (returning original events)", flush=True)
        return events

    return kept


def _is_cleaning_enabled(CFG: dict) -> bool:
    """Return True if post-transcription MIDI cleaning should run.

    Cleaning is ON by default. Disable it via:
      - CFG['no_clean'] = True
      - CFG['clean'] = False
      - CFG['cleaning']['enabled'] = False
      - environment variable NO_CLEAN=1
    """
    if CFG is None:
        CFG = {}

    # Explicit disables win
    if bool(CFG.get("no_clean", False)):
        return False

    cleaning_cfg = CFG.get("cleaning", {})
    if isinstance(cleaning_cfg, dict) and "enabled" in cleaning_cfg:
        enabled = bool(cleaning_cfg.get("enabled", True))
    else:
        enabled = bool(CFG.get("clean", True))

    env = os.getenv("NO_CLEAN") or os.getenv("NO_CLEANING")
    if env is not None and env.strip().lower() in {"1", "true", "yes", "y", "on"}:
        return False

    return enabled

def transcribe_pitched_tracks(stems: dict, CFG: dict, manifest: dict):
    """
    Use Basic Pitch (+ midi_tempo) on:
      - vocals -> voxlead, voxbg (with vocal-specific cleanup)
      - bass   -> bass
      - guitar -> guitar
      - other  -> other (as pad/synth-ish via program)

    Cleaning / post-processing is ON by default. To bypass it, set one of:
      - CFG['no_clean'] = True
      - CFG['clean'] = False
      - CFG['cleaning'] = {'enabled': False}
      - environment variable NO_CLEAN=1

    Returns:
      dict[name -> pretty_midi.Instrument]
    """
    pitched = {}
    status = {}

    clean_enabled = _is_cleaning_enabled(CFG)
    print(f"[cleaning] enabled={clean_enabled}", flush=True)

    # ---------- BASS ----------
    b_path = stems.get("bass")
    if b_path and os.path.exists(b_path):
        try:
            # 1) Basic Pitch bass events
            b_events = _bp_predict_events(b_path, manifest)
            raw_b_events = list(b_events)  # fallback

            print(f"[bass] HAS_TORCHCREPE={HAS_TORCHCREPE}", flush=True)
            print(f"[bass] bp events={len(raw_b_events)} stem={b_path}", flush=True)

            if clean_enabled:
                # Clamp to a sane bass range (adjust as you like)
                BASS_MIN = 28   # ~E1
                BASS_MAX = 72   # ~C5
                b_events = [(s, e, p, v) for (s, e, p, v) in b_events if BASS_MIN <= int(p) <= BASS_MAX]
                raw_b_events = list(b_events)

                # 2) TorchCREPE F0 gate (optional)
                y_bass, sr_bass = load_audio_mono(b_path)
                print(
                    f"[bass] y_ok={y_bass is not None} n={0 if y_bass is None else y_bass.size} sr={sr_bass}",
                    flush=True,
                )

                if y_bass is not None and y_bass.size > 0 and HAS_TORCHCREPE:
                    print("[bass] entering TorchCREPE F0...", flush=True)

                    crepe_device = "cpu"
                    if torch is not None:
                        if torch.backends.mps.is_available():
                            crepe_device = "mps"
                        elif torch.cuda.is_available():
                            crepe_device = "cuda"
                    print(f"[bass] crepe_device={crepe_device}", flush=True)

                    try:
                        f0_times, f0_hz, f0_conf = _run_crepe_f0(
                            y_bass,
                            sr_bass,
                            step_size_ms=20.0,
                            device=crepe_device,
                        )

                        before = len(b_events)
                        b_events = _gate_with_crepe(
                            b_events, f0_times, f0_hz, f0_conf,
                            conf_thresh=0.25,
                            voiced_ratio_thresh=0.10,
                            lookahead_s=0.03,
                            win_s=0.20,
                            min_conf_mean=0.12,
                            min_note_dur_s=0.10,
                            max_semitone_diff=4.0,
                            do_pitch_snap=True,
                            vel_bypass=90,
                            allow_pitch_change=True,
                            tag="[bass gate]"
                        )

                        print(f"[bass gate] {before} -> {len(b_events)} events", flush=True)

                    except Exception as fe:
                        print(f"[bass] TorchCREPE failed ({fe}); falling back to Basic Pitch", flush=True)
                        b_events = raw_b_events

                # Safety: if gating killed everything, fall back
                if not b_events and raw_b_events:
                    b_events = raw_b_events

                # 3) Optional: kill high-pitch + low-velocity spray (after gate/fallback, before merge)
                print(f"[bass] pre-hi/low filter: {len(b_events)}", flush=True)
                b_events = _filter_high_pitch_low_vel(
                    b_events,
                    pitch_hi=55,     # try 55–62 depending on your material
                    vel_max=50,      # try 25–45
                    max_dur_s=0.18,  # try 0.12–0.25
                    keep_if_long=True,
                )
                print(f"[bass] post-hi/low filter: {len(b_events)}", flush=True)

                # 4) Merge micro-chops
                print(f"[bass] pre-merge: {len(b_events)}", flush=True)
                b_events = _merge_same_pitch(b_events, max_gap=0.005)
                print(f"[bass] post-merge: {len(b_events)}", flush=True)

                # 5) Silence filter (last)
                b_events = _filter_bass_silence(
                    b_path,
                    b_events,
                    rms_thresh_db=-50.0,
                    min_active_ratio=0.1,
                )

                # Final safety fallback if silence filter wipes everything
                if not b_events and raw_b_events:
                    b_events = _merge_same_pitch(raw_b_events, max_gap=0.08)

            if b_events:
                pitched["bass"] = _events_to_instrument(b_events, program=34, name="bass")
                status["bass"] = True
            else:
                status["bass"] = "no_notes"

        except Exception as e:
            status["bass"] = f"error: {e}"
    else:
        status["bass"] = "missing_stem"

    # ---------- GUITAR ----------
    g_path = stems.get("guitar")
    if g_path and os.path.exists(g_path):
        try:
            # Keep BP only mildly stricter than default so real notes don't disappear
            g_events = _bp_predict_events(
                g_path,
                manifest,
                onset_threshold=0.55,   # if still too noisy -> 0.60
                frame_threshold=0.35,   # if still too noisy -> 0.40
                min_note_len=0.03,      # DON'T raise too much for guitar
            )
            print(f"[guitar] after BP: {len(g_events)}", flush=True)

            if clean_enabled:
                # 1) Kill tiny ultra-quiet specks (guitar-safe settings)
                g_events = _drop_low_vel_short(
                    g_events,
                    vel_floor=23,       # try 15–22
                    max_dur_s=0.06,     # try 0.05–0.08
                    tag="[guitar] drop_low_vel_short",
                )
                print(f"[guitar] after drop_low_vel_short: {len(g_events)}", flush=True)

                # 2) Mild confetti filter: only if BOTH short AND quiet-ish
                g_events = _drop_short_quiet(
                    g_events,
                    vel_max=30,         # try 20–35 (higher = more aggressive)
                    max_dur_s=0.04,     # try 0.03–0.06
                    tag="[guitar] drop_short_quiet",
                )
                print(f"[guitar] after drop_short_quiet: {len(g_events)}", flush=True)

                # 3) Density cap that ONLY drops quiet notes (prevents legit notes vanishing)
                g_events = _cap_density_drop_quiet_only(
                    g_events,
                    bin_s=0.05,
                    max_notes_per_bin=10,  # try 8 if still too dense
                    never_drop_vel=40,     # try 30–50
                    tag="[guitar] cap_density_quiet_only",
                )
                print(f"[guitar] after cap_density: {len(g_events)}", flush=True)

                # 4) Merge micro-chops
                g_events = _merge_same_pitch(g_events, max_gap=0.02)
                print(f"[guitar] after merge: {len(g_events)}", flush=True)

                # 5) Demucs dropout / silence-glitch filter (last)
                g_events = _filter_stem_silence(
                    g_path,
                    g_events,
                    rms_thresh_db=-55.0,
                    min_active_ratio=0.12,
                    tag="[guitar] silence",
                )
                print(f"[guitar] after silence: {len(g_events)}", flush=True)

            if g_events:
                pitched["guitar"] = _events_to_instrument(g_events, program=28, name="guitar")
                status["guitar"] = True
            else:
                status["guitar"] = "no_notes"

        except Exception as e:
            import traceback
            traceback.print_exc()
            status["guitar"] = f"error: {e}"
    else:
        status["guitar"] = "missing_stem"

    # ---------- OTHER (synth/extra melodic) ----------
    o_path = stems.get("other")
    if o_path and os.path.exists(o_path):
        try:
            o_events = _bp_predict_events(
                o_path,
                manifest,
                onset_threshold=0.55,
                frame_threshold=0.35,
                min_note_len=0.03,
            )
            print(f"[other] after BP: {len(o_events)}", flush=True)

            if clean_enabled:
                o_events = _drop_low_vel_short(
                    o_events,
                    vel_floor=20,
                    max_dur_s=0.12,
                    tag="[other] drop_low_vel_short",
                )
                print(f"[other] after drop_low_vel_short: {len(o_events)}", flush=True)

                o_events = _drop_short_quiet(
                    o_events,
                    vel_max=35,
                    max_dur_s=0.06,
                    tag="[other] drop_short_quiet",
                )
                print(f"[other] after drop_short_quiet: {len(o_events)}", flush=True)

                o_events = _cap_density_drop_quiet_only(
                    o_events,
                    bin_s=0.05,
                    max_notes_per_bin=10,
                    never_drop_vel=39,
                    tag="[other] cap_density_quiet_only",
                )
                print(f"[other] after cap_density: {len(o_events)}", flush=True)

                o_events = _merge_same_pitch(o_events, max_gap=0.02)
                print(f"[other] after merge: {len(o_events)}", flush=True)

                # Demucs dropout / silence-glitch filter (last)
                _pre_silence = list(o_events)
                o_events = _filter_stem_silence(
                    o_path,
                    o_events,
                    rms_thresh_db=-55.0,
                    min_active_ratio=0.12,
                    tag="[other] silence",
                )
                print(f"[other] after silence: {len(o_events)}", flush=True)

                # Optional safety fallback if silence filter wipes everything
                if not o_events and _pre_silence:
                    print("[other] silence filter wiped all events -> fallback to pre-silence", flush=True)
                    o_events = _pre_silence

            if o_events:
                pitched["other"] = _events_to_instrument(o_events, program=88, name="other")
                status["other"] = True
            else:
                status["other"] = "no_notes"

        except Exception as e:
            import traceback
            traceback.print_exc()
            status["other"] = f"error: {e}"
    else:
        status["other"] = "missing_stem"

    # ---------- VOCALS ----------
    v_path = stems.get("vocals")
    if v_path and os.path.exists(v_path):
        try:
            # 1) Basic Pitch (more conservative for vocals)
            v_events = _bp_predict_events(
                v_path,
                manifest,
                onset_threshold=0.6,
                frame_threshold=0.4,
                min_note_len=0.08,
            )
            print(f"[vocals] bp events={len(v_events)} stem={v_path}", flush=True)

            if not v_events:
                status["voxlead"] = "no_notes"
                status["voxbg"] = "no_notes"
            else:
                if clean_enabled:
                    # 2) Vocal-specific cleanup (pre-split)
                    v_events = _merge_same_pitch(v_events, max_gap=0.07)
                    v_events = _squash_vibrato(v_events, semitone_tol=1, max_span=0.30)

                # 3) Split lead vs harmony (always: downstream expects voxlead/voxbg)
                lead_ev, harm_ev = _split_lead_harmony(v_events)

                # If split yields nothing useful, treat everything as lead
                if not lead_ev and v_events:
                    lead_ev = v_events
                    harm_ev = []

                print(f"[vocals] lead={len(lead_ev)} harm={len(harm_ev)} (pre-gate)", flush=True)

                if clean_enabled:
                    # Pitch range clamp
                    VOX_MIN = 40   # E2
                    VOX_MAX = 88   # C6
                    lead_ev = [(s, e, p, v) for (s, e, p, v) in lead_ev if VOX_MIN <= int(p) <= VOX_MAX]
                    harm_ev = [(s, e, p, v) for (s, e, p, v) in harm_ev if VOX_MIN <= int(p) <= VOX_MAX]

                    # 4) Ensure manifest has a key (optional; safe)
                    key_dict = manifest.get("key") or {}
                    have_key = bool(key_dict.get("detected_tonic")) or bool(key_dict.get("detected_key"))

                    if not have_key:
                        # Prefer accompaniment instruments (vocals are last, so these usually exist)
                        assigned_instruments = {
                            name: pitched[name]
                            for name in ("bass", "guitar", "other")
                            if pitched.get(name) is not None
                        }

                        # Fallback: if nothing exists yet, detect from voxlead only (better than crashing)
                        if not assigned_instruments and lead_ev:
                            assigned_instruments = {
                                "voxlead_tmp": _events_to_instrument(lead_ev, program=0, name="voxlead_tmp")
                            }

                        if assigned_instruments:
                            try:
                                detect_key_only(assigned_instruments, manifest)
                                print("[key] detected (pre-vox gate):", manifest.get("key", {}), flush=True)
                            except Exception as ke:
                                print(f"[key] detect_key_only failed: {ke} (continuing without key)", flush=True)

                    # 5) CREPE gate + OPTIONAL keysnap on lead only
                    if lead_ev and HAS_TORCHCREPE:
                        y_v, sr_v = load_audio_mono(v_path)
                        if y_v is not None and y_v.size > 0:
                            crepe_device = "cpu"
                            if torch is not None:
                                if torch.backends.mps.is_available():
                                    crepe_device = "mps"
                                elif torch.cuda.is_available():
                                    crepe_device = "cuda"

                            f0_times, f0_hz, f0_conf = _run_crepe_f0(
                                y_v,
                                sr_v,
                                step_size_ms=10.0,
                                device=crepe_device,
                                fmin=80.0,
                                fmax=1200.0,
                                model="tiny",
                                batch_size=1024,
                            )

                            tonic_pc, mode = _parse_key_mode(manifest)
                            pre_gate_lead = list(lead_ev)

                            if tonic_pc is not None:
                                lead_ev = _gate_with_crepe_then_optional_keysnap(
                                    lead_ev,
                                    f0_times, f0_hz, f0_conf,
                                    tonic_pc=int(tonic_pc),
                                    mode=mode,
                                    conf_thresh=0.30,
                                    voiced_ratio_thresh=0.15,
                                    lookahead_s=0.01,
                                    win_s=0.15,
                                    min_conf_mean=0.18,
                                    min_note_dur_s=0.06,
                                    max_semitone_diff=2.5,
                                    vel_bypass=None,
                                    tag="[voxlead gate+keysnap]",
                                    keysnap_enable=True,
                                    keysnap_max_semitones=2,
                                    keysnap_only_if_weak_vel=65,
                                    outkey_dist_apply=1,
                                    keysnap_require_f0_agreement=True,
                                    hard_snap_enable=True,
                                    hard_outkey_dist=4,
                                    hard_snap_max_semitones=6,
                                    hard_only_if_weak_vel=70,
                                    hard_only_if_short_dur_s=0.35,
                                    hard_require_f0_agreement=True,
                                )
                            else:
                                # No key known -> gate without pitch changes
                                lead_ev = _gate_with_crepe(
                                    lead_ev,
                                    f0_times, f0_hz, f0_conf,
                                    conf_thresh=0.30,
                                    voiced_ratio_thresh=0.15,
                                    lookahead_s=0.01,
                                    win_s=0.15,
                                    min_conf_mean=0.18,
                                    min_note_dur_s=0.06,
                                    max_semitone_diff=2.5,
                                    do_pitch_snap=False,
                                    allow_pitch_change=False,
                                    vel_bypass=None,
                                    tag="[voxlead gate]",
                                )

                            # Safety fallback: if gate wipes everything, keep pre-gate
                            if (not lead_ev) and pre_gate_lead:
                                print("[voxlead] gate wiped all -> fallback to pre-gate", flush=True)
                                lead_ev = pre_gate_lead

                    # 6) Harmony cleanup (no CREPE)
                    if harm_ev:
                        harm_ev = _drop_low_vel_short(
                            harm_ev, vel_floor=23, max_dur_s=0.10, tag="[voxbg] drop_low_vel_short"
                        )
                        harm_ev = _drop_short_quiet(
                            harm_ev, vel_max=28, max_dur_s=0.06, tag="[voxbg] drop_short_quiet"
                        )
                        harm_ev = _cap_density_drop_quiet_only(
                            harm_ev,
                            bin_s=0.05,
                            max_notes_per_bin=10,
                            never_drop_vel=40,
                            tag="[voxbg] cap_density_quiet_only",
                        )
                        harm_ev = _merge_same_pitch(harm_ev, max_gap=0.03)

                    # 7) Silence filter (last) + fallback
                    _pre_silence_lead = list(lead_ev) if lead_ev else []
                    _pre_silence_bg = list(harm_ev) if harm_ev else []

                    if lead_ev:
                        lead_ev = _filter_stem_silence(
                            v_path, lead_ev, rms_thresh_db=-55.0, min_active_ratio=0.12, tag="[voxlead] silence"
                        )
                        print(f"[voxlead] after silence: {len(lead_ev)}", flush=True)
                        if not lead_ev and _pre_silence_lead:
                            print("[voxlead] silence wiped all -> fallback pre-silence", flush=True)
                            lead_ev = _pre_silence_lead

                    if harm_ev:
                        harm_ev = _filter_stem_silence(
                            v_path, harm_ev, rms_thresh_db=-55.0, min_active_ratio=0.12, tag="[voxbg] silence"
                        )
                        print(f"[voxbg] after silence: {len(harm_ev)}", flush=True)
                        if not harm_ev and _pre_silence_bg:
                            print("[voxbg] silence wiped all -> fallback pre-silence", flush=True)
                            harm_ev = _pre_silence_bg

                # 8) Write instruments
                if lead_ev:
                    pitched["voxlead"] = _events_to_instrument(lead_ev, program=0, name="voxlead")
                    status["voxlead"] = True
                else:
                    status["voxlead"] = "no_notes"

                if harm_ev:
                    pitched["voxbg"] = _events_to_instrument(harm_ev, program=0, name="voxbg")
                    status["voxbg"] = True
                else:
                    status["voxbg"] = "no_notes"

        except Exception as e:
            import traceback
            traceback.print_exc()
            status["voxlead"] = f"error: {e}"
            status["voxbg"] = f"error: {e}"
    else:
        status["voxlead"] = "missing_stem"
        status["voxbg"] = "missing_stem"

    # record status
    manifest.setdefault("transcription", {})["pitched"] = status
    return pitched
