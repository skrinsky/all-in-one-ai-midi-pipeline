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

    # (batch, time)
    audio = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)

    # Make sure the tensor is on the same device TorchCREPE will use
    if device != "cpu":
        audio = audio.to(device)

    hop_length = int(round((step_size_ms / 1000.0) * sr))
    hop_length = max(1, hop_length)

    with torch.inference_mode():
        f0_hz, periodicity = torchcrepe.predict(
            audio, sr,
            hop_length=hop_length,
            fmin=30.0, fmax=600.0,      # bass range
            model="tiny",               # faster than "full"
            batch_size=1024,
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

def _bass_gate_with_crepe(
    events,
    f0_times, f0_hz, f0_conf,
    conf_thresh=0.25,           # threshold for a frame to count as "voiced"
    voiced_ratio_thresh=0.10,   # (alias) require at least this fraction voiced frames
    lookahead_s=0.03,           # (alias) skip onset transient before analyzing
    win_s=0.20,                 # analysis window length (seconds)
    min_conf_mean=0.12,         # if mean confidence in window is below this, treat CREPE as uninformative
    min_note_dur_s=0.10,        # when CREPE is uninformative, drop notes shorter than this (spray killer)
    max_semitone_diff=4.0,      # if CREPE is confident, require BP pitch to agree (after octave snap)
    do_pitch_snap=True,
    debug_n=20,
):
    """
    Gate BasicPitch bass events using TorchCREPE.

    Rules:
    - If CREPE is confident (enough mean conf + enough voiced frames):
        keep only if BP pitch (optionally octave-snapped) matches CREPE median within tolerance.
    - If CREPE is NOT confident:
        do NOT trust it for pitch gating; instead, drop very short notes (spray) and keep longer notes.
    - Safety: if we somehow keep nothing, return original events.
    """
    if (not events) or (f0_times is None) or (f0_hz is None) or (f0_conf is None):
        return events

    events = sorted(events, key=lambda x: x[0])

    def hz_to_midi(f_hz: float) -> float:
        return 69.0 + 12.0 * np.log2(f_hz / 440.0)

    def snap_to_nearest_octave(p_bp: int, f0_midi: float) -> int:
        f0_round = int(round(f0_midi))
        candidates = [f0_round - 24, f0_round - 12, f0_round, f0_round + 12, f0_round + 24]
        return min(candidates, key=lambda c: abs(c - p_bp))

    kept = []
    dropped = 0
    changed = 0

    for i, (s, e, p_bp, v) in enumerate(events):
        s = float(s); e = float(e)
        p_bp = int(p_bp); v = int(v)
        dur = e - s
        if dur <= 0:
            continue

        # Analyze after onset transient
        t0 = s + float(lookahead_s)
        t1 = min(e, t0 + float(win_s))
        if t1 <= t0:
            # too short to analyze; treat as "uninformative"
            if dur < min_note_dur_s:
                dropped += 1
                if i < debug_n:
                    print(f"[gate dbg] i={i} dur={dur:.3f} -> DROP (too short, no window)", flush=True)
                continue
            kept.append((s, e, p_bp, v))
            if i < debug_n:
                print(f"[gate dbg] i={i} dur={dur:.3f} -> KEEP (too short, no window)", flush=True)
            continue

        i0 = int(np.searchsorted(f0_times, t0, side="left"))
        i1 = int(np.searchsorted(f0_times, t1, side="right"))
        i0 = max(0, min(i0, len(f0_times)))
        i1 = max(i0, min(i1, len(f0_times)))

        if i1 - i0 < 2:
            # CREPE uninformative
            if dur < min_note_dur_s:
                dropped += 1
                decision = "DROP (crepe sparse + short)"
            else:
                kept.append((s, e, p_bp, v))
                decision = "KEEP (crepe sparse)"
            if i < debug_n:
                print(f"[gate dbg] i={i} dur={dur:.3f} {decision}", flush=True)
            continue

        seg_hz = f0_hz[i0:i1]
        seg_c  = f0_conf[i0:i1]

        conf_mean = float(np.nanmean(seg_c)) if seg_c.size else 0.0
        good = (seg_c >= conf_thresh) & np.isfinite(seg_hz) & (seg_hz > 0)
        voiced_ratio = float(np.mean(good)) if seg_c.size else 0.0

        f0_med = None
        if np.sum(good) >= 2:
            f0_med = float(np.nanmedian(hz_to_midi(seg_hz[good])))

        # Debug print
        if i < debug_n:
            print(
                f"[gate dbg] i={i} s={s:.3f} e={e:.3f} dur={dur:.3f} "
                f"t0={t0:.3f} t1={t1:.3f} f0_med={None if f0_med is None else round(f0_med,2)} "
                f"vr={voiced_ratio:.2f} conf_mean={conf_mean:.3f} p_bp={p_bp}",
                flush=True,
            )

        # Decide whether CREPE is trustworthy in this window
        crepe_confident = (f0_med is not None) and (conf_mean >= min_conf_mean) and (voiced_ratio >= voiced_ratio_thresh)

        if not crepe_confident:
            # Don't do pitch gating. Only kill obvious spray (very short notes).
            if dur < min_note_dur_s:
                dropped += 1
                continue
            kept.append((s, e, p_bp, v))
            continue

        # CREPE confident: enforce pitch agreement
        p_out = p_bp
        if do_pitch_snap:
            p_out = snap_to_nearest_octave(p_bp, f0_med)

        if abs(p_out - f0_med) <= max_semitone_diff:
            if p_out != p_bp:
                changed += 1
            kept.append((s, e, p_out, v))
        else:
            dropped += 1

    kept.sort(key=lambda x: x[0])
    print(f"[bass gate] in={len(events)} kept={len(kept)} dropped={dropped} changed_pitch={changed}", flush=True)

    # Safety: never return empty unless input was empty
    if len(kept) == 0:
        print("[bass gate] kept=0 -> skipping gate (returning original BP events)", flush=True)
        return events

    return kept


def transcribe_pitched_tracks(stems: dict, CFG: dict, manifest: dict):
    """
    Use Basic Pitch (+ midi_tempo) on:
      - vocals -> voxlead, voxbg (with vocal-specific cleanup)
      - bass   -> bass
      - guitar -> guitar
      - other  -> other (as pad/synth-ish via program)
    Returns:
      dict[name -> pretty_midi.Instrument]
    """
    pitched = {}
    status = {}

    # ---------- VOCALS ----------
    v_path = stems.get("vocals")
    if v_path and os.path.exists(v_path):
        try:
            # More conservative for vocals
            v_events = _bp_predict_events(
                v_path,
                manifest,
                onset_threshold=0.6,
                frame_threshold=0.4,
                min_note_len=0.08,
            )
            # Vocal-specific cleanup
            v_events = _merge_same_pitch(v_events, max_gap=0.07)
            v_events = _squash_vibrato(v_events, semitone_tol=1, max_span=0.30)

            if v_events:
                lead_ev, harm_ev = _split_lead_harmony(v_events)

                if lead_ev:
                    pitched["voxlead"] = _events_to_instrument(
                        lead_ev, program=0, name="voxlead"
                    )
                    status["voxlead"] = True
                else:
                    status["voxlead"] = "no_notes"

                if harm_ev:
                    pitched["voxbg"] = _events_to_instrument(
                        harm_ev, program=0, name="voxbg"
                    )
                    status["voxbg"] = True
                else:
                    status["voxbg"] = "no_notes"
            else:
                status["voxlead"] = "no_notes"
                status["voxbg"] = "no_notes"
        except Exception as e:
            status["voxlead"] = f"error: {e}"
            status["voxbg"] = f"error: {e}"
    else:
        status["voxlead"] = "missing_stem"
        status["voxbg"] = "missing_stem"

    # ---------- BASS ----------
    b_path = stems.get("bass")
    if b_path and os.path.exists(b_path):
        try:
            # 1) Basic Pitch bass events
            b_events = _bp_predict_events(b_path, manifest)

            # Clamp to a sane bass range (adjust as you like)
            BASS_MIN = 28   # ~E1
            BASS_MAX = 72   # ~C5
            b_events = [(s, e, p, v) for (s, e, p, v) in b_events if BASS_MIN <= int(p) <= BASS_MAX]

            raw_b_events = list(b_events)  # fallback

            print(f"[bass] HAS_TORCHCREPE={HAS_TORCHCREPE}", flush=True)
            print(f"[bass] bp events={len(raw_b_events)} stem={b_path}", flush=True)

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
                    b_events = _bass_gate_with_crepe(
                        b_events,
                        f0_times, f0_hz, f0_conf,
                        conf_thresh=0.25,
                        voiced_ratio_thresh=0.10,
                        lookahead_s=0.03,
                        win_s=0.20,
                        max_semitone_diff=4.0,
                        do_pitch_snap=True,
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
                status["bass"] = "no_notes_after_filter"

        except Exception as e:
            status["bass"] = f"error: {e}"
    else:
        status["bass"] = "missing_stem"


    # ---------- GUITAR ----------
    g_path = stems.get("guitar")
    if g_path and os.path.exists(g_path):
        try:
            g_events = _bp_predict_events(g_path, manifest)
            if g_events:
                pitched["guitar"] = _events_to_instrument(
                    g_events, program=28, name="guitar"
                )
                status["guitar"] = True
            else:
                status["guitar"] = "no_notes"
        except Exception as e:
            status["guitar"] = f"error: {e}"
    else:
        status["guitar"] = "missing_stem"

    # ---------- OTHER (synth/extra melodic) ----------
    o_path = stems.get("other")
    if o_path and os.path.exists(o_path):
        try:
            o_events = _bp_predict_events(o_path, manifest)
            if o_events:
                # Use a pad-like GM program so it imports as a pad
                pitched["other"] = _events_to_instrument(
                    o_events, program=88, name="other"
                )
                status["other"] = True
            else:
                status["other"] = "no_notes"
        except Exception as e:
            status["other"] = f"error: {e}"
    else:
        status["other"] = "missing_stem"

    # ---------- record status ----------
    manifest.setdefault("transcription", {})["pitched"] = status

    return pitched
