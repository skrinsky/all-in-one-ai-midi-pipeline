import os
import pretty_midi
import numpy as np
from utils.audio_utils import load_audio_mono


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
            b_events = _bp_predict_events(b_path, manifest)

            # 1) basic harmonic/junk filter (optional, keep if it helped at all)
            # from the earlier helper; if you didn't keep it, you can skip this line.
            # b_events = _filter_bass_events(b_events,
            #                                min_velocity=30,
            #                                min_duration=0.04,
            #                                max_pitch=60)

            # 2) NEW: drop notes where the bass stem is effectively silent
            b_events = _filter_bass_silence(
                b_path,
                b_events,
                rms_thresh_db=-45.0,  # raise toward -40 if it's still too generous
                min_active_ratio=0.2, # require at least 20% of frames to be above threshold
            )

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
