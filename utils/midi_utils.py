import pretty_midi, numpy as np

def new_midi(tempo=120.0):
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    return pm

def add_tempo_changes(pm: pretty_midi.PrettyMIDI, times, bpms):
    pm._PrettyMIDI__tick_scales = None  # force recompute
    pm.adjust_times([0.0], [0.0])       # no-op to ensure internal init
    pm.remove_tempi(0, 1e9)
    pm._PrettyMIDI__tick_scales = None
    # pretty_midi supports a single global tempo or localized changes via `estimate_tempo` behavior.
    # For simplicity we set the first tempo. For full maps, consider writing meta events via mido.
    if len(bpms) > 0:
        pm._PrettyMIDI__tempi = np.array([bpms[0]])
        pm._PrettyMIDI__tempo_changes = np.array([times[0]])

def add_time_signature_meta(mf, numerator=4, denominator=4):
    # pretty_midi has limited TS support; for exact meta events, assemble with mido at write time.
    pass
