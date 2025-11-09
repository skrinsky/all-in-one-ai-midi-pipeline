import pretty_midi


def assign_seven_classes(pitched_midis, drums_inst, stems, CFG, manifest):
    """
    Map model outputs into our 7 canonical classes for writing:

        drums, voxlead, voxbg, bass, guitar, keys, other

    Inputs:
      - pitched_midis: dict[name -> pretty_midi.Instrument]
                       from transcribe_pitched_tracks
      - drums_inst:    pretty_midi.Instrument or None
                       from transcribe_drums_to_midi
      - stems, CFG, manifest: unused here except for bookkeeping

    Returns:
      dict[name -> pretty_midi.Instrument] used by assemble_and_write_midi.
    """
    assigned = {}

    # --- DRUMS ---
    if drums_inst is not None:
        drums_inst.is_drum = True
        drums_inst.name = "drums"
        assigned["drums"] = drums_inst

    # --- PITCHED CLASSES ---
    # We only add if instrument exists and has at least one note.
    for name in ["voxlead", "voxbg", "bass", "guitar", "keys", "other"]:
        inst = (pitched_midis or {}).get(name)
        if inst is None:
            continue
        notes = getattr(inst, "notes", [])
        if not notes:
            # skip truly empty tracks; if you want them visible, remove this check
            continue
        inst.name = name
        assigned[name] = inst

    # Record what we actually routed
    manifest.setdefault("assignment", {})["tracks"] = list(assigned.keys())

    return assigned
