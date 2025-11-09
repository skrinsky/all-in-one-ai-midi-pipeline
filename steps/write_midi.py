import os
import pretty_midi


def assemble_and_write_midi(assigned_map, meter_info, out_mid, CFG, manifest):
    """
    Tempo priority:
      1. manifest["meter_key"]["tempo"]
      2. meter_info["tempo"]
      3. 120.0
    """
    tempo = None

    mk = manifest.get("meter_key") or {}
    if isinstance(mk, dict):
        t = mk.get("tempo")
        if t:
            tempo = float(t)

    if tempo is None and isinstance(meter_info, dict):
        t = meter_info.get("tempo")
        if t:
            tempo = float(t)

    if tempo is None or tempo <= 0:
        tempo = 120.0

    print(f"[assemble_and_write_midi] Using tempo: {tempo}")

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))

    for name, inst in (assigned_map or {}).items():
        if not inst or not getattr(inst, "notes", None):
            continue
        inst.name = name
        pm.instruments.append(inst)

    os.makedirs(os.path.dirname(out_mid), exist_ok=True)
    pm.write(out_mid)

    manifest.setdefault("meter_key", {})["tempo"] = float(tempo)
    manifest.setdefault("output", {})["midi"] = out_mid
