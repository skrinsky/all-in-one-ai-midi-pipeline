def insert_time_signatures(assigned_map, meter_info, CFG, manifest):
    # pretty_midi lacks rich TS writing; we'll carry meter in manifest for now.
    manifest.setdefault('meter_key', {})['time_signature_written'] = False
    return assigned_map
