import numpy as np

def gentle_cleanup(assigned_map, CFG, manifest):
    # Placeholder: real cleanup would quantize onsets to a grid with max window and strip micro-notes
    # Here we just pass-through and record the policy in the manifest.
    manifest.setdefault('cleanup', {})['applied'] = False
    return assigned_map
