import json, os, pathlib, yaml

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def song_id_from_path(audio_path: str):
    base = os.path.basename(audio_path)
    return os.path.splitext(base)[0]

def read_manifest(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def write_manifest(path: str, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
