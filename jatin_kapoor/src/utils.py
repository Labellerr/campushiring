import os, json, yaml
from pathlib import Path

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def resolve_dataset_yaml(yaml_path: str, out_path: str) -> str:
    repo_root = Path(__file__).resolve().parent.parent
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    for key in ["train", "val", "test"]:
        if key in data and isinstance(data[key], str):
            p = Path(data[key])
            if not p.is_absolute():
                data[key] = str((repo_root / p).resolve())
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f)
    return str(Path(out_path).resolve())
