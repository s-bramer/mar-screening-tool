from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent.parent


def load(path: str | Path = None) -> dict:
    if path is None:
        path = ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(cfg: dict, key_path: str) -> Path:
    """Resolve a nested config path string (dot-separated) to an absolute Path."""
    keys = key_path.split(".")
    val = cfg
    for k in keys:
        val = val[k]
    p = Path(val)
    if not p.is_absolute():
        p = ROOT / p
    return p
