"""Config loader. Reads configs/default.yaml plus an optional override file."""
from __future__ import annotations
import os
import yaml
from pathlib import Path
from types import SimpleNamespace


_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_ns(x) for x in d]
    return d


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike | None = None) -> SimpleNamespace:
    with open(_DEFAULT_PATH) as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path).exists():
        with open(path) as f:
            cfg = _merge(cfg, yaml.safe_load(f))
    cfg["_project_root"] = str(Path(__file__).resolve().parents[1])
    return _to_ns(cfg)


def resolve_path(cfg: SimpleNamespace, rel: str) -> str:
    p = Path(rel)
    if p.is_absolute():
        return str(p)
    return str(Path(cfg._project_root) / p)
