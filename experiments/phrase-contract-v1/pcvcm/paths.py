"""Runtime path containment."""
from pathlib import Path


def below(root: Path, relative: str) -> Path:
    candidate = (Path(root) / relative).resolve()
    if not candidate.is_relative_to(Path(root).resolve()):
        raise ValueError('Manifest path escapes its declared data root')
    return candidate


