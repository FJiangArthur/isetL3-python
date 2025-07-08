from pathlib import Path


def l3_root_path() -> Path:
    """Return the path to the repository root directory."""
    return Path(__file__).resolve().parent.parent
