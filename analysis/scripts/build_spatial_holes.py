"""Build the spatial-holes paper: copy figures from outputs/holes/figures and render the PDF."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
FIG_SRC = ROOT / "paper" / "figures"
MARKDOWN_FILE = PAPER_DIR / "spatial_holes.md"
PDF_FILE = PAPER_DIR / "spatial_holes.pdf"


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    (PAPER_DIR / "figures").mkdir(parents=True, exist_ok=True)
    for f in FIG_SRC.glob("*.png"):
        shutil.copy2(f, PAPER_DIR / "figures" / f.name)
    run(["pandoc", MARKDOWN_FILE.name, "--resource-path=.", "-V", "papersize:letter", "-V", "geometry:margin=1.2in",
         "-o", PDF_FILE.name], cwd=PAPER_DIR)
    print(f"Paper rebuilt at {PDF_FILE}")


if __name__ == "__main__":
    main()
