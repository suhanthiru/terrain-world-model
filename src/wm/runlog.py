"""Letting a long-running script own its own log file.

Launching a detached process on Windows and handing it a redirected stdout gives the
child a file handle that belongs to whatever launched it. When that launcher exits --
a shell that has served its purpose, say -- the handle goes invalid, and the child dies
on its next print with ``OSError: [WinError 87] The parameter is incorrect``. That killed
both a dataset generation run and a training sweep here, hours in, for no reason other
than having printed a progress line at the wrong moment.

Opening the file inside the process removes the dependency entirely: the handle belongs
to the process doing the writing, and nothing outside can invalidate it.
"""

from __future__ import annotations

import sys
from pathlib import Path


def redirect_output(path: str | Path | None) -> None:
    """Send stdout and stderr to ``path``, line buffered, opened by this process."""
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = open(target, "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = handle
    sys.stderr = handle
