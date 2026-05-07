"""Utility functions for preparing the environment for PyTorch imports, especially on Windows."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def prepare_torch_import() -> None:
    """Prepares the environment for importing torch on Windows to avoid DLL loading issues."""
    if os.name != "nt":
        return

    torch_lib_dir = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib_dir.exists():
        os.environ["PATH"] = str(torch_lib_dir) + os.pathsep + os.environ.get("PATH", "")
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            add_dll_directory(str(torch_lib_dir))

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Workaround for a Windows runtime issue observed in this environment:
    # importing a small scikit-learn module first stabilizes DLL loading for torch.
    import sklearn.metrics  # noqa: F401
