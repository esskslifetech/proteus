"""
Optional C++ acceleration adapter (pybind11 module ``proteus_cpp``).

The extension is built out-of-tree with MSYS2 + CMake (see ``cpp/README``
and the build commands in the project README). Because a CPython extension
must match the interpreter that loads it, this adapter is strictly optional:
every helper falls back to a pure-Python implementation when the module
cannot be imported.

To point at a build, set ``PROTEUS_CPP_PATH`` to the directory containing
``proteus_cpp*.pyd`` (multiple directories may be separated by ``os.pathsep``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

_CPP_MODULE: Any | None = None
_IMPORT_ERROR: str | None = None
_IMPORT_ATTEMPTED = False


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []

    env = os.environ.get("PROTEUS_CPP_PATH", "")
    for entry in env.split(os.pathsep):
        if entry:
            dirs.append(Path(entry))

    here = Path(__file__).resolve()
    # .../proteus/python/proteus/cpp_adapter.py -> repo root is parents[2]
    repo_root = here.parents[2] if len(here.parents) > 2 else here.parent
    build_dirs = [
        repo_root / "cpp" / "build",
        repo_root / "cpp" / "build-msvc",
    ]
    for build in build_dirs:
        dirs.extend([
            build / "lib",
            build / "bin",
            # Multi-config generators (Visual Studio) append the configuration
            build / "lib" / "Release",
            build / "bin" / "Release",
            build / "Release",
        ])

    if sys.platform == "win32":
        dirs.extend([Path(r"C:\msys64\ucrt64\bin"), Path(r"C:\msys64\mingw64\bin")])
        for base in ("ucrt64", "mingw64"):
            dirs.append(Path(rf"C:\msys64\{base}\lib\python3.11\site-packages"))

    return [d for d in dirs if d.is_dir()]


def _prepare_dll_search() -> None:
    if not hasattr(os, "add_dll_directory"):
        return
    for directory in _candidate_dirs():
        try:
            os.add_dll_directory(str(directory))  # type: ignore[attr-defined]
        except OSError:
            pass
        if str(directory) not in sys.path:
            sys.path.append(str(directory))


def _load() -> Any | None:
    global _CPP_MODULE, _IMPORT_ERROR, _IMPORT_ATTEMPTED
    if _IMPORT_ATTEMPTED:
        return _CPP_MODULE
    _IMPORT_ATTEMPTED = True

    _prepare_dll_search()
    try:
        import proteus_cpp  # type: ignore

        _CPP_MODULE = proteus_cpp
    except Exception as exc:  # ImportError or Windows DLL load error
        _CPP_MODULE = None
        _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    return _CPP_MODULE


def cpp_available() -> bool:
    """True when the optional C++ extension can be imported."""
    return _load() is not None


def cpp_status() -> dict[str, Any]:
    """Diagnostic information about the C++ bridge."""
    module = _load()
    # Compute a stable hash of the module file content (or version) instead of exposing the temp path
    module_hash = ""
    module_file = getattr(module, "__file__", None)
    if module_file and Path(module_file).exists():
        import hashlib
        with open(module_file, "rb") as f:
            module_hash = "sha256:" + hashlib.sha256(f.read()).hexdigest()[:16]
    return {
        "available": module is not None,
        "module_hash": module_hash or "N/A",
        "import_error": _IMPORT_ERROR,
    }


def empirical_ddg(
    wild_type_seq: str,
    position: int,
    mutant_aa: str,
    solvent_accessibility: Sequence[float] | None = None,
    secondary_structure: Sequence[int] | None = None,
) -> float | None:
    """FoldX-style empirical ΔΔG (kcal/mol) from the C++ kernel, or None."""
    module = _load()
    if module is None:
        return None
    try:
        return float(module.energy.predict_ddg_empirical(
            wild_type_seq,
            int(position),
            mutant_aa,
            list(solvent_accessibility or []),
            list(secondary_structure or []),
        ))
    except Exception:
        return None


def mean(values: Sequence[float]) -> float | None:
    module = _load()
    if module is None:
        return None
    try:
        return float(module.kernels.mean(list(values)))
    except Exception:
        return None


def stddev(values: Sequence[float]) -> float | None:
    module = _load()
    if module is None:
        return None
    try:
        return float(module.kernels.stddev(list(values)))
    except Exception:
        return None


def solve_ode(
    rhs: Callable[[float, list[float]], list[float]],
    y0: Sequence[float],
    t0: float,
    tf: float,
    t_eval: Sequence[float] | None = None,
) -> tuple[list[float], list[list[float]]]:
    """Integrate dy/dt = rhs(t, y) with the C++ CVODE solver.

    ``rhs`` receives ``(t, y)`` and must return the derivative sequence.
    Returns ``(times, values)``. Raises ``RuntimeError`` if unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"C++ extension unavailable: {_IMPORT_ERROR}")

    result = module.solvers.solve_ode(
        rhs,
        list(y0),
        float(t0),
        float(tf),
        [float(t) for t in (t_eval or [])],
    )
    if getattr(result, "flag", 0) < 0:
        raise RuntimeError(f"CVODE failed: {getattr(result, 'message', 'unknown error')}")
    return list(result.t), [list(row) for row in result.y]


__all__ = [
    "cpp_available",
    "cpp_status",
    "empirical_ddg",
    "mean",
    "stddev",
    "solve_ode",
]
