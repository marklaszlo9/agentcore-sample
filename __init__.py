"""Package initialization for strands-gcuk.

This project does not currently expose any public objects at the package level,
but pytest attempts to import the package when discovering tests. The original
file tried to import :mod:`agent` which doesn't exist in this repository and
caused import errors during test collection. The import is therefore wrapped in
a try/except block so the package can be imported safely in test environments.
"""

try:  # pragma: no cover - optional dependency
    from .agent import StrandsAgent  # type: ignore
except Exception:  # pragma: no cover - allow missing module
    pass
