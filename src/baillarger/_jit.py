"""numba.njit with a typed signature.

numba has no type stubs, so mypy sees numba.njit as untyped. This wrapper
tells mypy that the jitted function keeps the signature of the Python one.
"""

from collections.abc import Callable
from typing import Any, cast

import numba


def njit[F: Callable[..., Any]](f: F) -> F:
    """Compile f with numba.njit (nopython mode, compiled on the first call)."""
    return cast(F, numba.njit(f))
