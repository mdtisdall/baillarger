"""numba.njit with a typed signature.

A copy of baillarger._jit, so that baillarger.fronts imports nothing from the
rest of baillarger. numba has no type stubs, so mypy sees numba.njit as
untyped; this wrapper tells mypy that a jitted function keeps the signature
of the Python one.
"""

from collections.abc import Callable
from typing import Any, overload

import numba


@overload
def njit[F: Callable[..., Any]](f: F, /) -> F: ...


@overload
def njit[F: Callable[..., Any]](*, cache: bool = ...) -> Callable[[F], F]: ...


def njit(f: Any = None, /, **options: Any) -> Any:
    """Compile f with numba.njit (nopython mode, compiled on the first call).

    With keyword options only, return a decorator.
    """
    if f is None:
        return numba.njit(**options)
    return numba.njit(f)
