"""numba.njit and numba.prange with typed signatures.

numba has no type stubs, so mypy sees numba.njit as untyped. These wrappers
tell mypy that a jitted function keeps the signature of the Python one.
"""

from collections.abc import Callable
from typing import Any, overload

import numba


@overload
def njit[F: Callable[..., Any]](f: F, /) -> F: ...


@overload
def njit[F: Callable[..., Any]](
    *, parallel: bool = ..., inline: str = ..., cache: bool = ...
) -> Callable[[F], F]: ...


def njit(f: Any = None, /, **options: Any) -> Any:
    """Compile f with numba.njit (nopython mode, compiled on the first call).

    With keyword options only (parallel, inline, cache), return a decorator.
    """
    if f is None:
        return numba.njit(**options)
    return numba.njit(f)


# numba.prange, typed as range: numba recognises the object itself
prange: type[range] = numba.prange
