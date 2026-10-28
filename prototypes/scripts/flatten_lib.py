"""Flattening in the column volume: moved to the package (baillarger.flatten); this
name stays for the prototype scripts that import it."""

from baillarger.flatten import *  # noqa: F403
from baillarger.flatten import (  # noqa: F401
    dp_lags,
    dynamic_shifts,
    fit_warp,
    interp_last,
    masked_smooth,
    pair_weights,
    resample,
    residuals,
    shifts,
    tau_of,
)
