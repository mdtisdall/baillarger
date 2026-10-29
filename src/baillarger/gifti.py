"""GIfTI files with their data outside the XML (Encoding="ExternalFileBinary").

The XML is small; each data array is raw binary (little-endian, row-major)
in one companion .dat file, at an offset that the XML records, so a reader
can memory-map the arrays and read any range of rows without decoding the
rest. nibabel reads these files (memory-mapped) but cannot write them, so
write_external writes the XML itself, following the GIfTI 1.0 format.
ExternalFileName is the .dat's bare name, which nibabel and the MATLAB
gifti toolbox resolve next to the .gii; gifticlib (FreeSurfer, AFNI) opens
it from the working directory, so run those tools in the files' directory.
"""

from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np
import numpy.typing as npt

#: numpy dtype -> GIfTI DataType
_DTYPES = {
    np.dtype("<f4"): "NIFTI_TYPE_FLOAT32",
    np.dtype("<i4"): "NIFTI_TYPE_INT32",
    np.dtype("u1"): "NIFTI_TYPE_UINT8",
}


@dataclass(frozen=True)
class DataArray:
    """One GIfTI data array: 2-D data (float32, int32 or uint8), its intent
    (NIFTI_INTENT_*), metadata, and for a point set the 4 x 4 transform to
    scanner coordinates (identity when the points are already there)."""

    data: npt.NDArray[np.generic]
    intent: str
    meta: dict[str, str] = field(default_factory=dict)
    xform: npt.NDArray[np.float64] | None = None


def dat_path(gii: Path) -> Path:
    """The .dat next to a .gii: <name without .surf.gii/.func.gii>.dat."""
    return gii.with_name(gii.name.split(".", 1)[0] + ".dat")


def _meta(meta: dict[str, str]) -> str:
    if not meta:
        return "<MetaData/>"
    md = "".join(
        f"<MD><Name><![CDATA[{k}]]></Name><Value><![CDATA[{v}]]></Value></MD>"
        for k, v in meta.items()
    )
    return f"<MetaData>{md}</MetaData>"


def write_external(
    gii: Path, arrays: list[DataArray], meta: dict[str, str] | None = None
) -> Path:
    """Write gii (the XML) and its .dat (the arrays, one after another).
    Returns the .dat's path."""
    dat = dat_path(gii)
    gii.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    offset = 0
    with dat.open("wb") as fh:
        for a in arrays:
            d = np.asarray(a.data)
            if d.ndim != 2:
                raise ValueError(f"{a.intent}: data must be 2-D, not {d.ndim}-D")
            d = np.ascontiguousarray(d, d.dtype.newbyteorder("<"))
            if d.dtype not in _DTYPES:
                raise ValueError(f"{a.intent}: dtype {d.dtype} is not supported")
            d.tofile(fh)
            attrs = dict(
                Intent=a.intent,
                DataType=_DTYPES[d.dtype],
                ArrayIndexingOrder="RowMajorOrder",
                Dimensionality="2",
                Dim0=str(d.shape[0]),
                Dim1=str(d.shape[1]),
                Encoding="ExternalFileBinary",
                Endian="LittleEndian",
                ExternalFileName=dat.name,
                ExternalFileOffset=str(offset),
            )
            offset += d.nbytes
            xf = ""
            if a.xform is not None:
                # one row per line: nibabel reads MatrixData with loadtxt
                m = "\n".join(
                    " ".join(f"{x:.10g}" for x in row) for row in np.asarray(a.xform)
                )
                xf = (
                    "<CoordinateSystemTransformMatrix>"
                    "<DataSpace><![CDATA[NIFTI_XFORM_SCANNER_ANAT]]></DataSpace>"
                    "<TransformedSpace><![CDATA[NIFTI_XFORM_SCANNER_ANAT]]>"
                    "</TransformedSpace>"
                    f"<MatrixData>{m}</MatrixData>"
                    "</CoordinateSystemTransformMatrix>"
                )
            at = " ".join(f"{k}={quoteattr(v)}" for k, v in attrs.items())
            parts.append(
                f"<DataArray {at}>\n{_meta(a.meta)}\n{xf}<Data></Data>\n</DataArray>"
            )
    body = "\n".join(parts)
    gii.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE GIFTI SYSTEM "http://www.nitrc.org/frs/download.php/115/gifti.dtd">\n'
        f'<GIFTI Version="1.0" NumberOfDataArrays="{len(arrays)}">\n'
        f"{_meta(meta or {})}\n<LabelTable/>\n{body}\n</GIFTI>\n"
    )
    return dat
