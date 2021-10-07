import numpy as np
from pathlib import Path

from opdxread import opdxtype

from typing import Any, Dict, Union


class OPDxFile(object):
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.filesize = self.path.stat().st_size

        self.data: Dict[str, Any] = {}

        self.read()

    def read(self):
        with self.path.open("rb") as fp:
            assert fp.read(12) == b"VCA DATA\x01\x00\x00\x55"

            while fp.tell() < self.filesize:
                item = opdxtype.NamedValue(fp)
                if item.value is not None:
                    self.data[item.name] = item.value

    def get_1d_polynomial_fit(self, xf: np.ndarray, deg: int = 1) -> np.ndarray:
        extent = self.data["1D_Data"]["Raw"]["Extent"].value
        scale = self.data["1D_Data"]["Raw"]["DataScale"].value
        x = self.data["1D_Data"]["Raw"]["PositionFunction"].data
        y = self.data["1D_Data"]["Raw"]["Array"].array
        xdiv = extent / x.size

        ixf = np.clip(xf / xdiv, 0, x.size - 1).astype(int)

        coefs = np.polynomial.polynomial.polyfit(x[ixf], y[ixf], deg)
        return np.polynomial.polynomial.polyval(x, coefs) * scale

    def get_1d_data(self, r: float = None, m: float = None) -> np.ndarray:
        """Return the profilometric data as [:, (x, y)] array.

        If `r` or `m` are passed a linear fit at these x positions is perfromed and subtracted from the data.
        """
        extent = self.data["1D_Data"]["Raw"]["Extent"].value
        scale = self.data["1D_Data"]["Raw"]["DataScale"].value
        x = self.data["1D_Data"]["Raw"]["PositionFunction"].data
        y = self.data["1D_Data"]["Raw"]["Array"].array * scale
        if r is not None or m is not None:
            r = 0.0 if r is None else (r if r >= 0.0 else extent + r)
            m = extent if m is None else (m if m >= 0.0 else extent + m)
            assert m > r
            y -= self.get_1d_polynomial_fit(np.array([r, m]), 1)
        return np.stack((x, y), axis=1)
