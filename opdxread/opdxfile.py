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

    def get_1d_linear_fit(self, r: float = None, m: float = None) -> np.ndarray:
        extent = self.data["1D_Data"]["Raw"]["Extent"].value
        x = self.data["1D_Data"]["Raw"]["PositionFunction"].data
        y = self.data["1D_Data"]["Raw"]["Array"].array
        ydiv = extent / y.size

        r = 0.0 if r is None else (r if r >= 0.0 else extent + r)
        m = extent if m is None else (m if m >= 0.0 else extent + m)
        assert m > r

        ir = np.clip(int(r / ydiv), 0, y.size - 1)
        im = np.clip(int(m / ydiv), 0, y.size - 1)

        coefs = np.polynomial.polynomial.polyfit([r, m], [y[ir], y[im]], 1)
        return np.polynomial.polynomial.polyval(x, coefs)

    def get_1d_data(self, r: float = None, m: float = None) -> np.ndarray:
        scale = self.data["1D_Data"]["Raw"]["DataScale"].value
        x = self.data["1D_Data"]["Raw"]["PositionFunction"].data
        y = self.data["1D_Data"]["Raw"]["Array"].array.copy()
        if r is not None or m is not None:
            y -= self.get_1d_linear_fit(r, m)
        return np.stack((x, y * scale), axis=1)
