from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from opdxread import opdxtype


class OPDxFile(object):
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.filesize = self.path.stat().st_size

        self.data: Dict[str, Any] = {}

        self.read()

    @property
    def extent(self) -> float:
        return self.data["1D_Data"]["Raw"]["Extent"].value

    @property
    def scale(self) -> float:
        return self.data["1D_Data"]["Raw"]["DataScale"].value

    @property
    def x(self) -> np.ndarray:
        return self.data["1D_Data"]["Raw"]["PositionFunction"].data

    @property
    def y(self) -> np.ndarray:
        return self.data["1D_Data"]["Raw"]["Array"].array * self.scale

    def read(self):
        with self.path.open("rb") as fp:
            assert fp.read(12) == b"VCA DATA\x01\x00\x00\x55"

            while fp.tell() < self.filesize:
                item = opdxtype.NamedValue(fp)
                if item.value is not None:
                    self.data[item.name] = item.value

    def get_1d_polynomial_fit(self, xf: np.ndarray, deg: int = 1) -> np.ndarray:
        """Return a scaled polynomial fit of data.

        Fitting of a `deg` degree polynomial is performed at all `xf` x positions.
        """
        idx = np.searchsorted(self.x, xf)
        coefs = np.polynomial.polynomial.polyfit(self.x[idx], self.y[idx], deg)
        return np.polynomial.polynomial.polyval(self.x, coefs)

    def get_1d_data(self, r: float | None = None, m: float | None = None) -> np.ndarray:
        """Return the scaled profilometric data.

        Args:
            r: start x pos of linear fit, defaults to 0
            m: end x pos of linear fit, defaults to end

        Returns:
            array (:, [x, y])
        """
        if r is None:
            r = 0.0
        if m is None:
            m = self.extent
        elif m < 0.0:
            m = self.extent - m
        assert m > r

        return np.stack(
            (self.x, self.y - self.get_1d_polynomial_fit(np.array([r, m]), 1)), axis=1
        )
