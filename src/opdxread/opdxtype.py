import numpy as np
import struct

from typing import BinaryIO, Type


class OPDxValue(object):
    code = 0xFF

    @staticmethod
    def read_name(fp: BinaryIO) -> str:
        size = struct.unpack("I", fp.read(4))[0]
        return fp.read(size).decode()

    @staticmethod
    def read_size(fp: BinaryIO) -> int:
        size = struct.unpack("B", fp.read(1))[0]
        if size == 1:
            return struct.unpack("B", fp.read(size))[0]
        elif size == 2:
            return struct.unpack("H", fp.read(size))[0]
        elif size == 4:
            return struct.unpack("I", fp.read(size))[0]
        else:
            raise ValueError("Bad size")

    @staticmethod
    def read_simple(fp: BinaryIO, dtype: str, size: int):
        return struct.unpack(dtype, fp.read(size))[0]

    @staticmethod
    def read(fp: BinaryIO):
        raise NotImplementedError


class Matrix(OPDxValue):
    code = 0x00


class Boolean(OPDxValue):
    code = 0x01

    @staticmethod
    def read(fp: BinaryIO):
        return OPDxValue.read_simple(fp, "?", 1)


class Int32(OPDxValue):
    code = 0x06

    @staticmethod
    def read(fp: BinaryIO):
        return OPDxValue.read_simple(fp, "i", 4)


class Uint32(OPDxValue):
    code = 0x07

    @staticmethod
    def read(fp: BinaryIO):
        return OPDxValue.read_simple(fp, "I", 4)


class Int64(OPDxValue):
    code = 0x0A

    @staticmethod
    def read(fp: BinaryIO):
        return OPDxValue.read_simple(fp, "l", 8)


class Uint64(OPDxValue):
    code = 0x0B

    @staticmethod
    def read(fp: BinaryIO):
        return OPDxValue.read_simple(fp, "L", 8)


class Float(OPDxValue):
    code = 0x0C

    @staticmethod
    def read(fp: BinaryIO):
        return OPDxValue.read_simple(fp, "f", 4)


class Double(OPDxValue):
    code = 0x0D

    @staticmethod
    def read(fp: BinaryIO):
        return OPDxValue.read_simple(fp, "d", 8)


class DType(OPDxValue):
    code = 0x0E
    type_size_dict = {1: "B", 2: "H", 4: "I"}

    def __init__(self, name: str, typeid: int):
        self.name = name
        self.typeid = typeid

    @staticmethod
    def read(fp: BinaryIO):
        name = OPDxValue.read_name(fp)
        size = OPDxValue.read_size(fp)
        return DType(name, struct.unpack(DType.type_size_dict[size], fp.read(size))[0])

    def __repr__(self):
        return f"DType[{self.name}={self.typeid}]"


class String(OPDxValue):
    code = 0x12

    @staticmethod
    def read(fp: BinaryIO):
        size = OPDxValue.read_size(fp)
        return fp.read(size).decode()


class Quanity(OPDxValue):
    code = 0x13

    def __init__(self, value, unit, symbol):
        self.value = value
        self.unit = unit
        self.symbol = symbol

    @staticmethod
    def read(fp: BinaryIO):
        size = OPDxValue.read_size(fp)
        pos = fp.tell()
        value = struct.unpack("d", fp.read(8))[0]
        unit = OPDxValue.read_name(fp)
        symbol = OPDxValue.read_name(fp)
        fp.seek(pos + size)

        return Quanity(value, unit, symbol)

    def __repr__(self):
        return f"Quantity[{self.value} {self.unit} ({self.symbol})]"


class TimeStamp(OPDxValue):
    code = 0x15

    @staticmethod
    def read(fp: BinaryIO):
        return struct.unpack("BBBBBBBBB", fp.read(9))


class Unit(OPDxValue):
    code = 0x18

    def __init__(self, value, unit, symbol):
        self.value = value
        self.unit = unit
        self.symbol = symbol

    @staticmethod
    def read(fp: BinaryIO):
        size = OPDxValue.read_size(fp)
        pos = fp.tell()
        unit = OPDxValue.read_name(fp)
        symbol = OPDxValue.read_name(fp)
        value = struct.unpack("d", fp.read(8))[0]
        fp.seek(pos + size)

        return Unit(value, unit, symbol)

    def __repr__(self):
        return f"Unit[{self.value} {self.unit} ({self.symbol})]"


class Array(OPDxValue):
    code = 0x40

    def __init__(self, name, array):
        self.name = name
        self.array = array

    @staticmethod
    def read(fp: BinaryIO):
        name = OPDxValue.read_name(fp)
        size = OPDxValue.read_size(fp)
        data = np.frombuffer(fp.read(size), offset=5, dtype=np.float64)
        return Array(name, data)

    def __repr__(self):
        return f"Array[{self.name}, size={self.array.size}]"


class StringList(OPDxValue):
    code = 0x42

    def __init__(self, name, strings):
        self.name = name
        self.strings = strings

    @staticmethod
    def read(fp: BinaryIO):
        name = OPDxValue.read_name(fp)
        size = OPDxValue.read_size(fp)
        pos = fp.tell()

        strings = []
        while fp.tell() < pos + size:
            strings.append(OPDxValue.read_name(fp))
        return StringList(name, strings)

    def __repr__(self):
        return f"StringList[{self.name}={self.strings}]"


class AnonMatrix(OPDxValue):
    code = 0x45


class RawData(OPDxValue):
    code = 0x46

    def __init__(self, data):
        self.data = data

    @staticmethod
    def read(fp: BinaryIO):
        size = OPDxValue.read_size(fp)
        pos = fp.tell()

        data = {}
        while fp.tell() < pos + size:
            item = NamedValue(fp)
            if item.value is not None:
                data[item.name] = item.value
        return RawData(data)

    def __repr__(self):
        return f"RawData[{self.data}]"

    def __getitem__(self, key: str):
        return self.data[key]

    def __setitem__(self, key: str, value):
        self.data[key] = value


class RawData2D(OPDxValue):
    code = 0x47


class PosData(OPDxValue):
    code = 0x7C

    def __init__(self, name, unit, data):
        self.name = name
        self.unit = unit
        self.data = data

    @staticmethod
    def read(fp: BinaryIO):
        name = OPDxValue.read_name(fp)
        _ = OPDxValue.read_size(fp)
        unit = OPDxValue.read_name(fp)
        symbol = OPDxValue.read_name(fp)
        divisor = OPDxValue.read_simple(fp, "d", 8)

        fp.read(12)

        length = OPDxValue.read_simple(fp, "L", 8)
        data = np.frombuffer(fp.read(length * 8), dtype=np.float64)

        return PosData(name, Unit(divisor, unit, symbol), data)

    def __repr__(self):
        return f"PosData[{self.name}={self.unit}, size={self.data.size}]"

    def scaled(self):
        return self.data * self.unit.value


class Dict(RawData):
    code = 0x7D

    def __repr__(self):
        return f"Dict[{self.data}]"


class Terminator(OPDxValue):
    code = 0x7F

    @staticmethod
    def read(fp: BinaryIO):
        assert fp.read(2) == b"\xff\xff"
        return None


class NamedValue(object):
    DTYPES = {
        Matrix.code: Matrix,
        Boolean.code: Boolean,
        Int32.code: Int32,
        Uint32.code: Uint32,
        Int64.code: Int64,
        Uint64.code: Uint64,
        Float.code: Float,
        Double.code: Double,
        DType.code: DType,
        String.code: String,
        Quanity.code: Quanity,
        TimeStamp.code: TimeStamp,
        Unit.code: Unit,
        Array.code: Array,
        StringList.code: StringList,
        AnonMatrix.code: AnonMatrix,
        RawData.code: RawData,
        RawData2D.code: RawData2D,
        PosData.code: PosData,
        Dict.code: Dict,
        Terminator.code: Terminator,
    }

    def __init__(self, fp: BinaryIO):
        self._pos = fp.tell()
        self.name = OPDxValue.read_name(fp)

        dtype: Type[OPDxValue] = NamedValue.DTYPES[struct.unpack("B", fp.read(1))[0]]

        self.pos = fp.tell()
        self.value = dtype.read(fp)

    def __repr__(self):
        return f"{self.name}={self.value}"
