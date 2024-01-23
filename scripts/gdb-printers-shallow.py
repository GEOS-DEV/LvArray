import abc
import logging
import math
from typing import (
    Any,
    Callable,
    Iterable,
)
import re

import gdb


class Buffer(metaclass=abc.ABCMeta):
    """Base class for accessing buffer contents"""

    def __init__(self, val: gdb.Value, data_key: str):
        self._val: gdb.Value = val
        self._data_key: str = data_key

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    def __getitem__(self, key) -> Any:
        return self.data()[key]

    def data(self) -> gdb.Value:
        return self._val[self._data_key]


class StackBuffer(Buffer):

    def __init__(self, val: gdb.Value):
        super().__init__(val, "m_data")

    def __len__(self) -> int:
        return int(self._val.type.template_argument(1))


class ChaiBuffer(Buffer):

    def __init__(self, val: gdb.Value):
        super().__init__(val, "m_pointer")

    def __len__(self) -> int:
        return int(self._val['m_capacity'])


class MallocBuffer(Buffer):

    def __init__(self, val: gdb.Value):
        super().__init__(val, "m_data")

    def __len__(self) -> int:
        return int(self._val['m_capacity'])


def extract_buffer(val: gdb.Value) -> Buffer:
    # Use self.val.type.fields() to know the fields.
    # You can also have a look at the fields of the fields.
    t: str = str(val.type)
    if re.match('LvArray::ChaiBuffer<.*>', t):
        return ChaiBuffer(val)
    elif re.match('LvArray::StackBuffer<.*>', t):
        return StackBuffer(val)
    elif re.match('LvArray::MallocBuffer<.*>', t):
        return MallocBuffer(val)
    else:
        raise ValueError(f"Could not build buffer from `{val.type}`.")


class LvArrayPrinter:
    """Base printer for LvArray classes"""

    def __init__(self, val: gdb.Value):
        self.val: gdb.Value = val

    def real_type(self) -> gdb.Type:
        return self.val.type.strip_typedefs()

    def display_hint(self) -> str:
        return 'array'


class __ArrayPrinter(LvArrayPrinter):
    """Utility class for code factorization"""

    def __init__(self,
                 val: gdb.Value,
                 data_extractor: Callable[[gdb.Value], gdb.Value],
                 dimension_extractor: Callable[[gdb.Value], gdb.Value]):
        """
        val: The initial `gdb.Value`. Typically a `LvArray::Array`.
        data_extractor: How to access the raw data from the initial `val`.
        dimension_extractor: How to access the dimensions (since this is a multi-dimensional array) from the initial `val`.
        """
        super().__init__(val)

        self.data = data_extractor(self.val)
        dimensions: gdb.Value = dimension_extractor(self.val)

        num_dimensions: int = int(self.val.type.template_argument(1))
        dimensions: Iterable[gdb.Value] = map(dimensions.__getitem__, range(num_dimensions))
        self.dimensions: tuple[int] = tuple(map(int, dimensions))

    def to_string(self) -> str:
        dimensions = map(str, self.dimensions)
        return f'{self.real_type()} of shape [{", ".join(dimensions)}]'

    def children(self) ->  Iterable[tuple[str, gdb.Value]]:
        d0, ds = self.dimensions[0], self.dimensions[1:]

        # The main idea of this loop is to build the mutli-dimensional array type to the data (e.g. `int[2][3]`).
        # Then we'll cast the raw pointer into this n-d array and gdb will be able to manage it.
        array_type: gdb.Type = self.data.type.target()
        for d in reversed(ds):  # Note that we ditch the first dimension from our loop in order to have it as first level children.
            array_type = array_type.array(d - 1)

        # We manage the first level children ourselves, so we need to manage the position of the data ourselves too.
        stride: int = math.prod(ds)

        for i in range(d0):
            array = (self.data + i * stride).dereference().cast(array_type)
            yield '[%d]' % i, array


class ArrayPrinter(__ArrayPrinter):
    """Pretty-print for Array(View)"""

    def __init__(self, val: gdb.Value):
        super().__init__(val, lambda v: extract_buffer(v["m_dataBuffer"]).data(), lambda v: v["m_dims"]["data"])


class ArraySlicePrinter(__ArrayPrinter):
    """Pretty-print for ArraySlice"""

    def __init__(self, val: gdb.Value):
        super().__init__(val, lambda v: v["m_data"], lambda v: v["m_dims"])


class ArrayOfArraysPrinter(LvArrayPrinter):
    """Pretty-print for ArrayOfArrays(View)"""

    def __len__(self) -> int:
        return int(self.val['m_numArrays'])

    def to_string(self) -> str:
        return '%s of size %d' % (self.real_type(), len(self))

    def children(self) -> Iterable[tuple[str, gdb.Value]]:
        # In this function, we are walking along the "sub" arrays ourselves.
        # To do this, we manipulate the raw pointer/offsets information ourselves.
        data = extract_buffer(self.val["m_values"]).data()
        offsets = extract_buffer(self.val["m_offsets"])
        sizes = extract_buffer(self.val["m_sizes"])
        for i in range(len(self)):  # Iterating over all the "sub" arrays.
            # Converting a raw pointer `T*` to an equivalent type including the size `T[N]` that gdb will manage.
            array_type = data.type.target().array(sizes[i] - 1)
            array = (data + offsets[i]).dereference().cast(array_type)
            yield '[%d]' % i, array


def build_array_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter("my-LvArray-arrays")
    pp.add_printer('LvArray::Array', '^LvArray::Array(View)?<.*>$', ArrayPrinter)
    pp.add_printer('LvArray::ArraySlice', '^LvArray::ArraySlice<.*>$', ArraySlicePrinter)
    pp.add_printer('LvArray::ArrayOfArrays', '^LvArray::ArrayOfArrays(View)?<.*>$', ArrayOfArraysPrinter)
    return pp


try:
    import gdb.printing
    gdb.printing.register_pretty_printer(gdb.current_objfile(), build_array_printer())
except ImportError:
    logging.warning("Could not register LvArray pretty printers.")
