import gdb
import itertools


def array_length(arr):
    return int(arr.type.sizeof / arr.type.target().sizeof)


def format_array(arr):
    return '[' + ']['.join([str(arr[i]) for i in range(array_length(arr))]) + ']'


class LvArrayPrinter(gdb.printing.PrettyPrinter):
    """Base printer for LvArray classes"""

    def __init__(self, val):
        self.val = val

    def real_type(self):
        return self.val.type.strip_typedefs()

    def to_string(self):
        return str(self.real_type())

    def children(self):
        return [(f.name, self.val.cast(f.type)) for f in self.real_type().fields() if f.is_base_class] \
             + [(f.name, self.val[f.name]) for f in self.real_type().fields() if not f.is_base_class]


class BufferPrinter(LvArrayPrinter):
    """Base class for printing buffer contents"""

    def size(self):
        raise NotImplementedError("Must override size()")

    def data(self):
        raise NotImplementedError("Must override data()")

    def value_type(self):
        return self.val.type.template_argument(0)

    def __getitem__(self, key):
        return self.data()[key]

    def __setitem__(self, key, value):
        self.data()[key] = value

    def to_string(self):
        return '%s of size %d' % (self.real_type(), self.size())

    def children(self):
        if self.data() != 0:
            array_type = self.value_type().array(self.size()-1)
            arr = self.data().dereference().cast(array_type)
        else:
            arr = self.data()
        return [('gdb_view', arr)] + LvArrayPrinter.children(self)


class StackBufferPrinter(BufferPrinter):
    """Pretty-print a StackBuffer"""

    def size(self):
        return self.val.type.template_argument(1)

    def data(self):
        return self.val['m_data']


class ChaiBufferPrinter(BufferPrinter):
    """Pretty-print a ChaiBuffer"""

    def size(self):
        return self.val['m_capacity']

    def data(self):
        return self.val['m_pointer']


class MallocBufferPrinter(BufferPrinter):
    """Pretty-print a MallocBuffer"""

    def size(self):
        return self.val['m_capacity']

    def data(self):
        return self.val['m_data']


def build_buffer_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter("LvArray-buffers")
    pp.add_printer('LvArray::StackBuffer', 'LvArray::StackBuffer<.*>', StackBufferPrinter)
    pp.add_printer('LvArray::ChaiBuffer', 'LvArray::ChaiBuffer<.*>', ChaiBufferPrinter)
    pp.add_printer('LvArray::MallocBuffer', 'LvArray::MallocBuffer<.*>', MallocBufferPrinter)
    return pp


try:
    import gdb.printing
    buffer_printer = build_buffer_printer()
    gdb.printing.register_pretty_printer(gdb.current_objfile(), buffer_printer)
except ImportError:
    pass


class ArraySlicePrinter(LvArrayPrinter):
    """Pretty-print an ArraySlice"""

    def value_type(self):
        return self.val.type.template_argument(0)

    def ndim(self):
        return self.val.type.template_argument(1)

    def dims(self):
        return self.val['m_dims']

    def strides(self):
        return self.val['m_strides']

    def data(self):
        return self.val['m_data']

    def to_string(self):
        return '%s of size %s' % (self.real_type(), format_array(self.dims()))

    def children(self):
        array_type = self.value_type()
        for i in range(self.ndim()):
            array_type = array_type.array(self.dims()[self.ndim() - i - 1] - 1)
        return [('gdb_view', self.data().dereference().cast(array_type))] + LvArrayPrinter.children(self)


class ArrayViewPrinter(LvArrayPrinter):
    """Pretty-print an ArrayView"""

    def value_type(self):
        return self.val.type.template_argument(0)

    def ndim(self):
        return self.val.type.template_argument(1)

    def dims(self):
        return self.val['m_dims']['data']

    def strides(self):
        return self.val['m_strides']['data']

    def data(self):
        return buffer_printer(self.val['m_dataBuffer']).data()

    def to_string(self):
        return '%s of size %s' % (self.real_type(), format_array(self.dims()))

    def children(self):
        array_type = self.value_type()
        for i in range(self.ndim()):
            array_type = array_type.array(self.dims()[self.ndim() - i - 1] - 1)
        return [('gdb_view', self.data().dereference().cast(array_type))] + LvArrayPrinter.children(self)


class ArrayOfArraysViewPrinter(LvArrayPrinter):
    """Pretty-print an ArrayOfArraysView"""

    def value_type(self):
        return self.val.type.template_argument(0)

    def data(self):
        return buffer_printer(self.val['m_values']).data()

    def size(self):
        return self.val['m_numArrays']

    def size_of_array(self, i):
        return buffer_printer(self.val['m_sizes'])[i]

    def offset_of_array(self, i):
        return buffer_printer(self.val['m_offsets'])[i]

    def ptr_to_array(self, i):
        return self.data() + self.offset_of_array(i)

    def get_array(self, i):
        return self.ptr_to_array(i).dereference().cast(self.value_type().array(self.size_of_array(i)-1))

    def to_string(self):
        return '%s of size %d' % (self.real_type(), self.size())

    def child_arrays(self):
        for i in range(self.size()):
            yield str(i), self.get_array(i)

    def children(self):
        return itertools.chain(LvArrayPrinter.children(self), self.child_arrays())


def build_array_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter("LvArray-arrays")
    pp.add_printer('LvArray::ArraySlice', '^LvArray::ArraySlice<.*>$', ArraySlicePrinter)
    pp.add_printer('LvArray::ArrayView', '^LvArray::ArrayView<.*>$', ArrayViewPrinter)
    pp.add_printer('LvArray::ArrayOfArraysView', '^LvArray::ArrayOfArraysView<.*>$', ArrayOfArraysViewPrinter)
    return pp


try:
    import gdb.printing
    gdb.printing.register_pretty_printer(gdb.current_objfile(), build_array_printer())
except ImportError:
    pass
