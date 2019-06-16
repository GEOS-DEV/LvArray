import gdb


def array_length(arr):
    return int(arr.type.sizeof / arr.type.target().sizeof)


def format_array(arr):
    return '[' + ']['.join(['%d' % arr[i] for i in range(array_length(arr))]) + ']'


class ArrayViewPrinter(gdb.printing.PrettyPrinter):
    """Pretty-print an Array/View/Slice"""

    def __init__(self, val):
        self.val = val

    def real_type(self):
        return self.val.type.strip_typedefs()

    def value_type(self):
        return self.val.type.template_argument(0)

    def ndim(self):
        return self.val.type.template_argument(1)

    def index_type(self):
        return self.val.type.template_argument(2)

    def dims(self):
        return self.val['m_dims']

    def strides(self):
        return self.val['m_strides']

    def data(self):
        return self.val['m_data']

    def to_string(self):
        return '%s of size %s' % (self.real_type().name, format_array(self.dims()))

    def children(self):

        array_type = self.value_type()
        for i in range(self.ndim()):
            array_type = array_type.array(self.dims()[self.ndim() - i - 1] - 1)

        values = [('gdb(m_data)',   self.data().dereference().cast(array_type)),
                  ('m_data',    self.data()),
                  ('m_dims',    self.dims()),
                  ('m_strides', self.strides())]

        try:
            values.append(('m_dataVector', self.val['m_dataVector']))
        except gdb.error:
            pass

        try:
            values.append(('m_singleParameterResizeIndex', self.val['m_singleParameterResizeIndex']))
        except gdb.error:
            pass

        return values


def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter("cxx-utilities")
    pp.add_printer('Array', '^LvArray::Array<.*>$', ArrayViewPrinter)
    pp.add_printer('ArrayView', '^LvArray::ArrayView<.*>$', ArrayViewPrinter)
    pp.add_printer('ArraySlice', '^LvArray::ArraySlice<.*>$', ArrayViewPrinter)
    return pp


try:
    import gdb.printing
    gdb.printing.register_pretty_printer(
        gdb.current_objfile(),
        build_pretty_printer())
except ImportError:
    pass
