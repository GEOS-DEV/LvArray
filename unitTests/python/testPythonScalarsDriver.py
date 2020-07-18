"""Tests for the `lvarray` extension module"""

import unittest

import numpy as np
from numpy import testing

import testPythonScalars as lvarray


class ScalarTests(unittest.TestCase):
    """Tests for the `lvarray` extension module"""

    GETTERS = (
        lvarray.get_schar,
        lvarray.get_ulong,
        lvarray.get_longdouble,
        lvarray.get_ushort,
    )

    def test_size(self):
        for getter in list(self.GETTERS) + [lvarray.get_short_const]:
            arr = getter()
            self.assertEqual(arr.size, 1)
            self.assertEqual(arr.ndim, 1)

    def test_modify(self):
        for getter in self.GETTERS:
            arr = getter()
            arr_copy = np.array(arr)
            arr *= 2
            testing.assert_array_equal(arr, getter())
            testing.assert_array_equal(arr, arr_copy * 2)
            arr[0] = 5
            testing.assert_array_equal(arr, getter())

    def test_modify_const(self):
        arr = lvarray.get_short_const()
        with self.assertRaisesRegex(ValueError, "read-only"):
            arr *= 2
        with self.assertRaisesRegex(ValueError, "read-only"):
            arr[0] = 14

    def test_unconvertible_types(self):
        for getter in (lvarray.get_bool, lvarray.get_char16):
            with self.assertRaisesRegex(TypeError, "No NumPy type for"):
                getter()


if __name__ == "__main__":
    unittest.main()
