"""Tests for the SortedArray python wrapper."""

import unittest

import numpy as np
from numpy import testing

import pylvarray
from testPyArray import (
    get_array1d_int,
    get_array1d_double,
    get_array2d_ij_long,
    get_array2d_ji_float,
    get_array4d_kilj_double,
)


def clear(arr):
    arr.set_access_level(pylvarray.RESIZEABLE)
    view = arr.to_numpy()
    view[:] = arr.dtype(0)


class ArrayTests(unittest.TestCase):
    """Tests for the Array python wrapper."""

    lvarrays = (
        get_array1d_int,
        get_array1d_double,
        get_array2d_ij_long,
        get_array2d_ji_float,
        get_array4d_kilj_double,
    )

    def setUp(self):
        for getter in self.lvarrays:
            clear(getter())

    def test_modification(self):
        for getter in self.lvarrays:
            arr = getter()
            clear(arr)
            view = arr.to_numpy()
            testing.assert_array_equal(view, np.zeros_like(view))
            testing.assert_array_equal(view, getter().to_numpy())
            view[:] = arr.dtype(5)
            testing.assert_array_equal(view, np.ones_like(view) * 5)
            testing.assert_array_equal(view, getter().to_numpy())

    def test_modification_read_only(self):
        """Test that calling insert or remove on a read only SortedArray raises an exception."""
        for getter in self.lvarrays:
            arr = getter()
            arr.set_access_level(pylvarray.READ_ONLY)
            view = arr.to_numpy()
            self.assertFalse(view.flags.writeable)
            arr.set_access_level(pylvarray.MODIFIABLE)
            view = arr.to_numpy()
            self.assertTrue(view.flags.writeable)
            arr.set_access_level(pylvarray.RESIZEABLE)
            self.assertEqual(arr.get_access_level(), pylvarray.RESIZEABLE)
            view = arr.to_numpy()
            self.assertTrue(view.flags.writeable)
            arr.set_access_level(pylvarray.READ_ONLY)
            self.assertEqual(arr.get_access_level(), pylvarray.READ_ONLY)
            view = arr.to_numpy()
            with self.assertRaisesRegex(ValueError, "read-only"):
                view[0] = arr.dtype(6)

    def test_resize_all_not_resizeable(self):
        for getter in self.lvarrays:
            arr = getter()
            arr.set_access_level(pylvarray.MODIFIABLE)
            original_dims = arr.to_numpy().shape
            with self.assertRaisesRegex(RuntimeError, "resizeable"):
                arr.resize_all(original_dims)

    def test_resize_all(self):
        for getter in self.lvarrays:
            arr = getter()
            arr.set_access_level(pylvarray.RESIZEABLE)
            original_dims = arr.to_numpy().shape
            new_dims = np.array(original_dims, dtype=np.int32) * 2
            arr.resize_all(new_dims)
            testing.assert_array_equal(new_dims, np.array(arr.to_numpy().shape))

    def test_resize_one_dim(self):
        for getter in self.lvarrays:
            arr = getter()
            arr.set_access_level(pylvarray.RESIZEABLE)
            for resize_val in (100, 75, 76):
                arr.resize(resize_val)
                self.assertEqual(resize_val, arr.to_numpy().shape[0])


if __name__ == "__main__":
    unittest.main()
