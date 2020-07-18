"""Tests for numpy views of LvArray::Array objects"""

import unittest
import itertools

import numpy as np
from numpy import testing

import testPythonArray as lvarray


class ArrayTests(unittest.TestCase):
    """Tests for numpy views of LvArray::Array objects"""

    LOWER_UPPER_PAIRS = ((0, 10), (1, 5), (-15, 15), (100, 200))
    ARRAY_GET_SET_MULT_DIM = (
        (lvarray.get_array1d, lvarray.set_array1d, lvarray.multiply_array1d, 1),
        (lvarray.get_array4d, lvarray.set_array4d, lvarray.multiply_array4d, 4),
        (lvarray.get_array2d, lvarray.set_array2d, lvarray.multiply_array2d, 2),
    )

    def test_resizing(self):
        """Test that the Numpy views can't be resized"""
        array1d = lvarray.set_array1d(0)
        with self.assertRaisesRegex(ValueError, "own its data"):
            array1d.resize((array1d.size * 2,))
        array4d = lvarray.set_array4d(0)
        with self.assertRaisesRegex(ValueError, "single-segment"):
            array4d.resize((array4d.size * 2,))
        array2d = lvarray.set_array2d(0)
        with self.assertRaisesRegex(ValueError, "own its data"):
            array2d.resize((array2d.size * 2,))

    def test_init(self):
        """Test initializing the arrays"""
        for getter, setter, _, dims in self.ARRAY_GET_SET_MULT_DIM:
            for offset_low, offset_high in self.LOWER_UPPER_PAIRS:
                array_low = np.array(setter(offset_low))
                array_high = np.array(setter(offset_high))
                self.assertEqual(len(array_low.shape), dims)
                self.assertEqual(array_low.shape, array_high.shape)
                testing.assert_array_equal(
                    array_low + (offset_high - offset_low), array_high
                )

    def test_multiply(self):
        """Test that multiplying the values of the lvarray change the numpy representation"""
        for getter, setter, multiplier, _ in self.ARRAY_GET_SET_MULT_DIM:
            for offset in range(7):
                for factor in range(1, 5):
                    array_from_c = setter(offset)
                    array_copy = np.array(array_from_c)
                    testing.assert_array_equal(array_from_c, array_copy)
                    multiplier(factor)
                    testing.assert_array_equal(getter(), factor * array_copy)

    def test_modification(self):
        """Test that modifying a numpy view of an array modifies the underlying LvArray"""
        for getter, setter, _, _ in self.ARRAY_GET_SET_MULT_DIM:
            for offset in range(7):
                for factor in range(2, 6):
                    arr = setter(offset)
                    unmodified_copy = np.array(arr)
                    # modify the array
                    arr *= factor
                    testing.assert_array_equal(arr, unmodified_copy * factor)
                    testing.assert_array_equal(getter(), arr)


if __name__ == "__main__":
    unittest.main()
