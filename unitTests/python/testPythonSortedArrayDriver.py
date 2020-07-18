"""Tests for the `lvarray` extension module"""

import unittest
import itertools

import numpy as np
from numpy import testing

import testPythonSortedArray as lvarray


class ArrayTests(unittest.TestCase):
    """Tests for the `lvarray` extension module"""

    LOWER_UPPER_PAIRS = ((0, 10), (1, 5), (-15, 15), (100, 200))
    GETTER_SETTER_PAIRS = (
        (lvarray.get_sorted_array_int, lvarray.set_sorted_array_int),
    )

    def test_init(self):
        """Test that the array is properly initialized"""
        for getter, setter in self.GETTER_SETTER_PAIRS:
            for lowerbound, upperbound in self.LOWER_UPPER_PAIRS:
                array_from_c = setter(lowerbound, upperbound)
                array_from_python = np.array(
                    range(lowerbound, upperbound), dtype=array_from_c.dtype
                )
                testing.assert_array_equal(array_from_c, array_from_python)
                testing.assert_array_equal(array_from_c, getter())

    def test_modification(self):
        """Test that SortedArrays can't be modified"""
        for getter, setter in self.GETTER_SETTER_PAIRS:
            for lowerbound, upperbound in self.LOWER_UPPER_PAIRS:
                array_from_c = setter(lowerbound, upperbound)
                testing.assert_array_equal(array_from_c, np.sort(array_from_c))
                with self.assertRaisesRegex(ValueError, "read-only"):
                    array_from_c *= 2
                with self.assertRaisesRegex(ValueError, "read-only"):
                    array_from_c[0] = 1

    def test_resizing(self):
        """Test that the SortedArrays can't be resized"""
        for _, setter in self.GETTER_SETTER_PAIRS:
            array_from_c = setter(0, 10)
            with self.assertRaisesRegex(ValueError, "own its data"):
                array_from_c.resize((100,))

    def test_multiply(self):
        """Test that the array is properly initialized"""
        for lowerbound, upperbound in self.LOWER_UPPER_PAIRS:
            for factor in range(1, 5):
                array_from_c = lvarray.set_sorted_array_int(lowerbound, upperbound)
                array_copy = np.array(array_from_c)
                testing.assert_array_equal(array_from_c, array_copy)
                lvarray.multiply_sorted_array_int(factor)
                testing.assert_array_equal(
                    lvarray.get_sorted_array_int(), factor * array_copy
                )


if __name__ == "__main__":
    unittest.main()
