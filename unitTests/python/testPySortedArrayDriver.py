"""Tests for the SortedArray python wrapper."""

import unittest

import numpy as np
from numpy import testing

from testPySortedArray import get_sorted_array_int, get_sorted_array_long
import pylvarray


def clear(arr):
    """Test the insert method."""
    while len(arr.to_numpy()) > 0:
        arr.set_access_level(pylvarray.RESIZEABLE)
        arr.remove(arr.to_numpy()[0])


class SortedArrayTests(unittest.TestCase):
    """Tests for the SortedArray python wrapper."""

    lvarrays = (get_sorted_array_int, get_sorted_array_long)

    def setUp(self):
        """Test the insert method."""
        for getter in self.lvarrays:
            clear(getter())
            self.assertEqual(len(getter().to_numpy()), 0)

    def test_insert(self):
        for getter in self.lvarrays:
            arr = getter()
            arr.set_access_level(pylvarray.RESIZEABLE)
            for value in range(-5, 15):
                arr.insert(arr.dtype(value))
                self.assertIn(arr.dtype(value), arr.to_numpy())
            self.assertEqual(len(arr.to_numpy()), 20)
            testing.assert_array_equal(arr.to_numpy(), np.sort(arr.to_numpy()))

    def test_remove(self):
        """Test the remove method."""
        for getter in self.lvarrays:
            arr = getter()
            arr.set_access_level(pylvarray.RESIZEABLE)
            for value in range(-5, 15):
                arr.insert(arr.dtype(value))
            self.assertEqual(len(arr.to_numpy()), 20)
            for value in range(-5, 15):
                arr.remove(arr.dtype(value))
            self.assertEqual(len(arr.to_numpy()), 0)

    def test_modification_read_only(self):
        """Test that calling insert or remove on a read only SortedArray raises an exception."""
        for getter in self.lvarrays:
            arr = getter()
            arr.set_access_level(pylvarray.RESIZEABLE)
            arr.insert(arr.dtype(5))
            self.assertEqual(arr.to_numpy()[0], arr.dtype(5))
            with self.assertRaisesRegex(ValueError, "read-only"):
                arr.to_numpy()[0] = arr.dtype(6)

    def test_modification_unsafe_conversion(self):
        """Test that calling insert or remove that involves an unsafe type conversion raises an exception."""
        for getter in self.lvarrays:
            arr = getter()
            arr.set_access_level(pylvarray.RESIZEABLE)
            with self.assertRaisesRegex(TypeError, "Cannot safely convert"):
                arr.insert(5.6)
            with self.assertRaisesRegex(TypeError, "Cannot safely convert"):
                arr.insert("foobar")


if __name__ == "__main__":
    unittest.main()
