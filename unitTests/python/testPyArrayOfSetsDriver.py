import unittest

import numpy as np
from numpy import testing

import pylvarray
from testPyArrayOfSets import get_array_of_sets


def clear(array):
    array.set_access_level(pylvarray.RESIZEABLE)
    while array:
        del array[0]


class LvArrayArrayOfArraysTests(unittest.TestCase):
    def setUp(self):
        clear(get_array_of_sets())
        self.assertEqual(len(get_array_of_sets()), 0)

    def populate(self, item=np.array((1, 2, 3)), num_entries=5):
        arr = get_array_of_sets()
        clear(arr)
        for i in range(num_entries):
            arr.insert(i, len(item))
            arr.insert_into(i, item)
        return arr

    def test_modify(self):
        arr = self.populate()
        for view in arr:
            with self.assertRaisesRegex(ValueError, "read-only"):
                view *= 2
            with self.assertRaisesRegex(ValueError, "read-only"):
                view[0] = 20

    def test_bad_delitem(self):
        arr = get_array_of_sets()
        with self.assertRaisesRegex(RuntimeError, "resizeable"):
            del arr[0]
        arr.set_access_level(pylvarray.RESIZEABLE)
        with self.assertRaises(IndexError):
            del arr[-1]
        with self.assertRaises(IndexError):
            del arr[6]
        with self.assertRaises(TypeError):
            del arr["no string indices"]

    def test_delitem(self):
        arr = self.populate()
        while arr:
            size = len(arr)
            del arr[0]
            self.assertEqual(len(arr), size - 1)

    def test_erase_from(self):
        arr = self.populate()
        for i in range(5):
            arr.erase_from(i, 2)
            testing.assert_array_equal(np.array((1, 3)), arr[i])

    def test_iter(self):
        arr = self.populate()
        for i, subarray in enumerate(arr):
            testing.assert_array_equal(subarray, arr[i])

    def test_setitem(self):
        arr = get_array_of_sets()
        with self.assertRaisesRegex(RuntimeError, "resizeable"):
            arr[0] = 5
        arr.set_access_level(pylvarray.RESIZEABLE)
        with self.assertRaises(TypeError):
            arr[0] = 5

    def test_insert(self):
        arr = self.populate()
        for i in range(len(arr)):
            arr.insert(i, i + 10)
            self.assertEqual(len(arr[i]), 0)

    def test_bad_insert(self):
        arr = get_array_of_sets()
        arr.set_access_level(pylvarray.RESIZEABLE)
        with self.assertRaises(IndexError):
            arr.insert(-1, 10)
        with self.assertRaises(IndexError):
            arr.insert(len(arr) + 1, 10)

    def test_insert_into(self):
        arr = self.populate()
        for i in range(len(arr)):
            arr.insert_into(i, np.array((5, 6, 7)))
            testing.assert_array_equal(arr[i], np.array((1, 2, 3, 5, 6, 7)))
        with self.assertRaises(IndexError):
            arr.insert_into(-1, np.array((1, 2)))
        with self.assertRaises(IndexError):
            arr.insert_into(len(arr), np.array((1, 2)))


if __name__ == "__main__":
    unittest.main()
