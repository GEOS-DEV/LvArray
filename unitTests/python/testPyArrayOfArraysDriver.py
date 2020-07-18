import unittest

import numpy as np
from numpy import testing

import pylvarray
from testPyArrayOfArrays import get_array_of_arrays


def clear(array):
    array.set_access_level(pylvarray.RESIZEABLE)
    while array:
        del array[0]


class LvArrayArrayOfArraysTests(unittest.TestCase):
    def setUp(self):
        clear(get_array_of_arrays())
        self.assertEqual(len(get_array_of_arrays()), 0)

    def populate(self, item=np.array((1, 2, 3)), num_entries=5):
        arr = get_array_of_arrays()
        arr.set_access_level(pylvarray.RESIZEABLE)
        for i in range(num_entries):
            arr.insert(i, item)
        return arr

    def test_modify(self):
        arr = self.populate()
        arr.set_access_level(pylvarray.READ_ONLY)
        with self.assertRaisesRegex(ValueError, "read-only"):
            arr[0][0] = 6
        arr.set_access_level(pylvarray.MODIFIABLE)
        self.assertTrue(arr[0].flags.writeable)
        arr[0][0] = 6
        arr.set_access_level(pylvarray.RESIZEABLE)
        self.assertTrue(arr[0].flags.writeable)
        arr[0][0] = 6

    def test_bad_delitem(self):
        arr = get_array_of_arrays()
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
            arr.erase_from(i, 1)
            testing.assert_array_equal(np.array((1, 3)), arr[i])
        with self.assertRaises(IndexError):
            arr.erase_from(i, 10)
        with self.assertRaises(IndexError):
            arr.erase_from(i, -1)

    def test_iter(self):
        arr = self.populate()
        for i, subarray in enumerate(arr):
            testing.assert_array_equal(subarray, arr[i])

    def test_setitem(self):
        arr = get_array_of_arrays()
        arr.set_access_level(pylvarray.RESIZEABLE)
        with self.assertRaises(TypeError):
            arr[0] = 5

    def test_insert(self):
        arr = get_array_of_arrays()
        arr.set_access_level(pylvarray.RESIZEABLE)
        size = len(arr)
        to_insert = np.array((1, 2, 3))
        for i in range(1, 6):
            arr.insert(0, to_insert)
            self.assertEqual(len(arr), size + i)
            testing.assert_array_equal(arr[i - 1], to_insert)

    def test_bad_insert(self):
        arr = get_array_of_arrays()
        arr.set_access_level(pylvarray.RESIZEABLE)
        with self.assertRaises(IndexError):
            arr.insert(-1, np.array((1, 2, 3)))
        with self.assertRaises(IndexError):
            arr.insert(len(arr) + 1, np.array((1, 2, 3)))

    def test_insert_into(self):
        arr = self.populate()
        for i in range(len(arr)):
            arr.insert_into(i, 0, np.array((5, 6, 7)))
            testing.assert_array_equal(arr[i], np.array((5, 6, 7, 1, 2, 3)))
        with self.assertRaises(IndexError):
            arr.insert_into(0, len(arr[0]) + 1, np.array((1, 2)))
        with self.assertRaises(IndexError):
            arr.insert_into(0, -1, np.array((1, 2)))
        with self.assertRaises(IndexError):
            arr.insert_into(-1, 0, np.array((1, 2)))
        with self.assertRaises(IndexError):
            arr.insert_into(len(arr), 0, np.array((1, 2)))

    def test_getitem(self):
        arr = get_array_of_arrays()
        arr.set_access_level(pylvarray.RESIZEABLE)
        to_insert = np.array((1, 2, 3))
        for i in range(6):
            arr.insert(i, np.array((1, 2, i)))
        for i in range(1, 6):
            testing.assert_array_equal(arr[-i], arr[len(arr) - i])


if __name__ == "__main__":
    unittest.main()
