"""Tests for the `lvarray` extension module"""

import unittest

import numpy as np

import testPythonListOfStrings as lvarray


class ListOfStringsTests(unittest.TestCase):
    """Tests for the `lvarray` extension module"""

    def test_set_vector(self):
        for initializer in ("foobar", "barfoo", "hello world!"):
            strlist = lvarray.setvector(initializer)
            self.assertEqual(len(strlist), lvarray.ARR_SIZE)
            self.assertEqual(
                [initializer for _ in range(lvarray.ARR_SIZE)], strlist
            )

    def test_get_vector(self):
        lvarray.setvector("barfoo")
        self.assertEqual(lvarray.getvector(), lvarray.getvector())
        self.assertFalse(lvarray.getvector() is lvarray.getvector())

    def test_get_array(self):
        lvarray.setarray("barfoo")
        self.assertIs(lvarray.getarray().dtype, np.object_)
        self.assertEqual(lvarray.getarray().to_numpy(), lvarray.getarray().to_numpy())
        self.assertFalse(lvarray.getarray() is lvarray.getarray())

    def test_set_array(self):
        for initializer in ("foobar", "barfoo", "hello world!"):
            strlist = lvarray.setarray(initializer).to_numpy()
            self.assertIsInstance(strlist, list)
            self.assertEqual(len(strlist), lvarray.ARR_SIZE)
            self.assertEqual(
                [initializer for _ in range(lvarray.ARR_SIZE)], strlist
            )


if __name__ == "__main__":
    unittest.main()
