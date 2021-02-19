import unittest

import testPyCallback
import pylvarray


class CustomException(Exception):
    pass


class LvArrayCallbackTests(unittest.TestCase):
    def test_modify(self):
        to_modify = 12

        def callback(*args):
            nonlocal to_modify
            to_modify += 1

        testPyCallback.callback = callback
        for _ in range(10):
            unmodified = to_modify
            testPyCallback.call()
            self.assertEqual(unmodified + 1, to_modify)

    def test_custom_exception(self):
        def callback(*args):
            raise CustomException("foobar")

        testPyCallback.callback = callback
        with self.assertRaisesRegex(CustomException, "foobar"):
            testPyCallback.call()

    def test_arg1(self):
        def callback(arg):
            self.assertTrue(isinstance(arg, pylvarray.SortedArray))
            dtype = arg.to_numpy().dtype.type
            arg.set_access_level(pylvarray.RESIZEABLE)
            for i in range(10):
                arg.insert(dtype(i))
                self.assertIn(i, arg.to_numpy())

        testPyCallback.callback = callback
        testPyCallback.call()


if __name__ == "__main__":
    unittest.main()
