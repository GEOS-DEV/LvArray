Version vxx.yy.zz -- Release date 20yy-mm-dd
============================================

* New features:

* API Changes:

* Build changes/improvements:

* Bug fixes:

Version v0.2.0 -- Release date 2020-05-05
============================================

* New features:
  * Various tensorOps functions.
  * A Python library with interfaces to most container classes.
  * Can create an `Array` that begins life on the device.
  * Can create an `Array` with specific Umpire allocators.
  * Math functions for CUDA's `__half` and `__half2`.
  * Better CUDA error messages.
  * Added in wrappers around Umpire's `memcpy` and `memset`.

* API Changes:
  * You will now get a compilation error when performing certain undefined operations on rvalues, for example calling `toView` on an `Array` rvalue.
  * Now use `camp::resources::Platform` for `MemorySpace`.

* Build changes/improvements:
  * config-build.py works with Python3.
  * Support for object libraries via the CMake variable `LVARRAY_BUILD_OBJ_LIBS`.
  * Support for RAJA 0.13.

* Bug fixes:
  * Fixed an indexing bug that produced a compilation error with GCC 10.
  * Fixed a memory leak in `CRSMatrix::assimilate`.
  * Fixed a memory leak in `ArrayOfArraysView::free` with non-trivially destructable values.
  * Fixed a compilation error with GCC 9.

Version v0.1.0 -- Release date 2020-10-13
=========================================

Initial release.
