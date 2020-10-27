Version vxx.yy.zz -- Release date 20yy-mm-dd
============================================

* New features:

* API Changes:
  * You will now get a compilation error when performing certain undefined operations on rvalues, for example calling `toView` on an `Array` rvalue.

* Build changes/improvements:

* Bug fixes:
  * Fixed indexing bug that produced a compilation error with GCC 10.
  * Fixed a memory leak in `CRSMatrix::assimilate`.
  * Fixed a memory leak in `ArrayOfArraysView::free` with non-trivially destructable values.

Version v0.1.0 -- Release date 2020-10-13
=========================================

Initial release.
