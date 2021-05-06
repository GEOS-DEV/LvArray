.. ##
.. ## Copyright (c) 2021, Lawrence Livermore National Security, LLC
.. ## and LvArray project contributors. See the LICENCE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

LvArray
=======

LvArray is a collection of container classes designed for performance portability in that they are usable on the host and device and provide performance similar to direct pointer manipulation. It consists of six main template classes:

- ``LvArray::Array``: A multidimensional array.
- ``LvArray::SortedArray``: A sorted unique collection of values stored in a contiguous allocation.
- ``LvArray::ArrayOfArrays``: Provides functionality similar to ``std::vector< std::vector< T > >`` except all the values are stored in one allocation.
- ``LvArray::ArrayOfSets``: Provides functionality similar to ``std::vector< std::set< T > >`` except all the values are stored in one allocation.
- ``LvArray::SparsityPattern``: A compressed row storage sparsity pattern, or equivalently a boolean CRS matrix.
- ``LvArray::CRSMatrix``: A compressed row storage matrix.

These template classes all take three common template arguments:

- The type of the value stored in the container.
- The integral type used for indexing calculations, the recommended type is ``std::ptrdiff_t``.
- The buffer type to use for allocation and de-allocation.

Like the standard template library containers the destructors destroy all the values in the container and release any acquired resources. Similarly the copy constructors and copy assignment operators perform a deep copy of the source container. In general the move constructors and move assignment operators perform a shallow copy and invalidate the source container.

The ``LvArray`` containers were designed specifically to integrate with `RAJA <https://github.com/LLNL/RAJA>`_ although they would also work with other kernel launching abstractions or native device code. In RAJA, variables are passed to device kernels by lambda capture. When using CUDA a ``__device__`` or ``__host__ __device__`` lambda can only capture variables by value, which means the lambda's closure object contains a copy of each captured variable created by copy construction. As mentioned above the copy constructor for the ``LvArray`` containers always perform a deep copy. So when a container gets captured by value in a lambda any modifications to it or the values it contains won't be reflected in the source container. Even if this works for your use case it is not optimal because it requires a new allocation each time the container gets captured and each time the closure object gets copied. To remedy this each ``LvArray`` container has an associated view class. Unlike the containers the views do not own their resources and in general have shallow copy semantics. This makes them suitable to be captured by value in a lambda. Furthermore almost every view method is ``const`` which means the lambda does not need to be made ``mutable``.

.. note::
  Not all buffer types support shallow copying. When using a container with such a buffer the move constructor and move assignment operator will create a deep copy of the source. Similarly the view copy constructor, copy assignment operator, move constructor and move assignment operator will all be deep copies.

.. toctree::
  :numbered:
  :maxdepth: 1

  buildGuide
  bufferClasses
  Array
  SortedArray
  ArrayOfArrays
  ArrayOfSets
  SparsityPatternAndCRSMatrix
  tensorOps
  extraThings
  developmentAids
  testing
  benchmarks
  python/index

Doxygen
=======
`Doxygen <doxygen/html/namespaces.html>`_
