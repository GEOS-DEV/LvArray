.. ##
.. ## Copyright (c) 2021, Lawrence Livermore National Security, LLC
.. ## and LvArray project contributors. See the LICENCE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

###############################################################################
``LvArray::Array``
###############################################################################

The ``LvArray::Array`` implements a multidimensional array of values. Unlike `Kokkos mdspan <https://github.com/kokkos/mdspan/wiki/A-Gentle-Introduction-to-mdspan>`_ or the `RAJA View <https://raja.readthedocs.io/en/v0.11.0/feature/view.html>`_ the ``LvArray::Array`` owns the allocation it is associated with and supports slicing with ``operator[]`` in addition to the standard ``operator()``. Further more a one dimensional ``LvArray::Array`` supports operations such as ``emplace_back`` and ``erase`` with functionality similar to ``std::vector``.

Template arguments
------------------
The ``LvArray::Array`` requires five template arguments.

1. ``T``: The type of values stored in the array.
2. ``NDIM``: The number of dimensionality of the array or the number of indices required to access a value.
3. ``PERMUTATION``: A ``camp::idx_seq`` which describes the mapping from the multidimensional index space to a linear index. Must be of length ``NDIM`` and contain all the values between ``0`` and ``NDIM - 1``. The way to read a permutation is that the indices go from the slowest on the left to the fastest on the right. Equivalently the left most index has the largest stride whereas the right most index has unit stride.
4. ``INDEX_TYPE``: An integral type used in index calculations, the suggested type is ``std::ptrdiff_t``.
5. ``BUFFER_TYPE``: A template template parameter specifying the buffer type used for allocation and de-allocation, the ``LvArray::Array`` contains a ``BUFFER_TYPE< T >``.

.. note::
  ``LvArray`` uses the same permutation conventions as the ``RAJA::View``, in fact it is recommended to use the types defined in `RAJA/Permutations.hpp <https://github.com/LLNL/RAJA/blob/v0.11.0/include/RAJA/util/Permutations.hpp>`_.

Creating and accessing a ``LvArray::Array``
-------------------------------------------
The ``LvArray::Array`` has two primary constructors a default constructor which creates an empty array and a constructor that takes the size of each dimension. When using the sized constructor the values of the array are default initialized.

.. literalinclude:: ../../examples/exampleArray.cpp
  :language: c++
  :start-after: // Sphinx start after constructors
  :end-before: // Sphinx end before constructors

*[Source: examples/exampleArray.cpp]*

``LvArray::Array`` supports two indexing methods ``operator()`` which takes all of the indices to a value and ``operator[]`` which takes a single index at a time and can be chained together.

.. literalinclude:: ../../examples/exampleArray.cpp
  :language: c++
  :start-after: // Sphinx start after accessors
  :end-before: // Sphinx end before accessors

*[Source: examples/exampleArray.cpp]*

The two indexing methods work consistently regardless of how the data is layed out in memory.

.. literalinclude:: ../../examples/exampleArray.cpp
  :language: c++
  :start-after: // Sphinx start after permutations
  :end-before: // Sphinx end before permutations

*[Source: examples/exampleArray.cpp]*

Resizing a ``LvArray::Array``
-----------------------------
``LvArray::Array`` supports a multitude of resizing options. You can resize all of the dimensions at once or specify a set of dimensions to resize. These methods default initialize newly created values and destroy any values no longer in the Array.

.. literalinclude:: ../../examples/exampleArray.cpp
  :language: c++
  :start-after: // Sphinx start after resize
  :end-before: // Sphinx end before resize

*[Source: examples/exampleArray.cpp]*

``LvArray::Array`` also has a method ``resizeWithoutInitializationOrDestruction`` that is only enabled if the value type of the array ``T`` is trivially destructible. This method does not initialize new values or destroy old values and as such it can be much faster for large allocations of trivial types.

It is important to note that unless the array being resized is one dimensional the resize methods above do not preserve the values in the array. That is if you have a two dimensional array ``A`` of size :math:`M \times N` and you resize it to :math:`P \times Q` using any of the methods above then you cannot rely on ``A( i, j )`` having the same value it did before the resize.

There is also a method ``resize`` which takes a single parameter and will resize the first dimension of the array.

.. literalinclude:: ../../examples/exampleArray.cpp
  :language: c++
  :start-after: // Sphinx start after resizeSingleDimension
  :end-before: // Sphinx end before resizeSingleDimension

*[Source: examples/exampleArray.cpp]*

The single dimension resize should only be used when it is necessary to preserve the values as it is a much more complicated operation than the multi-dimension resize methods.

The one dimensional ``LvArray::Array``
--------------------------------------
The one dimensional ``LvArray::Array`` supports a couple methods that are not available to multidimensional arrays. These methods are ``emplace_back``, ``emplace``, ``insert``, ``pop_back`` and ``erase``. They all behave exactly like their ``std::vector`` counter part, the only difference being that ``emplace``, ``insert`` and ``erase`` take an integer specifying the position to perform the operation instead of an iterator.

Lambda capture and ``LvArray::ArrayView``
-----------------------------------------
``LvArray::ArrayView`` is the parent class and the view class of ``LvArray::Array``. It shares the same template parameters as ``LvArray::Array`` except that the ``PERMUTATION`` type is replaced by an integer corresponding to the unit stride dimension (the last entry in the permutation).

There are multiple ways to create an ``LvArray::ArrayView`` from an ``LvArray::Array``. Because it is the parent class you can just create a reference to a ``LvArray::ArrayView`` from an existing ``LvArray::Array`` or by using the method ``toView``. ``LvArray::Array`` also has a user defined conversion operator to a ``LvArray::ArrayView`` with a const value type or you can use the ``toViewConst`` method.

The ``LvArray::ArrayView`` has a copy constructor, move constructor, copy assignment operator and move assignment operator all of which perform the same operation on the ``BUFFER_TYPE``. So ``ArrayView( ArrayView const & )`` calls ``BUFFER_TYPE< T >( BUFFER_TYPE< T > const & )`` and ``ArrayView::operator=( ArrayView && )`` calls ``BUFFER_TYPE< T >::operator=( BUFFER_TYPE< T > && )``. With the exception of the ``StackBuffer`` all of these operations are shallow copies.

.. literalinclude:: ../../examples/exampleArray.cpp
  :language: c++
  :start-after: // Sphinx start after arrayView
  :end-before: // Sphinx end before arrayView

*[Source: examples/exampleArray.cpp]*

An ``LvArray::ArrayView`` is almost always constructed from an existing ``LvArray::Array`` but it does has a default constructor which constructs an uninitialized ``ArrayView``. The only valid use of an uninitialized ``LvArray::ArrayView`` is to assign to it from an existing ``LvArray::ArrayView``. This behavior is primarily used when putting an ``LvArray::ArrayView`` into a container.

``LvArray::ArrayView`` supports the subset of operations on ``LvArray::Array`` that do not require reallocation or resizing. Every method in ``LvArray::ArrayView`` is ``const`` except the assignment operators. Most methods of are callable on device.

ArraySlice
----------
Up until now all the examples using ``operator[]`` have consumed all of the available indices, i.e. a two dimensional array has always had two immediate applications of ``operator[]`` to access a value. But what is the result of a single application of ``operator[]`` to a two dimensional array? Well it's a one dimensional ``LvArray::ArraySlice``! ``LvArray::ArraySlice`` shares the same template parameters as ``LvArray::ArrayView`` except that it doesn't need the ``BUFFER_TYPE`` parameter. ``LvArray::ArrayView`` is a feather weight object that consists only of three pointers: a pointer to the values, a pointer to the dimensions of the array and a pointer to the strides of the dimensions. ``LvArray::ArraySlice`` has shallow copy semantics but no assignment operators and every method is ``const``.

Unlike ``LvArray::Array`` and ``LvArray::ArrayView`` the ``LvArray::ArraySlice`` can refer to a non-contiguous set of values. As such it does not have a method ``data`` but instead has ``dataIfContiguous`` that performs a runtime check and aborts execution if the ``LvArray::ArraySlice`` does not refer to a contiguous set of values. Every method is ``const`` and callable on device.

.. literalinclude:: ../../examples/exampleArray.cpp
  :language: c++
  :start-after: // Sphinx start after arraySlice
  :end-before: // Sphinx end before arraySlice

*[Source: examples/exampleArray.cpp]*

.. note::
  A ``LvArray::ArraySlice`` should not be captured in a device kernel, even if the slice comes from an array that has be moved to the device. This is because ``LvArray::ArraySlice`` only contains a pointer to the dimension sizes and strides. Therefore when the slice is ``memcpy``'d to device the size and stride pointers will still point to host memory. This does not apply if you construct the slice manually and pass it device pointers but this is not a common use case.

Usage with ``LvArray::ChaiBuffer``
----------------------------------
When using the ``LvArray::ChaiBuffer`` as the buffer type the ``LvArray::Array`` can exist in multiple memory spaces. It can be explicitly moved between spaces with the method ``move`` and when the RAJA execution context is set the ``LvArray::ArrayView`` copy constructor will ensure that the newly constructed view's allocation is in the associated memory space.

.. literalinclude:: ../../examples/exampleArray.cpp
  :language: c++
  :start-after: // Sphinx start after chaiBuffer
  :end-before: // Sphinx end before chaiBuffer

*[Source: examples/exampleArray.cpp]*

Like the ``LvArray::ChaiBuffer`` a new allocation is created the first time the array is moved to a new space. Every time the array is moved between spaces it is touched in the new space unless the value type ``T`` is ``const`` or the array was moved with the ``move`` method and the optional ``touch`` parameter was set to ``false``. The data is only copied between memory spaces if the last space the array was touched in is different from the space to move to. The name associated with the Array can be set with ``setName``.

.. literalinclude:: ../../examples/exampleArray.cpp
  :language: c++
  :start-after: // Sphinx start after setName
  :end-before: // Sphinx end before setName

*[Source: examples/exampleArray.cpp]*

Output

::

  Moved    4.0 MB to the DEVICE: LvArray::Array<int, 2, camp::int_seq<long, 1l, 0l>, long, LvArray::ChaiBuffer> 
  Moved    4.0 MB to the HOST  : LvArray::Array<int, 2, camp::int_seq<long, 1l, 0l>, long, LvArray::ChaiBuffer> my_array

Interacting with an ``LvArray::Array`` of arbitrary dimension.
--------------------------------------------------------------
Often it can be useful to write a template function that can operate on an ``LvArray::Array`` with an arbitrary number of dimension. If the order the data is accessed in is not important you can use ``data()`` or the iterator interface ``begin`` and ``end``. For example you could write a function to sum up the values in an array as follows

.. literalinclude:: ../../examples/exampleArray.cpp
  :language: c++
  :start-after: // Sphinx start after sum int
  :end-before: // Sphinx end before sum int

*[Source: examples/exampleArray.cpp]*

However this same approach might not work as intended for floating point arrays because the order of additions and hence the answer would change with the data layout. To calculate the sum consistently you need to iterate over the multidimensional index space of the array. For an array with a fixed number of dimensions you could just nest the appropriate number of for loops. ``LvArray::forValuesInSlice`` defined in ``sliceHelpers.hpp`` solves this problem, it iterates over the values in a ``LvArray::ArraySlice`` in a consistent order regardless of the permutation. Using this ``sum`` could be written as

.. literalinclude:: ../../examples/exampleArray.cpp
  :language: c++
  :start-after: // Sphinx start after sum double
  :end-before: // Sphinx end before sum double

*[Source: examples/exampleArray.cpp]*

``sliceHelpers.hpp`` also provides a function ``forAllValuesInSliceWithIndices`` which calls the user provided callback with the indicies to each value in addition to the value. This can be used to easily copy data between arrays with different layouts.

.. literalinclude:: ../../examples/exampleArray.cpp
  :language: c++
  :start-after: // Sphinx start after copy
  :end-before: // Sphinx end before copy

*[Source: examples/exampleArray.cpp]*

Finally you can write a recursive function that operates on an ``LvArray::ArraySlice``. An example of this is the stream output method for ``LvArray::Array``.

.. literalinclude:: ../../src/output.hpp
  :language: c++
  :start-after: // Sphinx start after Array stream IO
  :end-before: // Sphinx end before Array stream IO

*[Source: src/output.hpp]*

Usage with ``LVARRAY_BOUNDS_CHECK``
-------------------------------------
When ``LVARRAY_BOUNDS_CHECK`` is defined all accesses via ``operator[]`` and ``operator()`` is checked to make sure the indices are valid. If invalid indices are detected an error message is printed to standard out and the program is aborted. It should be noted that access via ``operator()`` is able to provide a more useful error message upon an invalid access because it has access to all of the indices whereas ``operator[]`` only has access to a single index at a time. ``size( int dim )``, ``linearIndex``, ``emplace`` and ``insert`` will also check that their arguments are in bounds.

.. literalinclude:: ../../examples/exampleArray.cpp
  :language: c++
  :start-after: // Sphinx start after bounds check
  :end-before: // Sphinx end before bounds check

*[Source: examples/exampleArray.cpp]*

Guidelines
----------
In general you should opt to pass around the most restrictive array possible. If a function takes in an array and only needs to read and write the values then it should take in an ``LvArray::ArraySlice``. If it needs to read the values and captures the array in a kernel then it should take in an ``LvArray::ArrayView< T const, ... >``. Only when a function needs to resize or reallocate the array should it accept an ``LvArray::Array``.

Our benchmarks have shown no significant performance difference between using ``operator()`` and nested applications of ``operator[]``, if you do see a difference let us know!

Doxygen
-------
- `LvArray::Array <doxygen/html/class_lv_array_1_1_array.html>`_
- `LvArray::ArrayView <doxygen/html/class_lv_array_1_1_array_view.html>`_
- `LvArray::ArraySlice <doxygen/html/class_lv_array_1_1_array_slice.html>`_
