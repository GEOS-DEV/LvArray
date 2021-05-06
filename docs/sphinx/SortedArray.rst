.. ##
.. ## Copyright (c) 2021, Lawrence Livermore National Security, LLC
.. ## and LvArray project contributors. See the LICENCE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

########################
``LvArray::SortedArray``
########################

The ``LvArray::SortedArray`` functions similarly to a ``std::set`` except that unlike a ``std::set`` the values are stored contiguously in memory like a ``std::vector``. Like the ``std::set`` the cost of seeing if a ``LvArray::SortedArray`` ``a`` contains a value is :math:`O(log(N))` however the const of inserting or removing a value is :math:`O(N)` where ``N = a.size()``.

Template arguments
------------------
The ``LvArray::SortedArray`` requires three template arguments.

1. ``T``: The type of values stored in the set.
2. ``INDEX_TYPE``: An integral type used in index calculations, the suggested type is ``std::ptrdiff_t``.
3. ``BUFFER_TYPE``: A template template parameter specifying the buffer type used for allocation and de-allocation, the ``LvArray::SortedArray`` contains a ``BUFFER_TYPE< T >``.

.. note::
  Unlike ``std::set``, ``LvArray::SortedArray`` does not yet support a custom comparator, it us currently hard coded to ``operator<`` to sort values from least to greatest.

Creating and accessing a ``LvArray::SortedArray``
-------------------------------------------------
The ``LvArray::SortedArray`` has a single default constructor which creates an empty set. The only way to modify the set is through the methods ``insert`` and ``remove``. These methods are similar to the ``std::set`` methods of the same name except that when inserting or removing multiple values at once the values need to be sorted and unique. ``LvArray::SortedArray`` also has an ``operator[]`` and ``data`` method that provide read only access to the values.

.. literalinclude:: ../../examples/exampleSortedArray.cpp
  :language: c++
  :start-after: // Sphinx start after construction
  :end-before: // Sphinx end before construction

*[Source: examples/exampleSortedArray.cpp]*

``LvArray::SortedArrayView``
----------------------------
``LvArray::SortedArrayView`` is the view class of ``LvArray::SortedArray`` and it shares all the same template parameters. To construct a ``LvArray::SortedArrayView`` you must call ``toView()`` on an existing ``LvArray::SortedArray`` or create a copy of an existing ``LvArray::SortedArrayView``. The ``LvArray::SortedArrayView`` is not allowed to insert or remove values. As such the return type of ``LvArray::SortedArray< T, ... >::toView()`` is an ``LvArray::SortedArrayView< T const, ... >``

.. note::
  Unlike ``LvArray::ArrayView`` the ``LvArray::SortedArrayView`` does not yet support default construction.

Usage with ``LvArray::ChaiBuffer``
----------------------------------
When using the ``LvArray::ChaiBuffer`` as the buffer type the ``LvArray::SortedArray`` can exist in multiple memory spaces. It can be explicitly moved between spaces with the method ``move``. Because the ``SortedArrayView`` cannot modify the values the data is never touched when moving to device, even if the optional ``touch`` parameter is set to false.

It is worth noting that after a ``LvArray::SortedArray`` is moved to the device it must be explicitly moved back to the host by calling ``move( MemorySpace::host )`` before it can be safely modified. This won't actually trigger a memory copy since the values weren't touched on device, its purpose is to let CHAI know that the values were touched on the host so that the next time it is moved to device it will copy the values back over.

.. literalinclude:: ../../examples/exampleSortedArray.cpp
  :language: c++
  :start-after: // Sphinx start after ChaiBuffer
  :end-before: // Sphinx end before ChaiBuffer

*[Source: examples/exampleSortedArray.cpp]*

Usage with ``LVARRAY_BOUNDS_CHECK``
-------------------------------------
Like ``LvArray::Array`` when ``LVARRAY_BOUNDS_CHECK`` is defined access via ``operator[]`` is checked for invalid access. If an out of bounds access is detected the program is aborted. In addition calls to insert and remove multiple values will error out if the values to insert or remove aren't sorted and unique.

.. literalinclude:: ../../examples/exampleSortedArray.cpp
  :language: c++
  :start-after: // Sphinx start after bounds check
  :end-before: // Sphinx end before bounds check

*[Source: examples/exampleSortedArray.cpp]*

Guidelines
----------------
Batch insertion and removal is much faster than inserting or removing each value individually. For example calling ``a.insert( 5 )`` has complexity ``O( a.size() )`` but calling ``a.insert( first, last )`` only has complexity ``O( a.size() + std::distance( first, last ) )``. The function ``LvArray::sortedArrayManipulation::makeSortedUnique`` can help with this as it takes a range, sorts it and removes any duplicate values. When possible it is often faster to append values to a temporary container, sort the values, remove duplicates and then perform the operation.

.. literalinclude:: ../../examples/exampleSortedArray.cpp
  :language: c++
  :start-after: // Sphinx start after fast construction
  :end-before: // Sphinx end before fast construction

*[Source: examples/exampleSortedArray.cpp]*

Doxygen
-------
- `LvArray::SortedArray <doxygen/html/class_lv_array_1_1_sorted_array.html>`_
- `LvArray::SortedArrayView <doxygen/html/class_lv_array_1_1_sorted_array_view.html>`_
