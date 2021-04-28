.. ##
.. ## Copyright (c) 2021, Lawrence Livermore National Security, LLC
.. ## and LvArray project contributors. See the LICENCE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

###############################################################################
``LvArray::ArrayOfSets``
###############################################################################

The ``LvArray::ArrayOfSets`` is very similar to the ``LvArray::ArrayOfArrays`` except that the values of the inner arrays are sorted an unique like the ``LvArray::SortedArray``. If you are familiar with both of these classes the functionality of the ``LvArray::ArrayOfSets`` should be pretty straightforward.

Template arguments
------------------
The ``LvArray::ArrayOfArrays`` requires three template arguments.

#. ``T``: The type of values stored in the inner arrays.
#. ``INDEX_TYPE``: An integral type used in index calculations, the suggested type is ``std::ptrdiff_t``.
#. ``BUFFER_TYPE``: A template template parameter specifying the buffer type used for allocation and de-allocation, the ``LvArray::ArrayOfArrays`` contains a ``BUFFER_TYPE< T >`` along with two ``BUFFER_TYPE< INDEX_TYPE >``.

Usage
-----
All of the functionality for modifying the outer array from ``LvArray::ArrayOfArrays`` is present in the ``LvArray::ArrayOfSets`` although it might go by a different name. For example instead of ``appendArray`` there is ``appendSet``. However like ``LvArray::SortedArray`` the only options for modifying the inner sets are through either ``insertIntoSet` or ``removeFromSet``.

.. literalinclude:: ../../examples/exampleArrayOfSets.cpp
  :language: c++
  :start-after: // Sphinx start after examples
  :end-before: // Sphinx end before examples

*[Source: examples/exampleArrayOfSets.cpp]*

``LvArray::ArrayOfSets`` also has a method ``assimilate`` which takes an rvalue-reference to an ``LvArray::ArrayOfArrays`` and converts it to a ``LvArray::ArrayOfSets``. ``LvArray::ArrayOfArrays`` also has a similar method.

.. literalinclude:: ../../examples/exampleArrayOfSets.cpp
  :language: c++
  :start-after: // Sphinx start after assimilate
  :end-before: // Sphinx end before assimilate

*[Source: examples/exampleArrayOfSets.cpp]*


``LvArray::ArrayOfSetsView``
----------------------------
The ``LvArray::ArrayOfSetsView`` is the view counterpart to ``LvArray::ArrayOfSets``. Functionally it is very similar to the ``LvArray::ArrayOfArraysView`` in that it cannot modify the outer array but it can modify the inner sets as long as their capacities aren't exceeded. Unlike ``LvArray::ArrayOfArraysView`` however it doesn't have the ``CONST_SIZES`` template parameter. This is because if you can't change the size of the inner arrays then you also can't modify their values. Specifically an ``LvArray::ArrayOfSetsView< T, INDEX_TYPE, BUFFER_TYPE >`` contains the following buffers:

#. ``BUFFER_TYPE< T > values``: Contains the values of each inner set.
#. ``BUFFER-TYPE< std::conditional_t< std::is_const< T >::value, INDEX_TYPE const, INDEX_TYPE > >``: Of length ``array.size()``, ``sizes[ i ]`` contains the size of the inner set ``i``.
#. ``BUFFER_TYPE< INDEX_TYPE > offsets``: Of length ``array.size() + 1``, inner set ``i`` begins at ``values[ offsets[ i ] ]`` and has capacity ``offsets[ i + 1 ] - offsets[ i ]``.

From an ``LvArray::ArrayOfSets< T, INDEX_TYPE, BUFFER_TYPE >`` you can get three view types by calling the following methods

- ``toView()`` returns an ``LvArray::ArrayOfSetsView< T, INDEX_TYPE const, BUFFER_TYPE >``.
- ``toViewConst()`` returns an ``LvArray::ArrayOfSetsView< T const, INDEX_TYPE const, BUFFER_TYPE >``.
- ``toArrayOfArraysView()`` returns an ``LvArray::ArrayOfArraysView< T const, INDEX_TYPE const, true, BUFFER_TYPE >``.

Usage with ``LvArray::ChaiBuffer``
----------------------------------
The two types of ``LvArray::ArrayOfSetsView`` obtainable from an ``LvArray::ArrayOfSets`` act differently when moved to a new memory space.

- ``LvArray::ArrayOfSetsView< T, INDEX_TYPE const, LvArray::ChaiBuffer >``, obtained by calling ``toView()``. When it is moved to a new space the values are touched as well as the sizes. The offsets are not touched.
- ``LvArray::ArrayOfSetsView< T const, INDEX_TYPE const, LvArray::ChaiBuffer >``, obtained by calling ``toViewConst()``. None of the buffers are touched in the new space.

Calling the explicit ``move`` method with the touch parameter set to ``true`` on a view type has the behavior described above. However calling ``move( MemorySpace::host )`` on an ``LvArray::ArrayOfSets`` will also touch the offsets (if moving to the GPU the offsets aren't touched). This is the only way to touch the offsets so if an ``LvArray::ArrayOfSets`` was previously on the device then it must be explicitly moved and touched on the host before any modification to the offsets can safely take place.

Usage with ``LVARRAY_BOUNDS_CHECK``
-------------------------------------
When ``LVARRAY_BOUNDS_CHECK`` is defined access via ``operator[]`` and ``operator()`` is checked. If an invalid access is detected the program is aborted. Methods such as ``sizeOfArray``, ``insertArray`` and ``emplace`` are also checked. The values passed to ``insertIntoSet`` and ``removeFromSet`` are also checked to ensure they are sorted and contain no duplicates.

Guidelines
----------
Like ``LvArray::SortedArray`` batch insertion and removal from an inner set is much faster than inserting or removing each value individually.

All the tips for efficiently constructing an ``LvArray::ArrayOfArrays`` apply to constructing an ``LvArray::ArrayOfSets``. The main difference is that ``LvArray::ArrayOfSets`` doesn't support concurrent modification of an inner set. Often if the sorted-unique properties of the inner sets aren't used during construction it can be faster to first construct a ``LvArray::ArrayOfArrays`` where each inner array can contain duplicates and doesn't have to be sorted and then create the ``LvArray::ArrayOfSets`` via a call to ``assimilate``.

Doxygen
-------
- `LvArray::ArrayOfSets <doxygen/html/class_lv_array_1_1_array_of_sets.html>`_
- `LvArray::ArrayOfSetsView <doxygen/html/class_lv_array_1_1_array_of_sets_view.html>`_
