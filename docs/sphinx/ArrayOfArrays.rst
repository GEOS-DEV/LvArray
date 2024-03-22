.. ##
.. ## Copyright (c) 2021, Lawrence Livermore National Security, LLC
.. ## and LvArray project contributors. See the LICENCE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

###############################################################################
``LvArray::ArrayOfArrays``
###############################################################################

The ``LvArray::ArrayOfArrays`` provides functionality similar to a ``std::vector< std::vector< T > >`` but with just three allocations. The primary purpose is to reduce the number of memory transfers when moving between memory spaces but it also comes with the added benefit of reduced memory fragmentation.

Template arguments
------------------
The ``LvArray::ArrayOfArrays`` requires three template arguments.

#. ``T``: The type of values stored in the inner arrays.
#. ``INDEX_TYPE``: An integral type used in index calculations, the suggested type is ``std::ptrdiff_t``.
#. ``BUFFER_TYPE``: A template template parameter specifying the buffer type used for allocation and de-allocation, the ``LvArray::ArrayOfArrays`` contains a ``BUFFER_TYPE< T >`` along with two ``BUFFER_TYPE< INDEX_TYPE >``.

Usage
-----
``LvArray::ArrayOfArrays`` has a single constructor which takes two optional parameters; the number of inner arrays and the capacity to allocate for each inner array. Both values default to zero.

Given an ``ArrayOfArrays< T, ... > array``, an equivalent ``std::vector< std::vector< T > > vector``, a non negative integer ``n``, integer ``i`` such that ``0 <= i < vector.size()``, integer ``j`` such that ``0 <= j < vector[ i ].size()``, two iterators ``first`` and ``last`` and a variadic pack of parameters ``... args`` then following are equivalent:

==================================== ===================================
Getting information about the outer array or a specific inner array
------------------------------------------------------------------------
``LvArray::ArrayOfArrays< T, ... >`` ``std::vector< std::vector< T > >``
==================================== ===================================
``array.size()``                     ``vector.size()``
``array.capacity()``                 ``vector.capacity()``
``array.sizeOfArray( i )``           ``vector[ i ].size()``
``array.capacityOfArray( i )``       ``vector[ i ].capacity()``
==================================== ===================================


======================================= ========================================================================
Modifying the outer array
----------------------------------------------------------------------------------------------------------------
``LvArray::ArrayOfArrays< T, ... >``    ``std::vector< std::vector< T > >``
======================================= ========================================================================
``array.reserve( n )``                  ``vector.reserve( n )``
``array.resize( n )``                   ``vector.resize( n )``
``array.appendArray( n )``              ``vector.push_back( std::vector< T >( n ) )``
``array.appendArray( first, last )``    ``vector.push_back( std::vector< T >( first, last ) )``
``array.insertArray( i, first, last )`` ``vector.insert( vector.begin() + i, std::vector< T >( first, last ) )``
``array.eraseArray( i )``               ``vector.erase( vector.first() + i )``
======================================= ========================================================================


=============================================== ==============================================================================
Modifying an inner array
------------------------------------------------------------------------------------------------------------------------------
``LvArray::ArrayOfArrays< T, ... >``            ``std::vector< std::vector< T > >``
=============================================== ==============================================================================
``array.resizeArray( i, n, args ... )``         ``vector[ i ].resize( n, T( args ... ) )``
``array.clearArray( i )``                       ``vector[ i ].clear()``
``array.emplaceBack( i, args ... )``            ``vector[ i ].emplace_back( args ... )``
``array.appendToArray( i, first, last )``       ``vector[ i ].insert( vector[ i ].end(), first, last )``
``array.emplace( i, j , args ... )``            ``vector[ i ].emplace( j, args ... )```
``array.insertIntoArray( i, j , first, last )`` ``vector[ i ].insert( vector[ i ].begin() + j, first, last )``
``eraseFromArray( i, j, n )``                   ``vector[ i ].erase( vector[ i ].begin() + j, vector[ i ].begin() + j + n )``.
=============================================== ==============================================================================


==================================== ===================================
Accessing the data
------------------------------------------------------------------------
``LvArray::ArrayOfArrays< T, ... >`` ``std::vector< std::vector< T > >``
==================================== ===================================
``array( i, j )``                    ``vector[ i ][ j ]``
``array[ i ][ j ]``                  ``vector[ i ][ j ]``
==================================== ===================================

It is worth noting that ``operator[]`` returns a one dimensional ``LvArray::ArraySlice``.

.. literalinclude:: ../../examples/exampleArrayOfArrays.cpp
  :language: c++
  :start-after: // Sphinx start after examples
  :end-before: // Sphinx end before examples

*[Source: examples/exampleArrayOfArrays.cpp]*

``LvArray::ArrayOfArraysView``
------------------------------
``LvArray::ArrayOfArraysView`` is the view counterpart to ``LvArray::ArrayOfArrays``. The ``LvArray::ArrayOfArraysView`` shares the three template parameters of ``LvArray::ArrayOfArraysView`` but it has an additional boolean parameter ``CONST_SIZES`` which specifies if the view can change the sizes of the inner arrays. If ``CONST_SIZES`` is true the ``sizes`` buffer has type ``BUFFER_TYPE< INDEX_TYPE const >`` otherwise it has type ``BUFFER_TYPE< std::remove_const_t< INDEX_TYPE > >``.

The ``LvArray::ArrayOfArraysView`` doesn't have a default constructor, it must always be created from an existing ``LvArray::ArrayOfArrays``. From a ``LvArray::ArrayOfArrays< T, INDEX_TYPE, BUFFER_TYPE >`` you can get three different view types. 

- ``toView()`` returns a ``LvArray::ArrayOfArraysView< T, INDEX_TYPE const, false, BUFFER_TYPE >`` which can modify existing values and change the size of the inner arrays as long as their size doesn't exceed their capacity. - 
- ``toViewConstSizes()`` returns a ``LvArray::ArrayOfArraysView< T, INDEX_TYPE const, true, BUFFER_TYPE >`` which can modify existing values but cannot change the size of the inner arrays. 
- ``toViewConst()`` returns a ``LvArray::ArrayOfArraysView< T const, INDEX_TYPE const, true, BUFFER_TYPE >`` which provides read only access.

.. literalinclude:: ../../examples/exampleArrayOfArrays.cpp
  :language: c++
  :start-after: // Sphinx start after view
  :end-before: // Sphinx end before view

*[Source: examples/exampleArrayOfArrays.cpp]*

``LvArray::ArrayView`` also has a method ``emplaceBackAtomic`` which is allows multiple threads to append to a single inner array in a thread safe manner.

.. literalinclude:: ../../examples/exampleArrayOfArrays.cpp
  :language: c++
  :start-after: // Sphinx start after atomic
  :end-before: // Sphinx end before atomic

*[Source: examples/exampleArrayOfArrays.cpp]*

Data structure
--------------
In order to use the ``LvArray::ArrayOfArrays`` efficiently it is important to understand the underlying data structure. A ``LvArray::ArrayOfArrays< T, INDEX_TYPE, BUFFER_TYPE > array`` contains three buffers:

#. ``BUFFER_TYPE< T > values``: Contains the values of each inner array.
#. ``BUFFER_TYPE< INDEX_TYPE > sizes``: Of length ``array.size()``, ``sizes[ i ]`` contains the size of inner array ``i``.
#. ``BUFFER_TYPE< INDEX_TYPE > offsets``: Of length ``array.size() + 1``, inner array ``i`` begins at ``values[ offsets[ i ] ]`` and has capacity ``offsets[ i + 1 ] - offsets[ i ]``.

Given ``M = offsets[ array.size() ]`` which is the sum of the capacities of the inner arrays then 

- ``array.appendArray( n )`` is O( n ) because it entails appending a new entry to ``sizes`` and ``offsets`` and then appending the ``n`` new values to ``values``.
- ``array.insertArray( i, first, last )`` is ``O( array.size() + M + std::distance( first, last )``. It involves inserting an entry into ``sizes`` and ``offsets`` which is ``O( array.size() )``, making room in ``values`` for the new array which is ``O( M )`` and finally copying over the new values which is ``O( std::distance( first, last ) )``.
- ``array.eraseArray( i )`` is ``O( array.size() + M )``. It involves removing an entry from ``sizes`` and ``offsets`` which is ``O( array.size() )`` and then it involves shifting the entries in ``values`` down which is ``O( M )``.
- The methods which modify an inner array have the same complexity as their ``std::vector`` counterparts **if** the capacity of the inner array won't be exceeded by the operation. Otherwise they have an added cost of ``O( M )`` because new space will have to be made in ``values``. So for example ``array.emplaceBack( i, args ... )`` is ``O( 1 )`` if ``array.sizeOfArray( i ) < array.capacityOfArray( i )`` and ``O( M )`` otherwise.

``LvArray::ArrayOfArrays`` also supports two methods that don't have an ``std::vector< std::vector >`` counterpart. The first is ``compress`` which shrinks the capacity of each inner array to match it's size. This ensures that the inner arrays are contiguous in memory with no extra space between them.

.. literalinclude:: ../../examples/exampleArrayOfArrays.cpp
  :language: c++
  :start-after: // Sphinx start after compress
  :end-before: // Sphinx end before compress

*[Source: examples/exampleArrayOfArrays.cpp]*

The second is ``resizeFromCapacities`` which takes in a number of inner arrays and the capacity of each inner array. It clears the ``ArrayOfArrays`` and reconstructs it with the given number of empty inner arrays each with the provided capacity. It also takes a RAJA execution policy as a template parameter which specifies how to compute the offsets array from the capacities. Currently only host execution policies are supported.

.. literalinclude:: ../../examples/exampleArrayOfArrays.cpp
  :language: c++
  :start-after: // Sphinx start after resizeFromCapacities
  :end-before: // Sphinx end before resizeFromCapacities

*[Source: examples/exampleArrayOfArrays.cpp]*

Usage with ``LvArray::ChaiBuffer``
----------------------------------
The three types of ``LvArray::ArrayOfArrayView`` obtainable from an ``LvArray::ArrayOfArrays`` all act differently when moved to a new memory space.

- ``LvArray::ArrayOfArraysView< T, INDEX_TYPE const, false, LvArray::ChaiBuffer >``, obtained by calling ``toView()``. When it is moved to a new space the values are touched as well as the sizes. The offsets are not touched.
- ``LvArray::ArrayOfArraysView< T, INDEX_TYPE const, true, LvArray::ChaiBuffer >``, obtained by calling ``toViewConstSizes()``. When it is moved to a new space the values are touched but the sizes and offsets aren't.
- ``LvArray::ArrayOfArraysView< T const, INDEX_TYPE const, true, LvArray::ChaiBuffer >``, obtained by calling ``toViewConst()``. None of the buffers are touched in the new space.

Calling the explicit ``move`` method with the touch parameter set to ``true`` on a view type has the behavior described above. However calling ``move( MemorySpace::host )`` on an ``LvArray::ArrayOfArrays`` will also touch the offsets (if moving to the GPU the offsets aren't touched). This is the only way to touch the offsets so if an ``LvArray::ArrayOfArrays`` was previously on the device then it must be explicitly moved and touched on the host before any modification to the offsets can safely take place.

.. literalinclude:: ../../examples/exampleArrayOfArrays.cpp
  :language: c++
  :start-after: // Sphinx start after ChaiBuffer
  :end-before: // Sphinx end before ChaiBuffer

*[Source: examples/exampleArrayOfArrays.cpp]*

Usage with ``LVARRAY_BOUNDS_CHECK``
-------------------------------------
When ``LVARRAY_BOUNDS_CHECK`` is defined access via ``operator[]`` and ``operator()`` is checked. If an invalid access is detected the program is aborted. Methods such as ``sizeOfArray``, ``insertArray`` and ``emplace`` are also checked.

.. literalinclude:: ../../examples/exampleArrayOfArrays.cpp
  :language: c++
  :start-after: // Sphinx start after bounds check
  :end-before: // Sphinx end before bounds check

*[Source: examples/exampleArrayOfArrays.cpp]*

Guidelines
----------
Like with the ``LvArray::Array`` it is a good idea to pass around the most restrictive ``LvArray::ArrayOfArrays`` object possible. If a function only reads the values of an array it should accept an ``LvArray::ArrayOfArraysView< T const, INDEX_TYPE const, true, BUFFER_TYPE >``. If it only reads and writes to the values of the inner arrays it should accept an ``LvArray::ArrayOfArraysView< T, INDEX_TYPE const, true, BUFFER_TYPE >``. If it will modify the size of the inner arrays and can guarantee that their capacity won't be exceeded it should accept a ``LvArray::ArrayOfArraysView< T, INDEX_TYPE const, false, BUFFER_TYPE >``. Only when the function needs to modify the outer array or can't guarantee that the capacity of an inner array won't be exceeded should it accept an ``LvArray::ArrayOfArrays``.

Examining the computational complexities listed above resizing the inner arrays of an ``LvArray::ArrayOfArrays`` can be as fast as the equivalent operations on a ``std::vector< std::vector< T > >`` only if the capacity of the inner arrays is not exceeded. Whenever possible preallocate space for the inner arrays. The following examples demonstrate the impact this can have.

One use case for the ``LvArray::ArrayOfArrays`` is to represent the node-to-element map of a computational mesh. Usually this is constructed from a provided element-to-node map which is a map from an element index to a list of node indices that make up the element. For a structured mesh or an unstructured mesh made up of a single element type this map can be represented as a two dimensional ``LvArray::Array`` since every element has the same number of nodes. However the inverse node-to-element map cannot be easily represented this way since in general not all nodes are a part of the same number of elements. In a two dimensional structured mesh an interior node is part of four elements while the four corner nodes are only part of a single element.

This is an example of how to construct the node-to-element map represented as a ``std::vector< std::vector >`` from the element-to-node map.

.. literalinclude:: ../../benchmarks/benchmarkArrayOfArraysNodeToElementMapConstructionKernels.cpp
  :language: c++
  :start-after: // Sphinx start after vector
  :end-before: // Sphinx end before vector

*[Source: benchmarks/benchmarkArrayOfArraysNodeToElementMapConstructionKernels.cpp]*

For a mesh with ``N`` nodes this construction has complexity ``O( N )`` since there are ``N`` calls to ``std::vector::emplace_back`` each of which is ``O( 1 )``. The same approach works when the node-to-element map is a ``LvArray::ArrayOfArrays`` however the complexity is much higher.

.. literalinclude:: ../../benchmarks/benchmarkArrayOfArraysNodeToElementMapConstructionKernels.cpp
  :language: c++
  :start-after: // Sphinx start after naive
  :end-before: // Sphinx end before naive

*[Source: benchmarks/benchmarkArrayOfArraysNodeToElementMapConstructionKernels.cpp]*

Since nothing is done to preallocate space for each inner array every call to ``LvArray::ArrayOfArrays::emplaceBack`` has complexity ``O( M )`` where ``M`` is the sum of the capacities of the inner arrays. ``M`` is proportional to the number of nodes in the mesh ``N`` so the entire algorithm runs in ``O( N * N )``.

This can be sped up considerably with some preallocation.

.. literalinclude:: ../../benchmarks/benchmarkArrayOfArraysNodeToElementMapConstructionKernels.cpp
  :language: c++
  :start-after: // Sphinx start after overAllocation
  :end-before: // Sphinx end before overAllocation

*[Source: benchmarks/benchmarkArrayOfArraysNodeToElementMapConstructionKernels.cpp]*

Since this method guarantees that the capacity of each inner arrays won't be exceeded the complexity is reduced back down to ``O( N )``. In addition the loop appending to the inner arrays can be parallelized.

One problem with this approach is that it may allocate significantly more memory than is necessary to store the map since most of the inner arrays may not have the maximal length. Another potential issue is that the array is not compressed since some inner arrays will have a capacity that exceeds their size. Precomputing the size of each inner array (the number of elements each node is a part of) and using ``resizeFromCapacities`` solves both of these problems and is almost as fast as over allocating.

.. literalinclude:: ../../benchmarks/benchmarkArrayOfArraysNodeToElementMapConstructionKernels.cpp
  :language: c++
  :start-after: // Sphinx start after resizeFromCapacities
  :end-before: // Sphinx end before resizeFromCapacities

*[Source: benchmarks/benchmarkArrayOfArraysNodeToElementMapConstructionKernels.cpp]*

The following timings are from a clang 10 release build on LLNL's Quartz system run with a ``200 x 200 x 200`` element structured mesh. The reported time is the best of ten iterations.

======================== ========================= ======
Function                 RAJA Policy               Time
======================== ========================= ======
``vector``               N/A                       0.99s
``overAllocation``       ``seq_exec``             0.49s
``resizeFromCapacities`` ``seq_exec``             0.58s
``overAllocation``       ``omp_parallel_for_exec`` 0.11s
``resizeFromCapacities`` ``omp_parallel_for_exec`` 0.17s
======================== ========================= ======

The ``naive`` method is much to slow to run on this size mesh. However on a ``30 x 30 x 30`` mesh it takes 1.28 seconds.

Doxygen
-------
- `LvArray::ArrayOfArrays <doxygen/html/class_lv_array_1_1_array_of_arrays.html>`_
- `LvArray::ArrayOfArraysView <doxygen/html/class_lv_array_1_1_array_of_arrays_view.html>`_
