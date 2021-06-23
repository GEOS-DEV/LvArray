.. ##
.. ## Copyright (c) 2021, Lawrence Livermore National Security, LLC
.. ## and LvArray project contributors. See the LICENCE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

###############################################################################
``LvArray::SparsityPattern`` and ``LvArray::CRSMatrix``
###############################################################################
``LvArray::SparsityPattern`` represents just the sparsity pattern of a matrix while the ``LvArray::CRSMatrix`` represents a sparse matrix. Both use a slightly modified version of the `compressed row storage`_ format. The modifications to the standard format are as follows:

#. The columns of each row are sorted.
#. Each row can have a capacity in addition to a size which means that the rows aren't necessarily adjacent in memory.

Template arguments
------------------
The ``LvArray::SparsityPattern`` has three template arguments

#. ``COL_TYPE``: The integral type used to enumerate the columns of the matrix.
#. ``INDEX_TYPE``: An integral type used in index calculations, the suggested type is ``std::ptrdiff_t``.
#. ``BUFFER_TYPE``: A template template parameter specifying the buffer type used for allocation and de-allocation, the ``LvArray::SparsityPattern`` contains a ``BUFFER_TYPE< COL_TYPE >`` along with two ``BUFFER_TYPE< INDEX_TYPE >``.

The ``LvArray::CRSMatrix`` adds an additional template argument ``T`` which is the type of the entries in the matrix. It also has an addition member of type ``BUFER_TYPE< T >``.

Usage
-----
The ``LvArray::SparsityPattern`` is just a ``LvArray::ArrayOfSets`` by a different name. The only functional difference is that it doesn't support inserting or removing rows from the matrix (inserting or removing inner sets). Both the ``LvArray::SparsityPattern`` and ``LvArray::CRSMatrix`` support ``insertNonZero`` and ``insertNonZeros`` for inserting entries into a row as well as ``removeNonZero`` and ``removeNonZeros`` for removing entries from a row. ``LvArray::CRSMatrix`` also supports various ``addToRow`` methods which will add to existing entries in a specific row.

It is worth noting that neither ``LvArray::SparsityPattern`` nor ``LvArray::CRSMatrix`` have an ``operator()`` or ``operator[]``. Instead they both support ``getColumns`` which returns a ``LvArray::ArraySlice< COL_TYPE const, 1, 0, INDEX_TYPE >`` with the columns of the row and ``LvArray::CRSMatrix`` supports ``getEntries`` which returns a ``LvArray::ArraySlice< T, 1, 0, INDEX_TYPE >`` with the entries of the row.

.. literalinclude:: ../../examples/exampleSparsityPatternAndCRSMatrix.cpp
  :language: c++
  :start-after: // Sphinx start after examples
  :end-before: // Sphinx end before examples

*[Source: examples/exampleSparsityPatternAndCRSMatrix.cpp]*

Third party packages such as MKL or cuSparse expect the rows of the CRS matrix to be adjacent in memory. Specifically they don't accept a pointer that gives the size of each row they only accept an offsets pointer and calculate the sizes from that. To make an ``LvArray::CRSMatrix`` conform to this layout you can call ``compress`` which will set the capacity of each row equal to its size making the rows adjacent in memory.

.. literalinclude:: ../../examples/exampleSparsityPatternAndCRSMatrix.cpp
  :language: c++
  :start-after: // Sphinx start after compress
  :end-before: // Sphinx end before compress

*[Source: examples/exampleSparsityPatternAndCRSMatrix.cpp]*

``LvArray::CRSMatrix`` also has an ``assimilate`` method which takes an r-values reference to a ``LvArray::SparsityPattern`` and converts it into a ``LvArray::CRSMatrix``. It takes a RAJA execution policy as a template parameter.

.. literalinclude:: ../../examples/exampleSparsityPatternAndCRSMatrix.cpp
  :language: c++
  :start-after: // Sphinx start after assimilate
  :end-before: // Sphinx end before assimilate

*[Source: examples/exampleSparsityPatternAndCRSMatrix.cpp]*

``LvArray::CRSMatrixView``
--------------------------
The ``LvArray::SparsityPatternView`` and ``LvArray::CRSMatrixView`` are the view counterparts to ``LvArray::SparsityPattern`` and ``LvArray::CRSMatrix``. They both have the same template arguments as their counterparts. The ``LvArray::SparsityPatternView`` behaves exactly like an ``LvArray::ArrayOfSetsView``. From a ``LvArray::CRSMatrix< T, COL_TYPE, INDEX_TYPE, BUFFER_TYPE >`` there are four view types you can create

- ``toView()`` returns a ``LvArray::CRSMatrixView< T, COL_TYPE, INDEX_TYPE const, BUFFER_TYPE >`` which can modify the entries as well as insert and remove columns from the rows as long as the size of each row doesn't exceed its capacity.
- ``toViewConstSizes()`` returns a ``LvArray::CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >`` which can modify existing values but cannot add or remove columns from the rows.
- ``toViewConst()`` returns a ``LvArray::CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >`` which provides read only access to both the columns and values of the rows.
- ``toSparsityPatternView()`` returns a ``LvArray::SparsityPatternView< COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >`` which provides read only access to the columns of the rows.

.. literalinclude:: ../../examples/exampleSparsityPatternAndCRSMatrix.cpp
  :language: c++
  :start-after: // Sphinx start after views
  :end-before: // Sphinx end before views

*[Source: examples/exampleSparsityPatternAndCRSMatrix.cpp]*

Usage with ``LvArray::ChaiBuffer``
----------------------------------
The three types of ``LvArray::CRSMatrixView`` obtainable from an ``LvArray::CRSMatrixs`` all act differently when moved to a new memory space.

- ``LvArray::CRSMatrixsView< T, COL_TYPE, INDEX_TYPE const, LvArray::ChaiBuffer >``, obtained by calling ``toView()``. When it is moved to a new space the values, columns and sizes are all touched. The offsets are not touched.
- ``LvArray::CRSMatrixsView< T, COL_TYPE const, INDEX_TYPE const, LvArray::ChaiBuffer >``, obtained by calling ``toViewConstSizes()``. When it is moved to a new space the values are touched but the columns, sizes and offsets aren't.
- ``LvArray::CRSMatrixsView< T const, COL_TYPE const, INDEX_TYPE const, LvArray::ChaiBuffer >``, obtained by calling ``toViewConst()``. None of the buffers are touched in the new space.

Calling the explicit ``move`` method with the touch parameter set to ``true`` on a view type has the behavior described above. However calling ``move( MemorySpace::host )`` on an ``LvArray::CRSMatrix`` or ``LvArray::SparsityPattern`` will also touch the offsets (if moving to the GPU the offsets aren't touched). This is the only way to touch the offsets so if an ``LvArray::CRSMatrix`` was previously on the device then it must be explicitly moved and touched on the host before any modification to the offsets can safely take place.

Usage with ``LVARRAY_BOUNDS_CHECK``
-------------------------------------
When ``LVARRAY_BOUNDS_CHECK`` is defined access all row and column access is checked. Methods which expect a sorted unique set of columns check that the columns are indeed sorted and unique. In addition if ``addToRow`` checks that all the given columns are present in the row.

Guidelines
----------
As with all the ``LvArray`` containers it is important to pass around the most restrictive form. A function should only accept a ``LvArray::CRSMatrix`` if it needs to resize the matrix or might bust the capacity of a row. If a function only needs to be able to modify existing entries it should accept a ``LvArray::CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >``. If a function only needs to examine the sparsity pattern of the matrix it should accept a ``LvArray::SparsityPatternView< COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >``.

Like the ``LvArray::ArrayOfArrays`` and ``LvArray::ArrayOfSets`` when constructing an ``LvArray::SparsityPattern`` or ``LvArray::CRSMatrix`` it is important to preallocate space for each row in order to achieve decent performance.

A common pattern with sparse matrices is that the sparsity pattern need only be assembled once but the matrix is used multiple times with different values. For example when using the finite element method on an unstructured mesh you can generate the sparsity pattern once at the beginning of the simulation and then each time step you repopulate the entries of the matrix. When this is the case it is usually best to do the sparsity generation with a ``LvArray::SparsityPattern`` and then assimilate that into a ``LvArray::CRSMatrix``. Once you have a matrix with the proper sparsity pattern create a ``LvArray::CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >`` via ``toViewConstSizes()`` and you can then assemble into the matrix in parallel with the ``addToRow`` methods. Finally before beginning the next iteration you can zero out the entries in the matrix by calling ``setValues``.

.. _`compressed row storage`: https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)

Doxygen
-------
- `LvArray::SparsityPattern <doxygen/html/class_lv_array_1_1_sparsity_pattern.html>`_
- `LvArray::SparsityPatternView <doxygen/html/class_lv_array_1_1_sparsity_pattern_view.html>`_
- `LvArray::CRSMatrix <doxygen/html/class_lv_array_1_1_c_r_s_matrix.html>`_
- `LvArray::CRSMatrixView <doxygen/html/class_lv_array_1_1_c_r_s_matrix_view.html>`_
