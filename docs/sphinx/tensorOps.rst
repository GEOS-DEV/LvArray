.. ##
.. ## Copyright (c) 2021, Lawrence Livermore National Security, LLC
.. ## and LvArray project contributors. See the LICENCE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

###############################################################################
``LvArray::tensorOps``
###############################################################################

``LvArray::tensorOps`` is a collection of free template functions which perform common linear algebra operations on compile time sized matrices and vectors. All of the functions work on device.

Object Representation
---------------------
The ``LvArray::tensorOps`` methods currently operate on four different types of object; scalars, vectors, matrices, and symmetric matrices.

Vectors are represented by either a one dimensional c-array or a one dimensional ``LvArray::Array`` object. Examples of acceptable vector types are

- ``double[ 4 ]``
- ``int[ 55 ]``
- ``LvArray::Array< float, 1, camp::idx_seq< 0 >, std::ptrdiff_t, LvArray::MallocBuffer >``
- ``LvArray::ArrayView< long, 1, 0, std::ptrdiff_t, LvArray::MallocBuffer >``
- ``LvArray::ArraySlice< double, 1, 0, std::ptrdiff_t >``
- ``LvArray::ArraySlice< double, 1, -1, std::ptrdiff_t >``

Matrices are represented by either a two dimensional c-array or a two dimensional ``LvArray::Array`` object. Examples of acceptable matrix types

- ``double[ 3 ][ 3 ]``
- ``int[ 5 ][ 2 ]``
- ``LvArray::Array< float, 2, camp::idx_seq< 0, 1 >, std::ptrdiff_t, LvArray::MallocBuffer >``
- ``LvArray::ArrayView< long, 2, 0, std::ptrdiff_t, LvArray::MallocBuffer >``: 
- ``LvArray::ArraySlice< double, 2, 1, std::ptrdiff_t >``
- ``LvArray::ArraySlice< double, 2, 0, std::ptrdiff_t >``
- ``LvArray::ArraySlice< double, 2, -1, std::ptrdiff_t >``

Symmetric matrices are represented in `Voigt notation`_ as a vector. This means that a :math:`2 \times 2` symmetric matrix is represented as vector of length three as :math:`[a_{00}, a_{11}, a_{01}]` and that a :math:`3 \times 3` symmetric matrix is represented as vector of length six as :math:`[a_{00}, a_{11}, a_{22}, a_{12}, a_{02}, a_{01}]`. One consequence of this is that you can use the vector methods to for example add one symmetric matrix to another.

Common operations
-----------------

====================================================== ========================================================
Variables
---------------------------------------------------------------------------------------------------------------
:math:`\alpha`: A scalar.
---------------------------------------------------------------------------------------------------------------
:math:`x`: A vector in :math:`\mathbb{R}^m \times \mathbb{R}^1`.
---------------------------------------------------------------------------------------------------------------
:math:`y`: A vector in :math:`\mathbb{R}^m \times \mathbb{R}^1`.
---------------------------------------------------------------------------------------------------------------
:math:`z`: A vector in :math:`\mathbb{R}^m \times \mathbb{R}^1`.
---------------------------------------------------------------------------------------------------------------
:math:`\mathbf{A}`: A matrix in :math:`\mathbb{R}^m \times \mathbb{R}^n`.
---------------------------------------------------------------------------------------------------------------
:math:`\mathbf{B}`: A matrix in :math:`\mathbb{R}^m \times \mathbb{R}^n`.
---------------------------------------------------------------------------------------------------------------
Operation                                              Function 
====================================================== ========================================================
:math:`x \leftarrow y`                                 ``tensorOps::copy< m >( x, y )``
:math:`\mathbf{A} \leftarrow \mathbf{B}`               ``tensorOps::copy< m, n >( A, B )``
:math:`x \leftarrow \alpha x`                          ``tensorOps::scale< m >( x, alpha )``
:math:`\mathbf{A} \leftarrow \alpha \mathbf{A}`        ``tensorOps::scale< m, n >( A, alpha )``
:math:`x \leftarrow \alpha y`                          ``tensorOps::scaledCopy< m >( x, y, alpha )``
:math:`\mathbf{A} \leftarrow \alpha \mathbf{B}`        ``tensorOps::scaledCopy< m, n >( A, B, alpha )``
:math:`x \leftarrow x + y`                             ``tensorOps::add< m >( x, y )``
:math:`\mathbf{A} \leftarrow \mathbf{A} + \mathbf{B}`  ``tensorOps::add< m, n >( A, B, alpha )``
:math:`x \leftarrow x - y`                             ``tensorOps::subtract< m >( x, y )``
:math:`x \leftarrow x + \alpha y`                      ``tensorOps::scaledAdd< m >( x, y, alpha )``
:math:`x \leftarrow y \circ z`                         ``tensorOps::hadamardProduct< m >( x, y, z )``
====================================================== ========================================================

There is also a function ``fill`` that will set all the entries of the object to a specific value. For a vector it is called as ``tensorOps::fill< n >( x )`` and for a matrix as ``tensorOps::fill< m, n >( A )``.

Vector operations
------------------

==================================== =========================================
Variables
------------------------------------------------------------------------------
:math:`\alpha`: A scalar.
------------------------------------------------------------------------------
:math:`x`: A vector in :math:`\mathbb{R}^m \times \mathbb{R}^1`.
------------------------------------------------------------------------------
:math:`y`: A vector in :math:`\mathbb{R}^m \times \mathbb{R}^1`.
------------------------------------------------------------------------------
:math:`z`: A vector in :math:`\mathbb{R}^m \times \mathbb{R}^1`.
------------------------------------------------------------------------------
Operation                            Function 
==================================== =========================================
:math:`|x|_\infty`                   ``tensorOps::maxAbsoluteEntry< m >( x )``
:math:`|x|_2^2`                      ``tensorOps::l2NormSquared< m >( x )``
:math:`|x|_2`                        ``tensorOps::l2Norm< m >( x )``
:math:`x \leftarrow \hat{x}`         ``tensorOps::normalize< m >( x )``
:math:`x^T y`                        ``tensorOps::AiBi( x, y )``
==================================== =========================================

If :math:`x`, :math:`y` and :math:`z` are all in :math:`\mathbb{R}^3 \times \mathbb{R}^1` then you can perform the operation :math:`x \leftarrow y \times z` as ``tensorOps::crossProduct( x, y, z )``.

Matrix vector operations
------------------------

====================================================== ===========================================
Variables
--------------------------------------------------------------------------------------------------
:math:`x`: A vector in :math:`\mathbb{R}^m \times \mathbb{R}^1`.
--------------------------------------------------------------------------------------------------
:math:`y`: A vector in :math:`\mathbb{R}^n \times \mathbb{R}^1`.
--------------------------------------------------------------------------------------------------
:math:`\mathbf{A}`: A matrix in :math:`\mathbb{R}^m \times \mathbb{R}^n`.
--------------------------------------------------------------------------------------------------
Operation                                              Function 
====================================================== ===========================================
:math:`\mathbf{A} \leftarrow x y^T`                    ``tensorOps::Rij_eq_AiBj< m, n >( A, x, y )``
:math:`\mathbf{A} \leftarrow \mathbf{A} + x y^T`       ``tensorOps::Rij_add_AiBj< m, n >( A, x, y )``
:math:`x \leftarrow \mathbf{A} y`                      ``tensorOps::Ri_eq_AijBj< m, n >( x, A, y )``
:math:`x \leftarrow x + \mathbf{A} y`                  ``tensorOps::Ri_add_AijBj< m, n >( x, A, y )``
:math:`y \leftarrow \mathbf{A}^T x`                    ``tensorOps::Ri_eq_AjiBj< n, m >( y, A, x )``
:math:`y \leftarrow y + \mathbf{A}^T x`                ``tensorOps::Ri_add_AjiBj< n, m >( y, A, x )``
====================================================== ===========================================

Matrix operations
-----------------

=================================================================== ===============================================
Variables
-------------------------------------------------------------------------------------------------------------------
:math:`\mathbf{A}`: A matrix in :math:`\mathbb{R}^m \times \mathbb{R}^n`.
-------------------------------------------------------------------------------------------------------------------
:math:`\mathbf{B}`: A matrix in :math:`\mathbb{R}^m \times \mathbb{R}^p`.
-------------------------------------------------------------------------------------------------------------------
:math:`\mathbf{C}`: A matrix in :math:`\mathbb{R}^p \times \mathbb{R}^n`.
-------------------------------------------------------------------------------------------------------------------
:math:`\mathbf{D}`: A matrix in :math:`\mathbb{R}^n \times \mathbb{R}^m`.
-------------------------------------------------------------------------------------------------------------------
:math:`\mathbf{E}`: A matrix in :math:`\mathbb{R}^m \times \mathbb{R}^m`.
-------------------------------------------------------------------------------------------------------------------
Operation                                              Function 
=================================================================== ===============================================
:math:`\mathbf{A} \leftarrow \mathbf{D}^T`                          ``tensorOps::transpose< m, n >( A, D )``
:math:`\mathbf{A} \leftarrow \mathbf{B} \mathbf{C}`                 ``tensorOps::Rij_eq_AikBkj< m, n, p >( A, B, C )``
:math:`\mathbf{A} \leftarrow \mathbf{A} + \mathbf{B} \mathbf{C}`    ``tensorOps::Rij_add_AikBkj< m, n, p >( A, B, C )``
:math:`\mathbf{B} \leftarrow \mathbf{A} \mathbf{C}^T`               ``tensorOps::Rij_eq_AikBjk< m, p, n >( B, A, C )``
:math:`\mathbf{B} \leftarrow \mathbf{B} + \mathbf{A} \mathbf{C}^T`  ``tensorOps::Rij_add_AikBjk< m, p, n >( B, A, C )``
:math:`\mathbf{E} \leftarrow \mathbf{E} + \mathbf{A} \mathbf{A}^T`  ``tensorOps::Rij_add_AikAjk< m, n >( E, A )``
:math:`\mathbf{C} \leftarrow \mathbf{B}^T \mathbf{A}`               ``tensorOps::Rij_eq_AkiBkj< p, n, m >( C, B, A )``
:math:`\mathbf{C} \leftarrow \mathbf{C} + \mathbf{B}^T \mathbf{A}`  ``tensorOps::Rij_add_AkiBkj< p, n, m >( C, B, A )``
=================================================================== ===============================================

Square matrix operations
------------------------

=============================================================== =======================================================
Variables
-----------------------------------------------------------------------------------------------------------------------
:math:`\alpha`: A scalar.
-----------------------------------------------------------------------------------------------------------------------
:math:`\mathbf{A}`: A matrix in :math:`\mathbb{R}^m \times \mathbb{R}^m`.
-----------------------------------------------------------------------------------------------------------------------
:math:`\mathbf{B}`: A matrix in :math:`\mathbb{R}^m \times \mathbb{R}^m`.
-----------------------------------------------------------------------------------------------------------------------
Operation                                                       Function 
=============================================================== =======================================================
:math:`\mathbf{A} \leftarrow \mathbf{A}^T`                      ``tensorOps::transpose< m >( A )``
:math:`\mathbf{A} \leftarrow \mathbf{A} + \alpha \mathbf{I}`    ``tensorOps::tensorOps::addIdentity< m >( A, alpha )``
:math:`tr(\mathbf{A})`                                          ``tensorOps::trace< m >( A )``
:math:`|\mathbf{A}|`                                            ``tensorOps::determinant< m >( A )``
:math:`\mathbf{A} \leftarrow \mathbf{B}^-1`                     ``tensorOps::invert< m >( A, B )``
:math:`\mathbf{A} \leftarrow \mathbf{A}^-1`                     ``tensorOps::invert< m >( A )``
=============================================================== =======================================================

.. note::
  Apart from ``tensorOps::determinant`` and ``tensorOps::invert`` are only implemented for matrices of size :math:`2 \times 2` and :math:`3 \times 3`.

Symmetric matrix operations
---------------------------

================================================================================ ==============================================
Variables
-------------------------------------------------------------------------------------------------------------------------------
:math:`\alpha`: A scalar.
-------------------------------------------------------------------------------------------------------------------------------
:math:`x`: A vector in :math:`\mathbb{R}^m \times \mathbb{R}^1`.
-------------------------------------------------------------------------------------------------------------------------------
:math:`y`: A vector in :math:`\mathbb{R}^m \times \mathbb{R}^1`.
-------------------------------------------------------------------------------------------------------------------------------
:math:`\mathbf{A}`: A matrix in :math:`\mathbb{R}^m \times \mathbb{R}^m`.
-------------------------------------------------------------------------------------------------------------------------------
:math:`\mathbf{B}`: A matrix in :math:`\mathbb{R}^m \times \mathbb{R}^m`.
-------------------------------------------------------------------------------------------------------------------------------
:math:`\mathbf{S}`: A symmetric matrix in :math:`\mathbb{R}^m \times \mathbb{R}^m`.
-------------------------------------------------------------------------------------------------------------------------------
:math:`\mathbf{Q}`: A symmetric matrix in :math:`\mathbb{R}^m \times \mathbb{R}^m`.
-------------------------------------------------------------------------------------------------------------------------------
Operation                                                                        Function
================================================================================ ==============================================
:math:`\mathbf{S} \leftarrow \mathbf{S} + \alpha \mathbf{I}`                     ``tensorOps::symAddIdentity< m >( S, alpha )``
:math:`tr(\mathbf{S})`                                                           ``tensorOps::symTrace< m >( S )``
:math:`x \leftarrow \mathbf{S} y`                                                ``tensorOps::Ri_eq_symAijBj< m >( x, S ,y )``
:math:`x \leftarrow x + \mathbf{S} y`                                            ``tensorOps::Ri_add_symAijBj< m >( x, S ,y )``
:math:`\mathbf{A} \leftarrow \mathbf{S} \mathbf{B}^T`                            ``tensorOps::Rij_eq_symAikBjk< m >( A, S, B )``
:math:`\mathbf{S} \leftarrow \mathbf{A} \mathbf{Q} \mathbf{A}^T`                 ``tensorOps::Rij_eq_AikSymBklAjl< m >( S, A, Q )``
:math:`|\mathbf{S}|`                                                             ``tensorOps::symDeterminant< m >( S )``
:math:`\mathbf{S} \leftarrow \mathbf{Q}^-1`                                      ``tensorOps::symInvert< m >( S, Q )``
:math:`\mathbf{S} \leftarrow \mathbf{S}^-1`                                      ``tensorOps::symInvert< m >( S )``
:math:`x \leftarrow \mathbf{S} \mathbf{A}^T = diag(x) \mathbf{A}^T`              ``tensorOps::symEigenvalues< M >( x, S )``
:math:`x, \mathbf{A} \leftarrow \mathbf{S} \mathbf{A}^T = diag(x) \mathbf{A}^T`  ``tensorOps::symEigenvectors< M >( x, S )``
================================================================================ ==============================================

There are also two function ``tensorOps::denseToSymmetric`` and ``tensorOps::symmetricToDense`` which convert between dense and symmetric matrix representation.

.. note::
  Apart from ``tensorOps::symAddIdentity`` and ``tensorOps::symTrace`` the symmetric matrix operations are only implemented for matrices of size :math:`2 \times 2` and :math:`3 \times 3`.

Examples
--------

.. literalinclude:: ../../examples/exampleTensorOps.cpp
  :language: c++
  :start-after: // Sphinx start after inner product
  :end-before: // Sphinx end before inner product

*[Source: examples/exampleTensorOps.cpp]*

You can mix and match the data types of the objects and also call the ``tensorOps`` methods on device.

.. literalinclude:: ../../examples/exampleTensorOps.cpp
  :language: c++
  :start-after: // Sphinx start after device
  :end-before: // Sphinx end before device

*[Source: examples/exampleTensorOps.cpp]*

Bounds checking
---------------
Whatever the argument type the number of dimensions is checked at compile time. For example if you pass a ``double[ 3 ][ 3 ]`` or a three dimensional ``LvArray::ArraySlice`` to ``LvArray::tensorOps::crossProduct`` you will get a compilation error since that function is only implemented for vectors. When passing a c-array as an argument the size of the array is checked at compile time. For example if you pass ``int[ 2 ][ 3 ]`` to ``LvArray::tensorOps::addIdentity`` you will get a compilation error because that function only operates on square matrices. However when passing an ``LvArray::Array*`` object the size is only checked at runtime if ``LVARRAY_BOUNDS_CHECK`` is defined.

.. literalinclude:: ../../examples/exampleTensorOps.cpp
  :language: c++
  :start-after: // Sphinx start after bounds check
  :end-before: // Sphinx end before bounds check

*[Source: examples/exampleTensorOps.cpp]*


.. _`Voigt notation`: https://en.wikipedia.org/wiki/Voigt_notation

Doxygen
-------
- `LvArray::tensorOps <doxygen/html/namespace_lv_array_1_1tensor_ops.html>`_
