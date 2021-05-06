/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file fixedSizeSquareMatrixOps.hpp
 * @brief Contains tensorOps functions for operating on 2x2 and 3x3 matrices.
 */

#pragma once

#include "genericTensorOps.hpp"
#include "fixedSizeSquareMatrixOpsImpl.hpp"

namespace LvArray
{
namespace tensorOps
{

/**
 * @name Fixed size square matrix operations
 * @brief Functions that are overloaded to operate square matrices of a fixed size.
 * @note Currently all functions only support matrices of sizes 2x2 and 3x3.
 */
///@{

/**
 * @return Return the determinant of the matrix @p matrix.
 * @tparam M The size of @p matrix.
 * @tparam MATRIX The type of @p matrix.
 * @param matrix The M x M matrix to get the determinant of.
 */
template< std::ptrdiff_t M, typename MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
auto determinant( MATRIX const & matrix )
{ return internal::SquareMatrixOps< M >::determinant( matrix ); }

/**
 * @brief Invert the source matrix @p srcMatrix and store the result in @p dstMatrix.
 * @tparam M The size of the matrices @p dstMatrix and @p srcMatrix.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam SRC_MATRIX The type of @p srcMatrix.
 * @param dstMatrix The M x M matrix to write the inverse to.
 * @param srcMatrix The M x M matrix to take the inverse of.
 * @return The determinant.
 * @note @p srcMatrix can contain integers but @p dstMatrix must contain floating point values.
 */
template< std::ptrdiff_t M, typename DST_MATRIX, typename SRC_MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
auto invert( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
             SRC_MATRIX const & LVARRAY_RESTRICT_REF srcMatrix )
{
  static_assert( std::is_floating_point< std::decay_t< decltype( dstMatrix[ 0 ][ 0 ] ) > >::value,
                 "The destination matrix must be contain floating point values." );
  return internal::SquareMatrixOps< M >::invert( std::forward< DST_MATRIX >( dstMatrix ), srcMatrix );
}

/**
 * @brief Invert the matrix @p matrix overwritting it.
 * @tparam M The size of the matrix @p matrix.
 * @tparam MATRIX The type of @p matrix.
 * @param matrix The M x M matrix to take the inverse of and overwrite.
 * @return The determinant.
 * @note @p matrix must contain floating point values.
 */
template< std::ptrdiff_t M, typename MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
auto invert( MATRIX && matrix )
{
  static_assert( std::is_floating_point< std::decay_t< decltype( matrix[ 0 ][ 0 ] ) > >::value,
                 "The matrix must be contain floating point values." );
  return internal::SquareMatrixOps< M >::invert( std::forward< MATRIX >( matrix ) );
}

///@}

/**
 * @name Fixed size symmetric matrix operations
 * @brief Functions that are overloaded to operate symmetric matrices of a fixed size.
 * @note Currently all functions only support matrices of sizes 2x2 and 3x3.
 */
///@{

/**
 * @brief Multiply the vector @p vectorB by the symmetric matrix @p symMatrixA and store the result in @p dstVector.
 * @tparam M The size of @p dstVector, @p symMatrixA, and @p vectorB.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam SYM_MATRIX_A The type of @p symMatrixA.
 * @tparam VECTOR_B The type of @p vectorB.
 * @param dstVector The vector of length M to write the result to.
 * @param symMatrixA The M x M symmetric matrix to multiply @p vectorB by.
 * @param vectorB The vector of length M to be multiplied by @p symMatrixA.
 * @details Performs the operation @code dstVector[ i ] = symMatrixA[ i ][ j ] * vectorB[ j ] @endcode
 */
template< std::ptrdiff_t M, typename DST_VECTOR, typename SYM_MATRIX_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void Ri_eq_symAijBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                     SYM_MATRIX_A const & LVARRAY_RESTRICT_REF symMatrixA,
                     VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  internal::SquareMatrixOps< M >::Ri_eq_symAijBj( std::forward< DST_VECTOR >( dstVector ),
                                                  symMatrixA,
                                                  vectorB );
}

/**
 * @brief Multiply the vector @p vectorB by the symmetric matrix @p symMatrixA and add the result to @p dstVector.
 * @tparam M The size of @p dstVector, @p symMatrixA, and @p vectorB.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam SYM_MATRIX_A The type of @p symMatrixA.
 * @tparam VECTOR_B The type of @p vectorB.
 * @param dstVector The vector of length M to add the result to.
 * @param symMatrixA The M x M symmetric matrix to multiply @p vectorB by.
 * @param vectorB The vector of length M to be multiplied by @p symMatrixA.
 * @details Performs the operation @code dstVector[ i ] += symMatrixA[ i ][ j ] * vectorB[ j ] @endcode
 */
template< std::ptrdiff_t M, typename DST_VECTOR, typename SYM_MATRIX_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void Ri_add_symAijBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                      SYM_MATRIX_A const & LVARRAY_RESTRICT_REF symMatrixA,
                      VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  internal::SquareMatrixOps< M >::Ri_add_symAijBj( std::forward< DST_VECTOR >( dstVector ),
                                                   symMatrixA,
                                                   vectorB );
}

/**
 * @brief Multiply the transpose of matrix @p matrixB by the symmetric matrix @p symMatrixA and store
 *   the result in @p dstMatrix.
 * @tparam M The size of @p dstMatrix, @p symMatrixA and @p matrixB.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam SYM_MATRIX_A The type of @p symMatrixA.
 * @tparam MATRIX_B The type of @p matrixB.
 * @param dstMatrix The M x M matrix to write the result to.
 * @param symMatrixA The M x M symmetric matrix to multiply @p matrixB by.
 * @param matrixB The M x M matrix to be multiplied by @p matrixB.
 * @details Performs the operation @code dstMatrix[ i ][ j ] = symMatrixA[ i ][ k ] * matrixB[ j ][ k ] @endcode
 */
template< std::ptrdiff_t M, typename DST_MATRIX, typename SYM_MATRIX_A, typename MATRIX_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void Rij_eq_symAikBjk( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                       SYM_MATRIX_A const & LVARRAY_RESTRICT_REF symMatrixA,
                       MATRIX_B const & LVARRAY_RESTRICT_REF matrixB )
{
  internal::SquareMatrixOps< M >::Rij_eq_symAikBjk( std::forward< DST_MATRIX >( dstMatrix ),
                                                    symMatrixA,
                                                    matrixB );
}

/**
 * @brief Multiply the transpose of matrix @p matrixA by the symmetric matrix @p symMatrixB then by @p matrixA
 *   and store the result in @p dstSymMatrix.
 * @tparam M The size of @p dstSymMatrix, @p matrixA and @p symMatrixB.
 * @tparam DST_SYM_MATRIX The type of @p dstSymMatrix.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam SYM_MATRIX_B The type of @p symMatrixB.
 * @param dstSymMatrix The M x M symmetric matrix to write the result to.
 * @param matrixA The M x M matrix to pre and post multiply @p symMatrixB by.
 * @param symMatrixB The M x M symmetric matrix that gets pre multiplied by @p matrixA
 *   and post postmultiplied by the transpose of @p matrixA.
 * @details Performs the operation
 *   @code dstSymMatrix[ i ][ j ] = matrixA[ i ][ k ] * symMatrixB[ k ][ l ] * matrixA[ j ][ l ] @endcode
 */
template< std::ptrdiff_t M, typename DST_SYM_MATRIX, typename MATRIX_A, typename SYM_MATRIX_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void Rij_eq_AikSymBklAjl( DST_SYM_MATRIX && LVARRAY_RESTRICT_REF dstSymMatrix,
                          MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                          SYM_MATRIX_B const & LVARRAY_RESTRICT_REF symMatrixB )
{
  internal::SquareMatrixOps< M >::Rij_eq_AikSymBklAjl( std::forward< DST_SYM_MATRIX >( dstSymMatrix ),
                                                       matrixA,
                                                       symMatrixB );
}


/**
 * @brief Perform the outer product of @p vectorA with itself writing the result to @p dstMatrix.
 * @tparam M The size of both dimensions of @p dstMatrix and the length of @p vectorA.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam VECTOR_A The type of @p vectorA.
 * @param dstMatrix The matrix the result is written to, of size M x M.
 * @param vectorA The first vector in the outer product, of length M.
 * @details Performs the operations @code dstMatrix[ i ][ j ] = vectorA[ i ] * vectorA[ j ] @endcode
 */
template< std::ptrdiff_t M, typename DST_SYM_MATRIX, typename VECTOR_A >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void symRij_eq_AiAj( DST_SYM_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                     VECTOR_A const & LVARRAY_RESTRICT_REF vectorA )
{
  internal::SquareMatrixOps< M >::symRij_eq_AiAj( std::forward< DST_SYM_MATRIX >( dstMatrix ),
                                                  vectorA );
}

/**
 * @brief Perform the unscaled symmetric outer product of @p vectorA and
 *   @p vectorB with itself writing the result to @p dstMatrix.
 * @tparam M The size of both dimensions of @p dstMatrix and the length of @p vectorA.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam VECTOR_A The type of @p vectorA.
 * @tparam VECTOR_B The type of @p vectorB.
 * @param dstMatrix The matrix the result is written to, of size M x N.
 * @param vectorA The first vector in the outer product, of length M.
 * @param vectorB The second vector in the outer product, of length M.
 * @details Performs the operations @code dstMatrix[ i ][ j ] = vectorA[ i ] * vectorB[ j ] + vectorA[ j ] * vectorB[ i ] @endcode
 */
template< std::ptrdiff_t M, typename DST_SYM_MATRIX, typename VECTOR_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void symRij_eq_AiBj_plus_AjBi( DST_SYM_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                               VECTOR_A const & LVARRAY_RESTRICT_REF vectorA,
                               VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  internal::SquareMatrixOps< M >::symRij_eq_AiBj_plus_AjBi( std::forward< DST_SYM_MATRIX >( dstMatrix ),
                                                            vectorA,
                                                            vectorB );
}


/**
 * @return Return the determinant of the symmetric matrix @p symMatrix.
 * @tparam SYM_MATRIX The type of @p symMatrix.
 * @param symMatrix The M x M symmetric matrix to get the determinant of.
 */
template< std::ptrdiff_t M, typename SYM_MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
auto symDeterminant( SYM_MATRIX const & symMatrix )
{ return internal::SquareMatrixOps< M >::symDeterminant( symMatrix ); }

/**
 * @brief Invert the symmetric matrix @p srcSymMatrix and store the result in @p dstSymMatrix.
 * @tparam M The size of the symmetric matrices @p dstSymMatrix and @p srcSymMatrix.
 * @tparam DST_SYM_MATRIX The type of @p dstSymMatrix.
 * @tparam SRC_SYM_MATRIX The type of @p srcSymMatrix.
 * @param dstSymMatrix The M x M symmetric matrix to write the inverse to.
 * @param srcSymMatrix The M x M symmetric matrix to take the inverse of.
 * @return The determinant.
 * @note @p srcSymMatrix can contain integers but @p dstMatrix must contain floating point values.
 */
template< std::ptrdiff_t M, typename DST_SYM_MATRIX, typename SRC_SYM_MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
auto symInvert( DST_SYM_MATRIX && LVARRAY_RESTRICT_REF dstSymMatrix,
                SRC_SYM_MATRIX const & LVARRAY_RESTRICT_REF srcSymMatrix )
{
  static_assert( std::is_floating_point< std::decay_t< decltype( dstSymMatrix[ 0 ] ) > >::value,
                 "The destination matrix must be contain floating point values." );
  return internal::SquareMatrixOps< M >::symInvert( std::forward< DST_SYM_MATRIX >( dstSymMatrix ), srcSymMatrix );
}

/**
 * @brief Invert the symmetric matrix @p symMatrix overwritting it.
 * @tparam M The size of the matrix @p symMatrix.
 * @tparam SYM_MATRIX The type of @p symMatrix.
 * @param symMatrix The M x M symmetric matrix to take the inverse of and overwrite.
 * @return The determinant.
 * @note @p symMatrix must contain floating point values.
 */
template< std::ptrdiff_t M, typename SYM_MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
auto symInvert( SYM_MATRIX && symMatrix )
{
  static_assert( std::is_floating_point< std::decay_t< decltype( symMatrix[ 0 ] ) > >::value,
                 "The matrix must be contain floating point values." );
  return internal::SquareMatrixOps< M >::symInvert( std::forward< SYM_MATRIX >( symMatrix ) );
}

/**
 * @brief Compute the eigenvalues of the symmetric matrix @p symMatrix.
 * @tparam DST_VECTOR The type of @p eigenvalues.
 * @tparam SYM_MATRIX The type of @p symMatrix.
 * @param eigenvalues The vector of length M to write the eigenvalues to.
 * @param symMatrix The MxM symmetric matrix to compute the eigenvalues of.
 * @details Computes the eigenvalues by directly solving the characteristic equation,
 *   they are returned sorted in ascending order.
 * @note The 3x3 matrices eigenvalues are computed to within the following relative tolerance:
 *     double precision: 3e-8
 *     single precision: 5e-4
 *   However for matrices with well spaced eigenvalues the precision is many orders greater.
 * @note @p symMatrix can contain integers but @p eigenvalues must contain floating point numbers.
 */
template< std::ptrdiff_t M, typename DST_VECTOR, typename SYM_MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void symEigenvalues( DST_VECTOR && LVARRAY_RESTRICT_REF eigenvalues,
                     SYM_MATRIX const & LVARRAY_RESTRICT_REF symMatrix )
{
  static_assert( std::is_floating_point< std::decay_t< decltype( eigenvalues[ 0 ] ) > >::value,
                 "eigenvalues must be contain floating point values." );
  internal::SquareMatrixOps< M >::symEigenvalues( std::forward< DST_VECTOR >( eigenvalues ), symMatrix );
}

/**
 * @brief Compute the eigenvalues and eigenvectors of the symmetric matrix @p symMatrix.
 * @tparam DST_VECTOR The type of @p eigenvalues.
 * @tparam DST_MATRIX The type of @p eigenvectors
 * @tparam SYM_MATRIX The type of @p symMatrix.
 * @param eigenvalues The vector of length M to write the eigenvalues to.
 * @param eigenvectors The MxM matrix to write the eigenvectors to.
 * @param symMatrix The MxM symmetric matrix to compute the eigenvalues of.
 * @details Computes the eigenvalues by directly solving the characteristic equation,
 *   they are returned sorted in ascending order. The row @code eigenvectors[ i ] @endcode contains
 *   the eigenvector corresponding to @code eigenvalues[ i ] @endcode.
 * @note For 3x3 matrices the eigenvalues are computed to within the following relative tolerance:
 *     double precision: 3e-8
 *     single precision: 5e-4
 *   However for matrices with well spaced eigenvalues the precision is many orders greater.
 * @note @p symMatrix can contain integers but @p eigenvalues must contain floating point numbers.
 */
template< std::ptrdiff_t M, typename DST_VECTOR, typename DST_MATRIX, typename SYM_MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void symEigenvectors( DST_VECTOR && LVARRAY_RESTRICT_REF eigenvalues,
                      DST_MATRIX && LVARRAY_RESTRICT_REF eigenvectors,
                      SYM_MATRIX const & LVARRAY_RESTRICT_REF symMatrix )
{
  static_assert( std::is_floating_point< std::decay_t< decltype( eigenvalues[ 0 ] ) > >::value,
                 "eigenvalues must be contain floating point values." );
  static_assert( std::is_floating_point< std::decay_t< decltype( eigenvectors[ 0 ][ 0 ] ) > >::value,
                 "eigenvectors must be contain floating point values." );
  internal::SquareMatrixOps< M >::symEigenvectors( std::forward< DST_VECTOR >( eigenvalues ),
                                                   std::forward< DST_MATRIX >( eigenvectors ),
                                                   symMatrix );
}

/**
 * @brief Convert the upper triangular part of @p srcMatrix to a symmetric matrix.
 * @tparam M The size of @p dstSymMatrix and @p srcMatrix.
 * @tparam DST_SYM_MATRIX The type of @p dstSymMatrix.
 * @tparam SRC_MATRIX The type of @p srcMatrix.
 * @param dstSymMatrix The resulting MxM symmetric matrix.
 * @param srcMatrix The MxM matrix to convert to a symmetric matrix.
 */
template< std::ptrdiff_t M, typename DST_SYM_MATRIX, typename SRC_MATRIX >
LVARRAY_HOST_DEVICE constexpr inline
void denseToSymmetric( DST_SYM_MATRIX && dstSymMatrix, SRC_MATRIX const & srcMatrix )
{
  return internal::SquareMatrixOps< M >::denseToSymmetric( std::forward< DST_SYM_MATRIX >( dstSymMatrix ),
                                                           srcMatrix );
}

/**
 * @brief Convert the @p srcSymMatrix into a dense matrix.
 * @tparam M The size of @p dstMatrix and @p srcSymMatrix.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam SRC_SYM_MATRIX The type of @p srcSymMatrix.
 * @param dstMatrix The resulting MxM matrix.
 * @param srcSymMatrix The MxM symmetric matrix to convert.
 */
template< std::ptrdiff_t M, typename DST_MATRIX, typename SRC_SYM_MATRIX >
LVARRAY_HOST_DEVICE constexpr inline
void symmetricToDense( DST_MATRIX && dstMatrix, SRC_SYM_MATRIX const & srcSymMatrix )
{
  return internal::SquareMatrixOps< M >::symmetricToDense( std::forward< DST_MATRIX >( dstMatrix ),
                                                           srcSymMatrix );
}

///@}

} // namespace tensorOps
} // namespace LvArray
