/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file fixedSizeSquareMatrixOpsImpl.hpp
 * @brief Contains the implementation of the 2x2 and 3x3 matrix operations.
 */

#pragma once

#include "genericTensorOps.hpp"

namespace LvArray
{
namespace tensorOps
{
namespace internal
{

/**
 * @brief Shift the and scale the MxM symmetric matrix @p matrix.
 * @tparam M The size of the matrix.
 * @tparam FloatingPoint A floating point type.
 * @param matrix The matrix to shift and scale.
 * @param shift The amount the matrix is shifted.
 * @param maxEntryAfterShift The amount the matrix is scaled by after the shift.
 * @note The resulting matrix is guaranteed to have a zero trace and its entries will be
 *   in [-2, 2].
 * @note The size of @p matrix is SYM_SIZE< M >, but the intel compiler complains so the calculation
 *   must be inlined.
 */
template< std::ptrdiff_t M, typename FloatingPoint >
LVARRAY_HOST_DEVICE inline
static void shiftAndScale( FloatingPoint (& matrix)[ ( M * ( M + 1 ) ) / 2 ],
                           FloatingPoint & shift,
                           FloatingPoint & maxEntryAfterShift )
{
  // Compute the average eigenvalue.
  shift = symTrace< M >( matrix ) / FloatingPoint( M );

  // Initialize the floating point copy of the matrix and shift the average eigenvalue to 0.
  symAddIdentity< M >( matrix, -shift );

  // Now scale the entires of the copy to between [-1, 1].
  maxEntryAfterShift = maxAbsoluteEntry< SYM_SIZE< M > >( matrix );
  if( maxEntryAfterShift > 0 )
  {
    scale< SYM_SIZE< M > >( matrix, 1 / maxEntryAfterShift );
  }

  // A second shift is necessary because of floating point round off and because the eigenvalue
  // algorithm expects the trace of the matrix to be zero. However it isn't necessary to export
  // this shift since when calculating the eigenvalues of the original matrix it will be lost
  // to roundoff.
  FloatingPoint const secondShift = symTrace< M >( matrix ) / FloatingPoint( M );
  symAddIdentity< M >( matrix, -secondShift );
}

/**
 * @struct SquareMatrixOps
 * @brief Performs operations on square matrices.
 * @tparam M The size of the matrix.
 * @note This class is intended to have methods that operate on generic sized square matrices with specializations
 *   for specific sizes.
 */
template< std::ptrdiff_t M >
struct SquareMatrixOps
{};

/**
 * @struct SquareMatrixOps< 2 >
 * @brief Performs operations on 2x2 square matrices.
 */
template<>
struct SquareMatrixOps< 2 >
{
  /**
   * @return Return the determinant of the matrix @p matrix.
   * @tparam MATRIX The type of @p matrix.
   * @param matrix The 2x2 matrix to get the determinant of.
   */
  template< typename MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static auto determinant( MATRIX const & matrix )
  {
    checkSizes< 2, 2 >( matrix );
    return matrix[ 0 ][ 0 ] * matrix[ 1 ][ 1 ] - matrix[ 0 ][ 1 ] * matrix[ 1 ][ 0 ];
  }

  /**
   * @brief Invert the source matrix @p srcMatrix and store the result in @p dstMatrix.
   * @tparam DST_MATRIX The type of @p dstMatrix.
   * @tparam SRC_MATRIX The type of @p srcMatrix.
   * @param dstMatrix The 2x2 matrix to write the inverse to.
   * @param srcMatrix The 2x2 matrix to take the inverse of.
   * @return The determinant.
   * @note @p srcMatrix can contain integers but @p dstMatrix must contain floating point values.
   */
  template< typename DST_MATRIX, typename SRC_MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static auto invert( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                      SRC_MATRIX const & LVARRAY_RESTRICT_REF srcMatrix )
  {
    checkSizes< 2, 2 >( dstMatrix );
    checkSizes< 2, 2 >( srcMatrix );

    using FloatingPoint = std::decay_t< decltype( dstMatrix[ 0 ][ 0 ] ) >;

    auto const det = determinant( srcMatrix );
    FloatingPoint const invDet = FloatingPoint( 1 ) / det;

    dstMatrix[ 0 ][ 0 ] = srcMatrix[ 1 ][ 1 ] * invDet;
    dstMatrix[ 1 ][ 1 ] = srcMatrix[ 0 ][ 0 ] * invDet;
    dstMatrix[ 0 ][ 1 ] = srcMatrix[ 0 ][ 1 ] * -invDet;
    dstMatrix[ 1 ][ 0 ] = srcMatrix[ 1 ][ 0 ] * -invDet;

    return det;
  }

  /**
   * @brief Invert the matrix @p srcMatrix overwritting it.
   * @tparam MATRIX The type of @p matrix.
   * @param matrix The 2x2 matrix to take the inverse of and overwrite.
   * @return The determinant.
   * @note @p matrix must contain floating point values.
   */
  template< typename MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static auto invert( MATRIX && matrix )
  {
    checkSizes< 2, 2 >( matrix );

    auto const det = determinant( matrix );
    auto const invDet = 1 / det;

    auto const temp = matrix[ 0 ][ 0 ];
    matrix[ 0 ][ 0 ] = matrix[ 1 ][ 1 ] * invDet;
    matrix[ 1 ][ 1 ] = temp * invDet;
    matrix[ 0 ][ 1 ] *= -invDet;
    matrix[ 1 ][ 0 ] *= -invDet;

    return det;
  }

  /**
   * @return Return the determinant of the symmetric matrix @p symMatrix.
   * @tparam SYM_MATRIX The type of @p symMatrix.
   * @param symMatrix The 2x2 symmetric matrix to get the determinant of.
   */
  template< typename SYM_MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static auto symDeterminant( SYM_MATRIX const & symMatrix )
  {
    checkSizes< 3 >( symMatrix );
    return symMatrix[ 0 ] * symMatrix[ 1 ] - symMatrix[ 2 ] * symMatrix[ 2 ];
  }

  /**
   * @brief Invert the symmetric matrix @p srcSymMatrix and store the result in @p dstSymMatrix.
   * @tparam DST_SYM_MATRIX The type of @p dstSymMatrix.
   * @tparam SRC_SYM_MATRIX The type of @p srcSymMatrix.
   * @param dstSymMatrix The 2x2 symmetric matrix to write the inverse to.
   * @param srcSymMatrix The 2x2 symmetric matrix to take the inverse of.
   * @return The determinant.
   * @note @p srcSymMatrix can contain integers but @p dstMatrix must contain floating point values.
   */
  template< typename DST_SYM_MATRIX, typename SRC_SYM_MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static auto symInvert( DST_SYM_MATRIX && LVARRAY_RESTRICT_REF dstSymMatrix,
                         SRC_SYM_MATRIX const & LVARRAY_RESTRICT_REF srcSymMatrix )
  {
    checkSizes< 3 >( dstSymMatrix );
    checkSizes< 3 >( srcSymMatrix );

    using FloatingPoint = std::decay_t< decltype( dstSymMatrix[ 0 ] ) >;

    auto const det = symDeterminant( srcSymMatrix );
    FloatingPoint const invDet = FloatingPoint( 1 ) / det;

    dstSymMatrix[ 0 ] = srcSymMatrix[ 1 ] * invDet;
    dstSymMatrix[ 1 ] = srcSymMatrix[ 0 ] * invDet;
    dstSymMatrix[ 2 ] = srcSymMatrix[ 2 ] * -invDet;

    return det;
  }

  /**
   * @brief Invert the symmetric matrix @p symMatrix overwritting it.
   * @tparam SYM_MATRIX The type of @p symMatrix.
   * @param symMatrix The 2x2 symmetric matrix to take the inverse of and overwrite.
   * @return The determinant.
   * @note @p symMatrix must contain floating point values.
   */
  template< typename SYM_MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static auto symInvert( SYM_MATRIX && symMatrix )
  {
    checkSizes< 3 >( symMatrix );

    auto const det = symDeterminant( symMatrix );
    auto const invDet = 1 / det;

    auto const temp = symMatrix[ 0 ];
    symMatrix[ 0 ] = symMatrix[ 1 ] * invDet;
    symMatrix[ 1 ] = temp * invDet;
    symMatrix[ 2 ] *= -invDet;

    return det;
  }

  /**
   * @brief Multiply the vector @p vectorB by the symmetric matrix @p symMatrixA and store the result
   *   in @p dstVector.
   * @tparam DST_VECTOR The type of @p dstVector.
   * @tparam SYM_MATRIX_A The type of @p symMatrixA.
   * @tparam VECTOR_B The type of @p vectorB.
   * @param dstVector The vector of length 2 to write the result to.
   * @param symMatrixA The 2x2 symmetric matrix (vector of length 3) to multiply @p vectorB by.
   * @param vectorB The vector of length 2 to be multiplied by @p symMatrixA.
   * @details Performs the operation @code dstVector[ i ] = symMatrixA[ i ][ j ] * vectorB[ j ] @endcode
   */
  template< typename DST_VECTOR, typename SYM_MATRIX_A, typename VECTOR_B >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void Ri_eq_symAijBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                              SYM_MATRIX_A const & LVARRAY_RESTRICT_REF symMatrixA,
                              VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
  {
    checkSizes< 2 >( dstVector );
    checkSizes< 3 >( symMatrixA );
    checkSizes< 2 >( vectorB );

    dstVector[ 0 ] = symMatrixA[ 0 ] * vectorB[ 0 ] + symMatrixA[ 2 ] * vectorB[ 1 ];
    dstVector[ 1 ] = symMatrixA[ 2 ] * vectorB[ 0 ] + symMatrixA[ 1 ] * vectorB[ 1 ];
  }

  /**
   * @brief Multiply the vector @p vectorB by the symmetric matrix @p symMatrixA and add the result to @p dstVector.
   * @tparam DST_VECTOR The type of @p dstVector.
   * @tparam SYM_MATRIX_A The type of @p symMatrixA.
   * @tparam VECTOR_B The type of @p vectorB.
   * @param dstVector The vector of length 2 to add the result to.
   * @param symMatrixA The 2x2 symmetric matrix (vector of length 3) to multiply @p vectorB by.
   * @param vectorB The vector of length 2 to be multiplied by @p symMatrixA.
   * @details Performs the operation @code dstVector[ i ] += symMatrixA[ i ][ j ] * vectorB[ j ] @endcode
   */
  template< typename DST_VECTOR, typename SYM_MATRIX_A, typename VECTOR_B >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void Ri_add_symAijBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                               SYM_MATRIX_A const & LVARRAY_RESTRICT_REF symMatrixA,
                               VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
  {
    checkSizes< 2 >( dstVector );
    checkSizes< 3 >( symMatrixA );
    checkSizes< 2 >( vectorB );

    dstVector[ 0 ] = dstVector[ 0 ] + symMatrixA[ 0 ] * vectorB[ 0 ] + symMatrixA[ 2 ] * vectorB[ 1 ];
    dstVector[ 1 ] = dstVector[ 1 ] + symMatrixA[ 2 ] * vectorB[ 0 ] + symMatrixA[ 1 ] * vectorB[ 1 ];
  }

  /**
   * @brief Multiply the transpose of matrix @p matrixB by the symmetric matrix @p symMatrixA and store
   *   the result in @p dstMatrix.
   * @tparam DST_MATRIX The type of @p dstMatrix.
   * @tparam SYM_MATRIX_A The type of @p symMatrixA.
   * @tparam MATRIX_B The type of @p matrixB.
   * @param dstMatrix The 2x2 matrix to write the result to.
   * @param symMatrixA The 2x2 symmetric matrix (vector of length 3) to multiply @p matrixB by.
   * @param matrixB The 2x2 matrix to be multiplied by @p matrixB.
   * @details Performs the operation @code dstMatrix[ i ][ j ] = symMatrixA[ i ][ k ] * matrixB[ j ][ k ] @endcode
   */
  template< typename DST_MATRIX, typename SYM_MATRIX_A, typename MATRIX_B >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void Rij_eq_symAikBjk( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                                SYM_MATRIX_A const & LVARRAY_RESTRICT_REF symMatrixA,
                                MATRIX_B const & LVARRAY_RESTRICT_REF matrixB )
  {
    checkSizes< 2, 2 >( dstMatrix );
    checkSizes< 3 >( symMatrixA );
    checkSizes< 2, 2 >( matrixB );

    dstMatrix[ 0 ][ 0 ] = symMatrixA[ 0 ] * matrixB[ 0 ][ 0 ] + symMatrixA[ 2 ] * matrixB[ 0 ][ 1 ];
    dstMatrix[ 0 ][ 1 ] = symMatrixA[ 0 ] * matrixB[ 1 ][ 0 ] + symMatrixA[ 2 ] * matrixB[ 1 ][ 1 ];

    dstMatrix[ 1 ][ 0 ] = symMatrixA[ 2 ] * matrixB[ 0 ][ 0 ] + symMatrixA[ 1 ] * matrixB[ 0 ][ 1 ];
    dstMatrix[ 1 ][ 1 ] = symMatrixA[ 2 ] * matrixB[ 1 ][ 0 ] + symMatrixA[ 1 ] * matrixB[ 1 ][ 1 ];
  }

  /**
   * @brief Multiply the transpose of matrix @p matrixA by the symmetric matrix @p symMatrixB then by @p matrixA
   *   and store the result in @p dstSymMatrix.
   * @tparam DST_SYM_MATRIX The type of @p dstSymMatrix.
   * @tparam MATRIX_A The type of @p matrixA.
   * @tparam SYM_MATRIX_B The type of @p symMatrixB.
   * @param dstSymMatrix The 2x2 symmetric matrix (vector of length 3) to write the result to.
   * @param matrixA The 2x2 matrix to pre and post multiply @p symMatrixB by.
   * @param symMatrixB The 2x2 symmetric matrix (vector of length 3) that gets pre multiplied by @p matrixA
   *   and post postmultiplied by the transpose of @p matrixA.
   * @details Performs the operation
   *   @code dstSymMatrix[ i ][ j ] = matrixA[ i ][ k ] * symMatrixB[ k ][ l ] * matrixA[ j ][ l ] @endcode
   */
  template< typename DST_SYM_MATRIX, typename MATRIX_A, typename SYM_MATRIX_B >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void Rij_eq_AikSymBklAjl( DST_SYM_MATRIX && LVARRAY_RESTRICT_REF dstSymMatrix,
                                   MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                                   SYM_MATRIX_B const & LVARRAY_RESTRICT_REF symMatrixB )
  {
    checkSizes< 3 >( dstSymMatrix );
    checkSizes< 2, 2 >( matrixA );
    checkSizes< 3 >( symMatrixB );

    // Calculate entry (0, 0).
    dstSymMatrix[ 0 ] = matrixA[ 0 ][ 0 ] * symMatrixB[ 0 ] * matrixA[ 0 ][ 0 ] +
                        matrixA[ 0 ][ 0 ] * symMatrixB[ 2 ] * matrixA[ 0 ][ 1 ] +
                        matrixA[ 0 ][ 1 ] * symMatrixB[ 2 ] * matrixA[ 0 ][ 0 ] +
                        matrixA[ 0 ][ 1 ] * symMatrixB[ 1 ] * matrixA[ 0 ][ 1 ];

    // Calculate entry (1, 1).
    dstSymMatrix[ 1 ] = matrixA[ 1 ][ 0 ] * symMatrixB[ 0 ] * matrixA[ 1 ][ 0 ] +
                        matrixA[ 1 ][ 0 ] * symMatrixB[ 2 ] * matrixA[ 1 ][ 1 ] +
                        matrixA[ 1 ][ 1 ] * symMatrixB[ 2 ] * matrixA[ 1 ][ 0 ] +
                        matrixA[ 1 ][ 1 ] * symMatrixB[ 1 ] * matrixA[ 1 ][ 1 ];

    // Calculate entry (1, 0) or (0, 1).
    dstSymMatrix[ 2 ] = matrixA[ 1 ][ 0 ] * symMatrixB[ 0 ] * matrixA[ 0 ][ 0 ] +
                        matrixA[ 1 ][ 0 ] * symMatrixB[ 2 ] * matrixA[ 0 ][ 1 ] +
                        matrixA[ 1 ][ 1 ] * symMatrixB[ 2 ] * matrixA[ 0 ][ 0 ] +
                        matrixA[ 1 ][ 1 ] * symMatrixB[ 1 ] * matrixA[ 0 ][ 1 ];
  }


  /**
   * @brief Perform the outer product of @p vectorA with itself writing the result to @p dstMatrix.
   * @tparam M The size of both dimensions of @p dstMatrix and the length of @p vectorA.
   * @tparam DST_MATRIX The type of @p dstMatrix.
   * @tparam VECTOR_A The type of @p vectorA.
   * @param dstMatrix The matrix the result is written to, of size M x N.
   * @param vectorA The first vector in the outer product, of length M.
   * @details Performs the operations @code dstMatrix[ i ][ j ] = vectorA[ i ] * vectorA[ j ] @endcode
   */
  template< typename DST_MATRIX, typename VECTOR_A >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void symRij_eq_AiAj( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                              VECTOR_A const & LVARRAY_RESTRICT_REF vectorA )
  {
    internal::checkSizes< 3 >( dstMatrix );
    internal::checkSizes< 2 >( vectorA );

    dstMatrix[ 0 ] = vectorA[ 0 ] * vectorA[ 0 ];
    dstMatrix[ 1 ] = vectorA[ 1 ] * vectorA[ 1 ];
    dstMatrix[ 2 ] = vectorA[ 0 ] * vectorA[ 1 ];
  }

  /**
   * @brief Perform the unscaled symmetric outer product of @p vectorA and
   *   @p vectorB writing the result to @p dstMatrix.
   * @tparam DST_MATRIX The type of @p dstMatrix.
   * @tparam VECTOR_A The type of @p vectorA.
   * @tparam VECTOR_B The type of @p vectorB.
   * @param dstMatrix The matrix the result is written to, of size M x N.
   * @param vectorA The first vector in the outer product, of length M.
   * @param vectorB The second vector in the outer product, of length M.
   * @details Performs the operations @code dstMatrix[ i ][ j ] = vectorA[ i ] * vectorB[ j ] + vectorA[ j ] * vectorB[ i ] @endcode
   */
  template< typename DST_SYM_MATRIX, typename VECTOR_A, typename VECTOR_B >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void symRij_eq_AiBj_plus_AjBi( DST_SYM_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                                        VECTOR_A const & LVARRAY_RESTRICT_REF vectorA,
                                        VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
  {
    internal::checkSizes< 3 >( dstMatrix );
    internal::checkSizes< 2 >( vectorA );
    internal::checkSizes< 2 >( vectorB );

    dstMatrix[ 0 ] = 2 * vectorA[ 0 ] * vectorB[ 0 ];
    dstMatrix[ 1 ] = 2 * vectorA[ 1 ] * vectorB[ 1 ];
    dstMatrix[ 2 ] = vectorA[ 0 ] * vectorB[ 1 ] + vectorA[ 1 ] * vectorB[ 0 ];
  }

  /**
   * @brief Compute the eigenvalues of the symmetric matrix @p symMatrix.
   * @tparam DST_VECTOR The type of @p eigenvalues.
   * @tparam SYM_MATRIX The type of @p symMatrix.
   * @param eigenvalues The vector of length 2 to write the eigenvalues to.
   * @param symMatrix The 2x2 symmetric matrix to compute the eigenvalues of.
   * @details Computes the eigenvalues directly, they are returned sorted in ascending order.
   * @note @p symMatrix can contain integers but @p eigenvalues must contain floating point numbers.
   */
  template< typename DST_VECTOR, typename SYM_MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void symEigenvalues( DST_VECTOR && LVARRAY_RESTRICT_REF eigenvalues,
                              SYM_MATRIX const & LVARRAY_RESTRICT_REF symMatrix )
  {
    checkSizes< 2 >( eigenvalues );
    checkSizes< 3 >( symMatrix );

    using FloatingPoint = std::decay_t< decltype( eigenvalues[ 0 ] ) >;

    // Shift the and scale the matrix
    FloatingPoint shift, maxEntryAfterShift;
    FloatingPoint fpCopy[ 3 ];
    copy< 3 >( fpCopy, symMatrix );
    shiftAndScale< 2 >( fpCopy, shift, maxEntryAfterShift );

    // Compute the eigenvalues of the shifted matrix.
    eigenvaluesOfShiftedMatrix( eigenvalues, fpCopy );

    // Rescale the eigenvalues.
    eigenvalues[ 0 ] = eigenvalues[ 0 ] * maxEntryAfterShift + shift;
    eigenvalues[ 1 ] = eigenvalues[ 1 ] * maxEntryAfterShift + shift;
  }

  /**
   * @brief Compute the eigenvalues and eigenvectors of the symmetric matrix @p symMatrix.
   * @tparam DST_VECTOR The type of @p eigenvalues.
   * @tparam DST_MATRIX The type of @p eigenvectors
   * @tparam SYM_MATRIX The type of @p symMatrix.
   * @param eigenvalues The vector of length 2 to write the eigenvalues to.
   * @param eigenvectors The 2x2 matrix to write the eigenvectors to.
   * @param symMatrix The 2x2 symmetric matrix to compute the eigenvalues of.
   * @details Computes the eigenvalues directly, they are returned sorted in ascending order.
   *   The row @code eigenvectors[ i ] @endcode contains the eigenvector corresponding to
   *   @code eigenvalues[ i ] @endcode.
   * @note @p symMatrix can contain integers but @p eigenvalues must contain floating point numbers.
   */
  template< typename DST_VECTOR, typename DST_MATRIX, typename SYM_MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void symEigenvectors( DST_VECTOR && LVARRAY_RESTRICT_REF eigenvalues,
                               DST_MATRIX && LVARRAY_RESTRICT_REF eigenvectors,
                               SYM_MATRIX const & LVARRAY_RESTRICT_REF symMatrix )
  {
    checkSizes< 2 >( eigenvalues );
    checkSizes< 2, 2 >( eigenvectors );
    checkSizes< 3 >( symMatrix );

    using FloatingPoint = std::decay_t< decltype( eigenvalues[ 0 ] ) >;

    using FloatingPoint = std::decay_t< decltype( eigenvalues[ 0 ] ) >;

    // Shift the and scale the matrix
    FloatingPoint shift, maxEntryAfterShift;
    FloatingPoint fpCopy[ 3 ];
    copy< 3 >( fpCopy, symMatrix );
    shiftAndScale< 2 >( fpCopy, shift, maxEntryAfterShift );

    // Compute the eigenvalues of the shifted matrix.
    eigenvaluesOfShiftedMatrix( eigenvalues, fpCopy );

    // If the eigenvalues are equal
    if( ( eigenvalues[ 1 ] - eigenvalues[ 0 ] ) <= NumericLimits< FloatingPoint >::epsilon )
    {
      LVARRAY_TENSOROPS_ASSIGN_2x2( eigenvectors,
                                    1, 0,
                                    0, 1 );
    }
    else
    {
      // Compute the eigenvector corresponding to the largest eigenvalue.
      // Done by constructing a rank 1 matrix and extracting the kernel.
      symAddIdentity< 2 >( fpCopy, -eigenvalues[ 1 ] );
      FloatingPoint const a2 = fpCopy[ 0 ] * fpCopy[ 0 ];
      FloatingPoint const c2 = fpCopy[ 1 ] * fpCopy[ 1 ];
      FloatingPoint const b2 = fpCopy[ 2 ] * fpCopy[ 2 ];

      // Pick the row with the greatest magnitude.
      if( a2 > c2 )
      {
        FloatingPoint const inv = math::invSqrt( a2 + b2 );
        eigenvectors[ 0 ][ 1 ] = -fpCopy[ 2 ] * inv;
        eigenvectors[ 1 ][ 1 ] = fpCopy[ 0 ] * inv;
      }
      else
      {
        FloatingPoint const inv = math::invSqrt( c2 + b2 );
        eigenvectors[ 0 ][ 1 ] = -fpCopy[ 1 ] * inv;
        eigenvectors[ 1 ][ 1 ] = fpCopy[ 2 ] * inv;
      }

      // The other eigenvector is orthonormal to the one just computed.
      eigenvectors[ 0 ][ 0 ] = -eigenvectors[ 1 ][ 1 ];
      eigenvectors[ 1 ][ 0 ] = eigenvectors[ 0 ][ 1 ];
    }

    // Rescale the eigenvalues.
    eigenvalues[ 0 ] = eigenvalues[ 0 ] * maxEntryAfterShift + shift;
    eigenvalues[ 1 ] = eigenvalues[ 1 ] * maxEntryAfterShift + shift;
  }

  /**
   * @brief Convert the upper triangular part of @p srcMatrix to a symmetric matrix.
   * @tparam DST_SYM_MATRIX The type of @p dstSymMatrix.
   * @tparam SRC_MATRIX The type of @p srcMatrix.
   * @param dstSymMatrix The resulting 2x2 symmetric matrix.
   * @param srcMatrix The 2x2 matrix to convert to a symmetric matrix.
   */
  template< typename DST_SYM_MATRIX, typename SRC_MATRIX >
  LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
  static void denseToSymmetric( DST_SYM_MATRIX && dstSymMatrix, SRC_MATRIX const & srcMatrix )
  {
    tensorOps::internal::checkSizes< 3 >( dstSymMatrix );
    tensorOps::internal::checkSizes< 2, 2 >( srcMatrix );

    dstSymMatrix[ 0 ] = srcMatrix[ 0 ][ 0 ];
    dstSymMatrix[ 1 ] = srcMatrix[ 1 ][ 1 ];
    dstSymMatrix[ 2 ] = srcMatrix[ 0 ][ 1 ];
  }

  /**
   * @brief Convert the @p srcSymMatrix into a dense matrix.
   * @tparam DST_MATRIX The type of @p dstMatrix.
   * @tparam SRC_SYM_MATRIX The type of @p srcSymMatrix.
   * @param dstMatrix The resulting 2x2 matrix.
   * @param srcSymMatrix The 2x2 symmetric matrix to convert.
   */
  template< typename DST_MATRIX, typename SRC_SYM_MATRIX >
  LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
  static void symmetricToDense( DST_MATRIX && dstMatrix, SRC_SYM_MATRIX const & srcSymMatrix )
  {
    tensorOps::internal::checkSizes< 2, 2 >( dstMatrix );
    tensorOps::internal::checkSizes< 3 >( srcSymMatrix );

    dstMatrix[ 0 ][ 0 ] = srcSymMatrix[ 0 ];
    dstMatrix[ 1 ][ 1 ] = srcSymMatrix[ 1 ];

    dstMatrix[ 0 ][ 1 ] = srcSymMatrix[ 2 ];
    dstMatrix[ 1 ][ 0 ] = srcSymMatrix[ 2 ];
  }

private:

  /**
   * @brief Compute the eigenvalues of the 2x2 symmetric matrix @p matrix.
   * @tparam FloatingPoint A floating point type.
   * @tparam VECTOR The type of @p eigenvalues.
   * @param eigenvalues The resulting eigenvalues.
   * @param matrix A 2x2 symmetric matrix with trace 0.
   */
  template< typename FloatingPoint, typename VECTOR >
  LVARRAY_HOST_DEVICE inline
  static void eigenvaluesOfShiftedMatrix( VECTOR && eigenvalues, FloatingPoint const ( &matrix )[ 3 ] )
  {
    /*
       For a 2x2 symmetric matrix
          a0, a2
          a2, a1
       And eigenvalue x the characteristic equation is
          x^2 - (a0 + a1) * x + a0 * a1 - a2^2 = 0
       However the shifted matrix has a trace of 0 so this simplifies to
          x^2 + a0 * a1 - a2^2 = 0
     */
    eigenvalues[ 1 ] = math::sqrt( matrix[ 2 ] * matrix[ 2 ] - matrix[ 0 ] * matrix[ 1 ] );
    eigenvalues[ 0 ] = -eigenvalues[ 1 ];
  }
};

/**
 * @struct SquareMatrixOps< 3 >
 * @brief Performs operations on 3x3 square matrices.
 */
template<>
struct SquareMatrixOps< 3 >
{
  /**
   * @return Return the determinant of the matrix @p matrix.
   * @tparam MATRIX The type of @p matrix.
   * @param matrix The 3x3 matrix to get the determinant of.
   */
  template< typename MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static auto determinant( MATRIX const & matrix )
  {
    checkSizes< 3, 3 >( matrix );

    return matrix[ 0 ][ 0 ] * ( matrix[ 1 ][ 1 ] * matrix[ 2 ][ 2 ] - matrix[ 1 ][ 2 ] * matrix[ 2 ][ 1 ] ) +
           matrix[ 1 ][ 0 ] * ( matrix[ 0 ][ 2 ] * matrix[ 2 ][ 1 ] - matrix[ 0 ][ 1 ] * matrix[ 2 ][ 2 ] ) +
           matrix[ 2 ][ 0 ] * ( matrix[ 0 ][ 1 ] * matrix[ 1 ][ 2 ] - matrix[ 0 ][ 2 ] * matrix[ 1 ][ 1 ] );
  }

  /**
   * @brief Invert the source matrix @p srcMatrix and store the result in @p dstMatrix.
   * @tparam DST_MATRIX The type of @p dstMatrix.
   * @tparam SRC_MATRIX The type of @p srcMatrix.
   * @param dstMatrix The 3x3 matrix to write the inverse to.
   * @param srcMatrix The 3x3 matrix to take the inverse of.
   * @return The determinant.
   * @note @p srcMatrix can contain integers but @p dstMatrix must contain floating point values.
   */
  template< typename DST_MATRIX, typename SRC_MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static auto invert( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                      SRC_MATRIX const & LVARRAY_RESTRICT_REF srcMatrix )
  {
    checkSizes< 3, 3 >( dstMatrix );
    checkSizes< 3, 3 >( srcMatrix );

    using FloatingPoint = std::decay_t< decltype( dstMatrix[ 0 ][ 0 ] ) >;

    dstMatrix[ 0 ][ 0 ] = srcMatrix[ 1 ][ 1 ] * srcMatrix[ 2 ][ 2 ] - srcMatrix[ 1 ][ 2 ] * srcMatrix[ 2 ][ 1 ];
    dstMatrix[ 0 ][ 1 ] = srcMatrix[ 0 ][ 2 ] * srcMatrix[ 2 ][ 1 ] - srcMatrix[ 0 ][ 1 ] * srcMatrix[ 2 ][ 2 ];
    dstMatrix[ 0 ][ 2 ] = srcMatrix[ 0 ][ 1 ] * srcMatrix[ 1 ][ 2 ] - srcMatrix[ 0 ][ 2 ] * srcMatrix[ 1 ][ 1 ];

    auto const det = srcMatrix[ 0 ][ 0 ] * dstMatrix[ 0 ][ 0 ] +
                     srcMatrix[ 1 ][ 0 ] * dstMatrix[ 0 ][ 1 ] +
                     srcMatrix[ 2 ][ 0 ] * dstMatrix[ 0 ][ 2 ];
    FloatingPoint const invDet = FloatingPoint( 1 ) / det;

    dstMatrix[ 0 ][ 0 ] *= invDet;
    dstMatrix[ 0 ][ 1 ] *= invDet;
    dstMatrix[ 0 ][ 2 ] *= invDet;
    dstMatrix[ 1 ][ 0 ] = ( srcMatrix[ 1 ][ 2 ] * srcMatrix[ 2 ][ 0 ] - srcMatrix[ 1 ][ 0 ] * srcMatrix[ 2 ][ 2 ] ) * invDet;
    dstMatrix[ 1 ][ 1 ] = ( srcMatrix[ 0 ][ 0 ] * srcMatrix[ 2 ][ 2 ] - srcMatrix[ 0 ][ 2 ] * srcMatrix[ 2 ][ 0 ] ) * invDet;
    dstMatrix[ 1 ][ 2 ] = ( srcMatrix[ 0 ][ 2 ] * srcMatrix[ 1 ][ 0 ] - srcMatrix[ 0 ][ 0 ] * srcMatrix[ 1 ][ 2 ] ) * invDet;
    dstMatrix[ 2 ][ 0 ] = ( srcMatrix[ 1 ][ 0 ] * srcMatrix[ 2 ][ 1 ] - srcMatrix[ 1 ][ 1 ] * srcMatrix[ 2 ][ 0 ] ) * invDet;
    dstMatrix[ 2 ][ 1 ] = ( srcMatrix[ 0 ][ 1 ] * srcMatrix[ 2 ][ 0 ] - srcMatrix[ 0 ][ 0 ] * srcMatrix[ 2 ][ 1 ] ) * invDet;
    dstMatrix[ 2 ][ 2 ] = ( srcMatrix[ 0 ][ 0 ] * srcMatrix[ 1 ][ 1 ] - srcMatrix[ 0 ][ 1 ] * srcMatrix[ 1 ][ 0 ] ) * invDet;

    return det;
  }

  /**
   * @brief Invert the matrix @p srcMatrix overwritting it.
   * @tparam MATRIX The type of @p matrix.
   * @param matrix The 3x3 matrix to take the inverse of and overwrite.
   * @return The determinant.
   * @note @p srcMatrix must contain floating point values.
   */
  template< typename MATRIX >
  LVARRAY_HOST_DEVICE constexpr inline
  static auto invert( MATRIX && matrix )
  {
    using realType = std::remove_reference_t< decltype( matrix[ 0 ][ 0 ] ) >;
#if 0
    realType temp[ 3 ][ 3 ];
    copy< 3, 3 >( temp, matrix );
    return invert( matrix, temp );
#else
    // cuda kernels use a couple fewer registers in some cases with this implementation.
    realType const temp[3][3] =
    { { matrix[1][1]*matrix[2][2] - matrix[1][2]*matrix[2][1], matrix[0][2]*matrix[2][1] - matrix[0][1]*matrix[2][2], matrix[0][1]*matrix[1][2] - matrix[0][2]*matrix[1][1] },
      { matrix[1][2]*matrix[2][0] - matrix[1][0]*matrix[2][2], matrix[0][0]*matrix[2][2] - matrix[0][2]*matrix[2][0], matrix[0][2]*matrix[1][0] - matrix[0][0]*matrix[1][2] },
      { matrix[1][0]*matrix[2][1] - matrix[1][1]*matrix[2][0], matrix[0][1]*matrix[2][0] - matrix[0][0]*matrix[2][1], matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0] } };

    realType const det =  matrix[0][0] * temp[0][0] + matrix[1][0] * temp[0][1] + matrix[2][0] * temp[0][2];
    realType const invDet = 1.0 / det;

    for( int i=0; i<3; ++i )
    {
      for( int j=0; j<3; ++j )
      {
        matrix[i][j] = temp[i][j] * invDet;
      }
    }
    return det;
#endif
  }

  /**
   * @return Return the determinant of the symmetric matrix @p symMatrix.
   * @tparam SYM_MATRIX The type of @p symMatrix.
   * @param symMatrix The 3x3 symmetric matrix to get the determinant of.
   */
  template< typename SYM_MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static auto symDeterminant( SYM_MATRIX const & symMatrix )
  {
    checkSizes< 6 >( symMatrix );

    return symMatrix[ 0 ] * symMatrix[ 1 ] * symMatrix[ 2 ] +
           symMatrix[ 5 ] * symMatrix[ 4 ] * symMatrix[ 3 ] * 2 -
           symMatrix[ 0 ] * symMatrix[ 3 ] * symMatrix[ 3 ] -
           symMatrix[ 1 ] * symMatrix[ 4 ] * symMatrix[ 4 ] -
           symMatrix[ 2 ] * symMatrix[ 5 ] * symMatrix[ 5 ];
  }

  /**
   * @brief Invert the symmetric matrix @p srcSymMatrix and store the result in @p dstSymMatrix.
   * @tparam DST_SYM_MATRIX The type of @p dstSymMatrix.
   * @tparam SRC_SYM_MATRIX The type of @p srcSymMatrix.
   * @param dstSymMatrix The 3x3 symmetric matrix to write the inverse to.
   * @param srcSymMatrix The 3x3 symmetric matrix to take the inverse of.
   * @return The determinant.
   * @note @p srcSymMatrix can contain integers but @p dstMatrix must contain floating point values.
   */
  template< typename DST_SYM_MATRIX, typename SRC_SYM_MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static auto symInvert( DST_SYM_MATRIX && LVARRAY_RESTRICT_REF dstSymMatrix,
                         SRC_SYM_MATRIX const & LVARRAY_RESTRICT_REF srcSymMatrix )
  {
    checkSizes< 6 >( dstSymMatrix );
    checkSizes< 6 >( srcSymMatrix );

    using FloatingPoint = std::decay_t< decltype( dstSymMatrix[ 0 ] ) >;

    dstSymMatrix[ 0 ] = srcSymMatrix[ 1 ] * srcSymMatrix[ 2 ] - srcSymMatrix[ 3 ] * srcSymMatrix[ 3 ];
    dstSymMatrix[ 5 ] = srcSymMatrix[ 4 ] * srcSymMatrix[ 3 ] - srcSymMatrix[ 5 ] * srcSymMatrix[ 2 ];
    dstSymMatrix[ 4 ] = srcSymMatrix[ 5 ] * srcSymMatrix[ 3 ] - srcSymMatrix[ 4 ] * srcSymMatrix[ 1 ];

    auto const det = srcSymMatrix[ 0 ] * dstSymMatrix[ 0 ] +
                     srcSymMatrix[ 5 ] * dstSymMatrix[ 5 ] +
                     srcSymMatrix[ 4 ] * dstSymMatrix[ 4 ];
    FloatingPoint const invDet = FloatingPoint( 1 ) / det;

    dstSymMatrix[ 0 ] *= invDet;
    dstSymMatrix[ 5 ] *= invDet;
    dstSymMatrix[ 4 ] *= invDet;
    dstSymMatrix[ 1 ] = ( srcSymMatrix[ 0 ] * srcSymMatrix[ 2 ] - srcSymMatrix[ 4 ] * srcSymMatrix[ 4 ] ) * invDet;
    dstSymMatrix[ 3 ] = ( srcSymMatrix[ 5 ] * srcSymMatrix[ 4 ] - srcSymMatrix[ 0 ] * srcSymMatrix[ 3 ] ) * invDet;
    dstSymMatrix[ 2 ] = ( srcSymMatrix[ 0 ] * srcSymMatrix[ 1 ] - srcSymMatrix[ 5 ] * srcSymMatrix[ 5 ] ) * invDet;

    return det;
  }

  /**
   * @brief Invert the symmetric matrix @p symMatrix overwritting it.
   * @tparam SYM_MATRIX The type of @p symMatrix.
   * @param symMatrix The 3x3 symmetric matrix to take the inverse of and overwrite.
   * @return The determinant.
   * @note @p symMatrix can contain integers but @p dstMatrix must contain floating point values.
   */
  template< typename SYM_MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static auto symInvert( SYM_MATRIX && symMatrix )
  {
    std::remove_reference_t< decltype( symMatrix[ 0 ] ) > temp[ 6 ];
    auto const det = symInvert( temp, symMatrix );
    copy< 6 >( symMatrix, temp );

    return det;
  }

  /**
   * @brief Multiply the vector @p vectorB by the symmetric matrix @p symMatrixA and store the result in @p
   * dstVector.
   * @tparam DST_VECTOR The type of @p dstVector.
   * @tparam SYM_MATRIX_A The type of @p symMatrixA.
   * @tparam VECTOR_B The type of @p vectorB.
   * @param dstVector The vector of length 3 to write the result to.
   * @param symMatrixA The 3x3 symmetric matrix (vector of length 6) to multiply @p vectorB by.
   * @param vectorB The vector of length 3 to be multiplied by @p symMatrixA.
   * @details Performs the operation @code dstVector[ i ] = symMatrixA[ i ][ j ] * vectorB[ j ] @endcode
   */
  template< typename DST_VECTOR, typename SYM_MATRIX_A, typename VECTOR_B >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void Ri_eq_symAijBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                              SYM_MATRIX_A const & LVARRAY_RESTRICT_REF symMatrixA,
                              VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
  {
    checkSizes< 3 >( dstVector );
    checkSizes< 6 >( symMatrixA );
    checkSizes< 3 >( vectorB );

    dstVector[ 0 ] = symMatrixA[ 0 ] * vectorB[ 0 ] +
                     symMatrixA[ 5 ] * vectorB[ 1 ] +
                     symMatrixA[ 4 ] * vectorB[ 2 ];
    dstVector[ 1 ] = symMatrixA[ 5 ] * vectorB[ 0 ] +
                     symMatrixA[ 1 ] * vectorB[ 1 ] +
                     symMatrixA[ 3 ] * vectorB[ 2 ];
    dstVector[ 2 ] = symMatrixA[ 4 ] * vectorB[ 0 ] +
                     symMatrixA[ 3 ] * vectorB[ 1 ] +
                     symMatrixA[ 2 ] * vectorB[ 2 ];
  }

  /**
   * @brief Multiply the vector @p vectorB by the symmetric matrix @p symMatrixA and add the result to @p dstVector.
   * @tparam DST_VECTOR The type of @p dstVector.
   * @tparam SYM_MATRIX_A The type of @p symMatrixA.
   * @tparam VECTOR_B The type of @p vectorB.
   * @param dstVector The vector of length 3 to add the result to.
   * @param symMatrixA The 3x3 symmetric matrix (vector of length 6) to multiply @p vectorB by.
   * @param vectorB The vector of length 3 to be multiplied by @p symMatrixA.
   * @details Performs the operation @code dstVector[ i ] += symMatrixA[ i ][ j ] * vectorB[ j ] @endcode
   */
  template< typename DST_VECTOR, typename SYM_MATRIX_A, typename VECTOR_B >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void Ri_add_symAijBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                               SYM_MATRIX_A const & LVARRAY_RESTRICT_REF symMatrixA,
                               VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
  {
    checkSizes< 3 >( dstVector );
    checkSizes< 6 >( symMatrixA );
    checkSizes< 3 >( vectorB );

    dstVector[ 0 ] = dstVector[ 0 ] +
                     symMatrixA[ 0 ] * vectorB[ 0 ] +
                     symMatrixA[ 5 ] * vectorB[ 1 ] +
                     symMatrixA[ 4 ] * vectorB[ 2 ];
    dstVector[ 1 ] = dstVector[ 1 ] +
                     symMatrixA[ 5 ] * vectorB[ 0 ] +
                     symMatrixA[ 1 ] * vectorB[ 1 ] +
                     symMatrixA[ 3 ] * vectorB[ 2 ];
    dstVector[ 2 ] = dstVector[ 2 ] +
                     symMatrixA[ 4 ] * vectorB[ 0 ] +
                     symMatrixA[ 3 ] * vectorB[ 1 ] +
                     symMatrixA[ 2 ] * vectorB[ 2 ];
  }

  /**
   * @brief Multiply the transpose of matrix @p matrixB by the symmetric matrix @p symMatrixA and store
   *   the result in @p dstMatrix.
   * @tparam DST_MATRIX The type of @p dstMatrix.
   * @tparam SYM_MATRIX_A The type of @p symMatrixA.
   * @tparam MATRIX_B The type of @p matrixB.
   * @param dstMatrix The 3x3 matrix to write the result to.
   * @param symMatrixA The 3x3 symmetric matrix (vector of length 6) to multiply @p matrixB by.
   * @param matrixB The 3x3 matrix to be multiplied by @p matrixB.
   * @details Performs the operation @code dstMatrix[ i ][ j ] = symMatrixA[ i ][ k ] * matrixB[ j ][ k ] @endcode
   */
  template< typename DST_MATRIX, typename SYM_MATRIX_A, typename MATRIX_B >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void Rij_eq_symAikBjk( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                                SYM_MATRIX_A const & LVARRAY_RESTRICT_REF symMatrixA,
                                MATRIX_B const & LVARRAY_RESTRICT_REF matrixB )
  {
    checkSizes< 3, 3 >( dstMatrix );
    checkSizes< 6 >( symMatrixA );
    checkSizes< 3, 3 >( matrixB );

    dstMatrix[ 0 ][ 0 ] = symMatrixA[ 0 ] * matrixB[ 0 ][ 0 ] +
                          symMatrixA[ 5 ] * matrixB[ 0 ][ 1 ] +
                          symMatrixA[ 4 ] * matrixB[ 0 ][ 2 ];
    dstMatrix[ 0 ][ 1 ] = symMatrixA[ 0 ] * matrixB[ 1 ][ 0 ] +
                          symMatrixA[ 5 ] * matrixB[ 1 ][ 1 ] +
                          symMatrixA[ 4 ] * matrixB[ 1 ][ 2 ];
    dstMatrix[ 0 ][ 2 ] = symMatrixA[ 0 ] * matrixB[ 2 ][ 0 ] +
                          symMatrixA[ 5 ] * matrixB[ 2 ][ 1 ] +
                          symMatrixA[ 4 ] * matrixB[ 2 ][ 2 ];

    dstMatrix[ 1 ][ 0 ] = symMatrixA[ 5 ] * matrixB[ 0 ][ 0 ] +
                          symMatrixA[ 1 ] * matrixB[ 0 ][ 1 ] +
                          symMatrixA[ 3 ] * matrixB[ 0 ][ 2 ];
    dstMatrix[ 1 ][ 1 ] = symMatrixA[ 5 ] * matrixB[ 1 ][ 0 ] +
                          symMatrixA[ 1 ] * matrixB[ 1 ][ 1 ] +
                          symMatrixA[ 3 ] * matrixB[ 1 ][ 2 ];
    dstMatrix[ 1 ][ 2 ] = symMatrixA[ 5 ] * matrixB[ 2 ][ 0 ] +
                          symMatrixA[ 1 ] * matrixB[ 2 ][ 1 ] +
                          symMatrixA[ 3 ] * matrixB[ 2 ][ 2 ];

    dstMatrix[ 2 ][ 0 ] = symMatrixA[ 4 ] * matrixB[ 0 ][ 0 ] +
                          symMatrixA[ 3 ] * matrixB[ 0 ][ 1 ] +
                          symMatrixA[ 2 ] * matrixB[ 0 ][ 2 ];
    dstMatrix[ 2 ][ 1 ] = symMatrixA[ 4 ] * matrixB[ 1 ][ 0 ] +
                          symMatrixA[ 3 ] * matrixB[ 1 ][ 1 ] +
                          symMatrixA[ 2 ] * matrixB[ 1 ][ 2 ];
    dstMatrix[ 2 ][ 2 ] = symMatrixA[ 4 ] * matrixB[ 2 ][ 0 ] +
                          symMatrixA[ 3 ] * matrixB[ 2 ][ 1 ] +
                          symMatrixA[ 2 ] * matrixB[ 2 ][ 2 ];
  }

  /**
   * @brief Multiply the transpose of matrix @p matrixA by the symmetric matrix @p symMatrixB then by @p matrixA
   *   and store the result in @p dstSymMatrix.
   * @tparam DST_SYM_MATRIX The type of @p dstSymMatrix.
   * @tparam MATRIX_A The type of @p matrixA.
   * @tparam SYM_MATRIX_B The type of @p symMatrixB.
   * @param dstSymMatrix The 3x3 symmetric matrix (vector of length 6) to write the result to.
   * @param matrixA The 3x3 matrix to pre and post multiply @p symMatrixB by.
   * @param symMatrixB The 3x3 symmetric matrix (vector of length 6) that gets pre multiplied by @p matrixA
   *   and post postmultiplied by the transpose of @p matrixA.
   * @details Performs the operation
   *   @code dstSymMatrix[ i ][ j ] = matrixA[ i ][ k ] * symMatrixB[ k ][ l ] * matrixA[ j ][ l ] @endcode
   */
  template< typename DST_SYM_MATRIX, typename MATRIX_A, typename SYM_MATRIX_B >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void Rij_eq_AikSymBklAjl( DST_SYM_MATRIX && LVARRAY_RESTRICT_REF dstSymMatrix,
                                   MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                                   SYM_MATRIX_B const & LVARRAY_RESTRICT_REF symMatrixB )
  {
    checkSizes< 6 >( dstSymMatrix );
    checkSizes< 3, 3 >( matrixA );
    checkSizes< 6 >( symMatrixB );

    // Calculate entry (0, 0).
    dstSymMatrix[ 0 ] = matrixA[ 0 ][ 0 ] * symMatrixB[ 0 ] * matrixA[ 0 ][ 0 ] +
                        matrixA[ 0 ][ 0 ] * symMatrixB[ 5 ] * matrixA[ 0 ][ 1 ] +
                        matrixA[ 0 ][ 0 ] * symMatrixB[ 4 ] * matrixA[ 0 ][ 2 ] +
                        matrixA[ 0 ][ 1 ] * symMatrixB[ 5 ] * matrixA[ 0 ][ 0 ] +
                        matrixA[ 0 ][ 1 ] * symMatrixB[ 1 ] * matrixA[ 0 ][ 1 ] +
                        matrixA[ 0 ][ 1 ] * symMatrixB[ 3 ] * matrixA[ 0 ][ 2 ] +
                        matrixA[ 0 ][ 2 ] * symMatrixB[ 4 ] * matrixA[ 0 ][ 0 ] +
                        matrixA[ 0 ][ 2 ] * symMatrixB[ 3 ] * matrixA[ 0 ][ 1 ] +
                        matrixA[ 0 ][ 2 ] * symMatrixB[ 2 ] * matrixA[ 0 ][ 2 ];

    // Calculate entry (1, 1).
    dstSymMatrix[ 1 ] = matrixA[ 1 ][ 0 ] * symMatrixB[ 0 ] * matrixA[ 1 ][ 0 ] +
                        matrixA[ 1 ][ 0 ] * symMatrixB[ 5 ] * matrixA[ 1 ][ 1 ] +
                        matrixA[ 1 ][ 0 ] * symMatrixB[ 4 ] * matrixA[ 1 ][ 2 ] +
                        matrixA[ 1 ][ 1 ] * symMatrixB[ 5 ] * matrixA[ 1 ][ 0 ] +
                        matrixA[ 1 ][ 1 ] * symMatrixB[ 1 ] * matrixA[ 1 ][ 1 ] +
                        matrixA[ 1 ][ 1 ] * symMatrixB[ 3 ] * matrixA[ 1 ][ 2 ] +
                        matrixA[ 1 ][ 2 ] * symMatrixB[ 4 ] * matrixA[ 1 ][ 0 ] +
                        matrixA[ 1 ][ 2 ] * symMatrixB[ 3 ] * matrixA[ 1 ][ 1 ] +
                        matrixA[ 1 ][ 2 ] * symMatrixB[ 2 ] * matrixA[ 1 ][ 2 ];

    // Calculate entry (2, 2).
    dstSymMatrix[ 2 ] = matrixA[ 2 ][ 0 ] * symMatrixB[ 0 ] * matrixA[ 2 ][ 0 ] +
                        matrixA[ 2 ][ 0 ] * symMatrixB[ 5 ] * matrixA[ 2 ][ 1 ] +
                        matrixA[ 2 ][ 0 ] * symMatrixB[ 4 ] * matrixA[ 2 ][ 2 ] +
                        matrixA[ 2 ][ 1 ] * symMatrixB[ 5 ] * matrixA[ 2 ][ 0 ] +
                        matrixA[ 2 ][ 1 ] * symMatrixB[ 1 ] * matrixA[ 2 ][ 1 ] +
                        matrixA[ 2 ][ 1 ] * symMatrixB[ 3 ] * matrixA[ 2 ][ 2 ] +
                        matrixA[ 2 ][ 2 ] * symMatrixB[ 4 ] * matrixA[ 2 ][ 0 ] +
                        matrixA[ 2 ][ 2 ] * symMatrixB[ 3 ] * matrixA[ 2 ][ 1 ] +
                        matrixA[ 2 ][ 2 ] * symMatrixB[ 2 ] * matrixA[ 2 ][ 2 ];

    // Calculate entry (1, 2) or (2, 1).
    dstSymMatrix[ 3 ] = matrixA[ 1 ][ 0 ] * symMatrixB[ 0 ] * matrixA[ 2 ][ 0 ] +
                        matrixA[ 1 ][ 0 ] * symMatrixB[ 5 ] * matrixA[ 2 ][ 1 ] +
                        matrixA[ 1 ][ 0 ] * symMatrixB[ 4 ] * matrixA[ 2 ][ 2 ] +
                        matrixA[ 1 ][ 1 ] * symMatrixB[ 5 ] * matrixA[ 2 ][ 0 ] +
                        matrixA[ 1 ][ 1 ] * symMatrixB[ 1 ] * matrixA[ 2 ][ 1 ] +
                        matrixA[ 1 ][ 1 ] * symMatrixB[ 3 ] * matrixA[ 2 ][ 2 ] +
                        matrixA[ 1 ][ 2 ] * symMatrixB[ 4 ] * matrixA[ 2 ][ 0 ] +
                        matrixA[ 1 ][ 2 ] * symMatrixB[ 3 ] * matrixA[ 2 ][ 1 ] +
                        matrixA[ 1 ][ 2 ] * symMatrixB[ 2 ] * matrixA[ 2 ][ 2 ];

    // Calculate entry (0, 2) or (2, 0).
    dstSymMatrix[ 4 ] = matrixA[ 0 ][ 0 ] * symMatrixB[ 0 ] * matrixA[ 2 ][ 0 ] +
                        matrixA[ 0 ][ 0 ] * symMatrixB[ 5 ] * matrixA[ 2 ][ 1 ] +
                        matrixA[ 0 ][ 0 ] * symMatrixB[ 4 ] * matrixA[ 2 ][ 2 ] +
                        matrixA[ 0 ][ 1 ] * symMatrixB[ 5 ] * matrixA[ 2 ][ 0 ] +
                        matrixA[ 0 ][ 1 ] * symMatrixB[ 1 ] * matrixA[ 2 ][ 1 ] +
                        matrixA[ 0 ][ 1 ] * symMatrixB[ 3 ] * matrixA[ 2 ][ 2 ] +
                        matrixA[ 0 ][ 2 ] * symMatrixB[ 4 ] * matrixA[ 2 ][ 0 ] +
                        matrixA[ 0 ][ 2 ] * symMatrixB[ 3 ] * matrixA[ 2 ][ 1 ] +
                        matrixA[ 0 ][ 2 ] * symMatrixB[ 2 ] * matrixA[ 2 ][ 2 ];

    // Calculate entry (0, 1) or (1, 0).
    dstSymMatrix[ 5 ] = matrixA[ 0 ][ 0 ] * symMatrixB[ 0 ] * matrixA[ 1 ][ 0 ] +
                        matrixA[ 0 ][ 0 ] * symMatrixB[ 5 ] * matrixA[ 1 ][ 1 ] +
                        matrixA[ 0 ][ 0 ] * symMatrixB[ 4 ] * matrixA[ 1 ][ 2 ] +
                        matrixA[ 0 ][ 1 ] * symMatrixB[ 5 ] * matrixA[ 1 ][ 0 ] +
                        matrixA[ 0 ][ 1 ] * symMatrixB[ 1 ] * matrixA[ 1 ][ 1 ] +
                        matrixA[ 0 ][ 1 ] * symMatrixB[ 3 ] * matrixA[ 1 ][ 2 ] +
                        matrixA[ 0 ][ 2 ] * symMatrixB[ 4 ] * matrixA[ 1 ][ 0 ] +
                        matrixA[ 0 ][ 2 ] * symMatrixB[ 3 ] * matrixA[ 1 ][ 1 ] +
                        matrixA[ 0 ][ 2 ] * symMatrixB[ 2 ] * matrixA[ 1 ][ 2 ];
  }

  /**
   * @brief Perform the outer product of @p vectorA with itself writing the result to @p dstMatrix.
   * @tparam M The size of both dimensions of @p dstMatrix and the length of @p vectorA.
   * @tparam DST_MATRIX The type of @p dstMatrix.
   * @tparam VECTOR_A The type of @p vectorA.
   * @param dstMatrix The matrix the result is written to, of size M x N.
   * @param vectorA The first vector in the outer product, of length M.
   * @details Performs the operations @code dstMatrix[ i ][ j ] = vectorA[ i ] * vectorA[ j ] @endcode
   */
  template< typename DST_MATRIX, typename VECTOR_A >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void symRij_eq_AiAj( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                              VECTOR_A const & LVARRAY_RESTRICT_REF vectorA )
  {
    internal::checkSizes< 6 >( dstMatrix );
    internal::checkSizes< 3 >( vectorA );

    dstMatrix[ 0 ] = vectorA[ 0 ] * vectorA[ 0 ];
    dstMatrix[ 1 ] = vectorA[ 1 ] * vectorA[ 1 ];
    dstMatrix[ 2 ] = vectorA[ 2 ] * vectorA[ 2 ];
    dstMatrix[ 3 ] = vectorA[ 1 ] * vectorA[ 2 ];
    dstMatrix[ 4 ] = vectorA[ 0 ] * vectorA[ 2 ];
    dstMatrix[ 5 ] = vectorA[ 0 ] * vectorA[ 1 ];
  }

  /**
   * @brief Perform the unscaled symmetric outer product of @p vectorA and
   *   @p vectorB writing the result to @p dstMatrix.
   * @tparam DST_MATRIX The type of @p dstMatrix.
   * @tparam VECTOR_A The type of @p vectorA.
   * @tparam VECTOR_B The type of @p vectorB.
   * @param dstMatrix The matrix the result is written to, of size M x N.
   * @param vectorA The first vector in the outer product, of length M.
   * @param vectorB The second vector in the outer product, of length M.
   * @details Performs the operations @code dstMatrix[ i ][ j ] = vectorA[ i ] * vectorB[ j ] + vectorA[ j ] * vectorB[ i ] @endcode
   */
  template< typename DST_SYM_MATRIX, typename VECTOR_A, typename VECTOR_B >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void symRij_eq_AiBj_plus_AjBi( DST_SYM_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                                        VECTOR_A const & LVARRAY_RESTRICT_REF vectorA,
                                        VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
  {
    internal::checkSizes< 6 >( dstMatrix );
    internal::checkSizes< 3 >( vectorA );
    internal::checkSizes< 3 >( vectorB );

    dstMatrix[ 0 ] = 2 * vectorA[ 0 ] * vectorB[ 0 ];
    dstMatrix[ 1 ] = 2 * vectorA[ 1 ] * vectorB[ 1 ];
    dstMatrix[ 2 ] = 2 * vectorA[ 2 ] * vectorB[ 2 ];
    dstMatrix[ 3 ] = vectorA[ 1 ] * vectorB[ 2 ] + vectorA[ 2 ] * vectorB[ 1 ];
    dstMatrix[ 4 ] = vectorA[ 0 ] * vectorB[ 2 ] + vectorA[ 2 ] * vectorB[ 0 ];
    dstMatrix[ 5 ] = vectorA[ 0 ] * vectorB[ 1 ] + vectorA[ 1 ] * vectorB[ 0 ];
  }

  /**
   * @brief Compute the eigenvalues of the symmetric matrix @p symMatrix.
   * @tparam DST_VECTOR The type of @p eigenvalues.
   * @tparam SYM_MATRIX The type of @p symMatrix.
   * @param eigenvalues The vector of length 3 to write the eigenvalues to.
   * @param symMatrix The 3x3 symmetric matrix to compute the eigenvalues of.
   * @details Computes the eigenvalues directly, they are returned sorted in ascending order.
   * @note @p symMatrix can contain integers but @p eigenvalues must contain floating point numbers.
   */
  template< typename DST_VECTOR, typename SYM_MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void symEigenvalues( DST_VECTOR && LVARRAY_RESTRICT_REF eigenvalues,
                              SYM_MATRIX const & LVARRAY_RESTRICT_REF symMatrix )
  {
    checkSizes< 3 >( eigenvalues );
    checkSizes< 6 >( symMatrix );

    using FloatingPoint = std::decay_t< decltype( eigenvalues[ 0 ] ) >;

    FloatingPoint shift, maxEntryAfterShift;
    FloatingPoint fpCopy[ 6 ];
    copy< 6 >( fpCopy, symMatrix );
    shiftAndScale< 3 >( fpCopy, shift, maxEntryAfterShift );

    eigenvaluesOfShiftedMatrix( eigenvalues, fpCopy );

    // Rescale back to the original size.
    eigenvalues[ 0 ] = maxEntryAfterShift * eigenvalues[ 0 ] + shift;
    eigenvalues[ 1 ] = maxEntryAfterShift * eigenvalues[ 1 ] + shift;
    eigenvalues[ 2 ] = maxEntryAfterShift * eigenvalues[ 2 ] + shift;
  }

  /**
   * @brief Compute the eigenvalues and eigenvectors of the symmetric matrix @p symMatrix.
   * @tparam DST_VECTOR The type of @p eigenvalues.
   * @tparam DST_MATRIX The type of @p eigenvectors
   * @tparam SYM_MATRIX The type of @p symMatrix.
   * @param eigenvalues The vector of length 3 to write the eigenvalues to.
   * @param eigenvectors The 3x3 matrix to write the eigenvectors to.
   * @param symMatrix The 3x3 symmetric matrix to compute the eigenvalues of.
   * @details Computes the eigenvalues directly, they are returned sorted in ascending order.
   *   The row @code eigenvectors[ i ] @endcode contains the eigenvector corresponding to
   *   @code eigenvalues[ i ] @endcode.
   * @note @p symMatrix can contain integers but @p eigenvalues must contain floating point numbers.
   */
  template< typename DST_VECTOR, typename DST_MATRIX, typename SYM_MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void symEigenvectors( DST_VECTOR && LVARRAY_RESTRICT_REF eigenvalues,
                               DST_MATRIX && LVARRAY_RESTRICT_REF eigenvectors,
                               SYM_MATRIX const & LVARRAY_RESTRICT_REF symMatrix )
  {
    checkSizes< 3 >( eigenvalues );
    checkSizes< 3, 3 >( eigenvectors );
    checkSizes< 6 >( symMatrix );

    using FloatingPoint = std::decay_t< decltype( eigenvalues[ 0 ] ) >;

    FloatingPoint shift, maxEntryAfterShift;
    FloatingPoint fpCopy[ 6 ];
    copy< 6 >( fpCopy, symMatrix );
    shiftAndScale< 3 >( fpCopy, shift, maxEntryAfterShift );

    eigenvaluesOfShiftedMatrix( eigenvalues, fpCopy );

    // compute the eigenvectors
    FloatingPoint const eigenvalueDifference = eigenvalues[ 2 ] - eigenvalues[ 0 ];
    if( eigenvalueDifference <= NumericLimits< FloatingPoint >::epsilon )
    {
      // All three eigenvalues are numerically the same
      LVARRAY_TENSOROPS_ASSIGN_3x3( eigenvectors,
                                    1, 0, 0,
                                    0, 1, 0,
                                    0, 0, 1 );
    }
    else
    {
      // Compute the eigenvector of the most distinct eigenvalue first. Since the eigenvalues are sorted
      // this is the eigenvector corresponding to either the first or last eigenvalue.
      int const mostDistinct = ( ( eigenvalues[ 2 ] - eigenvalues[ 1 ] ) >
                                 ( eigenvalues[ 1 ] - eigenvalues[ 0 ] ) ) ? 2 : 0;
      int const secondMostDistinct = 2 - mostDistinct;

      // Compute the first eigenvector. This is done by subtracting the identity times the most distinct eigenvalue
      // from the matrix which gives us a rank 2 matrix where the nullspace is the corresponding eigenvector.
      FloatingPoint tmp[ 3 ][ 3 ] = { { fpCopy[ 0 ] - eigenvalues[ mostDistinct ], fpCopy[ 5 ], fpCopy[ 4 ] },
        { fpCopy[ 5 ], fpCopy[ 1 ] - eigenvalues[ mostDistinct ], fpCopy[ 3 ] },
        { fpCopy[ 4 ], fpCopy[ 3 ], fpCopy[ 2 ] - eigenvalues[ mostDistinct ] } };

      int const row = getNullVector( eigenvectors[ mostDistinct ], tmp );

      // Compute eigenvector of the second most distinct eigenvalue.
      FloatingPoint const minEigenvalueSpacing = math::abs( eigenvalues[ secondMostDistinct ] - eigenvalues[ 1 ] );
      if( minEigenvalueSpacing <= 2 * NumericLimits< FloatingPoint >::epsilon * eigenvalueDifference )
      {
        // If minEigenvalueSpacing is too small then the other two eigenvalues are numerically the same
        // we can use tmp[ row ].
        FloatingPoint const norm2 = l2NormSquared< 3 >( tmp[ row ] );
        scaledCopy< 3 >( eigenvectors[ secondMostDistinct ], tmp[ row ], math::invSqrt( norm2 ) );
      }
      else
      {
        // Otherwise repeat the procedure with the second most distinct eigenvalue.
        tmp[ 0 ][ 0 ] = fpCopy[ 0 ] - eigenvalues[ secondMostDistinct ];
        tmp[ 1 ][ 1 ] = fpCopy[ 1 ] - eigenvalues[ secondMostDistinct ];
        tmp[ 2 ][ 2 ] = fpCopy[ 2 ] - eigenvalues[ secondMostDistinct ];

        getNullVector( eigenvectors[ secondMostDistinct ], tmp );
      }

      // Compute last eigenvector from the other two
      crossProduct( eigenvectors[ 1 ], eigenvectors[ 2 ], eigenvectors[ 0 ] );
      normalize< 3 >( eigenvectors[ 1 ] );
    }

    // Rescale back to the original size.
    eigenvalues[ 0 ] = maxEntryAfterShift * eigenvalues[ 0 ] + shift;
    eigenvalues[ 1 ] = maxEntryAfterShift * eigenvalues[ 1 ] + shift;
    eigenvalues[ 2 ] = maxEntryAfterShift * eigenvalues[ 2 ] + shift;
  }

  /**
   * @brief Convert the upper triangular part of @p srcMatrix to a symmetric matrix.
   * @tparam DST_SYM_MATRIX The type of @p dstSymMatrix.
   * @tparam SRC_MATRIX The type of @p srcMatrix.
   * @param dstSymMatrix The resulting 3x3 symmetric matrix.
   * @param srcMatrix The 3x3 matrix to convert to a symmetric matrix.
   */
  template< typename DST_SYM_MATRIX, typename SRC_MATRIX >
  LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
  static void denseToSymmetric( DST_SYM_MATRIX && dstSymMatrix, SRC_MATRIX const & srcMatrix )
  {
    tensorOps::internal::checkSizes< 6 >( dstSymMatrix );
    tensorOps::internal::checkSizes< 3, 3 >( srcMatrix );

    dstSymMatrix[ 0 ] = srcMatrix[ 0 ][ 0 ];
    dstSymMatrix[ 1 ] = srcMatrix[ 1 ][ 1 ];
    dstSymMatrix[ 2 ] = srcMatrix[ 2 ][ 2 ];
    dstSymMatrix[ 3 ] = srcMatrix[ 1 ][ 2 ];
    dstSymMatrix[ 4 ] = srcMatrix[ 0 ][ 2 ];
    dstSymMatrix[ 5 ] = srcMatrix[ 0 ][ 1 ];
  }

  /**
   * @brief Convert the @p srcSymMatrix into a dense matrix.
   * @tparam DST_MATRIX The type of @p dstMatrix.
   * @tparam SRC_SYM_MATRIX The type of @p srcSymMatrix.
   * @param dstMatrix The resulting 3x3 matrix.
   * @param srcSymMatrix The 3x3 symmetric matrix to convert.
   */
  template< typename DST_MATRIX, typename SRC_SYM_MATRIX >
  LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
  static void symmetricToDense( DST_MATRIX && dstMatrix, SRC_SYM_MATRIX const & srcSymMatrix )
  {
    tensorOps::internal::checkSizes< 3, 3 >( dstMatrix );
    tensorOps::internal::checkSizes< 6 >( srcSymMatrix );

    dstMatrix[ 0 ][ 0 ] = srcSymMatrix[ 0 ];
    dstMatrix[ 1 ][ 1 ] = srcSymMatrix[ 1 ];
    dstMatrix[ 2 ][ 2 ] = srcSymMatrix[ 2 ];

    dstMatrix[ 0 ][ 1 ] = srcSymMatrix[ 5 ];
    dstMatrix[ 0 ][ 2 ] = srcSymMatrix[ 4 ];
    dstMatrix[ 1 ][ 2 ] = srcSymMatrix[ 3 ];

    dstMatrix[ 1 ][ 0 ] = srcSymMatrix[ 5 ];
    dstMatrix[ 2 ][ 0 ] = srcSymMatrix[ 4 ];
    dstMatrix[ 2 ][ 1 ] = srcSymMatrix[ 3 ];
  }

private:
  /**
   * @brief Compute the eigenvalues of the 3x3 symmetric matrix @p matrix.
   * @tparam FloatingPoint A floating point type.
   * @tparam VECTOR The type of @p eigenvalues.
   * @param eigenvalues The resulting eigenvalues.
   * @param matrix A 3x3 symmetric matrix with trace 0.
   */
  template< typename FloatingPoint, typename VECTOR >
  LVARRAY_HOST_DEVICE inline
  static void eigenvaluesOfShiftedMatrix( VECTOR && eigenvalues,
                                          FloatingPoint const ( &matrix )[ 6 ] )
  {
    /*
       For a 3x3 symmetric matrix A
          a0, a5, a4
          a5, a1, a3
          a4, a3, a2
       And eigenvalue x the characteristic equation is
          x^3 - tr(A) * x^2 - 0.5 * (tr(A^2) - tr(A)^2) * x - det( A ) = 0
       However the shifted matrix has a trace of 0 so this simplifies to
          x^3 - 0.5 * tr(A^2) * x - det( A ) = 0

       This approach is a hybrid of the wikipedia algorithm and Eigen's algorithm.
       The wikipedia algorithm uses one less call to sqrt to find theta but Eigen's
       conversion from theta to the roots is faster.

       https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3%C3%973_matrices.
       https://gitlab.com/libeigen/eigen/-/blob/3.3/Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h#L567
     */

    // The calculation of the trace(A^2) uses the fact that trace(A) = 0 and hence a2 = -a1 - a0.
    FloatingPoint const oneSixthTraceASquared =
      ( matrix[ 2 ] * matrix[ 2 ] - matrix[ 0 ] * matrix[ 1 ] + matrix[ 3 ] * matrix[ 3 ] +
        matrix[ 4 ] * matrix[ 4 ] + matrix[ 5 ] * matrix[ 5 ] ) / 3;

    FloatingPoint const p = math::sqrt( oneSixthTraceASquared );

    FloatingPoint const det = ( p > 0 ) ? symDeterminant( matrix ) / ( p * p * p ) : 0;

    // det should be in [-2, 2] but it may be slightly outside do to floating point error.
    FloatingPoint const halfDetB = det > 2 ? 1 : ( det < -2 ? -1 : det / 2 );
    FloatingPoint const theta = math::acos( halfDetB ) / 3;

    FloatingPoint sinTheta, cosTheta;
    math::sincos( theta, sinTheta, cosTheta );

    // roots are already sorted, since cos is monotonically decreasing on [0, pi]
    constexpr FloatingPoint squareRootThree = 1.73205080756887729352744;
    eigenvalues[ 0 ] = -p * ( cosTheta + squareRootThree * sinTheta ); // == 2 * p * cos( theta + 2pi/3 )
    eigenvalues[ 1 ] = -p * ( cosTheta - squareRootThree * sinTheta ); // == 2 * p * cos( theta +  pi/3 )
    eigenvalues[ 2 ] = 2 * p * cosTheta;
  }

  /**
   * @brief Extract the nullspace of the rank 2 3x3 symmetric matrix @p mat.
   * @tparam FLOAT The value type of @p mat.
   * @tparam VECTOR The type of @p nullVector.
   * @param nullVector The vector that spans the nullspace of @p mat.
   * @param mat The rank 2 3x3 symmetric matrix.
   * @note Since @p mat is rank 2 it's nullspace is spanned by a single vector.
   * @return The index of the row with the largest diagonal value. This will be orthogonal to @p nullVector.
   */
  template< typename FLOAT, typename VECTOR >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static int getNullVector( VECTOR && nullVector, FLOAT const (&mat)[ 3 ][ 3 ] )
  {
    // Find the row with the largest diagonal value.
    FLOAT const abs00 = math::abs( mat[ 0 ][ 0 ] );
    FLOAT const abs11 = math::abs( mat[ 1 ][ 1 ] );
    FLOAT const abs22 = math::abs( mat[ 2 ][ 2 ] );

    int const row = abs00 > abs11 ? ( abs00 > abs22 ? 0 : 2 ) :
                    ( abs11 > abs22 ? 1 : 2 );

    // From each of the other two rows in the matrix construct an orthogonal vector.
    crossProduct( nullVector, mat[ row ], mat[ ( row + 1 ) % 3 ] );
    FLOAT const n0 = l2NormSquared< 3 >( nullVector );

    FLOAT nullVectorCandidate[ 3 ];
    crossProduct( nullVectorCandidate, mat[ row ], mat[ ( row + 2 ) % 3 ] );
    FLOAT const n1 = l2NormSquared< 3 >( nullVectorCandidate );

    // Choose the null vector with the largest magnitude.
    if( n0 > n1 )
    { scale< 3 >( nullVector, math::invSqrt( n0 ) ); }
    else
    { scaledCopy< 3 >( nullVector, nullVectorCandidate, math::invSqrt( n1 ) ); }

    return row;
  }
};

} // namespace internal
} // namespace tensorOps
} // namespace LvArray
