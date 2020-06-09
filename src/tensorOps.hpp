/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2019, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-746361
 *
 * All rights reserved. See COPYRIGHT for details.
 *
 * This file is part of the GEOSX Simulation Framework.
 *
 * GEOSX is a free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

/**
 * @file tensorOps.hpp
 */

#include "ArraySlice.hpp"

#include <cmath>

#pragma once

/**
 * @brief Forward declaration of R1Tensor
 * TODO: Remove this.
 */
template< int T_dim >
class R1TensorT;

/**
 * @name Initialization macros.
 * @brief Macros that expand to initializer lists for common vector and matrix sizes.
 * @details Use these macros to initialize c-arrays from other c-arrays or ArraySlices.
 *   Usage:
 *   @code
 *   int vectorA[ 3 ] = { 0, 1, 2 };
 *   int const vectorACopy[ 3 ] = LVARRAY_TENSOROPS_INIT_LOCAL_3( vectorA );
 *
 *   Array< double, 3 > arrayOfMatrices( N, 3, 3 );
 *   double matrix[ 3 ][ 3 ] = LVARRAY_TENSOROPS_INIT_LOCAL_3x3( arrayOfMatrices[ 0 ] );
 *   @endcode
 *   Note that since these are macros expressions and not just variables can be passed.
 *   @code
 *   int vectorA[ 3 ] = { 0, 1, 2 };
 *   int const vectorATimes2[ 3 ] = LVARRAY_TENSOROPS_INIT_LOCAL_3( 2 * vectorA );
 *   @endcode
 * @note No bounds checking is done on the provided expression.
 */
///@{

/**
 * @brief Create an initializer list of length 2 from @p EXP.
 * @param EXP The expression used to create the initialize list.
 */
#define LVARRAY_TENSOROPS_INIT_LOCAL_2( EXP ) { EXP[ 0 ], EXP[ 1 ] }

/**
 * @brief Create an initializer list of length 3 from @p EXP.
 * @param EXP The expression used to create the initialize list.
 */
#define LVARRAY_TENSOROPS_INIT_LOCAL_3( EXP ) { EXP[ 0 ], EXP[ 1 ], EXP[ 2 ] }

/**
 * @brief Create an initializer list of length 6 from @p EXP.
 * @param EXP The expression used to create the initialize list.
 */
#define LVARRAY_TENSOROPS_INIT_LOCAL_6( EXP ) { EXP[ 0 ], EXP[ 1 ], EXP[ 2 ], EXP[ 3 ], EXP[ 4 ], EXP[ 5 ] }

/**
 * @brief Create an 2x2 initializer list from @p EXP.
 * @param EXP The expression used to create the initialize list.
 */
#define LVARRAY_TENSOROPS_INIT_LOCAL_2x2( EXP ) \
  { { EXP[ 0 ][ 0 ], EXP[ 0 ][ 1 ] }, \
    { EXP[ 1 ][ 0 ], EXP[ 1 ][ 1 ] } }

/**
 * @brief Create an 3x3 initializer list from @p EXP.
 * @param EXP The expression used to create the initialize list.
 */
#define LVARRAY_TENSOROPS_INIT_LOCAL_3x3( EXP ) \
  { { EXP[ 0 ][ 0 ], EXP[ 0 ][ 1 ], EXP[ 0 ][ 2 ] }, \
    { EXP[ 1 ][ 0 ], EXP[ 1 ][ 1 ], EXP[ 1 ][ 2 ] }, \
    { EXP[ 2 ][ 0 ], EXP[ 2 ][ 1 ], EXP[ 2 ][ 2 ] } }

///@}


namespace LvArray
{
namespace tensorOps
{
namespace internal
{

/// The size of a symmetric MxM matrix in Voigt notation.
template< int M >
constexpr std::ptrdiff_t SYM_SIZE = ( M * ( M + 1 ) ) / 2;

/**
 * @brief Verify at compile time that the size of the R1Tensor is as expected.
 * @tparam PROVIDED_SIZE The size the R1Tensor should be.
 * @tparam INFERRED_SIZE The size of the R1Tensor.
 * TODO: Remove this.
 */
template< std::ptrdiff_t PROVIDED_SIZE, int INFERRED_SIZE >
LVARRAY_HOST_DEVICE inline constexpr
void checkSizes( R1TensorT< INFERRED_SIZE > const & )
{
  static_assert( PROVIDED_SIZE == INFERRED_SIZE,
                 "Expected the first dimension of size PROVIDED_N, got an array of size INFERRED_N." );
}

/**
 * @brief Verify at compile time that the size of the c-array is as expected.
 * @tparam PROVIDED_SIZE The size the array should be.
 * @tparam T The type contained in the array.
 * @tparam INFERRED_SIZE The size of the array.
 */
template< std::ptrdiff_t PROVIDED_SIZE, typename T, std::ptrdiff_t INFERRED_SIZE >
LVARRAY_HOST_DEVICE inline constexpr
void checkSizes( T const ( & )[ INFERRED_SIZE ] )
{
  static_assert( PROVIDED_SIZE == INFERRED_SIZE,
                 "Expected the first dimension of size PROVIDED_N, got an array of size INFERRED_N." );
}

/**
 * @brief Verify at compile time that the size of the 2D c-array is as expected.
 * @tparam PROVIDED_M What the size of the first dimension should be.
 * @tparam PROVIDED_M What the size of the second dimension should be.
 * @tparam T The type contained in the array.
 * @tparam PROVIDED_M The size of the first dimension.
 * @tparam PROVIDED_M The size of the second dimension.
 */
template< std::ptrdiff_t PROVIDED_M,
          std::ptrdiff_t PROVIDED_N,
          typename T,
          std::ptrdiff_t INFERRED_M,
          std::ptrdiff_t INFERRED_N >
LVARRAY_HOST_DEVICE inline constexpr
void checkSizes( T const ( & )[ INFERRED_M ][ INFERRED_N ] )
{
  static_assert( PROVIDED_M == INFERRED_M, "Expected the first dimension of size PROVIDED_M, got an array of size INFERRED_M." );
  static_assert( PROVIDED_N == INFERRED_N, "Expected the second dimension of size PROVIDED_N, got an array of size INFERRED_N." );
}

/**
 * @brief Verify at compile time that @tparam ARRAY is 1D and at runtime verify the size when USE_ARRAY_BOUNDS_CHECK.
 * @tparam M The size expected size.
 * @tparam ARRAY The type o @p array, should be an Array, ArrayView or ArraySlice.
 * @param array The array to check.
 */
template< std::ptrdiff_t M, typename ARRAY >
LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
void checkSizes( ARRAY const & array )
{
  static_assert( ARRAY::ndim == 1, "Must be a 1D array." );
  #ifdef USE_ARRAY_BOUNDS_CHECK
  LVARRAY_ERROR_IF_NE( array.size( 0 ), M );
  #else
  LVARRAY_UNUSED_VARIABLE( array );
  #endif
}

/**
 * @brief Verify at compile time that @tparam ARRAY is 2D and at runtime verify the sizes when USE_ARRAY_BOUNDS_CHECK.
 * @tparam M The expected size of the first dimension.
 * @tparam N The expected size of the second dimension.
 * @tparam ARRAY The type o @p array, should be an Array, ArrayView or ArraySlice.
 * @param array The array to check.
 */
template< std::ptrdiff_t M, std::ptrdiff_t N, typename ARRAY >
LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
void checkSizes( ARRAY const & array )
{
  static_assert( ARRAY::ndim == 2, "Must be a 1D array." );
#ifdef USE_ARRAY_BOUNDS_CHECK
  LVARRAY_ERROR_IF_NE( array.size( 0 ), M );
  LVARRAY_ERROR_IF_NE( array.size( 1 ), N );
#else
  LVARRAY_UNUSED_VARIABLE( array );
#endif
}

/**
 * @struct SquareMatrixOps
 * @brief Performs operations on square matrices.
 * @tparam M The size of the matrix.
 * @note This class is intended to have its methods overwritten by template specializations
 *   for specific matrix sizes, however generic methods could go here.
 */
template< int M >
struct SquareMatrixOps
{
  static_assert( M > 0, "The size of the matrix must be greater than 0." );

  /**
   * @brief @return Return the determinant of the matrix @p matrix.
   * @tparam MATRIX The type of @p matrix.
   * @param matrix The matrix to get the determinant of.
   * @note This is not implemented for generic sized matrices and calling this is a compilation error.
   */
  template< typename MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static int determinant( MATRIX const & matrix )
  {
    LVARRAY_UNUSED_VARIABLE( matrix );
    static_assert( !M, "determinant is not implemented for this size matrix." );
    return 0;
  }

  /**
   * @brief Invert the source matrix @p srcMatrix and store the result in @p dstMatrix.
   * @tparam DST_MATRIX The type of @p dstMatrix.
   * @tparam SRC_MATRIX The type of @p srcMatrix.
   * @param dstMatrix The matrix to write the inverse to.
   * @param srcMatrix The matrix to take the inverse of.
   * @return The determinant.
   * @note This is not implemented for generic sized matrices and calling this is a compilation error.
   */
  template< typename DST_MATRIX, typename SRC_MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static int invert( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                     SRC_MATRIX const & LVARRAY_RESTRICT_REF srcMatrix )
  {
    LVARRAY_UNUSED_VARIABLE( dstMatrix );
    LVARRAY_UNUSED_VARIABLE( srcMatrix );
    static_assert( !M, "invert is not implemented for this size matrix." );
    return 0;
  }

  /**
   * @brief Invert the matrix @p srcMatrix overwritting it.
   * @tparam MATRIX The type of @p matrix.
   * @param matrix The matrix to take the inverse of and overwrite.
   * @return The determinant.
   * @note This is not implemented for generic sized matrices and calling this is a compilation error.
   */
  template< typename MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static auto invert( MATRIX && matrix )
  {
    LVARRAY_UNUSED_VARIABLE( matrix );
    static_assert( !M, "invert is not implemented for this size matrix." );
    return 0;
  }

  /**
   * @brief Multiply the vector @p vectorB by the symmetric matrix @p symMatrixA and store the result in @p
   * dstVector.
   * @tparam DST_VECTOR The type of @p dstVector.
   * @tparam SYM_MATRIX_A The type of @p symMatrixA.
   * @tparam VECTOR_B The type of @p vectorB.
   * @param dstVector The vector to write the result to.
   * @param symMatrixA The symmetric matrix to multiply @p vectorB by.
   * @param vectorB The vector to be multiplied by @p symMatrixA.
   * @details Performs the operation @code dstVector[ i ] = symMatrixA[ i ][ j ] * vectorB[ j ] @endcode
   * @note This is not implemented for generic sized matrices and calling this is a compilation error.
   */
  template< typename DST_VECTOR, typename SYM_MATRIX_A, typename VECTOR_B >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void symAijBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                        SYM_MATRIX_A const & LVARRAY_RESTRICT_REF symMatrixA,
                        VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
  {
    LVARRAY_UNUSED_VARIABLE( dstVector );
    LVARRAY_UNUSED_VARIABLE( symMatrixA );
    LVARRAY_UNUSED_VARIABLE( vectorB );
    static_assert( !M, "symAijBj not implemented for this size matrix." );
  }

  /**
   * @brief Multiply the vector @p vectorB by the symmetric matrix @p symMatrixA and add the result to @p dstVector.
   * @tparam DST_VECTOR The type of @p dstVector.
   * @tparam SYM_MATRIX_A The type of @p symMatrixA.
   * @tparam VECTOR_B The type of @p vectorB.
   * @param dstVector The vector to add the result to.
   * @param symMatrixA The symmetric matrix to multiply @p vectorB by.
   * @param vectorB The vector to be multiplied by @p symMatrixA.
   * @details Performs the operation @code dstVector[ i ] += symMatrixA[ i ][ j ] * vectorB[ j ] @endcode
   * @note This is not implemented for generic sized matrices and calling this is a compilation error.
   */
  template< typename DST_VECTOR, typename SYM_MATRIX_A, typename VECTOR_B >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void plusSymAijBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                            SYM_MATRIX_A const & LVARRAY_RESTRICT_REF symMatrixA,
                            VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
  {
    LVARRAY_UNUSED_VARIABLE( dstVector );
    LVARRAY_UNUSED_VARIABLE( symMatrixA );
    LVARRAY_UNUSED_VARIABLE( vectorB );
    static_assert( !M, "symAijBj not implemented for this size matrix." );
  }

  /**
   * @brief Multiply the transpose of matrix @p matrixB by the symmetric matrix @p symMatrixA and store
   *   the result in @p dstMatrix.
   * @tparam DST_MATRIX The type of @p dstMatrix.
   * @tparam SYM_MATRIX_A The type of @p symMatrixA.
   * @tparam MATRIX_B The type of @p matrixB.
   * @param dstMatrix The matrix to write the result to.
   * @param symMatrixA The symmetric matrix to multiply @p matrixB by.
   * @param matrixB The matrix to be multiplied by @p matrixB.
   * @details Performs the operation @code dstMatrix[ i ][ j ] = symMatrixA[ i ][ k ] * matrixB[ j ][ k ] @endcode
   * @note This is not implemented for generic sized matrices and calling this is a compilation error.
   */
  template< typename DST_MATRIX, typename SYM_MATRIX_A, typename MATRIX_B >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void symAikBjk( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                         SYM_MATRIX_A const & LVARRAY_RESTRICT_REF symMatrixA,
                         MATRIX_B const & LVARRAY_RESTRICT_REF matrixB )
  {
    LVARRAY_UNUSED_VARIABLE( dstMatrix );
    LVARRAY_UNUSED_VARIABLE( symMatrixA );
    LVARRAY_UNUSED_VARIABLE( matrixB );
    static_assert( !M, "symAikBjk not implemented for this size matrix." );
  }

  /**
   * @brief Multiply the transpose of matrix @p matrixA by the symmetric matrix @p symMatrixB then by @p matrixA
   *   and store the result in @p dstSymMatrix.
   * @tparam DST_SYM_MATRIX The type of @p dstSymMatrix.
   * @tparam MATRIX_A The type of @p matrixA.
   * @tparam SYM_MATRIX_B The type of @p symMatrixB.
   * @param dstSymMatrix The symmetric matrix to write the result to.
   * @param matrixA The matrix to pre and post multiply @p symMatrixB by.
   * @param symMatrixB The symmetric matrix that gets pre multiplied by @p matrixA and post postmultiplied
   *   by the transpose of @p matrixA.
   * @details Performs the operation
   *   @code dstSymMatrix[ i ][ j ] = matrixA[ i ][ k ] * symMatrixB[ k ][ l ] * matrixA[ j ][ l ] @endcode
   * @note This is not implemented for generic sized matrices and calling this is a compilation error.
   */
  template< typename DST_SYM_MATRIX, typename MATRIX_A, typename SYM_MATRIX_B >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void AikSymBklAjl( DST_SYM_MATRIX && LVARRAY_RESTRICT_REF dstSymMatrix,
                            MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                            SYM_MATRIX_B const & LVARRAY_RESTRICT_REF symMatrixB )
  {
    LVARRAY_UNUSED_VARIABLE( dstSymMatrix );
    LVARRAY_UNUSED_VARIABLE( matrixA );
    LVARRAY_UNUSED_VARIABLE( symMatrixB );
    static_assert( !M, "AikSymBklAjl not implemented for this size matrix." );
  }

};

/**
 * @struct SquareMatrixOps< 2 >
 * @brief Performs operations on 2x2 square matrices.
 */
template<>
struct SquareMatrixOps< 2 >
{
  /**
   * @brief @return Return the determinant of the matrix @p matrix.
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
   */
  template< typename DST_MATRIX, typename SRC_MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static auto invert( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                      SRC_MATRIX const & LVARRAY_RESTRICT_REF srcMatrix )
  {
    checkSizes< 2, 2 >( dstMatrix );
    checkSizes< 2, 2 >( srcMatrix );

    auto const det = determinant( srcMatrix );
    auto const invDet = 1.0 / det;

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
   */
  template< typename MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static auto invert( MATRIX && matrix )
  {
    checkSizes< 2, 2 >( matrix );

    auto const det = determinant( matrix );
    auto const invDet = 1.0 / det;

    auto const temp = matrix[ 0 ][ 0 ];
    matrix[ 0 ][ 0 ] *= matrix[ 1 ][ 1 ] * invDet;
    matrix[ 1 ][ 1 ] = temp * invDet;
    matrix[ 0 ][ 1 ] *= -invDet;
    matrix[ 1 ][ 0 ] *= -invDet;

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
  static void symAijBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
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
  static void plusSymAijBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
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
  static void symAikBjk( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
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
   * @param dstSymMatrix The 2x2 symmetric matrix (vector of lenght 3) to write the result to.
   * @param matrixA The 2x2 matrix to pre and post multiply @p symMatrixB by.
   * @param symMatrixB The 2x2 symmetric matrix (vector of length 3) that gets pre multiplied by @p matrixA
   *   and post postmultiplied by the transpose of @p matrixA.
   * @details Performs the operation
   *   @code dstSymMatrix[ i ][ j ] = matrixA[ i ][ k ] * symMatrixB[ k ][ l ] * matrixA[ j ][ l ] @endcode
   */
  template< typename DST_SYM_MATRIX, typename MATRIX_A, typename SYM_MATRIX_B >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static void AikSymBklAjl( DST_SYM_MATRIX && LVARRAY_RESTRICT_REF dstSymMatrix,
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
};

/**
 * @struct SquareMatrixOps< 3 >
 * @brief Performs operations on 3x3 square matrices.
 */
template<>
struct SquareMatrixOps< 3 >
{
  /**
   * @brief @return Return the determinant of the matrix @p matrix.
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
   */
  template< typename DST_MATRIX, typename SRC_MATRIX >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
  static auto invert( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                      SRC_MATRIX const & LVARRAY_RESTRICT_REF srcMatrix )
  {
    checkSizes< 3, 3 >( dstMatrix );
    checkSizes< 3, 3 >( srcMatrix );

    dstMatrix[ 0 ][ 0 ] = srcMatrix[ 1 ][ 1 ] * srcMatrix[ 2 ][ 2 ] - srcMatrix[ 1 ][ 2 ] * srcMatrix[ 2 ][ 1 ];
    dstMatrix[ 0 ][ 1 ] = srcMatrix[ 0 ][ 2 ] * srcMatrix[ 2 ][ 1 ] - srcMatrix[ 0 ][ 1 ] * srcMatrix[ 2 ][ 2 ];
    dstMatrix[ 0 ][ 2 ] = srcMatrix[ 0 ][ 1 ] * srcMatrix[ 1 ][ 2 ] - srcMatrix[ 0 ][ 2 ] * srcMatrix[ 1 ][ 1 ];

    auto const det = srcMatrix[ 0 ][ 0 ] * dstMatrix[ 0 ][ 0 ] +
                     srcMatrix[ 1 ][ 0 ] * dstMatrix[ 0 ][ 1 ] +
                     srcMatrix[ 2 ][ 0 ] * dstMatrix[ 0 ][ 2 ];
    auto const invDet = 1.0 / det;

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
   */
  template< typename MATRIX >
  LVARRAY_HOST_DEVICE constexpr inline
  static auto invert( MATRIX && matrix )
  {
    std::remove_reference_t< decltype( matrix[ 0 ][ 0 ] ) > temp[ 3 ][ 3 ];
    auto const det = invert( temp, matrix );

    for( std::ptrdiff_t i = 0; i < 3; ++i )
    {
      for( std::ptrdiff_t j = 0; j < 3; ++j )
      {
        matrix[ i ][ j ] = temp[ i ][ j ];
      }
    }

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
  static void symAijBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
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
  static void plusSymAijBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
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
  static void symAikBjk( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
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
  static void AikSymBklAjl( DST_SYM_MATRIX && LVARRAY_RESTRICT_REF dstSymMatrix,
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
};

/**
 * @brief @return The absolute value of the integer value @p value.
 * @tparam INTEGER an integral type, the type of @p value.
 * @param value The value to take the absolute value of.
 */
template< typename INTEGER >
LVARRAY_HOST_DEVICE inline constexpr
std::enable_if_t< std::is_integral< INTEGER >::value, INTEGER >
absoluteValue( INTEGER const value )
{ return abs( value ); }

/**
 * @brief @return The absolute value of floating point value @p value.
 * @tparam FLOATING_POINT a floating point type, the type of @p value.
 * @param value The value to take the absolute value of.
 */
template< typename FLOATING_POINT >
LVARRAY_HOST_DEVICE inline constexpr
std::enable_if_t< std::is_floating_point< FLOATING_POINT >::value, FLOATING_POINT >
absoluteValue( FLOATING_POINT const value )
{ return fabs( value ); }

} // namespace internal

/**************************************************************************************************
   Generic operations.
 ***************************************************************************************************/

/**
 * @brief @return Return the largest absolute entry of @p vector.
 * @tparam M The size of @p vector.
 * @tparam VECTOR The type of @p vector.
 * @param vector The vector to get the maximum entry of, of length M.
 */
template< std::ptrdiff_t M, typename VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
auto maxAbsoluteEntry( VECTOR && vector )
{
  static_assert( M > 0, "M must be greater than zero." );
  internal::checkSizes< M >( vector );

  auto maxVal = internal::absoluteValue( vector[ 0 ] );
  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    maxVal = max( maxVal, internal::absoluteValue( vector[ i ] ) );
  }

  return maxVal;
}

/**
 * @brief Set the entries of @p vector to @p value.
 * @tparam M The size of @p vector.
 * @tparam VECTOR The type of @p vector.
 * @param vector The vector to set the entries of, of length M.
 * @param value The value to set the entries to.
 * @details Performs the operation @code vector[ i ] = value @endcode
 */
template< std::ptrdiff_t M, typename VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void fill( VECTOR && vector, std::remove_reference_t< decltype( vector[ 0 ] ) > const value )
{
  static_assert( M > 0, "M must be greater than zero." );
  internal::checkSizes< M >( vector );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    vector[ i ] = value;
  }
}

/**
 * @brief Set the entries of @p matrix to @p value.
 * @tparam M The size of the first dimension of @p matrix.
 * @tparam N The size of the second dimension of @p matrix.
 * @tparam MATRIX The type of @p matrix.
 * @param matrix The matrix to set the entries of, of size M x N.
 * @param value The value to set the entries to.
 * @details Performs the operation @code matrix[ i ][ j ] = value @endcode
 */
template< std::ptrdiff_t M, std::ptrdiff_t N, typename MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void fill( MATRIX && matrix, std::remove_reference_t< decltype( matrix[ 0 ][ 0 ] ) > const value )
{
  static_assert( M > 0, "M must be greater than zero." );
  static_assert( N > 0, "N must be greater than zero." );
  internal::checkSizes< M, N >( matrix );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    for( std::ptrdiff_t j = 0; j < N; ++j )
    {
      matrix[ i ][ j ] = value;
    }
  }
}

/**
 * @brief Copy @p srcVector into @p dstVector.
 * @tparam M The length of @p dstVector and @p srcVector.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam SRC_VECTOR The type of @p srcVector.
 * @param dstVector The destination vector, of length M.
 * @param srcVector The source vector, of length M.
 * @details Performs the operation @code dstVector[ i ] = srcVector[ i ] @endcode
 */
template< std::ptrdiff_t M, typename DST_VECTOR, typename SRC_VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void copy( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
           SRC_VECTOR const & LVARRAY_RESTRICT_REF srcVector )
{
  static_assert( M > 0, "M must be greater than zero." );
  internal::checkSizes< M >( dstVector );
  internal::checkSizes< M >( srcVector );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    dstVector[ i ] = srcVector[ i ];
  }
}

/**
 * @brief Copy @p srcMatrix into @p dstMatrix.
 * @tparam M The size of the first dimension of @p dstMatrix and @p srcMatrix.
 * @tparam N The size of the second dimension of @p dstMatrix and @p srcMatrix.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam SRC_MATRIX The type of @p srcMatrix.
 * @param dstMatrix The destination matrix, of size M x N.
 * @param srcMatrix The source matrix, of size M x N.
 * @details Performs the operation @code dstMatrix[ i ][ j ] = srcMatrix[ i ][ j ] @endcode
 */
template< std::ptrdiff_t M, std::ptrdiff_t N, typename DST_MATRIX, typename SRC_MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void copy( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
           SRC_MATRIX const & LVARRAY_RESTRICT_REF srcMatrix )
{
  static_assert( M > 0, "M must be greater than zero." );
  static_assert( N > 0, "N must be greater than zero." );
  internal::checkSizes< M, N >( dstMatrix );
  internal::checkSizes< M, N >( srcMatrix );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    for( std::ptrdiff_t j = 0; j < N; ++j )
    {
      dstMatrix[ i ][ j ] = srcMatrix[ i ][ j ];
    }
  }
}

/**
 * @brief Multiply the entries of @p vector by @p scale.
 * @tparam M The size of @p vector.
 * @tparam VECTOR The type of @p vector.
 * @param vector The vector to scale, of length M.
 * @param scale The value to scale the entries of @p vector by.
 * @details Performs the operation @code vector[ i ] *= scale @endcode
 */
template< std::ptrdiff_t M, typename VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void scale( VECTOR && vector, std::remove_reference_t< decltype( vector[ 0 ] ) > const scale )
{
  static_assert( M > 0, "M must be greater than zero." );
  internal::checkSizes< M >( vector );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    vector[ i ] *= scale;
  }
}

/**
 * @brief Multiply the entries of @p matrix by @p scale.
 * @tparam M The size of the first dimension of @p matrix.
 * @tparam N The size of the second dimension of @p matrix.
 * @tparam MATRIX The type of @p matrix.
 * @param matrix The matrix to scale, of size M x N.
 * @param scale The value scale the entries of @p vector by.
 * @details Performs the operations @code matrix[ i ][ j ] *= scale @endcode
 */
template< std::ptrdiff_t M, std::ptrdiff_t N, typename MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void scale( MATRIX && matrix, std::remove_reference_t< decltype( matrix[ 0 ][ 0 ] ) > const scale )
{
  static_assert( M > 0, "M must be greater than zero." );
  static_assert( N > 0, "N must be greater than zero." );
  internal::checkSizes< M, N >( matrix );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    for( std::ptrdiff_t j = 0; j < N; ++j )
    {
      matrix[ i ][ j ] *= scale;
    }
  }
}

/**
 * @brief Copy @p srcVector scaled by @p scale into @p dstVector.
 * @tparam M The length of @p dstVector and @p srcVector.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam SRC_VECTOR The type of @p srcVector.
 * @param dstVector The destination vector, of length M.
 * @param srcVector The source vector, of length M.
 * @param scale The value to scale @p srcVector by.
 * @details Performs the operations @code dstVector[ i ] = scale * srcVector[ i ] @endcode
 */
template< std::ptrdiff_t M, typename DST_VECTOR, typename SRC_VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void scaledCopy( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                 SRC_VECTOR const & LVARRAY_RESTRICT_REF srcVector,
                 std::remove_reference_t< decltype( srcVector[ 0 ] ) > const scale )
{
  static_assert( M > 0, "M must be greater than zero." );
  internal::checkSizes< M >( dstVector );
  internal::checkSizes< M >( srcVector );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    dstVector[ i ] = scale * srcVector[ i ];
  }
}

/**
 * @brief Copy @p srcMatrix scaled by @p scale into @p dstMatrix.
 * @tparam M The size of the first dimension of @p dstMatrix and @p srcMatrix.
 * @tparam N The size of the second dimension of @p dstMatrix and @p srcMatrix.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam SRC_MATRIX The type of @p srcMatrix.
 * @param dstMatrix The destination matrix, of size M x N.
 * @param srcMatrix The source matrix, of size M x N.
 * @param scale The value to scale @p srcMatrix by.
 * @details Performs the operation @code dstMatrix[ i ][ j ] = scale * srcMatrix[ i ][ j ] @endcode
 */
template< std::ptrdiff_t M, std::ptrdiff_t N, typename DST_MATRIX, typename SRC_MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void scaledCopy( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                 SRC_MATRIX const & LVARRAY_RESTRICT_REF srcMatrix,
                 std::remove_reference_t< decltype( srcMatrix[ 0 ][ 0 ] ) > const scale )
{
  static_assert( M > 0, "M must be greater than zero." );
  static_assert( N > 0, "N must be greater than zero." );
  internal::checkSizes< M, N >( dstMatrix );
  internal::checkSizes< M, N >( srcMatrix );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    for( std::ptrdiff_t j = 0; j < N; ++j )
    {
      dstMatrix[ i ][ j ] = scale * srcMatrix[ i ][ j ];
    }
  }
}

/**
 * @brief Add @p srcVector to @p dstVector.
 * @tparam M The length of @p dstVector and @p srcVector.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam SRC_VECTOR The type of @p srcVector.
 * @param dstVector The destination vector, of length M.
 * @param srcVector The source vector, of length M.
 * @details Performs the operation @code dstVector[ i ] += srcVector[ i ] @endcode
 */
template< std::ptrdiff_t M, typename DST_VECTOR, typename SRC_VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void add( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
          SRC_VECTOR const & LVARRAY_RESTRICT_REF srcVector )
{
  static_assert( M > 0, "M must be greater than zero." );
  internal::checkSizes< M >( dstVector );
  internal::checkSizes< M >( srcVector );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    dstVector[ i ] += srcVector[ i ];
  }
}

/**
 * @brief Add @p srcMatrix to @p dstMatrix.
 * @tparam M The size of the first dimension of @p dstMatrix and @p srcMatrix.
 * @tparam N The size of the second dimension of @p dstMatrix and @p srcMatrix.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam SRC_MATRIX The type of @p srcMatrix.
 * @param dstMatrix The destination matrix, of size M x N.
 * @param srcMatrix The source matrix, of size M x N.
 * @details Performs the operation @code dstMatrix[ i ][ j ] += srcMatrix[ i ][ j ] @endcode
 */
template< std::ptrdiff_t M, std::ptrdiff_t N, typename DST_MATRIX, typename SRC_MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void add( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
          SRC_MATRIX const & LVARRAY_RESTRICT_REF srcMatrix )
{
  static_assert( M > 0, "M must be greater than zero." );
  static_assert( N > 0, "M must be greater than zero." );
  internal::checkSizes< M, N >( dstMatrix );
  internal::checkSizes< M, N >( srcMatrix );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    for( std::ptrdiff_t j = 0; j < N; ++j )
    {
      dstMatrix[ i ][ j ] += srcMatrix[ i ][ j ];
    }
  }
}

/**
 * @brief Subtract @p srcVector from @p dstVector.
 * @tparam M The length of @p dstVector and @p srcVector.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam SRC_VECTOR The type of @p srcVector.
 * @param dstVector The destination vector, of length M.
 * @param srcVector The source vector, of length M.
 * @details Performs the operation @code dstVector[ i ] -= srcVector[ i ] @endcode
 */
template< std::ptrdiff_t M, typename DST_VECTOR, typename SRC_VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void subtract( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
               SRC_VECTOR const & LVARRAY_RESTRICT_REF srcVector )
{
  static_assert( M > 0, "M must be greater than zero." );
  internal::checkSizes< M >( dstVector );
  internal::checkSizes< M >( srcVector );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    dstVector[ i ] -= srcVector[ i ];
  }
}

/**
 * @brief Add @p srcVector scaled by @p scale to @p dstVector.
 * @tparam M The length of @p dstVector and @p srcVector.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam SRC_VECTOR The type of @p srcVector.
 * @param dstVector The destination vector, of length M.
 * @param srcVector The source vector, of length M.
 * @param scale The value to scale the entries of @p srcVector by.
 * @details Performs the operation @code dstVector[ i ] += scale * srcVector[ i ] @endcode
 */
template< std::ptrdiff_t M, typename DST_VECTOR, typename SRC_VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void scaledAdd( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                SRC_VECTOR const & LVARRAY_RESTRICT_REF srcVector,
                std::remove_reference_t< decltype( srcVector[ 0 ] ) > const scale )
{
  static_assert( M > 0, "M must be greater than zero." );
  internal::checkSizes< M >( dstVector );
  internal::checkSizes< M >( srcVector );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    dstVector[ i ] = dstVector[ i ] + scale * srcVector[ i ];
  }
}

/**
 * @brief Multiply the elements of @p vectorA and @p vectorB putting the result into @p dstVector.
 * @tparam M The length of @p dstVector and @p srcVector.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam SRC_VECTOR The type of @p srcVector.
 * @param dstVector The destination vector, of length M.
 * @param vectorA The first source vector, of length M.
 * @param vectorB The second source vector, of length M.
 * @details Performs the operation @code dstVector[ i ] = vectorA[ i ] * vectorB[ i ] @endcode
 */
template< std::ptrdiff_t M, typename DST_VECTOR, typename VECTOR_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void elementWiseMultiplication( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                                VECTOR_A const & LVARRAY_RESTRICT_REF vectorA,
                                VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  static_assert( M > 0, "M must be greater than zero." );
  internal::checkSizes< M >( dstVector );
  internal::checkSizes< M >( vectorA );
  internal::checkSizes< M >( vectorB );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    dstVector[ i ] = vectorA[ i ] * vectorB[ i ];
  }
}

/**************************************************************************************************
   Vector operations.
 ***************************************************************************************************/

/**
 * @brief @return The L2 norm squared of @p vector.
 * @tparam M The length of @p vector.
 * @tparam VECTOR The type of @p vector.
 * @param vector The vector to get the L2 norm squared of, of length M.
 */
template< std::ptrdiff_t M, typename VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
auto l2NormSquared( VECTOR const & vector )
{
  static_assert( M > 0, "M must be greater than zero." );
  internal::checkSizes< M >( vector );

  auto norm = vector[ 0 ] * vector[ 0 ];
  for( std::ptrdiff_t i = 1; i < M; ++i )
  {
    norm = norm + vector[ i ] * vector[ i ];
  }
  return norm;
}

/**
 * @brief @return The L2 norm of @p vector.
 * @tparam M The length of @p vector.
 * @tparam VECTOR The type of @p vector.
 * @param vector The vector to get the L2 norm of, of length M.
 */
template< std::ptrdiff_t M, typename VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
double l2Norm( VECTOR const & vector )
{
  static_assert( M > 0, "M must be greater than zero." );
  internal::checkSizes< M >( vector );

  // The conversion to double is needed because nvcc only defines sqrt for floating point arguments.
  return sqrt( double( l2NormSquared< M >( vector ) ) );
}

/**
 * @brief Scale @p vector to a unit vector.
 * @tparam M The length of @p vector.
 * @tparam VECTOR The type of @p vector.
 * @param vector The vector normalize, of length M.
 * @return The previous L2 norm of @p vector.
 */
template< std::ptrdiff_t M, typename VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
double normalize( VECTOR && vector )
{
  static_assert( M > 0, "M must be greater than zero." );
  internal::checkSizes< M >( vector );

  double const norm = l2Norm< M >( vector );
  if( norm > 0 )
  {
    scale< M >( vector, 1 / norm );
  }

  return norm;
}

/**
 * @brief @return Return the inner product of @p vectorA and @p vectorB.
 * @tparam M The length of @p vectorA and @p vectorB.
 * @tparam VECTOR_A The type of @p vectorA.
 * @tparam VECTOR_B The type of @p vectorB.
 * @param vectorA The first vector in the inner product, of length M.
 * @param vectorB The second vector in the inner product, of length M.
 */
template< std::ptrdiff_t N, typename VECTOR_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
auto AiBi( VECTOR_A const & LVARRAY_RESTRICT_REF vectorA,
           VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  static_assert( N > 0, "N must be greater than zero." );
  internal::checkSizes< N >( vectorA );
  internal::checkSizes< N >( vectorB );

  auto result = vectorA[ 0 ] * vectorB[ 0 ];
  for( std::ptrdiff_t i = 1; i < N; ++i )
  {
    result = result + vectorA[ i ] * vectorB[ i ];
  }

  return result;
}

/**
 * @brief Compute the cross product of @p vectorA and @p vectorB and put it in @p dstVector.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam VECTOR_A The type of @p vectorA.
 * @tparam VECTOR_B The type of @p vectorB.
 * @param dstVector The vector to write the cross product to, of length 3.
 * @param vectorA The first vector in the cross product, of length 3.
 * @param vectorB THe second vector in the cross product, of length 3.
 */
template< typename DST_VECTOR, typename VECTOR_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void crossProduct( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                   VECTOR_A const & LVARRAY_RESTRICT_REF vectorA,
                   VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  internal::checkSizes< 3 >( dstVector );
  internal::checkSizes< 3 >( vectorA );
  internal::checkSizes< 3 >( vectorB );

  dstVector[ 0 ] = vectorA[ 1 ] * vectorB[ 2 ] - vectorA[ 2 ] * vectorB[ 1 ];
  dstVector[ 1 ] = vectorA[ 2 ] * vectorB[ 0 ] - vectorA[ 0 ] * vectorB[ 2 ];
  dstVector[ 2 ] = vectorA[ 0 ] * vectorB[ 1 ] - vectorA[ 1 ] * vectorB[ 0 ];
}

/**************************************************************************************************
   Matrix vector operations.
 ***************************************************************************************************/

/**
 * @brief Perform the outer product of @p vectorA and @p vectorB writing the result to @p dstMatrix.
 * @tparam M The size of the first dimension of @p dstMatrix and the length of @p vectorA.
 * @tparam N The size of the second dimension of @p dstMatrix and the length of @p vectorB.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam VECTOR_A The type of @p vectorA.
 * @tparam VECTOR_B The type of @p vectorB.
 * @param dstMatrix The matrix the result is written to, of size M x N.
 * @param vectorA The first vector in the outer product, of length M.
 * @param vectorB The second vector in the outer product, of length N.
 * @details Performs the operations @code dstMatrix[ i ][ j ] = vectorA[ i ] * vectorB[ j ] @endcode
 */
template< std::ptrdiff_t M, std::ptrdiff_t N, typename DST_MATRIX, typename VECTOR_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void AiBj( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
           VECTOR_A const & LVARRAY_RESTRICT_REF vectorA,
           VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  static_assert( M > 0, "M must be greater than zero." );
  static_assert( N > 0, "M must be greater than zero." );
  internal::checkSizes< M, N >( dstMatrix );
  internal::checkSizes< M >( vectorA );
  internal::checkSizes< N >( vectorB );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    for( std::ptrdiff_t j = 0; j < N; ++j )
    {
      dstMatrix[ i ][ j ] = vectorA[ i ] * vectorB[ j ];
    }
  }
}

/**
 * @brief Perform the outer product of @p vectorA and @p vectorB adding the result to @p dstMatrix.
 * @tparam M The size of the first dimension of @p dstMatrix and the length of @p vectorA.
 * @tparam N The size of the second dimension of @p dstMatrix and the length of @p vectorB.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam VECTOR_A The type of @p vectorA.
 * @tparam VECTOR_B The type of @p vectorB.
 * @param dstMatrix The matrix the result is added to, of size M x N.
 * @param vectorA The first vector in the outer product, of length M.
 * @param vectorB The second vector in the outer product, of length N.
 * @details Performs the operations @code dstMatrix[ i ][ j ] += vectorA[ i ] * vectorB[ j ] @endcode
 */
template< std::ptrdiff_t N, std::ptrdiff_t M, typename DST_MATRIX, typename VECTOR_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void plusAiBj( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
               VECTOR_A const & LVARRAY_RESTRICT_REF vectorA,
               VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  static_assert( N > 0, "N must be greater than zero." );
  static_assert( M > 0, "M must be greater than zero." );
  internal::checkSizes< N, M >( dstMatrix );
  internal::checkSizes< N >( vectorA );
  internal::checkSizes< M >( vectorB );

  for( std::ptrdiff_t i = 0; i < N; ++i )
  {
    for( std::ptrdiff_t j = 0; j < M; ++j )
    {
      dstMatrix[ i ][ j ] = dstMatrix[ i ][ j ] + vectorA[ i ] * vectorB[ j ];
    }
  }
}

/**
 * @brief Perform the matrix vector multiplication of @p matrixA and @p vectorB writing the result to @p dstVector.
 * @tparam M The length of @p dstVector and the size of the first dimension of @p matrixA.
 * @tparam N The size of the second dimension of @p matrixA and the length of @p vectorBB.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam VECTOR_B The type of @p vectorB.
 * @param dstVector The vector to write the result to, of length M.
 * @param matrixA The matrix to multiply @p vectorB by, of size M x N.
 * @param vectorB The vector multiplied by @p matrixA, of length N.
 * @details Performs the operations @code dstVector[ i ] = matrixA[ i ][ j ] * vectorB[ j ] @endcode
 */
template< std::ptrdiff_t M, std::ptrdiff_t N, typename DST_VECTOR, typename MATRIX_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void AijBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
            MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
            VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  static_assert( M > 0, "M must be greater than zero." );
  static_assert( N > 0, "N must be greater than zero." );
  internal::checkSizes< M >( dstVector );
  internal::checkSizes< M, N >( matrixA );
  internal::checkSizes< N >( vectorB );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    auto dot = matrixA[ i ][ 0 ] * vectorB[ 0 ];
    for( std::ptrdiff_t j = 1; j < N; ++j )
    {
      dot = dot + matrixA[ i ][ j ] * vectorB[ j ];
    }
    dstVector[ i ] = dot;
  }
}

/**
 * @brief Perform the matrix vector multiplication of @p matrixA and @p vectorB adding the result to @p dstVector.
 * @tparam M The length of @p dstVector and the size of the first dimension of @p matrixA.
 * @tparam N The size of the second dimension of @p matrixA and the length of @p vectorBB.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam VECTOR_B The type of @p vectorB.
 * @param dstVector The vector to add the result to, of length M.
 * @param matrixA The matrix to multiply @p vectorB by, of size M x N.
 * @param vectorB The vector multiplied by @p matrixA, of length N.
 * @details Performs the operations @code dstVector[ i ] += matrixA[ i ][ j ] * vectorB[ j ] @endcode
 */
template< std::ptrdiff_t M, std::ptrdiff_t N, typename DST_VECTOR, typename MATRIX_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void plusAijBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  static_assert( M > 0, "M must be greater than zero." );
  static_assert( N > 0, "M must be greater than zero." );
  internal::checkSizes< M >( dstVector );
  internal::checkSizes< M, N >( matrixA );
  internal::checkSizes< N >( vectorB );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    auto dot = matrixA[ i ][ 0 ] * vectorB[ 0 ];
    for( std::ptrdiff_t j = 1; j < N; ++j )
    {
      dot = dot + matrixA[ i ][ j ] * vectorB[ j ];
    }
    dstVector[ i ] += dot;
  }
}

/**
 * @brief Perform the matrix vector multiplication of the transpose of @p matrixA and @p vectorB
 *   writing the result to @p dstVector.
 * @tparam M The length of @p dstVector and the size of the second dimension of @p matrixA.
 * @tparam N The size of the first dimension of @p matrixA and the length of @p vectorBB.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam VECTOR_B The type of @p vectorB.
 * @param dstVector The vector to write the result to, of length M.
 * @param matrixA The matrix to multiply @p vectorB by, of size N x M.
 * @param vectorB The vector multiplied by @p matrixA, of length N.
 * @details Performs the operations @code dstVector[ i ] = matrixA[ j ][ i ] * vectorB[ j ] @endcode
 */
template< std::ptrdiff_t N, std::ptrdiff_t M, typename DST_VECTOR, typename MATRIX_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void AjiBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
            MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
            VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  static_assert( N > 0, "N must be greater than zero." );
  static_assert( M > 0, "M must be greater than zero." );
  internal::checkSizes< M >( dstVector );
  internal::checkSizes< N, M >( matrixA );
  internal::checkSizes< N >( vectorB );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    auto dot = matrixA[ 0 ][ i ] * vectorB[ 0 ];
    for( std::ptrdiff_t j = 1; j < N; ++j )
    {
      dot = dot + matrixA[ j ][ i ] * vectorB[ j ];
    }
    dstVector[ i ] = dot;
  }
}

/**
 * @brief Perform the matrix vector multiplication of the transpose of @p matrixA and @p vectorB
 *   adding the result to @p dstVector.
 * @tparam M The length of @p dstVector and the size of the second dimension of @p matrixA.
 * @tparam N The size of the first dimension of @p matrixA and the length of @p vectorB.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam VECTOR_B The type of @p vectorB.
 * @param dstVector The vector to add the result to, of length M.
 * @param matrixA The matrix to multiply @p vectorB by, of size N x M.
 * @param vectorB The vector multiplied by @p matrixA, of length N.
 * @details Performs the operations @code dstVector[ i ] += matrixA[ j ][ i ] * vectorB[ j ] @endcode
 */
template< std::ptrdiff_t N, std::ptrdiff_t M, typename DST_VECTOR, typename MATRIX_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void plusAjiBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  static_assert( N > 0, "N must be greater than zero." );
  static_assert( M > 0, "M must be greater than zero." );
  internal::checkSizes< M >( dstVector );
  internal::checkSizes< N, M >( matrixA );
  internal::checkSizes< N >( vectorB );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    auto dot = matrixA[ 0 ][ i ] * vectorB[ 0 ];
    for( std::ptrdiff_t j = 1; j < N; ++j )
    {
      dot = dot + matrixA[ j ][ i ] * vectorB[ j ];
    }
    dstVector[ i ] += dot;
  }
}

/**************************************************************************************************
   Matrix matrix operations
 ***************************************************************************************************/

/**
 * @brief Multiply @p matrixA with @p matrixB and put the result into @p dstMatrix.
 * @tparam M The size of the first dimension of @p dstMatrix and @p matrixA.
 * @tparam N The size of the second dimension of @p dstMatrix and @p matrixB.
 * @tparam P The size of the second dimension of @p matrixA and first dimension of @p matrixB.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam MATRIX_B The type of @p matrixB.
 * @param dstMatrix The matrix the result is written to, of size M x N.
 * @param matrixA The left matrix in the multiplication, of size M x P.
 * @param matrixB The right matrix in the multiplication, of size P x N.
 * @details Performs the operation
 *   @code dstMatrix[ i ][ j ] = matrixA[ i ][ k ] * matrixB[ k ][ j ] @endcode
 */
template< std::ptrdiff_t M,
          std::ptrdiff_t N,
          std::ptrdiff_t P,
          typename DST_MATRIX,
          typename MATRIX_A,
          typename MATRIX_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void AikBkj( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
             MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
             MATRIX_B const & LVARRAY_RESTRICT_REF matrixB )
{
  static_assert( N > 0, "M must be greater than zero." );
  static_assert( M > 0, "M must be greater than zero." );
  static_assert( P > 0, "P must be greater than zero." );
  internal::checkSizes< M, N >( dstMatrix );
  internal::checkSizes< M, P >( matrixA );
  internal::checkSizes< P, N >( matrixB );


  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    for( std::ptrdiff_t j = 0; j < N; ++j )
    {
      auto dot = matrixA[ i ][ 0 ] * matrixB[ 0 ][ j ];
      for( std::ptrdiff_t k = 1; k < P; ++k )
      {
        dot = dot + matrixA[ i ][ k ] * matrixB[ k ][ j ];
      }

      dstMatrix[ i ][ j ] = dot;
    }
  }
}

/**
 * @brief Multiply @p matrixA with @p matrixB and add the result to @p dstMatrix.
 * @tparam M The size of the first dimension of @p dstMatrix and @p matrixA.
 * @tparam N The size of the second dimension of @p dstMatrix and @p matrixB.
 * @tparam P The size of the second dimension of @p matrixA and first dimension of @p matrixB.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam MATRIX_B The type of @p matrixB.
 * @param dstMatrix The matrix the result is added to, of size M x N.
 * @param matrixA The left matrix in the multiplication, of size M x P.
 * @param matrixB The right matrix in the multiplication, of size P x N.
 * @details Performs the operation
 *   @code dstMatrix[ i ][ j ] += matrixA[ i ][ k ] * matrixB[ k ][ j ] @endcode
 */
template< std::ptrdiff_t M,
          std::ptrdiff_t N,
          std::ptrdiff_t P,
          typename DST_MATRIX,
          typename MATRIX_A,
          typename MATRIX_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void plusAikBkj( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                 MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                 MATRIX_B const & LVARRAY_RESTRICT_REF matrixB )
{
  static_assert( M > 0, "M must be greater than zero." );
  static_assert( N > 0, "N must be greater than zero." );
  static_assert( P > 0, "P must be greater than zero." );
  internal::checkSizes< M, N >( dstMatrix );
  internal::checkSizes< M, P >( matrixA );
  internal::checkSizes< P, N >( matrixB );


  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    for( std::ptrdiff_t j = 0; j < N; ++j )
    {
      auto dot = matrixA[ i ][ 0 ] * matrixB[ 0 ][ j ];
      for( std::ptrdiff_t k = 1; k < P; ++k )
      {
        dot = dot + matrixA[ i ][ k ] * matrixB[ k ][ j ];
      }

      dstMatrix[ i ][ j ] += dot;
    }
  }
}

/**
 * @brief Multiply @p matrixA with the transpose of @p matrixB and put the result into @p dstMatrix.
 * @tparam M The size of the first dimension of @p dstMatrix and @p matrixA.
 * @tparam N The size of the second dimension of @p dstMatrix and the first dimension of @p matrixB.
 * @tparam P The size of the second dimension of @p matrixA and @p matrixB.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam MATRIX_B The type of @p matrixB.
 * @param dstMatrix The matrix the result is written to, of size M x N.
 * @param matrixA The left matrix in the multiplication, of size M x P.
 * @param matrixB The right matrix in the multiplication, of size N x P.
 * @details Performs the operation
 *   @code dstMatrix[ i ][ j ] = matrixA[ i ][ k ] * matrixB[ j ][ k ] @endcode
 */
template< std::ptrdiff_t M,
          std::ptrdiff_t N,
          std::ptrdiff_t P,
          typename DST_MATRIX,
          typename MATRIX_A,
          typename MATRIX_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void AikBjk( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
             MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
             MATRIX_B const & LVARRAY_RESTRICT_REF matrixB )
{
  static_assert( M > 0, "M must be greater than zero." );
  static_assert( N > 0, "M must be greater than zero." );
  static_assert( P > 0, "P must be greater than zero." );
  internal::checkSizes< M, N >( dstMatrix );
  internal::checkSizes< M, P >( matrixA );
  internal::checkSizes< N, P >( matrixB );


  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    for( std::ptrdiff_t j = 0; j < N; ++j )
    {
      auto dot = matrixA[ i ][ 0 ] * matrixB[ j ][ 0 ];
      for( std::ptrdiff_t k = 1; k < P; ++k )
      {
        dot = dot + matrixA[ i ][ k ] * matrixB[ j ][ k ];
      }

      dstMatrix[ i ][ j ] = dot;
    }
  }
}

/**
 * @brief Multiply @p matrixA with the transpose of @p matrixB and put the result into @p dstMatrix.
 * @tparam M The size of the first dimension of @p dstMatrix and @p matrixA.
 * @tparam N The size of the second dimension of @p dstMatrix and the first dimension of @p matrixB.
 * @tparam P The size of the second dimension of @p matrixA and @p matrixB.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam MATRIX_B The type of @p matrixB.
 * @param dstMatrix The matrix the result is written to, of size M x N.
 * @param matrixA The left matrix in the multiplication, of size M x P.
 * @param matrixB The right matrix in the multiplication, of size N x P.
 * @details Performs the operation
 *   @code dstMatrix[ i ][ j ] += matrixA[ i ][ k ] * matrixB[ j ][ k ] @endcode
 */
template< std::ptrdiff_t M,
          std::ptrdiff_t N,
          std::ptrdiff_t P,
          typename DST_MATRIX,
          typename MATRIX_A,
          typename MATRIX_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void plusAikBjk( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                 MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                 MATRIX_B const & LVARRAY_RESTRICT_REF matrixB )
{
  static_assert( M > 0, "M must be greater than zero." );
  static_assert( N > 0, "N must be greater than zero." );
  static_assert( P > 0, "P must be greater than zero." );
  internal::checkSizes< M, N >( dstMatrix );
  internal::checkSizes< M, P >( matrixA );
  internal::checkSizes< N, P >( matrixB );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    for( std::ptrdiff_t j = 0; j < N; ++j )
    {
      auto dot = matrixA[ i ][ 0 ] * matrixB[ j ][ 0 ];
      for( std::ptrdiff_t k = 1; k < P; ++k )
      {
        dot = dot + matrixA[ i ][ k ] * matrixB[ j ][ k ];
      }

      dstMatrix[ i ][ j ] += dot;
    }
  }
}

/**
 * @brief Multiply the transpose of @p matrixA with @p matrixB and put the result into @p dstMatrix.
 * @tparam M The size of the first dimension of @p dstMatrix and the second dimension of @p matrixA.
 * @tparam N The size of the second dimension of @p dstMatrix and @p matrixB.
 * @tparam P The size of the first dimension of @p matrixA and @p matrixB.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam MATRIX_B The type of @p matrixB.
 * @param dstMatrix The matrix the result is written to, of size M x N.
 * @param matrixA The left matrix in the multiplication, of size P x M.
 * @param matrixB The right matrix in the multiplication, of size P x N.
 * @details Performs the operation
 *   @code dstMatrix[ i ][ j ] = matrixA[ k ][ i ] * matrixB[ k ][ j ] @endcode
 */
template< std::ptrdiff_t M,
          std::ptrdiff_t N,
          std::ptrdiff_t P,
          typename DST_MATRIX,
          typename MATRIX_A,
          typename MATRIX_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void AkiBkj( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
             MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
             MATRIX_B const & LVARRAY_RESTRICT_REF matrixB )
{
  static_assert( M > 0, "M must be greater than zero." );
  static_assert( N > 0, "M must be greater than zero." );
  static_assert( P > 0, "P must be greater than zero." );
  internal::checkSizes< M, N >( dstMatrix );
  internal::checkSizes< P, M >( matrixA );
  internal::checkSizes< P, N >( matrixB );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    for( std::ptrdiff_t j = 0; j < N; ++j )
    {
      auto dot = matrixA[ 0 ][ i ] * matrixB[ 0 ][ j ];
      for( std::ptrdiff_t k = 1; k < P; ++k )
      {
        dot = dot + matrixA[ k ][ i ] * matrixB[ k ][ j ];
      }

      dstMatrix[ i ][ j ] = dot;
    }
  }
}

/**
 * @brief Multiply the transpose of @p matrixA with @p matrixB and add the result into @p dstMatrix.
 * @tparam M The size of the first dimension of @p dstMatrix and the second dimension of @p matrixA.
 * @tparam N The size of the second dimension of @p dstMatrix and @p matrixB.
 * @tparam P The size of the first dimension of @p matrixA and @p matrixB.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam MATRIX_B The type of @p matrixB.
 * @param dstMatrix The matrix the result is written to, of size M x N.
 * @param matrixA The left matrix in the multiplication, of size P x M.
 * @param matrixB The right matrix in the multiplication, of size P x N.
 * @details Performs the operation
 *   @code dstMatrix[ i ][ j ] += matrixA[ k ][ i ] * matrixB[ k ][ j ] @endcode
 */
template< std::ptrdiff_t M,
          std::ptrdiff_t N,
          std::ptrdiff_t P,
          typename DST_MATRIX,
          typename MATRIX_A,
          typename MATRIX_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void plusAkiBkj( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                 MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                 MATRIX_B const & LVARRAY_RESTRICT_REF matrixB )
{
  static_assert( M > 0, "M must be greater than zero." );
  static_assert( N > 0, "M must be greater than zero." );
  static_assert( P > 0, "P must be greater than zero." );
  internal::checkSizes< M, N >( dstMatrix );
  internal::checkSizes< P, M >( matrixA );
  internal::checkSizes< P, N >( matrixB );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    for( std::ptrdiff_t j = 0; j < N; ++j )
    {
      auto dot = matrixA[ 0 ][ i ] * matrixB[ 0 ][ j ];
      for( std::ptrdiff_t k = 1; k < P; ++k )
      {
        dot = dot + matrixA[ k ][ i ] * matrixB[ k ][ j ];
      }

      dstMatrix[ i ][ j ] = dstMatrix[ i ][ j ] + dot;
    }
  }
}

/**************************************************************************************************
   Things that operator on a square matrices only.
 ***************************************************************************************************/

/**
 * @brief Add @p scale times the identity matrix to @p matrix.
 * @tparam M The size of @p matrix.
 * @tparam MATRIX The type of @p matrix.
 * @param matrix The M x M matrix to add the identity matrix to.
 * @param scale The amount to scale the identity matrix by.
 * @details Performs the operations @code matrix[ i ][ i ] += scale @endcode
 */
template< std::ptrdiff_t M, typename MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void addIdentity( MATRIX && matrix, std::remove_reference_t< decltype( matrix[ 0 ][ 0 ] ) > const scale )
{
  static_assert( M > 0, "M must be greater than zero." );
  internal::checkSizes< M, M >( matrix );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    matrix[ i ][ i ] += scale;
  }
}

/**
 * @brief Add @p scale times the identity matrix to @p symMatrix.
 * @tparam M The size of @p symMatrix.
 * @tparam SYM_MATRIX The type of @p symMatrix.
 * @param symMatrix The M x M symmetric matrix to add the identity matrix to.
 * @param scale The amount to scale the identity matrix by.
 * @details Performs the operations @code matrix[ i ][ i ] += scale @endcode
 */
template< std::ptrdiff_t M, typename SYM_MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void addIdentityToSymmetric( SYM_MATRIX && symMatrix, std::remove_reference_t< decltype( symMatrix[ 0 ] ) > const scale )
{
  static_assert( M > 0, "M must be greater than zero." );
  internal::checkSizes< internal::SYM_SIZE< M > >( symMatrix );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    symMatrix[ i ] += scale;
  }
}

/**
 * @brief @return Return the determinant of the matrix @p matrix.
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
 */
template< std::ptrdiff_t M, typename DST_MATRIX, typename SRC_MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
auto invert( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
             SRC_MATRIX const & LVARRAY_RESTRICT_REF srcMatrix )
{ return internal::SquareMatrixOps< M >::invert( std::forward< DST_MATRIX >( dstMatrix ), srcMatrix ); }

/**
 * @brief Invert the matrix @p matrix overwritting it.
 * @tparam M The size of the matrix @p matrix.
 * @tparam MATRIX The type of @p matrix.
 * @param matrix The M x M matrix to take the inverse of and overwrite.
 * @return The determinant.
 */
template< std::ptrdiff_t N, typename MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
auto invert( MATRIX && matrix )
{ return internal::SquareMatrixOps< N >::invert( std::forward< MATRIX >( matrix ) ); }


/**************************************************************************************************
   Symmetric matrix operations.
 ***************************************************************************************************/

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
void symAijBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
               SYM_MATRIX_A const & LVARRAY_RESTRICT_REF symMatrixA,
               VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  internal::SquareMatrixOps< M >::symAijBj( std::forward< DST_VECTOR >( dstVector ),
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
void plusSymAijBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                   SYM_MATRIX_A const & LVARRAY_RESTRICT_REF symMatrixA,
                   VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  internal::SquareMatrixOps< M >::plusSymAijBj( std::forward< DST_VECTOR >( dstVector ),
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
void symAikBjk( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                SYM_MATRIX_A const & LVARRAY_RESTRICT_REF symMatrixA,
                MATRIX_B const & LVARRAY_RESTRICT_REF matrixB )
{
  internal::SquareMatrixOps< M >::symAikBjk( std::forward< DST_MATRIX >( dstMatrix ),
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
void AikSymBklAjl( DST_SYM_MATRIX && LVARRAY_RESTRICT_REF dstSymMatrix,
                   MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                   SYM_MATRIX_B const & LVARRAY_RESTRICT_REF symMatrixB )
{
  internal::SquareMatrixOps< M >::AikSymBklAjl( std::forward< DST_SYM_MATRIX >( dstSymMatrix ),
                                                matrixA,
                                                symMatrixB );
}

} // namespace tensorOps
} // namespace LvArray
