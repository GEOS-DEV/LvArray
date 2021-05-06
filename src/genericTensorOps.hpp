/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file genericTensorOps.hpp
 * @brief Contains the implementation of arbitrary sized vector and matrix operations.
 */

#pragma once

// Source includes
#include "ArraySlice.hpp"
#include "math.hpp"

/**
 * @name Initialization and assignment macros.
 * @brief Macros that aid in to initializng and assigning to common vector and matrix sizes.
 * @details Use these macros to initialize c-arrays from other c-arrays or Array objects and
 *   to assign to either c-arrays or Array objects.
 *   Usage:
 *   @code
 *   int vectorA[ 3 ] = { 0, 1, 2 };
 *   int const vectorACopy[ 3 ] = LVARRAY_TENSOROPS_INIT_LOCAL_3( vectorA );
 *
 *   LVARRAY_TENSOROPS_ASSIGN_2( vectorA, 5, 3 );
 *
 *   Array< double, 2, ... > arrayOfMatrices( JSIZE, 2, 2 );
 *   double matrix[ 2 ][ 2 ] = LVARRAY_TENSOROPS_INIT_LOCAL_2x2( arrayOfMatrices[ 0 ] );
 *
 *   LVARRAY_TENSOROPS_ASSIGN_2x2( arrayOfMatrices[ 0 ], a00, a01, a10, a11 );
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
 * @brief Assign to the length 2 vector @p var.
 * @param var The vector of length 2 to assign to.
 * @param exp0 The value to assign to var[ 0 ].
 * @param exp1 The value to assign to var[ 1 ].
 */
#define LVARRAY_TENSOROPS_ASSIGN_2( var, exp0, exp1 ) \
  var[ 0 ] = exp0; var[ 1 ] = exp1

/**
 * @brief Create an initializer list of length 3 from @p EXP.
 * @param EXP The expression used to create the initialize list.
 */
#define LVARRAY_TENSOROPS_INIT_LOCAL_3( EXP ) { EXP[ 0 ], EXP[ 1 ], EXP[ 2 ] }

/**
 * @brief Assign to the length 3 vector @p var.
 * @param var The vector of length 3 to assign to.
 * @param exp0 The value to assign to var[ 0 ].
 * @param exp1 The value to assign to var[ 1 ].
 * @param exp2 The value to assign to var[ 2 ].
 */
#define LVARRAY_TENSOROPS_ASSIGN_3( var, exp0, exp1, exp2 ) \
  var[ 0 ] = exp0; var[ 1 ] = exp1; var[ 2 ] = exp2

/**
 * @brief Create an initializer list of length 6 from @p EXP.
 * @param EXP The expression used to create the initialize list.
 */
#define LVARRAY_TENSOROPS_INIT_LOCAL_6( EXP ) { EXP[ 0 ], EXP[ 1 ], EXP[ 2 ], EXP[ 3 ], EXP[ 4 ], EXP[ 5 ] }

/**
 * @brief Assign to the length 6 vector @p var.
 * @param var The vector of length 6 to assign to.
 * @param exp0 The value to assign to var[ 0 ].
 * @param exp1 The value to assign to var[ 1 ].
 * @param exp2 The value to assign to var[ 2 ].
 * @param exp3 The value to assign to var[ 3 ].
 * @param exp4 The value to assign to var[ 4 ].
 * @param exp5 The value to assign to var[ 5 ].
 */
#define LVARRAY_TENSOROPS_ASSIGN_6( var, exp0, exp1, exp2, exp3, exp4, exp5 ) \
  var[ 0 ] = exp0; var[ 1 ] = exp1; var[ 2 ] = exp2; var[ 3 ] = exp3; var[ 4 ] = exp4; var[ 5 ] = exp5

/**
 * @brief Create an 2x2 initializer list from @p EXP.
 * @param EXP The expression used to create the initialize list.
 */
#define LVARRAY_TENSOROPS_INIT_LOCAL_2x2( EXP ) \
  { { EXP[ 0 ][ 0 ], EXP[ 0 ][ 1 ] }, \
    { EXP[ 1 ][ 0 ], EXP[ 1 ][ 1 ] } }

/**
 * @brief Assign to the 2x2 matrix @p var.
 * @param var The 2x2 matrix to assign to.
 * @param exp00 The value to assign to var[ 0 ][ 0 ].
 * @param exp01 The value to assign to var[ 0 ][ 1 ].
 * @param exp10 The value to assign to var[ 1 ][ 0 ].
 * @param exp11 The value to assign to var[ 1 ][ 1 ].
 */
#define LVARRAY_TENSOROPS_ASSIGN_2x2( var, exp00, exp01, exp10, exp11 ) \
  var[ 0 ][ 0 ] = exp00; var[ 0 ][ 1 ] = exp01; \
  var[ 1 ][ 0 ] = exp10; var[ 1 ][ 1 ] = exp11

/**
 * @brief Create an 3x3 initializer list from @p EXP.
 * @param EXP The expression used to create the initialize list.
 */
#define LVARRAY_TENSOROPS_INIT_LOCAL_3x3( EXP ) \
  { { EXP[ 0 ][ 0 ], EXP[ 0 ][ 1 ], EXP[ 0 ][ 2 ] }, \
    { EXP[ 1 ][ 0 ], EXP[ 1 ][ 1 ], EXP[ 1 ][ 2 ] }, \
    { EXP[ 2 ][ 0 ], EXP[ 2 ][ 1 ], EXP[ 2 ][ 2 ] } }

/**
 * @brief Assign to the 3x3 matrix @p var.
 * @param var The 3x3 matrix to assign to.
 * @param exp00 The value to assign to var[ 0 ][ 0 ].
 * @param exp01 The value to assign to var[ 0 ][ 1 ].
 * @param exp02 The value to assign to var[ 0 ][ 2 ].
 * @param exp10 The value to assign to var[ 1 ][ 0 ].
 * @param exp11 The value to assign to var[ 1 ][ 1 ].
 * @param exp12 The value to assign to var[ 1 ][ 2 ].
 * @param exp20 The value to assign to var[ 2 ][ 0 ].
 * @param exp21 The value to assign to var[ 2 ][ 1 ].
 * @param exp22 The value to assign to var[ 2 ][ 2 ].
 */
#define LVARRAY_TENSOROPS_ASSIGN_3x3( var, exp00, exp01, exp02, exp10, exp11, exp12, exp20, exp21, exp22 ) \
  var[ 0 ][ 0 ] = exp00; var[ 0 ][ 1 ] = exp01; var[ 0 ][ 2 ] = exp02; \
  var[ 1 ][ 0 ] = exp10; var[ 1 ][ 1 ] = exp11; var[ 1 ][ 2 ] = exp12; \
  var[ 2 ][ 0 ] = exp20; var[ 2 ][ 1 ] = exp21; var[ 2 ][ 2 ] = exp22

///@}


namespace LvArray
{
namespace tensorOps
{
namespace internal
{

/**
 * @brief Expands to a static constexpr template bool @c HasStaticMember_SIZE which is true if @c T
 *   has a static member @c T::SIZE.
 * @tparam T The type to query.
 */
HAS_STATIC_MEMBER( SIZE );

/**
 * @brief Expands to a static constexpr template bool @c HasStaticMember_NDIM which is true if @c T
 *   has a static member @c T::NDIM.
 * @tparam T The type to query.
 */
HAS_STATIC_MEMBER( NDIM );

/**
 * @brief Verify at compile time that the size of a user-provided type is as expected.
 * @tparam ISIZE The size the array should be.
 * @tparam T The provided type.
 * @param src The value to check.
 * @note This overload is enabled for user-defined types that expose a compile-time size
 *   parameter through a public static integral member variable SIZE
 * @return None.
 */
template< std::ptrdiff_t ISIZE, typename T >
LVARRAY_HOST_DEVICE inline constexpr
std::enable_if_t< HasStaticMember_SIZE< T > >
checkSizes( T const & src )
{
  static_assert( ISIZE == T::SIZE,
                 "Expected the first dimension of size ISIZE, got an type of size T::SIZE." );
  LVARRAY_UNUSED_VARIABLE( src );
}

/**
 * @brief Verify at compile time that the size of the c-array is as expected.
 * @tparam PROVIDED_SIZE The size the array should be.
 * @tparam T The type contained in the array.
 * @tparam INFERRED_SIZE The size of the array.
 * @param src The 1D c-array to check.
 */
template< std::ptrdiff_t PROVIDED_SIZE, typename T, std::ptrdiff_t INFERRED_SIZE >
LVARRAY_HOST_DEVICE inline constexpr
void checkSizes( T const ( &src )[ INFERRED_SIZE ] )
{
  static_assert( PROVIDED_SIZE == INFERRED_SIZE,
                 "Expected the first dimension of size PROVIDED_N, got an array of size INFERRED_N." );
  LVARRAY_UNUSED_VARIABLE( src );
}

/**
 * @brief Verify at compile time that the size of the 2D c-array is as expected.
 * @tparam PROVIDED_M What the size of the first dimension should be.
 * @tparam PROVIDED_M What the size of the second dimension should be.
 * @tparam T The type contained in the array.
 * @tparam PROVIDED_M The size of the first dimension.
 * @tparam PROVIDED_M The size of the second dimension.
 * @param src The 2D c-array to check.
 */
template< std::ptrdiff_t PROVIDED_M,
          std::ptrdiff_t PROVIDED_N,
          typename T,
          std::ptrdiff_t INFERRED_M,
          std::ptrdiff_t INFERRED_N >
LVARRAY_HOST_DEVICE inline constexpr
void checkSizes( T const ( &src )[ INFERRED_M ][ INFERRED_N ] )
{
  static_assert( PROVIDED_M == INFERRED_M, "Expected the first dimension of size PROVIDED_M, got an array of size INFERRED_M." );
  static_assert( PROVIDED_N == INFERRED_N, "Expected the second dimension of size PROVIDED_N, got an array of size INFERRED_N." );
  LVARRAY_UNUSED_VARIABLE( src );
}

/**
 * @brief Verify at compile time that @tparam ARRAY is 1D and at runtime verify the size when LVARRAY_BOUNDS_CHECK.
 * @tparam ISIZE The size expected size.
 * @tparam ARRAY The type o @p array, should be an Array, ArrayView or ArraySlice.
 * @param array The array to check.
 * @return None.
 */
template< std::ptrdiff_t ISIZE, typename ARRAY >
LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
std::enable_if_t< HasStaticMember_NDIM< ARRAY > >
checkSizes( ARRAY const & array )
{
  static_assert( ARRAY::NDIM == 1, "Must be a 1D array." );

#ifdef LVARRAY_BOUNDS_CHECK
  LVARRAY_ERROR_IF_NE( array.size( 0 ), ISIZE );
#else
  LVARRAY_UNUSED_VARIABLE( array );
#endif
}

/**
 * @brief Verify at compile time that @tparam ARRAY is 2D and at runtime verify the sizes when LVARRAY_BOUNDS_CHECK.
 * @tparam ISIZE The expected size of the first dimension.
 * @tparam JSIZE The expected size of the second dimension.
 * @tparam ARRAY The type o @p array, should be an Array, ArrayView or ArraySlice.
 * @param array The array to check.
 * @return None.
 */
template< std::ptrdiff_t ISIZE, std::ptrdiff_t JSIZE, typename ARRAY >
LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
std::enable_if_t< HasStaticMember_NDIM< ARRAY > >
checkSizes( ARRAY const & array )
{
  static_assert( ARRAY::NDIM == 2, "Must be a 1D array." );
#ifdef LVARRAY_BOUNDS_CHECK
  LVARRAY_ERROR_IF_NE( array.size( 0 ), ISIZE );
  LVARRAY_ERROR_IF_NE( array.size( 1 ), JSIZE );
#else
  LVARRAY_UNUSED_VARIABLE( array );
#endif
}

} // namespace internal

/// The size of a symmetric MxM matrix in Voigt notation.
template< std::ptrdiff_t ISIZE >
constexpr std::ptrdiff_t SYM_SIZE = ( ISIZE * ( ISIZE + 1 ) ) / 2;

/**
 * @name Generic operations
 * @brief Functions that are overloaded to operate on both vectors and matrices.
 * @note Not all of the functions have been overloaded to operate on both vectors and
 *   matrices.
 */
///@{

/**
 * @return Return the largest absolute entry of @p vector.
 * @tparam ISIZE The size of @p vector.
 * @tparam VECTOR The type of @p vector.
 * @param vector The vector to get the maximum entry of, of length M.
 */
template< std::ptrdiff_t ISIZE, typename VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
auto maxAbsoluteEntry( VECTOR && vector )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  internal::checkSizes< ISIZE >( vector );

  auto maxVal = math::abs( vector[ 0 ] );
  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    maxVal = math::max( maxVal, math::abs( vector[ i ] ) );
  }

  return maxVal;
}

/**
 * @brief Set the entries of @p vector to @p value.
 * @tparam ISIZE The size of @p vector.
 * @tparam VECTOR The type of @p vector.
 * @param vector The vector to set the entries of, of length M.
 * @param value The value to set the entries to.
 * @details Performs the operation @code vector[ i ] = value @endcode
 */
template< std::ptrdiff_t ISIZE, typename VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void fill( VECTOR && vector, std::remove_reference_t< decltype( vector[ 0 ] ) > const value )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  internal::checkSizes< ISIZE >( vector );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    vector[ i ] = value;
  }
}

/**
 * @brief Set the entries of @p matrix to @p value.
 * @tparam ISIZE The size of the first dimension of @p matrix.
 * @tparam JSIZE The size of the second dimension of @p matrix.
 * @tparam MATRIX The type of @p matrix.
 * @param matrix The matrix to set the entries of, of size ISIZE x N.
 * @param value The value to set the entries to.
 * @details Performs the operation @code matrix[ i ][ j ] = value @endcode
 */
template< std::ptrdiff_t ISIZE, std::ptrdiff_t JSIZE, typename MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void fill( MATRIX && matrix, std::remove_reference_t< decltype( matrix[ 0 ][ 0 ] ) > const value )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  internal::checkSizes< ISIZE, JSIZE >( matrix );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < JSIZE; ++j )
    {
      matrix[ i ][ j ] = value;
    }
  }
}

/**
 * @brief Copy @p srcVector into @p dstVector.
 * @tparam ISIZE The length of @p dstVector and @p srcVector.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam SRC_VECTOR The type of @p srcVector.
 * @param dstVector The destination vector, of length ISIZE.
 * @param srcVector The source vector, of length ISIZE.
 * @details Performs the operation @code dstVector[ i ] = srcVector[ i ] @endcode
 */
template< std::ptrdiff_t ISIZE, typename DST_VECTOR, typename SRC_VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void copy( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
           SRC_VECTOR const & LVARRAY_RESTRICT_REF srcVector )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  internal::checkSizes< ISIZE >( dstVector );
  internal::checkSizes< ISIZE >( srcVector );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    dstVector[ i ] = srcVector[ i ];
  }
}

/**
 * @brief Copy @p srcMatrix into @p dstMatrix.
 * @tparam ISIZE The size of the first dimension of @p dstMatrix and @p srcMatrix.
 * @tparam JSIZE The size of the second dimension of @p dstMatrix and @p srcMatrix.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam SRC_MATRIX The type of @p srcMatrix.
 * @param dstMatrix The destination matrix, of size ISIZE x N.
 * @param srcMatrix The source matrix, of size ISIZE x N.
 * @details Performs the operation @code dstMatrix[ i ][ j ] = srcMatrix[ i ][ j ] @endcode
 */
template< std::ptrdiff_t ISIZE, std::ptrdiff_t JSIZE, typename DST_MATRIX, typename SRC_MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void copy( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
           SRC_MATRIX const & LVARRAY_RESTRICT_REF srcMatrix )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  internal::checkSizes< ISIZE, JSIZE >( dstMatrix );
  internal::checkSizes< ISIZE, JSIZE >( srcMatrix );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < JSIZE; ++j )
    {
      dstMatrix[ i ][ j ] = srcMatrix[ i ][ j ];
    }
  }
}

/**
 * @brief Multiply the entries of @p vector by @p scale.
 * @tparam ISIZE The size of @p vector.
 * @tparam VECTOR The type of @p vector.
 * @param vector The vector to scale, of length M.
 * @param scale The value to scale the entries of @p vector by.
 * @details Performs the operation @code vector[ i ] *= scale @endcode
 */
template< std::ptrdiff_t ISIZE, typename VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void scale( VECTOR && vector, std::remove_reference_t< decltype( vector[ 0 ] ) > const scale )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  internal::checkSizes< ISIZE >( vector );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    vector[ i ] *= scale;
  }
}

/**
 * @brief Multiply the entries of @p matrix by @p scale.
 * @tparam ISIZE The size of the first dimension of @p matrix.
 * @tparam JSIZE The size of the second dimension of @p matrix.
 * @tparam MATRIX The type of @p matrix.
 * @param matrix The matrix to scale, of size ISIZE x N.
 * @param scale The value scale the entries of @p vector by.
 * @details Performs the operations @code matrix[ i ][ j ] *= scale @endcode
 */
template< std::ptrdiff_t ISIZE, std::ptrdiff_t JSIZE, typename MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void scale( MATRIX && matrix, std::remove_reference_t< decltype( matrix[ 0 ][ 0 ] ) > const scale )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  internal::checkSizes< ISIZE, JSIZE >( matrix );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < JSIZE; ++j )
    {
      matrix[ i ][ j ] *= scale;
    }
  }
}

/**
 * @brief Copy @p srcVector scaled by @p scale into @p dstVector.
 * @tparam ISIZE The length of @p dstVector and @p srcVector.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam SRC_VECTOR The type of @p srcVector.
 * @param dstVector The destination vector, of length M.
 * @param srcVector The source vector, of length M.
 * @param scale The value to scale @p srcVector by.
 * @details Performs the operations @code dstVector[ i ] = scale * srcVector[ i ] @endcode
 */
template< std::ptrdiff_t ISIZE, typename DST_VECTOR, typename SRC_VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void scaledCopy( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                 SRC_VECTOR const & LVARRAY_RESTRICT_REF srcVector,
                 std::remove_reference_t< decltype( srcVector[ 0 ] ) > const scale )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  internal::checkSizes< ISIZE >( dstVector );
  internal::checkSizes< ISIZE >( srcVector );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    dstVector[ i ] = scale * srcVector[ i ];
  }
}

/**
 * @brief Copy @p srcMatrix scaled by @p scale into @p dstMatrix.
 * @tparam ISIZE The size of the first dimension of @p dstMatrix and @p srcMatrix.
 * @tparam JSIZE The size of the second dimension of @p dstMatrix and @p srcMatrix.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam SRC_MATRIX The type of @p srcMatrix.
 * @param dstMatrix The destination matrix, of size ISIZE x N.
 * @param srcMatrix The source matrix, of size ISIZE x N.
 * @param scale The value to scale @p srcMatrix by.
 * @details Performs the operation @code dstMatrix[ i ][ j ] = scale * srcMatrix[ i ][ j ] @endcode
 */
template< std::ptrdiff_t ISIZE, std::ptrdiff_t JSIZE, typename DST_MATRIX, typename SRC_MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void scaledCopy( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                 SRC_MATRIX const & LVARRAY_RESTRICT_REF srcMatrix,
                 std::remove_reference_t< decltype( srcMatrix[ 0 ][ 0 ] ) > const scale )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  internal::checkSizes< ISIZE, JSIZE >( dstMatrix );
  internal::checkSizes< ISIZE, JSIZE >( srcMatrix );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < JSIZE; ++j )
    {
      dstMatrix[ i ][ j ] = scale * srcMatrix[ i ][ j ];
    }
  }
}

/**
 * @brief Add @p value to @p dstVector.
 * @tparam M The length of @p dstVector.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @param dstVector The destination vector, of length M.
 * @param value The value to add.
 * @details Performs the operation @code dstVector[ i ] += value @endcode
 */
template< std::ptrdiff_t M, typename DST_VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void addScalar( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                std::remove_reference_t< decltype( dstVector[0] ) > const value )
{
  static_assert( M > 0, "M must be greater than zero." );
  internal::checkSizes< M >( dstVector );

  for( std::ptrdiff_t i = 0; i < M; ++i )
  {
    dstVector[ i ] += value;
  }
}

/**
 * @brief Add @p srcVector to @p dstVector.
 * @tparam ISIZE The length of @p dstVector and @p srcVector.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam SRC_VECTOR The type of @p srcVector.
 * @param dstVector The destination vector, of length M.
 * @param srcVector The source vector, of length M.
 * @details Performs the operation @code dstVector[ i ] += srcVector[ i ] @endcode
 */
template< std::ptrdiff_t ISIZE, typename DST_VECTOR, typename SRC_VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void add( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
          SRC_VECTOR const & LVARRAY_RESTRICT_REF srcVector )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  internal::checkSizes< ISIZE >( dstVector );
  internal::checkSizes< ISIZE >( srcVector );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    dstVector[ i ] += srcVector[ i ];
  }
}

/**
 * @brief Add @p srcMatrix to @p dstMatrix.
 * @tparam ISIZE The size of the first dimension of @p dstMatrix and @p srcMatrix.
 * @tparam JSIZE The size of the second dimension of @p dstMatrix and @p srcMatrix.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam SRC_MATRIX The type of @p srcMatrix.
 * @param dstMatrix The destination matrix, of size ISIZE x N.
 * @param srcMatrix The source matrix, of size ISIZE x N.
 * @details Performs the operation @code dstMatrix[ i ][ j ] += srcMatrix[ i ][ j ] @endcode
 */
template< std::ptrdiff_t ISIZE, std::ptrdiff_t JSIZE, typename DST_MATRIX, typename SRC_MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void add( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
          SRC_MATRIX const & LVARRAY_RESTRICT_REF srcMatrix )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  internal::checkSizes< ISIZE, JSIZE >( dstMatrix );
  internal::checkSizes< ISIZE, JSIZE >( srcMatrix );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < JSIZE; ++j )
    {
      dstMatrix[ i ][ j ] += srcMatrix[ i ][ j ];
    }
  }
}

/**
 * @brief Subtract @p srcVector from @p dstVector.
 * @tparam ISIZE The length of @p dstVector and @p srcVector.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam SRC_VECTOR The type of @p srcVector.
 * @param dstVector The destination vector, of length M.
 * @param srcVector The source vector, of length M.
 * @details Performs the operation @code dstVector[ i ] -= srcVector[ i ] @endcode
 */
template< std::ptrdiff_t ISIZE, typename DST_VECTOR, typename SRC_VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void subtract( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
               SRC_VECTOR const & LVARRAY_RESTRICT_REF srcVector )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  internal::checkSizes< ISIZE >( dstVector );
  internal::checkSizes< ISIZE >( srcVector );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    dstVector[ i ] -= srcVector[ i ];
  }
}

/**
 * @brief Add @p srcVector scaled by @p scale to @p dstVector.
 * @tparam ISIZE The length of @p dstVector and @p srcVector.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam SRC_VECTOR The type of @p srcVector.
 * @param dstVector The destination vector, of length M.
 * @param srcVector The source vector, of length M.
 * @param scale The value to scale the entries of @p srcVector by.
 * @details Performs the operation @code dstVector[ i ] += scale * srcVector[ i ] @endcode
 */
template< std::ptrdiff_t ISIZE, typename DST_VECTOR, typename SRC_VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void scaledAdd( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                SRC_VECTOR const & LVARRAY_RESTRICT_REF srcVector,
                std::remove_reference_t< decltype( srcVector[ 0 ] ) > const scale )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  internal::checkSizes< ISIZE >( dstVector );
  internal::checkSizes< ISIZE >( srcVector );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    dstVector[ i ] = dstVector[ i ] + scale * srcVector[ i ];
  }
}


/**
 * @brief Add @p srcMatrix scaled by @p scale to @p dstMatrix.
 * @tparam ISIZE The size of the first dimension of @p dstMatrix and @p srcMatrix.
 * @tparam JSIZE The size of the second dimension of @p dstMatrix and @p srcMatrix.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam SRC_MATRIX The type of @p srcMatrix.
 * @param dstMatrix The destination matrix, of size ISIZE x N.
 * @param srcMatrix The source matrix, of size ISIZE x N.
 * @param scale The value to scale the entries of @p srcMatrix by.
 * @details Performs the operation @code dstMatrix[ i ][ j ] += srcMatrix[ i ][ j ] @endcode
 */
template< std::ptrdiff_t ISIZE, std::ptrdiff_t JSIZE, typename DST_MATRIX, typename SRC_MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void scaledAdd( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                SRC_MATRIX const & LVARRAY_RESTRICT_REF srcMatrix,
                std::remove_reference_t< decltype( srcMatrix[ 0 ][ 0 ] ) > const scale )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  internal::checkSizes< ISIZE, JSIZE >( dstMatrix );
  internal::checkSizes< ISIZE, JSIZE >( srcMatrix );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < JSIZE; ++j )
    {
      dstMatrix[ i ][ j ] += scale * srcMatrix[ i ][ j ];
    }
  }
}

/**
 * @brief Multiply the elements of @p vectorA and @p vectorB putting the result into @p dstVector.
 * @tparam ISIZE The length of @p dstVector and @p srcVector.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam SRC_VECTOR The type of @p srcVector.
 * @param dstVector The destination vector, of length M.
 * @param vectorA The first source vector, of length M.
 * @param vectorB The second source vector, of length M.
 * @details Performs the operation @code dstVector[ i ] = vectorA[ i ] * vectorB[ i ] @endcode
 */
template< std::ptrdiff_t ISIZE, typename DST_VECTOR, typename VECTOR_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void hadamardProduct( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                      VECTOR_A const & LVARRAY_RESTRICT_REF vectorA,
                      VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  internal::checkSizes< ISIZE >( dstVector );
  internal::checkSizes< ISIZE >( vectorA );
  internal::checkSizes< ISIZE >( vectorB );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    dstVector[ i ] = vectorA[ i ] * vectorB[ i ];
  }
}

///@}

/**
 * @name Vector operations
 * @brief Functions that operate on vectors.
 */
///@{

/**
 * @return The L2 norm squared of @p vector.
 * @tparam ISIZE The length of @p vector.
 * @tparam VECTOR The type of @p vector.
 * @param vector The vector to get the L2 norm squared of, of length M.
 */
template< std::ptrdiff_t ISIZE, typename VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
auto l2NormSquared( VECTOR const & vector )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  internal::checkSizes< ISIZE >( vector );

  auto norm = vector[ 0 ] * vector[ 0 ];
  for( std::ptrdiff_t i = 1; i < ISIZE; ++i )
  {
    norm = norm + vector[ i ] * vector[ i ];
  }
  return norm;
}

/**
 * @return The L2 norm of @p vector.
 * @tparam ISIZE The length of @p vector.
 * @tparam VECTOR The type of @p vector.
 * @param vector The vector to get the L2 norm of, of length M.
 */
template< std::ptrdiff_t ISIZE, typename VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
auto l2Norm( VECTOR const & vector )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  internal::checkSizes< ISIZE >( vector );

  return math::sqrt( l2NormSquared< ISIZE >( vector ) );
}

/**
 * @brief Scale @p vector to a unit vector.
 * @tparam ISIZE The length of @p vector.
 * @tparam VECTOR The type of @p vector.
 * @param vector The vector normalize, of length M.
 * @return The previous L2 norm of @p vector.
 */
template< std::ptrdiff_t ISIZE, typename VECTOR >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
auto normalize( VECTOR && vector )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  internal::checkSizes< ISIZE >( vector );

  auto const normInv = math::invSqrt( l2NormSquared< ISIZE >( vector ) );
  scale< ISIZE >( vector, normInv );

  return 1 / normInv;
}

/**
 * @return Return the inner product of @p vectorA and @p vectorB.
 * @tparam ISIZE The length of @p vectorA and @p vectorB.
 * @tparam VECTOR_A The type of @p vectorA.
 * @tparam VECTOR_B The type of @p vectorB.
 * @param vectorA The first vector in the inner product, of length M.
 * @param vectorB The second vector in the inner product, of length M.
 */
template< std::ptrdiff_t JSIZE, typename VECTOR_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
auto AiBi( VECTOR_A const & LVARRAY_RESTRICT_REF vectorA,
           VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  internal::checkSizes< JSIZE >( vectorA );
  internal::checkSizes< JSIZE >( vectorB );

  auto result = vectorA[ 0 ] * vectorB[ 0 ];
  for( std::ptrdiff_t i = 1; i < JSIZE; ++i )
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

///@}

/**
 * @name Matrix-vector operations
 * @brief Functions that operate on matrices and vectors.
 */
///@{

/**
 * @brief Perform the outer product of @p vectorA and @p vectorB writing the result to @p dstMatrix.
 * @tparam ISIZE The size of the first dimension of @p dstMatrix and the length of @p vectorA.
 * @tparam JSIZE The size of the second dimension of @p dstMatrix and the length of @p vectorB.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam VECTOR_A The type of @p vectorA.
 * @tparam VECTOR_B The type of @p vectorB.
 * @param dstMatrix The matrix the result is written to, of size ISIZE x N.
 * @param vectorA The first vector in the outer product, of length M.
 * @param vectorB The second vector in the outer product, of length N.
 * @details Performs the operations @code dstMatrix[ i ][ j ] = vectorA[ i ] * vectorB[ j ] @endcode
 */
template< std::ptrdiff_t ISIZE, std::ptrdiff_t JSIZE, typename DST_MATRIX, typename VECTOR_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void Rij_eq_AiBj( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                  VECTOR_A const & LVARRAY_RESTRICT_REF vectorA,
                  VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  internal::checkSizes< ISIZE, JSIZE >( dstMatrix );
  internal::checkSizes< ISIZE >( vectorA );
  internal::checkSizes< JSIZE >( vectorB );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < JSIZE; ++j )
    {
      dstMatrix[ i ][ j ] = vectorA[ i ] * vectorB[ j ];
    }
  }
}

/**
 * @brief Perform the outer product of @p vectorA and @p vectorB adding the result to @p dstMatrix.
 * @tparam ISIZE The size of the first dimension of @p dstMatrix and the length of @p vectorA.
 * @tparam JSIZE The size of the second dimension of @p dstMatrix and the length of @p vectorB.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam VECTOR_A The type of @p vectorA.
 * @tparam VECTOR_B The type of @p vectorB.
 * @param dstMatrix The matrix the result is added to, of size ISIZE x N.
 * @param vectorA The first vector in the outer product, of length M.
 * @param vectorB The second vector in the outer product, of length N.
 * @details Performs the operations @code dstMatrix[ i ][ j ] += vectorA[ i ] * vectorB[ j ] @endcode
 */
template< std::ptrdiff_t JSIZE, std::ptrdiff_t ISIZE, typename DST_MATRIX, typename VECTOR_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void Rij_add_AiBj( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                   VECTOR_A const & LVARRAY_RESTRICT_REF vectorA,
                   VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  internal::checkSizes< JSIZE, ISIZE >( dstMatrix );
  internal::checkSizes< JSIZE >( vectorA );
  internal::checkSizes< ISIZE >( vectorB );

  for( std::ptrdiff_t i = 0; i < JSIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < ISIZE; ++j )
    {
      dstMatrix[ i ][ j ] = dstMatrix[ i ][ j ] + vectorA[ i ] * vectorB[ j ];
    }
  }
}

/**
 * @brief Perform the matrix vector multiplication of @p matrixA and @p vectorB writing the result to @p dstVector.
 * @tparam ISIZE The length of @p dstVector and the size of the first dimension of @p matrixA.
 * @tparam JSIZE The size of the second dimension of @p matrixA and the length of @p vectorBB.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam VECTOR_B The type of @p vectorB.
 * @param dstVector The vector to write the result to, of length M.
 * @param matrixA The matrix to multiply @p vectorB by, of size ISIZE x N.
 * @param vectorB The vector multiplied by @p matrixA, of length N.
 * @details Performs the operations @code dstVector[ i ] = matrixA[ i ][ j ] * vectorB[ j ] @endcode
 */
template< std::ptrdiff_t ISIZE, std::ptrdiff_t JSIZE, typename DST_VECTOR, typename MATRIX_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void Ri_eq_AijBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                  MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                  VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  internal::checkSizes< ISIZE >( dstVector );
  internal::checkSizes< ISIZE, JSIZE >( matrixA );
  internal::checkSizes< JSIZE >( vectorB );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    dstVector[ i ] = matrixA[ i ][ 0 ] * vectorB[ 0 ];
    for( std::ptrdiff_t j = 1; j < JSIZE; ++j )
    {
      dstVector[ i ] = dstVector[ i ] + matrixA[ i ][ j ] * vectorB[ j ];
    }
  }
}

/**
 * @brief Perform the matrix vector multiplication of @p matrixA and @p vectorB adding the result to @p dstVector.
 * @tparam ISIZE The length of @p dstVector and the size of the first dimension of @p matrixA.
 * @tparam JSIZE The size of the second dimension of @p matrixA and the length of @p vectorBB.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam VECTOR_B The type of @p vectorB.
 * @param dstVector The vector to add the result to, of length M.
 * @param matrixA The matrix to multiply @p vectorB by, of size ISIZE x N.
 * @param vectorB The vector multiplied by @p matrixA, of length N.
 * @details Performs the operations @code dstVector[ i ] += matrixA[ i ][ j ] * vectorB[ j ] @endcode
 */
template< std::ptrdiff_t ISIZE, std::ptrdiff_t JSIZE, typename DST_VECTOR, typename MATRIX_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void Ri_add_AijBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                   MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                   VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  internal::checkSizes< ISIZE >( dstVector );
  internal::checkSizes< ISIZE, JSIZE >( matrixA );
  internal::checkSizes< JSIZE >( vectorB );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < JSIZE; ++j )
    {
      dstVector[ i ] = dstVector[ i ] + matrixA[ i ][ j ] * vectorB[ j ];
    }
  }
}

/**
 * @brief Perform the matrix vector multiplication of the transpose of @p matrixA and @p vectorB
 *   writing the result to @p dstVector.
 * @tparam ISIZE The length of @p dstVector and the size of the second dimension of @p matrixA.
 * @tparam JSIZE The size of the first dimension of @p matrixA and the length of @p vectorB.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam VECTOR_B The type of @p vectorB.
 * @param dstVector The vector to write the result to, of length M.
 * @param matrixA The matrix to multiply @p vectorB by, of size JSIZE x M.
 * @param vectorB The vector multiplied by @p matrixA, of length N.
 * @details Performs the operations @code dstVector[ i ] = matrixA[ j ][ i ] * vectorB[ j ] @endcode
 */
template< std::ptrdiff_t ISIZE, std::ptrdiff_t JSIZE, typename DST_VECTOR, typename MATRIX_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void Ri_eq_AjiBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                  MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                  VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  internal::checkSizes< ISIZE >( dstVector );
  internal::checkSizes< JSIZE, ISIZE >( matrixA );
  internal::checkSizes< JSIZE >( vectorB );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    dstVector[ i ] = matrixA[ 0 ][ i ] * vectorB[ 0 ];
    for( std::ptrdiff_t j = 1; j < JSIZE; ++j )
    {
      dstVector[ i ] = dstVector[ i ] + matrixA[ j ][ i ] * vectorB[ j ];
    }
  }
}

/**
 * @brief Perform the matrix vector multiplication of the transpose of @p matrixA and @p vectorB
 *   adding the result to @p dstVector.
 * @tparam ISIZE The length of @p dstVector and the size of the second dimension of @p matrixA.
 * @tparam JSIZE The size of the first dimension of @p matrixA and the length of @p vectorB.
 * @tparam DST_VECTOR The type of @p dstVector.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam VECTOR_B The type of @p vectorB.
 * @param dstVector The vector to add the result to, of length M.
 * @param matrixA The matrix to multiply @p vectorB by, of size JSIZE x M.
 * @param vectorB The vector multiplied by @p matrixA, of length N.
 * @details Performs the operations @code dstVector[ i ] += matrixA[ j ][ i ] * vectorB[ j ] @endcode
 */
template< std::ptrdiff_t ISIZE, std::ptrdiff_t JSIZE, typename DST_VECTOR, typename MATRIX_A, typename VECTOR_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void Ri_add_AjiBj( DST_VECTOR && LVARRAY_RESTRICT_REF dstVector,
                   MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                   VECTOR_B const & LVARRAY_RESTRICT_REF vectorB )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  internal::checkSizes< ISIZE >( dstVector );
  internal::checkSizes< JSIZE, ISIZE >( matrixA );
  internal::checkSizes< JSIZE >( vectorB );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < JSIZE; ++j )
    {
      dstVector[ i ] = dstVector[ i ] + matrixA[ j ][ i ] * vectorB[ j ];
    }
  }
}

///@}

/**
 * @name Matrix operations
 * @brief Functions that operate matrices of any shape.
 */
///@{

/**
 * @brief Store the transpose of the NxM matrix @p srcMatrix in @p dstMatrix.
 * @tparam ISIZE The size of the first dimension of @p dstMatrix and the second dimension of @p srcMatrix.
 * @tparam JSIZE The size of the second dimension of @p dstMatrix and the first dimension of @p srcMatrix.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam SRC_MATRIX The type of @p srcMatrix.
 * @param dstMatrix The MxN matrix where the transpose of @p srcMatrix is written.
 * @param srcMatrix The NxM matrix to transpose.
 */
template< std::ptrdiff_t ISIZE, std::ptrdiff_t JSIZE, typename DST_MATRIX, typename SRC_MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void transpose( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                SRC_MATRIX const & LVARRAY_RESTRICT_REF srcMatrix )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  internal::checkSizes< ISIZE, JSIZE >( dstMatrix );
  internal::checkSizes< JSIZE, ISIZE >( srcMatrix );

  for( std::ptrdiff_t i = 0; i < JSIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < ISIZE; ++j )
    {
      dstMatrix[ j ][ i ] = srcMatrix[ i ][ j ];
    }
  }
}

/**
 * @brief Multiply @p matrixA with @p matrixB and put the result into @p dstMatrix.
 * @tparam ISIZE The size of the first dimension of @p dstMatrix and @p matrixA.
 * @tparam JSIZE The size of the second dimension of @p dstMatrix and @p matrixB.
 * @tparam KSIZE The size of the second dimension of @p matrixA and first dimension of @p matrixB.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam MATRIX_B The type of @p matrixB.
 * @param dstMatrix The matrix the result is written to, of size ISIZE x N.
 * @param matrixA The left matrix in the multiplication, of size ISIZE x P.
 * @param matrixB The right matrix in the multiplication, of size KSIZE x N.
 * @details Performs the operation
 *   @code dstMatrix[ i ][ j ] = matrixA[ i ][ k ] * matrixB[ k ][ j ] @endcode
 */
template< std::ptrdiff_t ISIZE,
          std::ptrdiff_t JSIZE,
          std::ptrdiff_t KSIZE,
          typename DST_MATRIX,
          typename MATRIX_A,
          typename MATRIX_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void Rij_eq_AikBkj( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                    MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                    MATRIX_B const & LVARRAY_RESTRICT_REF matrixB )
{
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( KSIZE > 0, "KSIZE must be greater than zero." );
  internal::checkSizes< ISIZE, JSIZE >( dstMatrix );
  internal::checkSizes< ISIZE, KSIZE >( matrixA );
  internal::checkSizes< KSIZE, JSIZE >( matrixB );


  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < JSIZE; ++j )
    {
      dstMatrix[ i ][ j ] = matrixA[ i ][ 0 ] * matrixB[ 0 ][ j ];
      for( std::ptrdiff_t k = 1; k < KSIZE; ++k )
      {
        dstMatrix[ i ][ j ] = dstMatrix[ i ][ j ] + matrixA[ i ][ k ] * matrixB[ k ][ j ];
      }
    }
  }
}

/**
 * @brief Multiply @p matrixA with @p matrixB and add the result to @p dstMatrix.
 * @tparam ISIZE The size of the first dimension of @p dstMatrix and @p matrixA.
 * @tparam JSIZE The size of the second dimension of @p dstMatrix and @p matrixB.
 * @tparam KSIZE The size of the second dimension of @p matrixA and first dimension of @p matrixB.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam MATRIX_B The type of @p matrixB.
 * @param dstMatrix The matrix the result is added to, of size ISIZE x N.
 * @param matrixA The left matrix in the multiplication, of size ISIZE x P.
 * @param matrixB The right matrix in the multiplication, of size KSIZE x N.
 * @details Performs the operation
 *   @code dstMatrix[ i ][ j ] += matrixA[ i ][ k ] * matrixB[ k ][ j ] @endcode
 */
template< std::ptrdiff_t ISIZE,
          std::ptrdiff_t JSIZE,
          std::ptrdiff_t KSIZE,
          typename DST_MATRIX,
          typename MATRIX_A,
          typename MATRIX_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void Rij_add_AikBkj( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                     MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                     MATRIX_B const & LVARRAY_RESTRICT_REF matrixB )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  static_assert( KSIZE > 0, "KSIZE must be greater than zero." );
  internal::checkSizes< ISIZE, JSIZE >( dstMatrix );
  internal::checkSizes< ISIZE, KSIZE >( matrixA );
  internal::checkSizes< KSIZE, JSIZE >( matrixB );


  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < JSIZE; ++j )
    {
      for( std::ptrdiff_t k = 0; k < KSIZE; ++k )
      {
        dstMatrix[ i ][ j ] = dstMatrix[ i ][ j ] + matrixA[ i ][ k ] * matrixB[ k ][ j ];
      }
    }
  }
}

/**
 * @brief Multiply @p matrixA with the transpose of @p matrixB and put the result into @p dstMatrix.
 * @tparam ISIZE The size of the first dimension of @p dstMatrix and @p matrixA.
 * @tparam JSIZE The size of the second dimension of @p dstMatrix and the first dimension of @p matrixB.
 * @tparam KSIZE The size of the second dimension of @p matrixA and @p matrixB.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam MATRIX_B The type of @p matrixB.
 * @param dstMatrix The matrix the result is written to, of size ISIZE x N.
 * @param matrixA The left matrix in the multiplication, of size ISIZE x P.
 * @param matrixB The right matrix in the multiplication, of size JSIZE x P.
 * @details Performs the operation
 *   @code dstMatrix[ i ][ j ] = matrixA[ i ][ k ] * matrixB[ j ][ k ] @endcode
 */
template< std::ptrdiff_t ISIZE,
          std::ptrdiff_t JSIZE,
          std::ptrdiff_t KSIZE,
          typename DST_MATRIX,
          typename MATRIX_A,
          typename MATRIX_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void Rij_eq_AikBjk( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                    MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                    MATRIX_B const & LVARRAY_RESTRICT_REF matrixB )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  static_assert( KSIZE > 0, "KSIZE must be greater than zero." );
  internal::checkSizes< ISIZE, JSIZE >( dstMatrix );
  internal::checkSizes< ISIZE, KSIZE >( matrixA );
  internal::checkSizes< JSIZE, KSIZE >( matrixB );


  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < JSIZE; ++j )
    {
      dstMatrix[ i ][ j ] = matrixA[ i ][ 0 ] * matrixB[ j ][ 0 ];
      for( std::ptrdiff_t k = 1; k < KSIZE; ++k )
      {
        dstMatrix[ i ][ j ] = dstMatrix[ i ][ j ] + matrixA[ i ][ k ] * matrixB[ j ][ k ];
      }
    }
  }
}

/**
 * @brief Multiply @p matrixA with the transpose of @p matrixB and put the result into @p dstMatrix.
 * @tparam ISIZE The size of the first dimension of @p dstMatrix and @p matrixA.
 * @tparam JSIZE The size of the second dimension of @p dstMatrix and the first dimension of @p matrixB.
 * @tparam KSIZE The size of the second dimension of @p matrixA and @p matrixB.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam MATRIX_B The type of @p matrixB.
 * @param dstMatrix The matrix the result is written to, of size ISIZE x N.
 * @param matrixA The left matrix in the multiplication, of size ISIZE x P.
 * @param matrixB The right matrix in the multiplication, of size JSIZE x P.
 * @details Performs the operation
 *   @code dstMatrix[ i ][ j ] += matrixA[ i ][ k ] * matrixB[ j ][ k ] @endcode
 */
template< std::ptrdiff_t ISIZE,
          std::ptrdiff_t JSIZE,
          std::ptrdiff_t KSIZE,
          typename DST_MATRIX,
          typename MATRIX_A,
          typename MATRIX_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void Rij_add_AikBjk( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                     MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                     MATRIX_B const & LVARRAY_RESTRICT_REF matrixB )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  static_assert( KSIZE > 0, "KSIZE must be greater than zero." );
  internal::checkSizes< ISIZE, JSIZE >( dstMatrix );
  internal::checkSizes< ISIZE, KSIZE >( matrixA );
  internal::checkSizes< JSIZE, KSIZE >( matrixB );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < JSIZE; ++j )
    {
      for( std::ptrdiff_t k = 0; k < KSIZE; ++k )
      {
        dstMatrix[ i ][ j ] = dstMatrix[ i ][ j ] + matrixA[ i ][ k ] * matrixB[ j ][ k ];
      }
    }
  }
}

/**
 * @brief Multiply @p matrixA with the transpose of itself and put the result into @p dstMatrix.
 * @tparam ISIZE The size of the first dimension of @p matrixA and both dimensions of @p dstMatrix.
 * @tparam JSIZE The size of the second dimension of matrixA.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam MATRIX_A The type of @p matrixA.
 * @param dstMatrix The matrix the result is written to, of size ISIZE x M.
 * @param matrixA The matrix in the multiplication, of size ISIZE x N.
 * @details Performs the operation
 *   @code dstMatrix[ i ][ j ] += matrixA[ i ][ k ] * matrixA[ j ][ k ] @endcode
 */
template< std::ptrdiff_t ISIZE,
          std::ptrdiff_t JSIZE,
          typename DST_MATRIX,
          typename MATRIX_A >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void Rij_add_AikAjk( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                     MATRIX_A const & LVARRAY_RESTRICT_REF matrixA )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  internal::checkSizes< ISIZE, ISIZE >( dstMatrix );
  internal::checkSizes< ISIZE, JSIZE >( matrixA );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < ISIZE; ++j )
    {
      for( std::ptrdiff_t k = 0; k < JSIZE; ++k )
      {
        dstMatrix[ i ][ j ] = dstMatrix[ i ][ j ] + matrixA[ i ][ k ] * matrixA[ j ][ k ];
      }
    }
  }
}

/**
 * @brief Multiply the transpose of @p matrixA with @p matrixB and put the result into @p dstMatrix.
 * @tparam ISIZE The size of the first dimension of @p dstMatrix and the second dimension of @p matrixA.
 * @tparam JSIZE The size of the second dimension of @p dstMatrix and @p matrixB.
 * @tparam KSIZE The size of the first dimension of @p matrixA and @p matrixB.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam MATRIX_B The type of @p matrixB.
 * @param dstMatrix The matrix the result is written to, of size ISIZE x N.
 * @param matrixA The left matrix in the multiplication, of size KSIZE x M.
 * @param matrixB The right matrix in the multiplication, of size KSIZE x N.
 * @details Performs the operation
 *   @code dstMatrix[ i ][ j ] = matrixA[ k ][ i ] * matrixB[ k ][ j ] @endcode
 */
template< std::ptrdiff_t ISIZE,
          std::ptrdiff_t JSIZE,
          std::ptrdiff_t KSIZE,
          typename DST_MATRIX,
          typename MATRIX_A,
          typename MATRIX_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void Rij_eq_AkiBkj( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                    MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                    MATRIX_B const & LVARRAY_RESTRICT_REF matrixB )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  static_assert( KSIZE > 0, "KSIZE must be greater than zero." );
  internal::checkSizes< ISIZE, JSIZE >( dstMatrix );
  internal::checkSizes< KSIZE, ISIZE >( matrixA );
  internal::checkSizes< KSIZE, JSIZE >( matrixB );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < JSIZE; ++j )
    {
      dstMatrix[ i ][ j ] = matrixA[ 0 ][ i ] * matrixB[ 0 ][ j ];
      for( std::ptrdiff_t k = 1; k < KSIZE; ++k )
      {
        dstMatrix[ i ][ j ] = dstMatrix[ i ][ j ] + matrixA[ k ][ i ] * matrixB[ k ][ j ];
      }
    }
  }
}

/**
 * @brief Multiply the transpose of @p matrixA with @p matrixA and put the result into @p dstMatrix.
 * @tparam ISIZE The size of both dimensions of @p dstMatrix and the second dimension of @p matrixA.
 * @tparam JSIZE The size of the first dimension of @p matrixA.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam MATRIX_A The type of @p matrixA.
 * @param dstMatrix The matrix the result is written to, of size ISIZE x ISIZE.
 * @param matrixA The left matrix in the multiplication, of size JSIZE x ISIZE.
 * @details Performs the operation
 *   @code dstMatrix[ i ][ j ] = matrixA[ k ][ i ] * matrixA[ k ][ j ] @endcode
 */
template< std::ptrdiff_t ISIZE,
          std::ptrdiff_t JSIZE,
          typename DST_MATRIX,
          typename MATRIX_A >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void Rij_eq_AkiAkj( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                    MATRIX_A const & LVARRAY_RESTRICT_REF matrixA )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  internal::checkSizes< ISIZE, ISIZE >( dstMatrix );
  internal::checkSizes< JSIZE, ISIZE >( matrixA );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < ISIZE; ++j )
    {
      dstMatrix[ i ][ j ] = matrixA[ 0 ][ i ] * matrixA[ 0 ][ j ];
      for( std::ptrdiff_t k = 1; k < JSIZE; ++k )
      {
        dstMatrix[ i ][ j ] = dstMatrix[ i ][ j ] + matrixA[ k ][ i ] * matrixA[ k ][ j ];
      }
    }
  }
}

/**
 * @brief Multiply the transpose of @p matrixA with @p matrixB and add the result into @p dstMatrix.
 * @tparam ISIZE The size of the first dimension of @p dstMatrix and the second dimension of @p matrixA.
 * @tparam JSIZE The size of the second dimension of @p dstMatrix and @p matrixB.
 * @tparam KSIZE The size of the first dimension of @p matrixA and @p matrixB.
 * @tparam DST_MATRIX The type of @p dstMatrix.
 * @tparam MATRIX_A The type of @p matrixA.
 * @tparam MATRIX_B The type of @p matrixB.
 * @param dstMatrix The matrix the result is written to, of size ISIZE x N.
 * @param matrixA The left matrix in the multiplication, of size KSIZE x M.
 * @param matrixB The right matrix in the multiplication, of size KSIZE x N.
 * @details Performs the operation
 *   @code dstMatrix[ i ][ j ] += matrixA[ k ][ i ] * matrixB[ k ][ j ] @endcode
 */
template< std::ptrdiff_t ISIZE,
          std::ptrdiff_t JSIZE,
          std::ptrdiff_t KSIZE,
          typename DST_MATRIX,
          typename MATRIX_A,
          typename MATRIX_B >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void Rij_add_AkiBkj( DST_MATRIX && LVARRAY_RESTRICT_REF dstMatrix,
                     MATRIX_A const & LVARRAY_RESTRICT_REF matrixA,
                     MATRIX_B const & LVARRAY_RESTRICT_REF matrixB )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  static_assert( JSIZE > 0, "JSIZE must be greater than zero." );
  static_assert( KSIZE > 0, "KSIZE must be greater than zero." );
  internal::checkSizes< ISIZE, JSIZE >( dstMatrix );
  internal::checkSizes< KSIZE, ISIZE >( matrixA );
  internal::checkSizes< KSIZE, JSIZE >( matrixB );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    for( std::ptrdiff_t j = 0; j < JSIZE; ++j )
    {
      for( std::ptrdiff_t k = 0; k < KSIZE; ++k )
      {
        dstMatrix[ i ][ j ] = dstMatrix[ i ][ j ] + matrixA[ k ][ i ] * matrixB[ k ][ j ];
      }
    }
  }
}

///@}

/**
 * @name Square matrix operations
 * @brief Functions that operate on square matrices of any size.
 */
///@{

/**
 * @brief Transpose the MxM matrix @p matrix.
 * @tparam ISIZE The size of @p matrix.
 * @tparam MATRIX The type of @p matrix.
 * @param matrix The MxM matrix to transpose.
 */
template< std::ptrdiff_t ISIZE, typename MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void transpose( MATRIX && LVARRAY_RESTRICT_REF matrix )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  internal::checkSizes< ISIZE, ISIZE >( matrix );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    for( std::ptrdiff_t j = i + 1; j < ISIZE; ++j )
    {
      auto const entryIJ = matrix[ i ][ j ];
      matrix[ i ][ j ] = matrix[ j ][ i ];
      matrix[ j ][ i ] = entryIJ;
    }
  }
}

/**
 * @brief Add @p scale times the identity matrix to @p matrix.
 * @tparam ISIZE The size of @p matrix.
 * @tparam MATRIX The type of @p matrix.
 * @param matrix The ISIZE x ISIZE matrix to add the identity matrix to.
 * @param scale The amount to scale the identity matrix by.
 * @details Performs the operations @code matrix[ i ][ i ] += scale @endcode
 */
template< std::ptrdiff_t ISIZE, typename MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void addIdentity( MATRIX && matrix, std::remove_reference_t< decltype( matrix[ 0 ][ 0 ] ) > const scale )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  internal::checkSizes< ISIZE, ISIZE >( matrix );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    matrix[ i ][ i ] += scale;
  }
}

/**
 * @return The trace of @p matrix.
 * @tparam ISIZE The size of @p matrix.
 * @tparam MATRIX The type of @p matrix.
 * @param matrix The ISIZE x ISIZE matrix to get the trace of.
 */
template< std::ptrdiff_t ISIZE, typename MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
auto trace( MATRIX const & matrix )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  internal::checkSizes< ISIZE, ISIZE >( matrix );

  auto trace = matrix[ 0 ][ 0 ];
  for( std::ptrdiff_t i = 1; i < ISIZE; ++i )
  {
    trace += matrix[ i ][ i ];
  }

  return trace;
}

///@}

/**
 * @name Symmetric matrix operations
 * @brief Functions that operate on symmetric matrices of any size.
 */
///@{

/**
 * @brief Add @p scale times the identity matrix to @p symMatrix.
 * @tparam ISIZE The size of @p symMatrix.
 * @tparam SYM_MATRIX The type of @p symMatrix.
 * @param symMatrix The ISIZE x ISIZE symmetric matrix to add the identity matrix to.
 * @param scale The amount to scale the identity matrix by.
 * @details Performs the operations @code matrix[ i ][ i ] += scale @endcode
 */
template< std::ptrdiff_t ISIZE, typename SYM_MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
void symAddIdentity( SYM_MATRIX && symMatrix, std::remove_reference_t< decltype( symMatrix[ 0 ] ) > const scale )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  internal::checkSizes< SYM_SIZE< ISIZE > >( symMatrix );

  for( std::ptrdiff_t i = 0; i < ISIZE; ++i )
  {
    symMatrix[ i ] += scale;
  }
}

/**
 * @return The trace of @p symMatrix.
 * @tparam ISIZE The size of @p symMatrix.
 * @tparam SYM_MATRIX The type of @p symMatrix.
 * @param symMatrix The ISIZE x ISIZE symmetric matrix to get the trace of.
 */
template< std::ptrdiff_t ISIZE, typename SYM_MATRIX >
LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK inline
auto symTrace( SYM_MATRIX const & symMatrix )
{
  static_assert( ISIZE > 0, "ISIZE must be greater than zero." );
  internal::checkSizes< SYM_SIZE< ISIZE > >( symMatrix );

  auto trace = symMatrix[ 0 ];
  for( std::ptrdiff_t i = 1; i < ISIZE; ++i )
  {
    trace += symMatrix[ i ];
  }

  return trace;
}

///@}

} // namespace tensorOps
} // namespace LvArray
