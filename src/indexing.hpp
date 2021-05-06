/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file indexing.hpp
 * @brief Contains functions to aid in multidimensional indexing.
 */

#pragma once

// Source includes
#include "Macros.hpp"
#include "typeManipulation.hpp"
#include "limits.hpp"

// TPL includes
#include <RAJA/RAJA.hpp>

namespace LvArray
{

/**
 * @brief Contains functions to aid in multidimensional indexing.
 */
namespace indexing
{

/**
 * @tparam B_IS_ONE If true the multiplication is skipped.
 * @brief A helper struct to multiply two numbers.
 */
template< bool B_IS_ONE >
struct ConditionalMultiply
{
  /**
   * @tparam A The type of @p a.
   * @tparam B the type of @p b.
   * @return Return the product of @p a and @p b.
   * @param a The left multiplication operand.
   * @param b The right multiplication operand.
   */
  template< typename A, typename B >
  static inline LVARRAY_HOST_DEVICE constexpr auto multiply( A const a, B const b )
  { return a * b; }
};

/**
 * @brief A specialization of ConditionalMultiply that skips the multiplication.
 */
template<>
struct ConditionalMultiply< true >
{
  /**
   * @tparam A The type of @p a.
   * @tparam B the type of @p b.
   * @return Return @p a since the second argument is assumed to be 1.
   * @param a The value to return.
   * @note The second argument is passed by reference so that you can do
   * @code  ConditionalMultiply< true >( 5, *static_cast< int * >( nullptr ) ) @endcode.
   */
  template< typename A, typename B >
  static inline LVARRAY_HOST_DEVICE constexpr A multiply( A const a, B const & )
  { return a; }
};

/**
 * @tparam SIZE The number of values to multiply together.
 * @tparam T The type of the values.
 * @return The product of the values in [ @p values, @p values + SIZE )
 * @param values A pointer to the values to multiply together.
 */
template< int SIZE, typename T >
LVARRAY_HOST_DEVICE inline constexpr
std::enable_if_t< (SIZE == 1), T >
multiplyAll( T const * const LVARRAY_RESTRICT values )
{ return values[ 0 ]; }

/**
 * @tparam SIZE The number of values to multiply together.
 * @tparam T The type of the values.
 * @return The product of the values in [ @p values, @p values + SIZE )
 * @param values A pointer to the values to multiply together.
 */
template< int SIZE, typename T >
LVARRAY_HOST_DEVICE inline constexpr
std::enable_if_t< (SIZE > 1), T >
multiplyAll( T const * const LVARRAY_RESTRICT values )
{ return values[ 0 ] * multiplyAll< SIZE - 1 >( values + 1 ); }

/**
 * @tparam USD The unit stride dimension of strides.
 * @tparam INDEX_TYPE The integral type of the strides and the type to return.
 * @tparam INDEX The integral type of the index.
 * @brief Get the index into a one dimensional space.
 * @param strides A pointer to the stride of the dimension.
 * @param index The index into the dimension.
 * @note If USD == 0 then strides[ 0 ] is assumed to equal 1.
 * @return The product of index with strides[ 0 ].
 */
template< int USD, typename INDEX_TYPE, typename INDEX >
LVARRAY_HOST_DEVICE inline constexpr
INDEX_TYPE getLinearIndex( INDEX_TYPE const * const LVARRAY_RESTRICT strides, INDEX const index )
{ return ConditionalMultiply< USD == 0 >::multiply( index, strides[ 0 ] ); }

/**
 * @tparam USD The unit stride dimension of strides.
 * @tparam INDEX_TYPE The integral type of the strides and the type to return.
 * @tparam INDEX The integral type of the first index.
 * @tparam REMAINING_INDICES A variadic pack of the integral types of the remaining indices.
 * @brief Get the index into a a multidimensional space.
 * @param strides A pointer to the strides of the dimension.
 * @param index The index into the first dimension.
 * @param indices A variadic pack of the indices to the remaining dimensions.
 * @note If 0 <= USD < the number of dimensions then strides[ USD ] is assumed to equal 1.
 * @return The dot product of @p strides with ( @p index, @p indices ... ).
 */
template< int USD, typename INDEX_TYPE, typename INDEX, typename ... REMAINING_INDICES >
LVARRAY_HOST_DEVICE inline constexpr
INDEX_TYPE getLinearIndex( INDEX_TYPE const * const LVARRAY_RESTRICT strides, INDEX const index, REMAINING_INDICES const ... indices )
{
  return ConditionalMultiply< USD == 0 >::multiply( index, strides[ 0 ] ) +
         getLinearIndex< USD - 1, INDEX_TYPE, REMAINING_INDICES... >( strides + 1, indices ... );
}

/// @return A string representing an empty set of indices.
inline
std::string getIndexString()
{ return "{}"; }

/**
 * @tparam INDEX The integral type of the first index.
 * @tparam REMAINING_INDICES A variadic pack of the integral types of the remaining indices.
 * @return A string representing the indices.
 * @param index The first index.
 * @param indices A variadic pack of indices to the remaining dimensions.
 */
template< typename INDEX, typename ... REMAINING_INDICES >
std::string getIndexString( INDEX const index, REMAINING_INDICES const ... indices )
{
  std::ostringstream oss;

  oss << "{ " << index;
  using expander = int[];
  (void) expander{ 0, ( void (oss << ", " << indices ), 0 )... };
  oss << " }";

  return oss.str();
}

/**
 * @tparam INDEX_TYPE The integral type of the dimensions.
 * @tparam INDICES A variadic pack of the integral types of the indices.
 * @return A string representing the dimensions of the multidimensional space and the indices into it.
 * @param dims A pointer to the dimensions of the space.
 * @param indices A variadic pack of indices.
 */
template< typename INDEX_TYPE, typename ... INDICES >
std::string printDimsAndIndices( INDEX_TYPE const * const LVARRAY_RESTRICT dims, INDICES const... indices )
{
  constexpr int NDIM = sizeof ... (INDICES);
  std::ostringstream oss;
  oss << "dimensions = { " << dims[ 0 ];
  for( int i = 1; i < NDIM; ++i )
  {
    oss << ", " << dims[ i ];
  }

  oss << " } indices = " << getIndexString( indices ... );

  return oss.str();
}

/**
 * @tparam INDEX_TYPE The integral type used for the dimensions of the space.
 * @tparam INDICES A variadic pack of the integral types of the indices.
 * @return If the indices into the multidimensional space are invalid.
 * @param dims A pointer to the dimensions of the space.
 * @param indices A variadic pack of the indices.
 */
template< typename INDEX_TYPE, typename ... INDICES >
LVARRAY_HOST_DEVICE inline constexpr
bool invalidIndices( INDEX_TYPE const * const LVARRAY_RESTRICT dims, INDICES const ... indices )
{
  int curDim = 0;
  bool invalid = false;
  typeManipulation::forEachArg( [dims, &curDim, &invalid]( auto const index )
  {
    invalid = invalid || ( index < 0 ) || ( index >= dims[ curDim ] );
    ++curDim;
  }, indices ... );

  return invalid;
}

/**
 * @tparam INDEX_TYPE The integral type used for the dimensions of the space.
 * @tparam INDICES A variadic pack of the integral types of the indices.
 * @brief Check that the @p indices are with @p dims , if not the program is aborted.
 * @param dims A pointer to the dimensions of the space.
 * @param indices A variadic pack of the indices.
 */
template< typename INDEX_TYPE, typename ... INDICES >
LVARRAY_HOST_DEVICE inline
void checkIndices( INDEX_TYPE const * const LVARRAY_RESTRICT dims, INDICES const ... indices )
{ LVARRAY_ERROR_IF( invalidIndices( dims, indices ... ), "Invalid indices. " << printDimsAndIndices( dims, indices ... ) ); }

/**
 * @brief Calculate the strides given the dimensions and permutation.
 * @tparam PERMUTATION The permutation to apply to the dimensions to calculate the strides.
 * @tparam INDEX_TYPE The integral type used for the dimensions of the space.
 * @tparam NDIM The number of dimensions.
 * @param dims The size of each dimension.
 * @return The strides of each dimension.
 * @note Adapted from RAJA::make_permuted_layout.
 */
template< typename PERMUTATION, typename INDEX_TYPE, camp::idx_t NDIM >
LVARRAY_HOST_DEVICE inline
typeManipulation::CArray< INDEX_TYPE, NDIM > calculateStrides( typeManipulation::CArray< INDEX_TYPE, NDIM > const & dims )
{
  constexpr typeManipulation::CArray< camp::idx_t, NDIM > perm = typeManipulation::asArray( PERMUTATION {} );
  INDEX_TYPE foldedStrides[ NDIM ];

  for( int i = 0; i < NDIM; ++i )
  {
    foldedStrides[ i ] = 1;
    for( int j = i + 1; j < NDIM; ++j )
    {
      foldedStrides[ i ] *= dims[ perm[ j ] ];
    }
  }

  typeManipulation::CArray< INDEX_TYPE, NDIM > strides;
  for( int i = 0; i < NDIM; ++i )
  {
    strides[ perm[ i ] ] = foldedStrides[ i ];
  }

  return strides;
}

} // namespace indexing
} // namespace LvArray
