/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file sliceHelpers.hpp
 * @brief Contains functions for interacting with ArraySlices of arbitrary dimension.
 */

#pragma once

// Source includes
#include "ArraySlice.hpp"

namespace LvArray
{

/**
 * @tparam T The type of @p value.
 * @tparam LAMBDA the type of the function @p f to apply.
 * @brief Apply the given function to the given value.
 * @param value the value to pass to the function.
 * @param f the function to apply to the value.
 */
DISABLE_HD_WARNING
template< typename T, typename LAMBDA >
LVARRAY_HOST_DEVICE
void forValuesInSlice( T & value, LAMBDA && f )
{ f( value ); }

/**
 * @tparam T The type of values stored in @p slice.
 * @tparam NDIM the dimension of @p slice.
 * @tparam USD the unit stride dimension of @p slice.
 * @tparam INDEX_TYPE the integer used to index into @p slice.
 * @tparam LAMBDA the type of the function @p f to apply.
 * @brief Iterate over the values in the slice in lexicographic order.
 * @param slice the slice to iterate over.
 * @param f the function to apply to each value.
 */
DISABLE_HD_WARNING
template< typename T, int NDIM, int USD, typename INDEX_TYPE, typename LAMBDA >
LVARRAY_HOST_DEVICE
void forValuesInSlice( ArraySlice< T, NDIM, USD, INDEX_TYPE > const slice, LAMBDA && f )
{
  INDEX_TYPE const bounds = slice.size( 0 );
  for( INDEX_TYPE i = 0; i < bounds; ++i )
  {
    forValuesInSlice( slice[ i ], f );
  }
}

/**
 * @tparam T The type of @p value.
 * @tparam INDICES variadic pack of indices.
 * @tparam LAMBDA the type of the function @p f to apply.
 * @brief Apply the function @p f to the value @p value also passing @p f any indices used to reach @p value.
 * @param value The value to apply @p f to.
 * @param f The function to apply to each value.
 * @param indices The previous sliced off indices.
 */
DISABLE_HD_WARNING
template< typename T, typename LAMBDA, typename ... INDICES >
LVARRAY_HOST_DEVICE
void forValuesInSliceWithIndices( T & value, LAMBDA && f, INDICES const ... indices )
{ f( value, indices ... ); }

/**
 * @tparam T The type of values stored in @p slice.
 * @tparam NDIM the dimension of @p slice.
 * @tparam USD the unit stride dimension of @p slice.
 * @tparam INDEX_TYPE the integer used to index into @p slice.
 * @tparam INDICES variadic pack of indices.
 * @tparam LAMBDA the type of the function @p f to apply.
 * @brief Iterate over the values in the slice in lexicographic order, passing the indices as well
 *   as the value to the lambda.
 * @param slice The slice to iterate over.
 * @param f The lambda to apply to each value.
 * @param indices The previous sliced off indices.
 */
DISABLE_HD_WARNING
template< typename T, int NDIM, int USD, typename INDEX_TYPE, typename LAMBDA, typename ... INDICES >
LVARRAY_HOST_DEVICE
void forValuesInSliceWithIndices( ArraySlice< T, NDIM, USD, INDEX_TYPE > const slice,
                                  LAMBDA && f,
                                  INDICES const ... indices )
{
  INDEX_TYPE const bounds = slice.size( 0 );
  for( INDEX_TYPE i = 0; i < bounds; ++i )
  {
    forValuesInSliceWithIndices( slice[ i ], f, indices ..., i );
  }
}

/**
 * @tparam T The type of values stored in @p src.
 * @tparam NDIM the dimension of @p src.
 * @tparam USD_SRC the unit stride dimension of @p src.
 * @tparam INDEX_TYPE the integer used to index into @p src.
 * @brief Add the values in @p src to @p dst.
 * @param src The array slice to sum over.
 * @param dst The value to add the sum to.
 */
template< typename T, int USD_SRC, typename INDEX_TYPE >
void sumOverFirstDimension( ArraySlice< T const, 1, USD_SRC, INDEX_TYPE > const src,
                            T & dst )
{
  INDEX_TYPE const bounds = src.size( 0 );
  for( INDEX_TYPE i = 0; i < bounds; ++i )
  {
    dst += src( i );
  }
}

/**
 * @tparam T The type of values stored in @p src.
 * @tparam NDIM the dimension of @p src.
 * @tparam USD_SRC the unit stride dimension of @p src.
 * @tparam USD_DST the unit stride dimension of @p dst.
 * @tparam INDEX_TYPE the integer used to index into @p src and @p dst.
 * @brief Sum over the first dimension of @p src adding the results to dst.
 * @param src The slice to sum over.
 * @param dst The slice to add to.
 */
template< typename T, int NDIM, int USD_SRC, int USD_DST, typename INDEX_TYPE >
void sumOverFirstDimension( ArraySlice< T const, NDIM, USD_SRC, INDEX_TYPE > const src,
                            ArraySlice< T, NDIM - 1, USD_DST, INDEX_TYPE > const dst )
{
#ifdef ARRAY_SLICE_CHECK_BOUNDS
  for( int i = 1; i < NDIM; ++i )
  {
    LVARRAY_ERROR_IF_NE( src.size( i ), dst.size( i - 1 ) );
  }
#endif

  forValuesInSliceWithIndices( src, [dst] ( T const & value, INDEX_TYPE const, auto const ... indices )
  {
    dst( indices ... ) += value;
  } );
}

} // namespace LvArray
