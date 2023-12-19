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
 * @tparam LAMBDA the type of the function @p f to apply.
 * @brief Iterate over the values in the slice in lexicographic order.
 * @param slice the slice to iterate over.
 * @param f the function to apply to each value.
 */
DISABLE_HD_WARNING
template< typename T, typename LAYOUT, typename LAMBDA >
LVARRAY_HOST_DEVICE
void forValuesInSlice( ArraySlice< T, LAYOUT > const slice, LAMBDA && f )
{
  auto const bounds = slice.template size< 0 >();
  for( typename LAYOUT::IndexType i = 0; i < bounds; ++i )
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
template< typename T, typename LAYOUT, typename LAMBDA, typename ... INDICES >
LVARRAY_HOST_DEVICE
void forValuesInSliceWithIndices( ArraySlice< T, LAYOUT > const slice,
                                  LAMBDA && f,
                                  INDICES const ... indices )
{
  auto const bounds = slice.template size< 0 >();
  for( typename LAYOUT::IndexType i = 0; i < bounds; ++i )
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
template< typename T, typename LAYOUT >
void sumOverFirstDimension( ArraySlice< T const, LAYOUT > const src,
                            T & dst )
{
  auto const bounds = src.template size< 0 >();
  for( typename LAYOUT::IndexType i = 0; i < bounds; ++i )
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
template< typename T, typename LAYOUT_SRC, typename LAYOUT_DST >
void sumOverFirstDimension( ArraySlice< T const, LAYOUT_SRC > const src,
                            ArraySlice< T, LAYOUT_DST > const dst )
{
  static_assert(LAYOUT_DST::NDIM == LAYOUT_SRC::NDIM - 1, "Destination slice must have dimension one less than source");
#ifdef ARRAY_SLICE_CHECK_BOUNDS
  tupleManipulation::forEach2( tupleManipulation::drop_front< 1 >( src.layout().extent() ), dst.layout().extent(), []( auto src_size, auto dst_size )
  {
    LVARRAY_ERROR_IF_NE( src_size, dst_size );
  } );
#endif

  forValuesInSliceWithIndices( src, [dst] ( T const & value, auto, auto const ... indices )
  {
    dst( indices ... ) += value;
  } );
}

} // namespace LvArray
