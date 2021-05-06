/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file memcpy.hpp
 * @brief Contains the implementation of LvArray::memcpy.
 */

#pragma once

// Source includes
#include "ArrayView.hpp"
#include "umpireInterface.hpp"

// System includes
#include <cstring>

namespace LvArray
{

namespace internal
{

/**
 * @brief Specialization for when all the indices have been consumed.
 * @return @p curSlice.
 * @tparam N_LEFT The number of indices left to consume, this is a specialization for @c N_LEFT == 0.
 * @tparam T The type of values in the slice.
 * @tparam NDIM The number of dimensions in the slice.
 * @tparam USD The unit stride dimension of the slice.
 * @tparam INDEX_TYPE The index type of the slice.
 * @param curSlice The slice to return.
 */
template< int N_LEFT, typename T, int NDIM, int USD, typename INDEX_TYPE >
std::enable_if_t< N_LEFT == 0, ArraySlice< T, NDIM, USD, INDEX_TYPE > >
slice( ArraySlice< T, NDIM, USD, INDEX_TYPE > const curSlice, INDEX_TYPE const * )
{ return curSlice; }

/**
 * @brief Slice @p curSlice until all of @p remainingIndices have been consumed.
 * @tparam N_LEFT The number of indices left to consume.
 * @tparam T The type of values in the slice.
 * @tparam NDIM The number of dimensions in the slice.
 * @tparam USD The unit stride dimension of the slice.
 * @tparam INDEX_TYPE The index type of the slice.
 * @param curSlice The ArraySlice to further slice.
 * @param remainingIndices The indices used to slice @p curSlice.
 * @return @code curSlice[ remainingIndices[ 0 ] ][ remainingIndices[ 1 ] ][ ... ] @endcode.
 */
template< int N_LEFT, typename T, int NDIM, int USD, typename INDEX_TYPE >
std::enable_if_t< N_LEFT != 0, ArraySlice< T, NDIM - N_LEFT, USD - N_LEFT, INDEX_TYPE > >
slice( ArraySlice< T, NDIM, USD, INDEX_TYPE > const curSlice, INDEX_TYPE const * const remainingIndices )
{ return slice< N_LEFT - 1 >( curSlice[ remainingIndices[ 0 ] ], remainingIndices + 1 ); }

} // namespace internal

/**
 * @brief Use memcpy to copy @p src in to @p dst.
 * @tparam T The type contined by @p dst and @p src.
 * @tparam NDIM The number of dimensions in @p dst and @p src.
 * @tparam DST_USD The unit stride dimension of @p dst and @p src.
 * @tparam DST_INDEX_TYPE The index type of @p dst.
 * @tparam SRC_INDEX_TYPE The index type of @p src.
 * @param dst The destination slice.
 * @param src The source slice.
 * @details If both @p src and @p dst were allocated with Umpire then the Umpire ResouceManager is used to perform
 *   the copy, otherwise std::memcpy is used.
 */
template< typename T, int NDIM, int USD, typename DST_INDEX_TYPE, typename SRC_INDEX_TYPE >
void memcpy( ArraySlice< T, NDIM, USD, DST_INDEX_TYPE > const dst,
             ArraySlice< T const, NDIM, USD, SRC_INDEX_TYPE > const src )
{
  LVARRAY_ERROR_IF_NE( dst.size(), src.size() );

  for( int i = 0; i < NDIM; ++i )
  {
    LVARRAY_ERROR_IF_NE( dst.stride( i ), src.stride( i ) );
  }

  T * const dstPointer = dst.dataIfContiguous();
  T * const srcPointer = const_cast< T * >( src.dataIfContiguous() );
  std::size_t numBytes = dst.size() * sizeof( T );

  umpireInterface::copy( dstPointer, srcPointer, numBytes );
}

/**
 * @brief Use memcpy to (maybe asynchronously) copy @p src in to @p dst.
 * @tparam T The type contined by @p dst and @p src.
 * @tparam NDIM The number of dimensions in @p dst and @p src.
 * @tparam DST_USD The unit stride dimension of @p dst and @p src.
 * @tparam DST_INDEX_TYPE The index type of @p dst.
 * @tparam SRC_INDEX_TYPE The index type of @p src.
 * @param resource The resource to use.
 * @param dst The destination slice.
 * @param src The source slice.
 * @return An event corresponding to the copy.
 * @details If both @p src and @p dst were allocated with Umpire then the Umpire ResouceManager is used to perform
 *   the copy, otherwise std::memcpy is used. Umpire does not currently support asynchronous copying with host
 *   resources, in fact it does not even support synchronous copying with host resources. As such if a @p resource
 *   wraps a resource of type @c camp::resouces::Host the method that doesn't take a resource is used.
 */
template< typename T, int NDIM, int USD, typename DST_INDEX_TYPE, typename SRC_INDEX_TYPE >
camp::resources::Event memcpy( camp::resources::Resource & resource,
                               ArraySlice< T, NDIM, USD, DST_INDEX_TYPE > const dst,
                               ArraySlice< T const, NDIM, USD, SRC_INDEX_TYPE > const src )
{
  LVARRAY_ERROR_IF_NE( dst.size(), src.size() );

  for( int i = 0; i < NDIM; ++i )
  {
    LVARRAY_ERROR_IF_NE( dst.stride( i ), src.stride( i ) );
  }

  T * const dstPointer = dst.dataIfContiguous();
  T * const srcPointer = const_cast< T * >( src.dataIfContiguous() );
  std::size_t numBytes = dst.size() * sizeof( T );

  return umpireInterface::copy( dstPointer, srcPointer, resource, numBytes );
}

/**
 * @brief Copy the values from a slice of @p src into a slice of @p dst.
 * @tparam N_DST_INDICES The number of indices used to slice @p dst.
 * @tparam N_DST_INDICES The number of indices used to slice @p src.
 * @tparam T The type of the values contained by @p dst and @p src.
 * @tparam DST_NDIM The number of dimensions in @p dst.
 * @tparam DST_USD The unit stride dimension of @p dst.
 * @tparam DST_INDEX_TYPE The index type of @p dst.
 * @tparam DST_BUFFER The buffer type of @p dst.
 * @tparam SRC_NDIM The number of dimensions in @p src.
 * @tparam SRC_USD The unit stride dimension of @p src.
 * @tparam SRC_INDEX_TYPE The index type of @p src.
 * @tparam SRC_BUFFER The buffer type of @p src.
 * @param dst The destination array.
 * @param dstIndices The indices used to slice @p dst.
 * @param src The source array.
 * @param srcIndices The indices used to slice @p src.
 * @details The copy occurs from the current space of @p src to the current space of @p dst.
 *   Each array is moved to their current space.
 * @code
 *   Array2d< T > x( 10, 10 );
 *   Array1d< T > y( 10 );
 *   // Set x[ 5 ][ : ] = y[ : ]
 *   memcpy< 1, 0 >( x, { 5 }, y, {} );
 * @endcode
 */
template< std::size_t N_DST_INDICES, std::size_t N_SRC_INDICES, typename T, int DST_NDIM, int DST_USD,
          typename DST_INDEX_TYPE, template< typename > class DST_BUFFER, int SRC_NDIM, int SRC_USD,
          typename SRC_INDEX_TYPE, template< typename > class SRC_BUFFER >
void memcpy( ArrayView< T, DST_NDIM, DST_USD, DST_INDEX_TYPE, DST_BUFFER > const & dst,
             std::array< DST_INDEX_TYPE, N_DST_INDICES > const & dstIndices,
             ArrayView< T const, SRC_NDIM, SRC_USD, SRC_INDEX_TYPE, SRC_BUFFER > const & src,
             std::array< SRC_INDEX_TYPE, N_SRC_INDICES > const & srcIndices )
{
#if !defined( LVARRAY_USE_UMPIRE )
  LVARRAY_ERROR_IF_NE_MSG( dst.getPreviousSpace(), MemorySpace::host, "Without Umpire only host memory is supported." );
  LVARRAY_ERROR_IF_NE_MSG( src.getPreviousSpace(), MemorySpace::host, "Without Umpire only host memory is supported." );
#endif

  dst.move( dst.getPreviousSpace(), true );
  src.move( src.getPreviousSpace(), false );

  ArraySlice< T, DST_NDIM - N_DST_INDICES, DST_USD - N_DST_INDICES, DST_INDEX_TYPE > const dstSlice =
    internal::slice< N_DST_INDICES >( dst.toSlice(), dstIndices.data() );

  ArraySlice< T const, SRC_NDIM - N_SRC_INDICES, SRC_USD - N_SRC_INDICES, SRC_INDEX_TYPE > const srcSlice =
    internal::slice< N_SRC_INDICES >( src.toSlice(), srcIndices.data() );

  memcpy( dstSlice, srcSlice );
}

/**
 * @brief Copy the values from a slice of @p src into a slice of @p dst using @p resource (may be asynchronous).
 * @tparam N_DST_INDICES The number of indices used to slice @p dst.
 * @tparam N_DST_INDICES The number of indices used to slice @p src.
 * @tparam T The type of the values contained by @p dst and @p src.
 * @tparam DST_NDIM The number of dimensions in @p dst.
 * @tparam DST_USD The unit stride dimension of @p dst.
 * @tparam DST_INDEX_TYPE The index type of @p dst.
 * @tparam DST_BUFFER The buffer type of @p dst.
 * @tparam SRC_NDIM The number of dimensions in @p src.
 * @tparam SRC_USD The unit stride dimension of @p src.
 * @tparam SRC_INDEX_TYPE The index type of @p src.
 * @tparam SRC_BUFFER The buffer type of @p src.
 * @param resource The resource to use.
 * @param dst The destination array.
 * @param dstIndices The indices used to slice @p dst.
 * @param src The source array.
 * @param srcIndices The indices used to slice @p src.
 * @return An event corresponding to the copy.
 * @details The copy occurs from the current space of @p src to the current space of @p dst.
 *   Each array is moved to their current space.
 */
template< std::size_t N_DST_INDICES, std::size_t N_SRC_INDICES, typename T, int DST_NDIM, int DST_USD,
          typename DST_INDEX_TYPE, template< typename > class DST_BUFFER, int SRC_NDIM, int SRC_USD,
          typename SRC_INDEX_TYPE, template< typename > class SRC_BUFFER >
camp::resources::Event memcpy( camp::resources::Resource & resource,
                               ArrayView< T, DST_NDIM, DST_USD, DST_INDEX_TYPE, DST_BUFFER > const & dst,
                               std::array< DST_INDEX_TYPE, N_DST_INDICES > const & dstIndices,
                               ArrayView< T const, SRC_NDIM, SRC_USD, SRC_INDEX_TYPE, SRC_BUFFER > const & src,
                               std::array< SRC_INDEX_TYPE, N_SRC_INDICES > const & srcIndices )
{
  dst.move( dst.getPreviousSpace(), true );
  src.move( src.getPreviousSpace(), false );

  ArraySlice< T, DST_NDIM - N_DST_INDICES, DST_USD - N_DST_INDICES, DST_INDEX_TYPE > const dstSlice =
    internal::slice< N_DST_INDICES >( dst.toSlice(), dstIndices.data() );

  ArraySlice< T const, SRC_NDIM - N_SRC_INDICES, SRC_USD - N_SRC_INDICES, SRC_INDEX_TYPE > const srcSlice =
    internal::slice< N_SRC_INDICES >( src.toSlice(), srcIndices.data() );

  return memcpy( resource, dstSlice, srcSlice );
}

} // namespace LvArray
