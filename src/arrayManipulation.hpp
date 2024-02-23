/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file arrayManipulation.hpp
 * @brief Contains functions for manipulating a contiguous array of values.
 */

#pragma once

// Source includes
#include "limits.hpp"
#include "Macros.hpp"
#include "typeManipulation.hpp"

// System includes
#include <cstring>

#ifdef LVARRAY_BOUNDS_CHECK

/**
 * @brief Check that @p index is a valid into into the array.
 * @param index The index to check.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define ARRAYMANIPULATION_CHECK_BOUNDS( index ) \
  LVARRAY_ERROR_IF( !isPositive( index ) || index >= size, \
                    "Array Bounds Check Failed: index=" << index << " size()=" << size )

/**
 * @brief Check that @p index is a valid insertion position in the array.
 * @param index The index to check.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index ) \
  LVARRAY_ERROR_IF( !isPositive( index ) || index > size, \
                    "Array Bounds Insert Check Failed: index=" << index << " size()=" << size )

#else // LVARRAY_BOUNDS_CHECK

/**
 * @brief Check that @p index is a valid into into the array.
 * @param index The index to check.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define ARRAYMANIPULATION_CHECK_BOUNDS( index )

/**
 * @brief Check that @p index is a valid insertion position in the array.
 * @param index The index to check.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index )

#endif // LVARRAY_BOUNDS_CHECK

namespace LvArray
{

/**
 * @brief Contains functions for operating on a contiguous array of values.
 * @details Most functions accept a pointer and a size as the first two arguments. Values
 *   in this range are expected to be in a valid state. Values past the end of the array
 *   are expected to be uninitialized. Functions that increase the size of the array
 *   expect the array to have a large enough capacity to handle the increase.
 */
namespace arrayManipulation
{

/**
 * @tparam INDEX_TYPE the integral type to check.
 * @return Return true iff @p i is greater than or equal to zero.
 * @param i the value to check.
 */
template< typename INDEX_TYPE >
LVARRAY_HOST_DEVICE inline constexpr
typename std::enable_if< std::is_signed< INDEX_TYPE >::value, bool >::type
isPositive( INDEX_TYPE const i )
{ return i >= 0; }

/**
 * @tparam INDEX_TYPE the integral type to check.
 * @return Returns true. This specialization for unsigned types avoids compiler warnings.
 */
template< typename INDEX_TYPE >
LVARRAY_HOST_DEVICE inline constexpr
typename std::enable_if< !std::is_signed< INDEX_TYPE >::value, bool >::type
isPositive( INDEX_TYPE )
{ return true; }

/**
 * @tparam ITER An iterator type.
 * @return The distance between two non-random access iterators.
 * @param first The iterator to the beginning of the range.
 * @param last The iterator to the end of the range.
 */
DISABLE_HD_WARNING
template< typename ITER >
inline constexpr LVARRAY_HOST_DEVICE
typename std::iterator_traits< ITER >::difference_type
iterDistance( ITER first, ITER const last, std::input_iterator_tag )
{
  typename std::iterator_traits< ITER >::difference_type n = 0;
  while( first != last )
  {
    ++first;
    ++n;
  }

  return n;
}

/**
 * @tparam ITER An iterator type.
 * @return The distance between two random access iterators.
 * @param first The iterator to the beginning of the range.
 * @param last The iterator to the end of the range.
 */
DISABLE_HD_WARNING
template< typename RandomAccessIterator >
inline constexpr LVARRAY_HOST_DEVICE
typename std::iterator_traits< RandomAccessIterator >::difference_type
iterDistance( RandomAccessIterator first, RandomAccessIterator last, std::random_access_iterator_tag )
{ return last - first; }

/**
 * @tparam ITER An iterator type.
 * @return The distance between two iterators.
 * @param first The iterator to the beginning of the range.
 * @param last The iterator to the end of the range.
 */
DISABLE_HD_WARNING
template< typename ITER >
inline constexpr LVARRAY_HOST_DEVICE
typename std::iterator_traits< ITER >::difference_type
iterDistance( ITER const first, ITER const last )
{ return iterDistance( first, last, typename std::iterator_traits< ITER >::iterator_category() ); }

/**
 * @tparam T the storage type of the array.
 * @brief Destory the values in the array.
 * @param ptr Pointer to the array.
 * @param size The size of the array.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
void destroy( T * const LVARRAY_RESTRICT ptr,
              std::ptrdiff_t const size )
{
  LVARRAY_ASSERT( ptr != nullptr || size == 0 );

  if( !std::is_trivially_destructible< T >::value )
  {
    for( std::ptrdiff_t i = 0; i < size; ++i )
    {
      ptr[ i ].~T();
    }
  }
}

/**
 * @tparam ITER An iterator type.
 * @tparam T the storage type of the array.
 * @brief Copy construct values from the source to the destination.
 * @param first Iterator to the first value to copy.
 * @param last Iterator to the end of the values to copy.
 * @param dst Pointer to the destination array, must be uninitialized memory.
 */
DISABLE_HD_WARNING
template< typename ITER, typename T >
LVARRAY_HOST_DEVICE inline
void uninitializedCopy( ITER first,
                        ITER const & last,
                        T * LVARRAY_RESTRICT dst )
{
  LVARRAY_ASSERT( dst != nullptr || first == last );

  while( first != last )
  {
    new ( dst ) T( *first );
    ++dst;
    ++first;
  }
}

/**
 * @tparam T the storage type of the array.
 * @brief Move construct values from the source to the destination.
 * @param dst pointer to the destination array, must be uninitialized memory.
 * @param size The number of values to copy.
 * @param src pointer to the source array.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
void uninitializedMove( T * const LVARRAY_RESTRICT dst,
                        std::ptrdiff_t const size,
                        T * const LVARRAY_RESTRICT src )
{
  LVARRAY_ASSERT( dst != nullptr || size == 0 );
  LVARRAY_ASSERT( isPositive( size ) );
  LVARRAY_ASSERT( src != nullptr || size == 0 );

  for( std::ptrdiff_t i = 0; i < size; ++i )
  {
    new (dst + i) T( std::move( src[ i ] ) );
  }
}

/**
 * @tparam T the storage type of the array.
 * @brief Shift values down into uninitialized memory.
 * @param ptr Pointer to the begining of the shift, values before this must be uninitialized.
 * @param size number of values to shift.
 * @param amount the amount to shift by.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
void uninitializedShiftDown( T * const LVARRAY_RESTRICT ptr,
                             std::ptrdiff_t const size,
                             std::ptrdiff_t const amount )
{
  LVARRAY_ASSERT( ptr != nullptr || size == 0 );
  LVARRAY_ASSERT( isPositive( size ) );
  LVARRAY_ASSERT( isPositive( amount ) );

  if( amount == 0 )
    return;

  for( std::ptrdiff_t j = 0; j < size; ++j )
  {
    new ( ptr + j - amount ) T( std::move( ptr[ j ] ) );
    ptr[ j ].~T();
  }
}

/**
 * @tparam T the storage type of the array.
 * @brief Shift values up into uninitialized memory.
 * @param ptr Pointer to the begining of the shift.
 * @param size number of values to shift, values after this must be uninitialized.
 * @param amount the amount to shift by.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
void uninitializedShiftUp( T * const LVARRAY_RESTRICT ptr,
                           std::ptrdiff_t const size,
                           std::ptrdiff_t const amount )
{
  LVARRAY_ASSERT( ptr != nullptr || size == 0 );
  LVARRAY_ASSERT( isPositive( size ) );
  LVARRAY_ASSERT( isPositive( amount ) );

  if( amount == 0 )
    return;

  for( std::ptrdiff_t j = size - 1; j >= 0; --j )
  {
    new ( ptr + amount + j ) T( std::move( ptr[ j ] ) );
    ptr[ j ].~T();
  }
}

/**
 * @tparam T the storage type of the array.
 * @tparam ARGS the types of the arguments to forward to the constructor.
 * @brief Resize the give array.
 * @param ptr Pointer to the array.
 * @param size The size of the array.
 * @param newSize the new size.
 * @param args the arguments to forward to construct any new elements with.
 */
DISABLE_HD_WARNING
template< typename T, typename ... ARGS >
LVARRAY_HOST_DEVICE inline
void resize( T * const LVARRAY_RESTRICT ptr,
             std::ptrdiff_t const size,
             std::ptrdiff_t const newSize,
             ARGS && ... args )
{
  LVARRAY_ASSERT( ptr != nullptr || (size == 0 && newSize == 0) );
  LVARRAY_ASSERT( isPositive( size ) );
  LVARRAY_ASSERT( isPositive( newSize ) );

  // Delete things between newSize and size.
  destroy( ptr + newSize, size - newSize );

  // Initialize things between size and newSize.
  if( sizeof ... ( ARGS ) == 0 && std::is_trivially_default_constructible< T >::value )
  {
    if( newSize - size > 0 )
    {
      std::size_t const sizeDiff = integerConversion< std::size_t >( newSize - size );
      memset( reinterpret_cast< void * >( ptr + size ), 0, ( sizeDiff ) * sizeof( T ) );
    }
  }
  else
  {
    // Use std::size_t so that when GCC optimizes this it doesn't produce sign warnings.
    for( std::size_t i = size; i < std::size_t( newSize ); ++i )
    {
      new ( ptr + i ) T( std::forward< ARGS >( args )... );
    }
  }
}

/**
 * @tparam T the storage type of the array.
 * @brief Shift the values in the array at or above the given position up by the given amount.
 *        New uninitialized values take their place.
 * @param ptr Pointer to the array.
 * @param size The size of the array.
 * @param index the index at which to begin the shift.
 * @param n the number of places to shift.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
void shiftUp( T * const LVARRAY_RESTRICT ptr,
              std::ptrdiff_t const size,
              std::ptrdiff_t const index,
              std::ptrdiff_t const n )
{
  LVARRAY_ASSERT( ptr != nullptr || (size == 0 && n == 0) );
  LVARRAY_ASSERT( isPositive( size ) );
  LVARRAY_ASSERT( isPositive( n ) );
  ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index );

  if( n == 0 )
    return;

  // Move the existing values up by n.
  for( std::ptrdiff_t i = size; i > index; --i )
  {
    std::ptrdiff_t const curIndex = i - 1;
    new ( ptr + curIndex + n ) T( std::move( ptr[ curIndex ] ) );
  }

  // Delete the values moved out of.
  std::ptrdiff_t const bounds = (index + n < size) ? index + n : size;
  destroy( ptr + index, bounds - index );
}

/**
 * @tparam T the storage type of the array.
 * @brief Shift the values in the array at or above the given position down by the given amount overwriting
 *        the existing values. The n entries at the end of the array are not destroyed.
 * @param ptr Pointer to the array.
 * @param size The size of the array.
 * @param index the index at which to begin the shift.
 * @param n the number of places to shift.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
void shiftDown( T * const LVARRAY_RESTRICT ptr,
                std::ptrdiff_t const size,
                std::ptrdiff_t const index,
                std::ptrdiff_t const n )
{
  ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index );
  LVARRAY_ASSERT( ptr != nullptr || (size == 0 && n == 0) );
  LVARRAY_ASSERT( isPositive( size ) );
  LVARRAY_ASSERT( isPositive( n ) );
  LVARRAY_ASSERT( index >= n );

  if( n == 0 )
    return;

  // Move the existing down by n.
  for( std::ptrdiff_t i = index; i < size; ++i )
  {
    ptr[i - n] = std::move( ptr[i] );
  }
}

/**
 * @tparam T the storage type of the array.
 * @brief Shift the values in the array at or above the given position down by the given amount overwriting
 *        the existing values. The n entries at the end of the array are then destroyed.
 * @param ptr Pointer to the array.
 * @param size The size of the array.
 * @param index the index at which to begin the shift.
 * @param n the number of places to shift.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
void erase( T * const LVARRAY_RESTRICT ptr,
            std::ptrdiff_t const size,
            std::ptrdiff_t const index,
            std::ptrdiff_t const n=1 )
{
  LVARRAY_ASSERT( isPositive( n ) );
  if( n == 0 )
    return;

  ARRAYMANIPULATION_CHECK_BOUNDS( index );
  ARRAYMANIPULATION_CHECK_BOUNDS( index + n - 1 );

  shiftDown( ptr, size, std::ptrdiff_t( index + n ), n );

  // Delete the values that were moved out of at the end of the array.
  destroy( ptr + size - n, n );
}

/**
 * @brief Append the to the array constructing the new value in place.
 * @tparam T The storage type of the array.
 * @tparam ARGS Variadic pack of types to construct T with, the types of @p args.
 * @param ptr Pointer to the array.
 * @param size The size of the array.
 * @param args The arguments to forward to construct the new value.
 */
DISABLE_HD_WARNING
template< typename T, typename ... ARGS >
LVARRAY_HOST_DEVICE inline
void emplaceBack( T * const LVARRAY_RESTRICT ptr,
                  std::ptrdiff_t const size,
                  ARGS && ... args )
{
  LVARRAY_ASSERT( ptr != nullptr );
  LVARRAY_ASSERT( isPositive( size ) );
  new ( ptr + size ) T( std::forward< ARGS >( args ) ... );
}

/**
 * @brief Append the given values to the array.
 * @tparam T The storage type of the array.
 * @tparam ITER The type of the iterators @p first and @p last.
 * @param ptr Pointer to the array.
 * @param size The size of the array.
 * @param first An iterator to the first value to append.
 * @param last An iterator to the end of the values to append.
 * @return The number of values appended.
 */
DISABLE_HD_WARNING
template< typename T, typename ITER >
LVARRAY_HOST_DEVICE inline
std::ptrdiff_t append( T * const LVARRAY_RESTRICT ptr,
                       std::ptrdiff_t const size,
                       ITER first,
                       ITER const last )
{
  LVARRAY_ASSERT( ptr != nullptr || (size == 0 && first == last) );
  LVARRAY_ASSERT( isPositive( size ) );

  std::ptrdiff_t i = 0;
  while( first != last )
  {
    new( ptr + size + i ) T( *first );
    ++first;
    ++i;
  }

  return i;
}

/**
 * @brief Insert into the array constructing the new value in place.
 * @tparam T The storage type of the array.
 * @tparam ARGS Variadic pack of types to construct T with, the types of @p args.
 * @param ptr Pointer to the array.
 * @param size The size of the array.
 * @param index The position to insert at.
 * @param args The arguments to forward to construct the new value.
 */
DISABLE_HD_WARNING
template< typename T, typename ... ARGS >
LVARRAY_HOST_DEVICE inline
void emplace( T * const LVARRAY_RESTRICT ptr,
              std::ptrdiff_t const size,
              std::ptrdiff_t const index,
              ARGS && ... args )
{
  LVARRAY_ASSERT( ptr != nullptr );
  LVARRAY_ASSERT( isPositive( size ) );
  ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index );

  // Create space for the new value.
  shiftUp( ptr, size, index, std::ptrdiff_t( 1 ) );
  new ( ptr + index ) T( std::forward< ARGS >( args ) ... );
}

/**
 * @brief Insert the given values into the array at the given position.
 * @tparam T The storage type of the array.
 * @tparam ITERATOR An iterator type, the type of @p first.
 * @param ptr Pointer to the array.
 * @param size The size of the array.
 * @param index The position to insert the value at.
 * @param first Iterator to the first value to insert.
 * @param n The number of values to insert.
 */
DISABLE_HD_WARNING
template< typename T, typename ITERATOR >
LVARRAY_HOST_DEVICE inline
void insert( T * const LVARRAY_RESTRICT ptr,
             std::ptrdiff_t const size,
             std::ptrdiff_t const index,
             ITERATOR first,
             std::ptrdiff_t const n )
{
  LVARRAY_ASSERT( ptr != nullptr );
  LVARRAY_ASSERT( isPositive( size ) );
  ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index );

  shiftUp( ptr, size, index, n );

  for( std::ptrdiff_t i = 0; i < n; ++i )
  {
    new ( ptr + index + i ) T( *first );
    ++first;
  }
}

/**
 * @tparam T the storage type of the array.
 * @brief Destroy the value at the end of the array.
 * @param ptr Pointer to the array.
 * @param size The size of the array.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
void popBack( T * const LVARRAY_RESTRICT ptr,
              std::ptrdiff_t const size )
{
  LVARRAY_ASSERT( ptr != nullptr );
  LVARRAY_ASSERT( size > 0 );
  ptr[ size - 1 ].~T();
}

} // namespace arrayManipulation
} // namespace LvArray
