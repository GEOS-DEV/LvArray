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
 * @file arrayManipulation.hpp
 * This file contains common array manipulation routines. Every function takes a
 * pointer and a size that define the array. Every function assumes that
 * the array has a capacity large enough for the given operation.
 */

#ifndef ARRAYMANIPULATION_HPP_
#define ARRAYMANIPULATION_HPP_

#include "Macros.hpp"

#ifdef USE_ARRAY_BOUNDS_CHECK

#define ARRAYMANIPULATION_CHECK_BOUNDS( index ) \
  LVARRAY_ERROR_IF( !isPositive( index ) || index >= size, \
                    "Array Bounds Check Failed: index=" << index << " size()=" << size )

#define ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index ) \
  LVARRAY_ERROR_IF( !isPositive( index ) || index > size, \
                    "Array Bounds Insert Check Failed: index=" << index << " size()=" << size )

#else // USE_ARRAY_BOUNDS_CHECK

#define ARRAYMANIPULATION_CHECK_BOUNDS( index )
#define ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index )

#endif // USE_ARRAY_BOUNDS_CHECK

namespace LvArray
{
namespace arrayManipulation
{

/**
 * @tparam INDEX_TYPE the integral type to check.
 * @brief Return true iff the given value is greater than or equal to zero.
 * @param [in] i the value to check.
 */
template< typename INDEX_TYPE >
LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
typename std::enable_if< std::is_signed< INDEX_TYPE >::value, bool >::type
isPositive( INDEX_TYPE const i )
{ return i >= 0; }

/**
 * @tparam INDEX_TYPE the integral type to check.
 * @brief Returns true. This specialization for unsigned types avoids compiler warnings.
 */
template< typename INDEX_TYPE >
LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
typename std::enable_if< !std::is_signed< INDEX_TYPE >::value, bool >::type
isPositive( INDEX_TYPE const )
{ return true; }

/**
 * @tparam T the storage type of the array.
 * @brief Destory the values in the array.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
void destroy( T * const LVARRAY_RESTRICT ptr,
              std::ptrdiff_t const size )
{
  LVARRAY_ASSERT( ptr != nullptr || size == 0 );

  for( std::ptrdiff_t i = 0 ; i < size ; ++i )
  {
    ptr[ i ].~T();
  }
}

/**
 * @tparam T the storage type of the array.
 * @brief Copy construct values from the source to the destination.
 * @param [in/out] dst pointer to the destination array, must be uninitialized memory.
 * @param [in] size the number of values to copy.
 * @param [in] src pointer to the source array.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
void uninitializedCopy( T * const LVARRAY_RESTRICT dst,
                        std::ptrdiff_t const size,
                        T const * const LVARRAY_RESTRICT src )
{
  LVARRAY_ASSERT( dst != nullptr || size == 0 );
  LVARRAY_ASSERT( isPositive( size ) );
  LVARRAY_ASSERT( src != nullptr || size == 0 );

  for( std::ptrdiff_t i = 0 ; i < size ; ++i )
  {
    new (dst + i) T( src[ i ] );
  }
}

/**
 * @tparam T the storage type of the array.
 * @brief Move construct values from the source to the destination.
 * @param [in/out] dst pointer to the destination array, must be uninitialized memory.
 * @param [in] size the number of values to copy.
 * @param [in/out] src pointer to the source array.
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

  for( std::ptrdiff_t i = 0 ; i < size ; ++i )
  {
    new (dst + i) T( std::move( src[ i ] ) );
  }
}

/**
 * @tparam T the storage type of the array.
 * @brief Shift values down into uninitialized memory.
 * @param [in/out] ptr pointer to the begining of the shift, values before this must be uninitialized.
 * @param [in] size number of values to shift.
 * @param [in] amount the amount to shift by.
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

  for( std::ptrdiff_t j = 0 ; j < size ; ++j )
  {
    new ( ptr + j - amount ) T( std::move( ptr[ j ] ) );
    ptr[ j ].~T();
  }
}

/**
 * @tparam T the storage type of the array.
 * @brief Shift values up into uninitialized memory.
 * @param [in/out] ptr pointer to the begining of the shift.
 * @param [in] size number of values to shift, values after this must be uninitialized.
 * @param [in] amount the amount to shift by.
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

  for( std::ptrdiff_t j = size - 1 ; j >= 0 ; --j )
  {
    new ( ptr + amount + j ) T( std::move( ptr[ j ] ) );
    ptr[ j ].~T();
  }
}

/**
 * @tparam T the storage type of the array.
 * @tparam ARGS the types of the arguments to forward to the constructor.
 * @brief Resize the give array.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] newSize the new size.
 * @param [in/out] args the arguments to forward to construct any new elements with.
 */
DISABLE_HD_WARNING
template< typename T, typename ... ARGS >
inline
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
  for( std::ptrdiff_t i = size ; i < newSize ; ++i )
  {
    new ( ptr + i ) T( std::forward< ARGS >( args )... );
  }
}

/**
 * @tparam T the storage type of the array.
 * @brief Shift the values in the array at or above the given position up by the given amount.
 *        New uninitialized values take their place.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] index the index at which to begin the shift.
 * @param [in] n the number of places to shift.
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
  for( std::ptrdiff_t i = size ; i > index ; --i )
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
 * @brief Shift the values in the array at or above the given position up by the given amount.
 *        New values take their place.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] index the index at which to begin the shift.
 * @param [in] n the number of places to shift.
 * @param [in] defaultValue the value to initialize the new entries with.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
void emplace( T * const LVARRAY_RESTRICT ptr,
              std::ptrdiff_t const size,
              std::ptrdiff_t const index,
              std::ptrdiff_t const n=1,
              T const & defaultValue=T() )
{
  ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index );

  shiftUp( ptr, size, index, n );

  // Initialize the empty values to the default value.
  for( std::ptrdiff_t i = index ; i < index + n ; ++i )
  {
    new( ptr + i ) T( defaultValue );
  }
}

/**
 * @tparam T the storage type of the array.
 * @brief Shift the values in the array at or above the given position down by the given amount overwriting
 *        the existing values. The n entries at the end of the array are not destroyed.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] index the index at which to begin the shift.
 * @param [in] n the number of places to shift.
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
  for( std::ptrdiff_t i = index ; i < size ; ++i )
  {
    ptr[i - n] = std::move( ptr[i] );
  }
}

/**
 * @tparam T the storage type of the array.
 * @brief Shift the values in the array at or above the given position down by the given amount overwriting
 *        the existing values. The n entries at the end of the array are then destroyed.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] index the index at which to begin the shift.
 * @param [in] n the number of places to shift.
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
 * @tparam T the storage type of the array.
 * @brief Append the given value to the array.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] value the value to append.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
void append( T * const LVARRAY_RESTRICT ptr,
             std::ptrdiff_t const size,
             T const & value )
{
  LVARRAY_ASSERT( ptr != nullptr );
  LVARRAY_ASSERT( isPositive( size ) );
  new ( ptr + size ) T( value );
}

/**
 * @tparam T the storage type of the array.
 * @brief Append the given value to the array.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in/out] value the value to append.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
void append( T * const LVARRAY_RESTRICT ptr,
             std::ptrdiff_t const size,
             T && value )
{
  LVARRAY_ASSERT( ptr != nullptr );
  LVARRAY_ASSERT( isPositive( size ) );
  new ( ptr + size ) T( std::move( value ) );
}

/**
 * @tparam T the storage type of the array.
 * @brief Append the given values to the array.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] values the values to append.
 * @param [in] n the number of values to append.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
void append( T * const LVARRAY_RESTRICT ptr,
             std::ptrdiff_t const size,
             T const * const LVARRAY_RESTRICT values,
             std::ptrdiff_t const n )
{
  LVARRAY_ASSERT( ptr != nullptr || (size == 0 && n == 0) );
  LVARRAY_ASSERT( isPositive( size ) );
  LVARRAY_ASSERT( isPositive( n ) );
  LVARRAY_ASSERT( values != nullptr || n == 0 );

  for( std::ptrdiff_t i = 0 ; i < n ; ++i )
  {
    new ( ptr + size + i ) T( values[i] );
  }
}

/**
 * @tparam T the storage type of the array.
 * @brief Insert the given value into the array at the given position.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] index the position to insert the value at.
 * @param [in] value the value to insert.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
void insert( T * const LVARRAY_RESTRICT ptr,
             std::ptrdiff_t const size,
             std::ptrdiff_t const index,
             T const & value )
{
  LVARRAY_ASSERT( ptr != nullptr );
  LVARRAY_ASSERT( isPositive( size ) );
  ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index );

  // Create space for the new value.
  shiftUp( ptr, size, index, std::ptrdiff_t( 1 ) );
  new ( ptr + index ) T( value );
}

/**
 * @tparam T the storage type of the array.
 * @brief Insert the given value into the array at the given position.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] index the position to insert the value at.
 * @param [in/out] value the value to insert.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
void insert( T * const LVARRAY_RESTRICT ptr,
             std::ptrdiff_t const size,
             std::ptrdiff_t const index,
             T && value )
{
  LVARRAY_ASSERT( ptr != nullptr );
  LVARRAY_ASSERT( isPositive( size ) );
  ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index );

  shiftUp( ptr, size, index, std::ptrdiff_t( 1 ) );
  new ( ptr + index ) T( std::move( value ) );
}

/**
 * @tparam T the storage type of the array.
 * @brief Insert the given values into the array at the given position.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] index the position to insert the value at.
 * @param [in] values the values to insert.
 * @param [in] n the number of values to insert.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
void insert( T * const LVARRAY_RESTRICT ptr,
             std::ptrdiff_t const size,
             std::ptrdiff_t const index,
             T const * const LVARRAY_RESTRICT values,
             std::ptrdiff_t const n )
{
  LVARRAY_ASSERT( ptr != nullptr );
  LVARRAY_ASSERT( isPositive( size ) );
  ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index );

  shiftUp( ptr, size, index, n );

  for( std::ptrdiff_t i = 0 ; i < n ; ++i )
  {
    new ( ptr + index + i ) T( values[ i ] );
  }
}

/**
 * @tparam T the storage type of the array.
 * @brief Destroy the value at the end of the array.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
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

#endif /* ARRAYMANIPULATION_HPP_ */
