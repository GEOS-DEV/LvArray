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

#include "Logger.hpp"

#ifdef USE_ARRAY_BOUNDS_CHECK

#define ARRAYMANIPULATION_CHECK_BOUNDS( index ) \
  GEOS_ERROR_IF( !isPositive( index ) || index >= size, \
                 "Array Bounds Check Failed: index=" << index << " size()=" << size )

#define ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index ) \
  GEOS_ERROR_IF( !isPositive( index ) || index > size, \
                 "Array Bounds Insert Check Failed: index=" << index << " size()=" << size )

#else // USE_ARRAY_BOUNDS_CHECK

#define ARRAYMANIPULATION_CHECK_BOUNDS( index )
#define ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index )

#endif // USE_ARRAY_BOUNDS_CHECK

namespace arrayManipulation
{

/**
 * @tparam INDEX_TYPE a signed integer.
 * @brief Return true iff the given value is greater than or equal to zero.
 * @param [in] i the value to check.
 */
template <class INDEX_TYPE>
LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
typename std::enable_if<std::is_signed<INDEX_TYPE>::value, bool>::type
isPositive( INDEX_TYPE const i )
{ return i >= 0; }

/**
 * @tparam INDEX_TYPE an unsigned integer.
 * @brief Returns true.
 * This specialization for unsigned types avoids compiler warnings.
 */
template <class INDEX_TYPE>
LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
typename std::enable_if<!std::is_signed<INDEX_TYPE>::value, bool>::type
isPositive( INDEX_TYPE const )
{ return true; }

/**
 * @tparam T the storage type of the array.
 * @tparam INDEX_TYPE the integer type used to index into the array.
 * @tparam ARGS the types of the arguments to forward to the constructor.
 * @brief Resize the give array.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] newSize the new size.
 * @param [in/out] args the arguments to forward to construct any new elements with.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE, class ...ARGS>
LVARRAY_HOST_DEVICE inline
void resize( T * const ptr, INDEX_TYPE const size, INDEX_TYPE const newSize, ARGS &&... args )
{
  GEOS_ASSERT( ptr != nullptr || (size == 0 && newSize == 0));
  GEOS_ASSERT( isPositive( size ));
  GEOS_ASSERT( isPositive( newSize ));

  // Delete things between newSize and size.
  for( INDEX_TYPE i = newSize ; i < size ; ++i )
  {
    ptr[ i ].~T();
  }

  // Initialize things between size and newSize.
  for( INDEX_TYPE i = size ; i < newSize ; ++i )
  {
    new (&ptr[i]) T(std::forward<ARGS>(args)...);
  }
}

/**
 * @tparam T the storage type of the array.
 * @tparam INDEX_TYPE the integer type used to index into the array.
 * @brief Shift the values in the array at or above the given position up by the given amount.
 *        New uninitialized values take their place.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] index the index at which to begin the shift.
 * @param [in] n the number of places to shift.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE>
LVARRAY_HOST_DEVICE inline
void shiftUp( T * const ptr, INDEX_TYPE const size, INDEX_TYPE const index, INDEX_TYPE const n )
{
  GEOS_ASSERT( ptr != nullptr || (size == 0 && n == 0));
  GEOS_ASSERT( isPositive( size ));
  GEOS_ASSERT( isPositive( n ));
  ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index );

  if( n == 0 )
    return;

  // Move the existing values up by n.
  for( INDEX_TYPE i = size ; i > index ; --i )
  {
    INDEX_TYPE const curIndex = i - 1;
    new (&ptr[curIndex + n]) T( std::move( ptr[curIndex] ));
  }

  // Delete the values moved out of.
  INDEX_TYPE const bounds = (index + n < size) ? index + n : size;
  for( INDEX_TYPE i = index ; i < bounds ; ++i )
  {
    ptr[i].~T();
  }
}

/**
 * @tparam T the storage type of the array.
 * @tparam INDEX_TYPE the integer type used to index into the array.
 * @brief Shift the values in the array at or above the given position up by the given amount.
 *        New values take their place.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] index the index at which to begin the shift.
 * @param [in] n the number of places to shift.
 * @param [in] defaultValue the value to initialize the new entries with.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE>
LVARRAY_HOST_DEVICE inline
void emplace( T * const ptr, INDEX_TYPE const size, INDEX_TYPE const index, INDEX_TYPE const n=1, T const & defaultValue=T())
{
  ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index );

  shiftUp( ptr, size, index, n );

  // Initialize the empty values to the default value.
  for( INDEX_TYPE i = index ; i < index + n ; ++i )
  {
    new(&ptr[i]) T( defaultValue );
  }
}

/**
 * @tparam T the storage type of the array.
 * @tparam INDEX_TYPE the integer type used to index into the array.
 * @brief Shift the values in the array at or above the given position down by the given amount overwriting
 *        the existing values. The n entries at the end of the array are not destroyed.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] index the index at which to begin the shift.
 * @param [in] n the number of places to shift.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE>
LVARRAY_HOST_DEVICE inline
void shiftDown( T * const ptr, INDEX_TYPE const size, INDEX_TYPE const index, INDEX_TYPE const n )
{
  ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index );
  GEOS_ASSERT( ptr != nullptr || (size == 0 && n == 0));
  GEOS_ASSERT( isPositive( size ));
  GEOS_ASSERT( isPositive( n ));
  GEOS_ASSERT( index >= n );

  if( n == 0 )
    return;

  // Move the existing down by n.
  for( INDEX_TYPE i = index ; i < size ; ++i )
  {
    ptr[i - n] = std::move( ptr[i] );
  }
}

/**
 * @tparam T the storage type of the array.
 * @tparam INDEX_TYPE the integer type used to index into the array.
 * @brief Shift the values in the array at or above the given position down by the given amount overwriting
 *        the existing values. The n entries at the end of the array are then destroyed.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] index the index at which to begin the shift.
 * @param [in] n the number of places to shift.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE>
LVARRAY_HOST_DEVICE inline
void erase( T * const ptr, INDEX_TYPE const size, INDEX_TYPE const index, INDEX_TYPE const n=1 )
{
  GEOS_ASSERT( isPositive( n ) );
  if( n == 0 ) return;

  ARRAYMANIPULATION_CHECK_BOUNDS( index );
  ARRAYMANIPULATION_CHECK_BOUNDS( index + n - 1 );

  shiftDown( ptr, size, INDEX_TYPE( index + n ), n );

  // Delete the values that were moved out of at the end of the array.
  for( INDEX_TYPE i = size - n ; i < size ; ++i )
  {
    ptr[ i ].~T();
  }
}

/**
 * @tparam T the storage type of the array.
 * @tparam INDEX_TYPE the integer type used to index into the array.
 * @brief Append the given value to the array.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] value the value to append.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE>
LVARRAY_HOST_DEVICE inline
void append( T * const ptr, INDEX_TYPE const size, T const & value )
{
  GEOS_ASSERT( ptr != nullptr );
  GEOS_ASSERT( isPositive( size ));
  new (&ptr[size]) T( value );
}

/**
 * @tparam T the storage type of the array.
 * @tparam INDEX_TYPE the integer type used to index into the array.
 * @brief Append the given value to the array.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in/out] value the value to append.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE>
LVARRAY_HOST_DEVICE inline
void append( T * const ptr, INDEX_TYPE const size, T && value )
{
  GEOS_ASSERT( ptr != nullptr );
  GEOS_ASSERT( isPositive( size ));
  new (&ptr[size]) T( std::move( value ));
}

/**
 * @tparam T the storage type of the array.
 * @tparam INDEX_TYPE the integer type used to index into the array.
 * @brief Append the given values to the array.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] values the values to append.
 * @param [in] n the number of values to append.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE>
LVARRAY_HOST_DEVICE inline
void append( T * const ptr, INDEX_TYPE const size, T const * const values, INDEX_TYPE const n )
{
  GEOS_ASSERT( ptr != nullptr || (size == 0 && n == 0));
  GEOS_ASSERT( isPositive( size ));
  GEOS_ASSERT( isPositive( n ));
  GEOS_ASSERT( values != nullptr || n == 0 );

  for( INDEX_TYPE i = 0 ; i < n ; ++i )
  {
    new (&ptr[size + i]) T( values[i] );
  }
}

/**
 * @tparam T the storage type of the array.
 * @tparam INDEX_TYPE the integer type used to index into the array.
 * @brief Insert the given value into the array at the given position.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] index the position to insert the value at.
 * @param [in] value the value to insert.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE>
LVARRAY_HOST_DEVICE inline
void insert( T * const ptr, INDEX_TYPE const size, INDEX_TYPE const index, T const & value )
{
  GEOS_ASSERT( ptr != nullptr );
  GEOS_ASSERT( isPositive( size ));
  ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index );

  // Create space for the new value.
  shiftUp( ptr, size, index, INDEX_TYPE( 1 ) );
  new (&ptr[index]) T( value );
}

/**
 * @tparam T the storage type of the array.
 * @tparam INDEX_TYPE the integer type used to index into the array.
 * @brief Insert the given value into the array at the given position.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] index the position to insert the value at.
 * @param [in/out] value the value to insert.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE>
LVARRAY_HOST_DEVICE inline
void insert( T * const ptr, INDEX_TYPE const size, INDEX_TYPE const index, T && value )
{
  GEOS_ASSERT( ptr != nullptr );
  GEOS_ASSERT( isPositive( size ));
  ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index );

  shiftUp( ptr, size, index, INDEX_TYPE( 1 ));
  new (&ptr[index]) T( std::move( value ));
}

/**
 * @tparam T the storage type of the array.
 * @tparam INDEX_TYPE the integer type used to index into the array.
 * @brief Insert the given values into the array at the given position.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] index the position to insert the value at.
 * @param [in] values the values to insert.
 * @param [in] n the number of values to insert.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE>
LVARRAY_HOST_DEVICE inline
void insert( T * const ptr, INDEX_TYPE const size, INDEX_TYPE const index, T const * const values, INDEX_TYPE const n )
{
  GEOS_ASSERT( ptr != nullptr );
  GEOS_ASSERT( isPositive( size ));
  ARRAYMANIPULATION_CHECK_INSERT_BOUNDS( index );

  shiftUp( ptr, size, index, n );

  for( INDEX_TYPE i = 0 ; i < n ; ++i )
  {
    new (&ptr[index + i]) T( values[i] );
  }
}

/**
 * @tparam T the storage type of the array.
 * @tparam INDEX_TYPE the integer type used to index into the array.
 * @brief Destroy the value at the end of the array.
 * @param [in/out] ptr pointer to the array.
 * @param [in] size the size of the array.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE>
LVARRAY_HOST_DEVICE inline
void popBack( T * const ptr, INDEX_TYPE const size )
{
  GEOS_ASSERT( ptr != nullptr );
  GEOS_ASSERT( size > 0 );
  ptr[size - 1].~T();
}

} // namespace arrayManipulation

#endif /* ARRAYMANIPULATION_HPP_ */
