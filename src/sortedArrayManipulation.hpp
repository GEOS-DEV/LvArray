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
 * @file sortedArrayManipulator.hpp
 * This file contains common sorted array manipulation routines.
 * Aside from the functions that take a callback every function assumes that
 * the array has a capacity large enough for the given operation.
 */

#ifndef SORTEDARRAYMANIPULATION_HPP_
#define SORTEDARRAYMANIPULATION_HPP_

// Source includes
#include "Macros.hpp"
#include "arrayManipulation.hpp"
#include "sortedArrayManipulationHelpers.hpp"

// System includes
#include <cstdlib>      // for std::malloc and std::free.
#include <algorithm>    // for std::sort

namespace LvArray
{
namespace sortedArrayManipulation
{

enum Description
{
  SORTED_UNIQUE,
  UNSORTED_NO_DUPLICATES,
  SORTED_WITH_DUPLICATES,
  UNSORTED_WITH_DUPLICATES
};

LVARRAY_HOST_DEVICE constexpr inline
bool isSorted( Description const desc )
{ return desc == SORTED_UNIQUE || desc == SORTED_WITH_DUPLICATES; }

LVARRAY_HOST_DEVICE constexpr inline
bool isUnique( Description const desc )
{ return desc == SORTED_UNIQUE || desc == UNSORTED_NO_DUPLICATES; }

/**
 * @class CallBacks
 * @brief This class provides a no-op callbacks interface for the ArrayManipulation sorted routines.
 */
template< typename T >
class CallBacks
{
public:

  /**
   * @brief Callback signaling that the size of the array has increased.
   * @param [in] curPtr the current pointer to the array.
   * @param [in] nToAdd the increase in the size of the array.
   * @return a pointer to the array.
   */
  LVARRAY_HOST_DEVICE inline
  T * incrementSize( T * const curPtr,
                     std::ptrdiff_t const LVARRAY_UNUSED_ARG( nToAdd ) ) LVARRAY_RESTRICT_THIS
  { return curPtr; }

  /**
   * @brief Callback signaling that a value was inserted at the given position.
   * @param [in] pos the position the value was inserted at.
   */
  LVARRAY_HOST_DEVICE inline
  void insert( std::ptrdiff_t const LVARRAY_UNUSED_ARG( pos ) ) LVARRAY_RESTRICT_THIS
  {}

  /**
   * @brief Callback signaling that the entry of the array at the first position was set to the value
   *        at the second position.
   * @param [in] pos the position in the array that was set.
   * @param [in] valuePos the position of the value that the entry in the array was set to.
   */
  LVARRAY_HOST_DEVICE inline
  void set( std::ptrdiff_t const LVARRAY_UNUSED_ARG( pos ),
            std::ptrdiff_t const LVARRAY_UNUSED_ARG( valuePos ) ) LVARRAY_RESTRICT_THIS
  {}

  /**
   * @brief Callback signaling that the that the given value was inserted at the given position.
   *        Further information is provided in order to make the insertion efficient.
   * @param [in] nLeftToInsert the number of insertions that occur after this one.
   * @param [in] valuePos the position of the value that was inserted.
   * @param [in] pos the position in the array the value was inserted at.
   * @param [in] prevPos the position the previous value was inserted at or the size of the array
   *             if it is the first insertion.
   */
  LVARRAY_HOST_DEVICE inline
  void insert( std::ptrdiff_t const LVARRAY_UNUSED_ARG( nLeftToInsert ),
               std::ptrdiff_t const LVARRAY_UNUSED_ARG( valuePos ),
               std::ptrdiff_t const LVARRAY_UNUSED_ARG( pos ),
               std::ptrdiff_t const LVARRAY_UNUSED_ARG( prevPos ) ) LVARRAY_RESTRICT_THIS
  {}

  /**
   * @brief Callback signaling that an entry was removed from the array at given position.
   * @param [in] pos the position of the entry that was removed.
   */
  LVARRAY_HOST_DEVICE inline
  void remove( std::ptrdiff_t const LVARRAY_UNUSED_ARG( pos ) ) LVARRAY_RESTRICT_THIS
  {}

  /**
   * @brief Callback signaling that the given entry was removed from the given position. Further information
   *        is provided in order to make the removal efficient.
   * @param [in] nRemoved the number of entries removed, starts at 1.
   * @param [in] curPos the position in the array the entry was removed at.
   * @param [in] nextPos the position the next entry will be removed at or the original size of the array
   *             if this was the last entry removed.
   */
  LVARRAY_HOST_DEVICE inline
  void remove( std::ptrdiff_t const LVARRAY_UNUSED_ARG( nRemoved ),
               std::ptrdiff_t const LVARRAY_UNUSED_ARG( curPos ),
               std::ptrdiff_t const LVARRAY_UNUSED_ARG( nextPos ) ) LVARRAY_RESTRICT_THIS
  {}
};

/**
 * @tparam T the type of the values to compare.
 * @class less
 * @brief This class operates as functor similar to std::less.
 */
template< typename T >
struct less
{
  /**
   * @brief Return true iff lhs < rhs.
   */
  DISABLE_HD_WARNING
  constexpr LVARRAY_HOST_DEVICE inline
  bool operator() ( T const & lhs, T const & rhs ) const LVARRAY_RESTRICT_THIS
  { return lhs < rhs; }
};

/**
 * @tparam T the type of the values to compare.
 * @class less
 * @brief This class operates as functor similar to std::greater.
 */
template< typename T >
struct greater
{
  /**
   * @brief Return true iff lhs > rhs.
   */
  DISABLE_HD_WARNING
  constexpr LVARRAY_HOST_DEVICE inline
  bool operator() ( T const & lhs, T const & rhs ) const LVARRAY_RESTRICT_THIS
  { return lhs > rhs; }
};

/**
 * @tparam RandomAccessIterator an iterator type that provides random access.
 * @tparam Compare the type of the comparison function.
 * @brief Sort the given values in place using the given comparator.
 * @param [in/out] first a RandomAccessIterator to the beginning of the values to sort.
 * @param [in/out] last a RandomAccessIterator to the end of the values to sort.
 * @param [in/out] comp a function that does the comparison between two objects.
 *
 * @note should be equivalent to std::sort(first, last, comp).
 */
DISABLE_HD_WARNING
template< typename RandomAccessIterator, typename Compare >
LVARRAY_HOST_DEVICE inline void makeSorted( RandomAccessIterator first, RandomAccessIterator last, Compare && comp )
{
#ifdef __CUDA_ARCH__
  if( last - first > internal::INTROSORT_THRESHOLD )
  {
    internal::introsortLoop( first, last, comp );
  }

  internal::insertionSort( first, last - first, comp );
#else
  std::sort( first, last, std::forward< Compare >( comp ) );
#endif
}

/**
 * @tparam RandomAccessIterator an iterator type that provides random access.
 * @brief Sort the given values in place from least to greatest.
 * @param [in/out] first a RandomAccessIterator to the beginning of the values to sort.
 * @param [in/out] last a RandomAccessIterator to the end of the values to sort.
 *
 * @note should be equivalent to std::sort(first, last).
 */
DISABLE_HD_WARNING
template< typename RandomAccessIterator >
LVARRAY_HOST_DEVICE inline void makeSorted( RandomAccessIterator first, RandomAccessIterator last )
{ return makeSorted( first, last, less< typename std::remove_reference< decltype(*first) >::type >() ); }

/**
 * @tparam RandomAccessIteratorA an iterator type that provides random access.
 * @tparam RandomAccessIteratorB an iterator type that provides random access.
 * @tparam Compare the type of the comparison function.
 * @brief Sort the given values in place using the given comparator and perform the same operations
 *        on the data array thus preserving the mapping between values[i] and data[i].
 * @param [in/out] valueFirst a RandomAccessIterator to the beginning of the values to sort.
 * @param [in/out] valueLast a RandomAccessIterator to the end of the values to sort.
 * @param [in/out] dataFirst a RandomAccessIterator to the beginning of the data.
 * @param [in/out] comp a function that does the comparison between two objects.
 */
DISABLE_HD_WARNING
template< typename RandomAccessIteratorA, typename RandomAccessIteratorB, typename Compare >
LVARRAY_HOST_DEVICE inline void dualSort( RandomAccessIteratorA valueFirst, RandomAccessIteratorA valueLast,
                                          RandomAccessIteratorB dataFirst, Compare && comp )
{
  std::ptrdiff_t const size = valueLast - valueFirst;
  internal::DualIterator< RandomAccessIteratorA, RandomAccessIteratorB > dualIter( valueFirst, dataFirst );

  auto dualCompare = dualIter.createComparator( std::forward< Compare >( comp ) );

  if( size > internal::INTROSORT_THRESHOLD )
  {
    internal::introsortLoop( dualIter, dualIter + size, dualCompare );
  }

  internal::insertionSort( dualIter, size, dualCompare );
}


/**
 * @tparam RandomAccessIteratorA an iterator type that provides random access.
 * @tparam RandomAccessIteratorB an iterator type that provides random access.
 * @brief Sort the given values in place from least to greatest and perform the same operations
 *        on the data array thus preserving the mapping between values[i] and data[i].
 * @param [in/out] valueFirst a RandomAccessIterator to the beginning of the values to sort.
 * @param [in/out] valueLast a RandomAccessIterator to the end of the values to sort.
 * @param [in/out] dataFirst a RandomAccessIterator to the beginning of the data.
 */
DISABLE_HD_WARNING
template< typename RandomAccessIteratorA, typename RandomAccessIteratorB >
LVARRAY_HOST_DEVICE inline
void dualSort( RandomAccessIteratorA valueFirst,
               RandomAccessIteratorA valueLast,
               RandomAccessIteratorB dataFirst )
{ return dualSort( valueFirst, valueLast, dataFirst, less< typename std::remove_reference< decltype(*valueFirst) >::type >() ); }

/**
 * @tparam T the type of the values stored in the buffer.
 * @tparam N the size of the local buffer.
 * @brief Create a copy of the values array in localBuffer if possible. If not an allocation is
 *        made and the values are copied into that.
 * @param [in] values pointer to the values to copy.
 * @param [in] nVals the number of values to copy.
 * @param [in/out] localBuffer a reference to an array T[N].
 * @return A pointer to the buffer containing a copy of the values array.
 *
 * @note freeTemporaryBuffer must be called with the pointer returned from this method, nVals, and localBuffer to
 *       deallocate the created buffer.
 */
DISABLE_HD_WARNING
template< typename T, int N >
LVARRAY_HOST_DEVICE inline
T * createTemporaryBuffer( T const * const LVARRAY_RESTRICT values,
                           std::ptrdiff_t const nVals,
                           T (& localBuffer)[N] )
{
  T * buffer = localBuffer;
  if( nVals <= N )
  {
    for( std::ptrdiff_t i = 0; i < nVals; ++i )
    {
      localBuffer[i] = values[i];
    }
  }
  else
  {
    buffer = static_cast< T * >( std::malloc( sizeof( T ) * nVals ) );

    for( std::ptrdiff_t i = 0; i < nVals; ++i )
    {
      new ( buffer + i ) T( values[i] );
    }
  }

  return buffer;
}

/**
 * @tparam T the type of the values stored in the buffer.
 * @tparam N the size of the local buffer.
 * @brief Deallocate a buffer created from createTemporaryBuffer.
 * @param [in] buffer the buffer returned by createTemporaryBuffer.
 * @param [in] nVals the number of values in the buffer.
 * @param [in] localBuffer a reference to an array T[N].
 */
DISABLE_HD_WARNING
template< typename T, int N >
LVARRAY_HOST_DEVICE inline
void freeTemporaryBuffer( T * const LVARRAY_RESTRICT buffer,
                          std::ptrdiff_t const nVals,
                          T const (&localBuffer)[N] )
{
  if( buffer == localBuffer )
  {
    return;
  }

  for( std::ptrdiff_t i = 0; i < nVals; ++i )
  {
    buffer[i].~T();
  }
  std::free( buffer );
}

/**
 * @tparam T the type of values in the array.
 * @tparam Compare the type of the comparison function, defaults to less<T>.
 * @brief Return true if the given array is sorted using comp.
 * @param [in] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in/out] the comparison method to use.
 *
 * @note Should be equivalent to std::is_sorted(ptr, ptr + size, comp).
 */
DISABLE_HD_WARNING
template< typename T, typename Compare=less< T > >
LVARRAY_HOST_DEVICE inline
bool isSorted( T const * const LVARRAY_RESTRICT ptr,
               std::ptrdiff_t const size,
               Compare && comp=Compare() )
{
  LVARRAY_ASSERT( ptr != nullptr || size == 0 );
  LVARRAY_ASSERT( arrayManipulation::isPositive( size ) );

  for( std::ptrdiff_t i = 0; i < size - 1; ++i )
  {
    if( comp( ptr[i + 1], ptr[i] ) )
    {
      return false;
    }
  }

  return true;
}

/**
 * @tparam T the type of the values stored in the array.
 * @brief Return true if the array contains no duplicates.
 * @param [in/out] ptr a pointer to the values.
 * @param [in] size the number of values.
 * @note The array must be sorted such that values that compare equal are adjacent.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
bool allUnique( T const * const LVARRAY_RESTRICT ptr, std::ptrdiff_t const size )
{
  for( std::ptrdiff_t i = 0; i < size - 1; ++i )
  {
    if( ptr[i] == ptr[i + 1] )
      return false;
  }

  return true;
}

/**
 * @tparam T the type of the values stored in the array.
 * @brief Remove duplicates from the array, duplicates aren't destroyed but they're moved out of.
 * @return The number of unique elements, such that ptr[0, returnValue) contains the unique elements.
 * @param [in/out] ptr a pointer to the values.
 * @param [in] size the number of values.
 * @note The array must be sorted such that values that compare equal are adjacent.
 * @note If you are using this and would like the duplicates to remain intact at the end of the array
 *       change the std::move to a portable implementation of std::swap.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
std::ptrdiff_t removeDuplicates( T * const LVARRAY_RESTRICT ptr, std::ptrdiff_t const size )
{
  LVARRAY_ASSERT( ptr != nullptr || size == 0 );
  LVARRAY_ASSERT( arrayManipulation::isPositive( size ) );

  // Note it doesn't have to be sorted under less<T>, so this assert is a bit too restrictive.
  LVARRAY_ASSERT( isSorted( ptr, size ) );

  if( size == 0 )
    return 0;

  std::ptrdiff_t shiftAmount = 0;
  std::ptrdiff_t curPos = 0;
  while( curPos + shiftAmount + 1 < size )
  {
    if( ptr[curPos] != ptr[curPos + shiftAmount + 1] )
    {
      ++curPos;
      if( shiftAmount != 0 )
      {
        ptr[curPos] = std::move( ptr[curPos + shiftAmount] );
      }
    }
    else
      ++shiftAmount;
  }

  return curPos + 1;
}

/**
 * @tparam T the type of values in the array.
 * @tparam Compare the type of the comparison function, defaults to less<T>.
 * @brief Return the index of the first value in the array greater than or equal to value.
 * @param [in] ptr pointer to the array, must be sorted under comp.
 * @param [in] size the size of the array.
 * @param [in] value the value to find.
 * @param [in/out] comp the comparison method to use.
 *
 * @note Should be equivalent to std::lower_bound(ptr, ptr + size, value, comp).
 */
DISABLE_HD_WARNING
template< typename T, typename Compare=less< T > >
LVARRAY_HOST_DEVICE inline
std::ptrdiff_t find( T const * const LVARRAY_RESTRICT ptr,
                     std::ptrdiff_t const size,
                     T const & value,
                     Compare && comp=Compare() )
{
  LVARRAY_ASSERT( ptr != nullptr || size == 0 );
  LVARRAY_ASSERT( arrayManipulation::isPositive( size ) );
  LVARRAY_ASSERT( isSorted( ptr, size, comp ) );

  std::ptrdiff_t lower = 0;
  std::ptrdiff_t upper = size;
  while( lower != upper )
  {
    std::ptrdiff_t const guess = (lower + upper) / 2;
    if( comp( ptr[guess], value ) )
    {
      lower = guess + 1;
    }
    else
    {
      upper = guess;
    }
  }

  return lower;
}

/**
 * @tparam T the type of values in the array.
 * @tparam Compare the type of the comparison function, defaults to less<T>.
 * @brief Return true if the array contains value.
 * @param [in] ptr pointer to the array, must be sorted under comp.
 * @param [in] size the size of the array.
 * @param [in] value the value to find.
 * @param [in/out] comp the comparison method to use.
 *
 * @note Should be equivalent to std::binary_search(ptr, ptr + size, value, comp).
 */
DISABLE_HD_WARNING
template< typename T, typename Compare=less< T > >
LVARRAY_HOST_DEVICE inline
bool contains( T const * const LVARRAY_RESTRICT ptr,
               std::ptrdiff_t const size,
               T const & value,
               Compare && comp=Compare() )
{
  std::ptrdiff_t const pos = find( ptr, size, value, std::forward< Compare >( comp ) );
  return (pos != size) && (ptr[pos] == value);
}

/**
 * @tparam T the type of values in the array.
 * @tparam CALLBACKS the type of the callBacks class.
 * @brief Remove the given value from the array if it exists.
 * @param [in] ptr pointer to the array, must be sorted under less<T>.
 * @param [in] size the size of the array.
 * @param [in] value the value to remove.
 * @param [in/out] callBacks class which must define a method equivalent to CallBacks::remove(std::ptrdiff_t).
 * @return True iff the value was removed.
 */
DISABLE_HD_WARNING
template< typename T, typename CALLBACKS >
LVARRAY_HOST_DEVICE inline
bool remove( T * const LVARRAY_RESTRICT ptr,
             std::ptrdiff_t const size,
             T const & value,
             CALLBACKS && callBacks )
{
  LVARRAY_ASSERT( ptr != nullptr || size == 0 );
  LVARRAY_ASSERT( arrayManipulation::isPositive( size ) );

  // Find the position of value.
  std::ptrdiff_t const index = find( ptr, size, value );

  // If it's not in the array we can return.
  if( index == size || ptr[index] != value )
  {
    return false;
  }

  // Remove the value and call the callback.
  arrayManipulation::erase( ptr, size, index );
  callBacks.remove( index );
  return true;
}

/**
 * @tparam T the type of values in the array.
 * @brief Remove the given value from the array if it exists.
 * @param [in] ptr pointer to the array, must be sorted under less<T>.
 * @param [in] size the size of the array.
 * @param [in] value the value to remove.
 * @return True iff the value was removed.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
bool remove( T * const LVARRAY_RESTRICT ptr,
             std::ptrdiff_t const size,
             T const & value )
{ return remove( ptr, size, value, CallBacks< T > {} ); }

/**
 * @tparam T the type of values in the array.
 * @tparam CALLBACKS the type of the callBacks class.
 * @brief Remove the given values from the array if they exist.
 * @param [in] ptr pointer to the array, must be sorted under less<T>.
 * @param [in] size the size of the array.
 * @param [in] values the values to remove, must be sorted under less<T>.
 * @param [in] nVals the number of values to remove.
 * @param [in/out] callBacks class which must define methods equivalent to
 *                 CallBacks::remove(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t).
 * @return The number of values removed.
 */
DISABLE_HD_WARNING
template< typename T, typename CALLBACKS >
LVARRAY_HOST_DEVICE inline
std::ptrdiff_t removeSorted( T * const LVARRAY_RESTRICT ptr,
                             std::ptrdiff_t const size,
                             T const * const LVARRAY_RESTRICT values,
                             std::ptrdiff_t const nVals,
                             CALLBACKS && callBacks )
{
  LVARRAY_ASSERT( ptr != nullptr || size == 0 );
  LVARRAY_ASSERT( arrayManipulation::isPositive( size ) );
  LVARRAY_ASSERT( values != nullptr || nVals == 0 );
  LVARRAY_ASSERT( arrayManipulation::isPositive( nVals ) );
  LVARRAY_ASSERT( isSorted( values, nVals ) );

  // If there are no values to remove we can return.
  if( nVals == 0 )
    return 0;

  // Find the position of the first value to remove and the position it's at in the array.
  std::ptrdiff_t firstValuePos = nVals;
  std::ptrdiff_t curPos = size;
  for( std::ptrdiff_t i = 0; i < nVals; ++i )
  {
    curPos = find( ptr, size, values[i] );

    // If the current value is larger than the largest value in the array we can return.
    if( curPos == size )
      return 0;

    if( ptr[curPos] == values[i] )
    {
      firstValuePos = i;
      break;
    }
  }

  // If we didn't find a value to remove we can return.
  if( firstValuePos == nVals )
    return 0;

  // Loop over the values
  std::ptrdiff_t nRemoved = 0;
  for( std::ptrdiff_t curValuePos = firstValuePos; curValuePos < nVals; )
  {
    // Find the next value to remove
    std::ptrdiff_t nextValuePos = nVals;
    std::ptrdiff_t nextPos = size;
    for( std::ptrdiff_t j = curValuePos + 1; j < nVals; ++j )
    {
      // Skip over duplicate values
      if( values[j] == values[j - 1] )
        continue;

      // Find the position
      std::ptrdiff_t const pos = find( ptr + curPos, size - curPos, values[j] ) + curPos;

      // If it's not in the array then neither are any of the rest of the values.
      if( pos == size )
        break;

      // If it's in the array then we can exit this loop.
      if( ptr[pos] == values[j] )
      {
        nextValuePos = j;
        nextPos = pos;
        break;
      }
    }

    // Shift the values down and call the callback.
    nRemoved += 1;
    arrayManipulation::shiftDown( ptr, nextPos, curPos + 1, nRemoved );
    callBacks.remove( nRemoved, curPos, nextPos );

    curValuePos = nextValuePos;
    curPos = nextPos;

    // If we've reached the end of the array we can exit the loop.
    if( curPos == size )
      break;
  }

  // Destroy the values moved out of at the end of the array.
  for( std::ptrdiff_t i = size - nRemoved; i < size; ++i )
  {
    ptr[i].~T();
  }

  return nRemoved;
}

/**
 * @tparam T the type of values in the array.
 * @brief Remove the given values from the array if they exist.
 * @param [in] ptr pointer to the array, must be sorted under less<T>.
 * @param [in] size the size of the array.
 * @param [in] values the values to remove, must be sorted under less<T>.
 * @param [in] nVals the number of values to remove.
 * @return The number of values removed.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
std::ptrdiff_t removeSorted( T * const LVARRAY_RESTRICT ptr,
                             std::ptrdiff_t const size,
                             T const * const LVARRAY_RESTRICT values,
                             std::ptrdiff_t const nVals )
{ return removeSorted( ptr, size, values, nVals, CallBacks< T > {} ); }

/**
 * @tparam T the type of values in the array.
 * @tparam CALLBACKS the type of the callBacks class.
 * @brief Remove the given values from the array if they exist.
 * @param [in] ptr pointer to the array, must be sorted under less<T>.
 * @param [in] size the size of the array.
 * @param [in] values the values to remove.
 * @param [in] nVals the number of values to remove.
 * @param [in/out] callBacks class which must define a method similar to
 *                 CallBacks::remove(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t).
 * @return The number of values removed.
 */
DISABLE_HD_WARNING
template< typename T, typename CALLBACKS >
LVARRAY_HOST_DEVICE inline
std::ptrdiff_t remove( T * const LVARRAY_RESTRICT ptr,
                       std::ptrdiff_t const size,
                       T const * const values,
                       std::ptrdiff_t const nVals,
                       CALLBACKS && callBacks )
{
  LVARRAY_ASSERT( ptr != nullptr || size == 0 );
  LVARRAY_ASSERT( arrayManipulation::isPositive( size ) );
  LVARRAY_ASSERT( values != nullptr || nVals == 0 );
  LVARRAY_ASSERT( arrayManipulation::isPositive( nVals ) );

  constexpr std::ptrdiff_t LOCAL_SIZE = 16;
  T localBuffer[LOCAL_SIZE];
  T * const buffer = createTemporaryBuffer( values, nVals, localBuffer );
  makeSorted( buffer, buffer + nVals );

  std::ptrdiff_t const nInserted = removeSorted( ptr, size, buffer, nVals, std::move( callBacks ) );

  freeTemporaryBuffer( buffer, nVals, localBuffer );
  return nInserted;
}

/**
 * @tparam T the type of values in the array.
 * @brief Remove the given values from the array if they exist.
 * @param [in] ptr pointer to the array, must be sorted under less<T>.
 * @param [in] size the size of the array.
 * @param [in] values the values to remove.
 * @param [in] nVals the number of values to remove.
 * @return The number of values removed.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
std::ptrdiff_t remove( T * const LVARRAY_RESTRICT ptr,
                       std::ptrdiff_t const size,
                       T const * const values,
                       std::ptrdiff_t const nVals )
{ return remove( ptr, size, values, nVals, CallBacks< T > {} ); }

/**
 * @tparam T the type of values in the array.
 * @tparam CALLBACKS the type of the callBacks class.
 * @brief Insert the given value into the array if it doesn't already exist.
 * @param [in/out] ptr pointer to the array, must be sorted under less<T>.
 * @param [in] size the size of the array.
 * @param [in] value the value to insert.
 * @param [in/out] callBacks class which must define methods similar to
 *                 CallBacks::incrementSize(std::ptrdiff_t) and CallBacks::insert(std::ptrdiff_t).
 * @return True iff the value was inserted.
 */
DISABLE_HD_WARNING
template< typename T, typename CALLBACKS >
LVARRAY_HOST_DEVICE inline
bool insert( T * const LVARRAY_RESTRICT ptr,
             std::ptrdiff_t const size,
             T const & value,
             CALLBACKS && callBacks )
{
  LVARRAY_ASSERT( ptr != nullptr || size == 0 );
  LVARRAY_ASSERT( arrayManipulation::isPositive( size ) );

  std::ptrdiff_t insertPos = std::ptrdiff_t( -1 );
  if( size == 0 || value < ptr[0] ) // Take care of the case of an empty array or inserting at the beginning.
  {
    insertPos = 0;
  }
  else if( ptr[size - 1] < value ) // The case of inserting at the end.
  {
    insertPos = size;
  }
  else // otherwise do a binary search
  {
    // We've already checked the first and last values.
    insertPos = find( ptr, size, value );
  }

  // If it's in the array we call the incrementSize callback and return false.
  if( insertPos != size && ptr[insertPos] == value )
  {
    callBacks.incrementSize( ptr, 0 );
    return false;
  }

  // Otherwise call the incrementSize callback, get the new pointer, insert and call the insert callback.
  T * const newPtr = callBacks.incrementSize( ptr, 1 );
  arrayManipulation::insert( newPtr, size, insertPos, value );
  callBacks.insert( insertPos );
  return true;
}

/**
 * @tparam T the type of values in the array.
 * @brief Insert the given value into the array if it doesn't already exist.
 * @param [in] ptr pointer to the array, must be sorted under less<T>.
 * @param [in] size the size of the array.
 * @param [in] value the value to insert.
 * @return True iff the value was inserted.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
bool insert( T const * const LVARRAY_RESTRICT ptr,
             std::ptrdiff_t const size,
             T const & value )
{ return insert( ptr, size, value, CallBacks< T > {} ); }

/**
 * @tparam T the type of values in the array.
 * @tparam CALLBACKS the type of the callBacks class.
 * @brief Insert the given values into the array if they don't already exist.
 * @param [in] ptr pointer to the array, must be sorted under less<T>.
 * @param [in] size the size of the array.
 * @param [in] values the values to insert, must be sorted under less<T>.
 * @param [in] nVals the number of values to insert.
 * @param [in/out] callBacks class which must define methods similar to
 *                 CallBacks::incrementSize(std::ptrdiff_t), CallBacks::set(std::ptrdiff_t, std::ptrdiff_t)
 *                 and CallBacks::insert(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, * std::ptrdiff_t).
 * @return The number of values inserted.
 */
DISABLE_HD_WARNING
template< typename T, typename CALLBACKS >
LVARRAY_HOST_DEVICE inline
std::ptrdiff_t insertSorted( T * const LVARRAY_RESTRICT ptr,
                             std::ptrdiff_t const size,
                             T const * const LVARRAY_RESTRICT values,
                             std::ptrdiff_t const nVals,
                             CALLBACKS && callBacks )
{
  LVARRAY_ASSERT( ptr != nullptr || size == 0 );
  LVARRAY_ASSERT( arrayManipulation::isPositive( size ) );
  LVARRAY_ASSERT( values != nullptr || nVals == 0 );
  LVARRAY_ASSERT( nVals >= 0 );
  LVARRAY_ASSERT( isSorted( values, nVals ) );

  // If there are no values to insert.
  if( nVals == 0 )
  {
    callBacks.incrementSize( ptr, 0 );
    return 0;
  }

  std::ptrdiff_t nToInsert = 0; // The number of values that will actually be inserted.

  // Special case for inserting into an empty array.
  if( size == 0 )
  {
    // Count up the number of unique values.
    nToInsert = 1;
    for( std::ptrdiff_t i = 1; i < nVals; ++i )
    {
      if( values[i] != values[i - 1] )
        ++nToInsert;
    }

    // Call callBack with the number to insert and get a new pointer back.
    T * const newPtr = callBacks.incrementSize( ptr, nToInsert );

    // Insert the first value.
    new ( newPtr ) T( values[0] );
    callBacks.set( 0, 0 );

    // Insert the remaining values, checking for duplicates.
    std::ptrdiff_t curInsertPos = 1;
    for( std::ptrdiff_t i = 1; i < nVals; ++i )
    {
      if( values[i] != values[i - 1] )
      {
        new ( newPtr + curInsertPos ) T( values[i] );
        callBacks.set( curInsertPos, i );
        ++curInsertPos;
      }
    }

    return nToInsert;
  }

  // We need to count up the number of values that will actually be inserted.
  // Doing so involves finding the position the value will be inserted at.
  // Below we store the first MAX_PRE_CALCULATED positions so that we don't need
  // to find them again.

  constexpr int MAX_PRE_CALCULATED = 32;
  std::ptrdiff_t valuePositions[MAX_PRE_CALCULATED];  // Positions of the first 32 values to insert.
  std::ptrdiff_t insertPositions[MAX_PRE_CALCULATED]; // Position at which to insert the first 32 values.

  // Loop over the values in reverse (from largest to smallest).
  std::ptrdiff_t curPos = size;
  for( std::ptrdiff_t i = nVals - 1; i >= 0; --i )
  {
    // Skip duplicate values.
    if( i != 0 && values[i] == values[i - 1] )
      continue;

    curPos = find( ptr, curPos, values[i] );

    // If values[i] isn't in the array.
    if( curPos == size || ptr[curPos] != values[i] )
    {
      // Store the value position and insert position if there is space left.
      if( nToInsert < MAX_PRE_CALCULATED )
      {
        valuePositions[nToInsert] = i;
        insertPositions[nToInsert] = curPos;
      }

      nToInsert += 1;
    }
  }

  // Call callBack with the number to insert and get a new pointer back.
  T * const newPtr = callBacks.incrementSize( ptr, nToInsert );

  // If there are no values to insert we can return.
  if( nToInsert == 0 )
    return 0;

  //
  std::ptrdiff_t const nPreCalculated = (nToInsert < MAX_PRE_CALCULATED) ?
                                        nToInsert : MAX_PRE_CALCULATED;

  // Insert pre-calculated values.
  std::ptrdiff_t prevInsertPos = size;
  for( std::ptrdiff_t i = 0; i < nPreCalculated; ++i )
  {
    // Shift the values up...
    arrayManipulation::shiftUp( newPtr, prevInsertPos, insertPositions[i], std::ptrdiff_t( nToInsert - i ) );

    // and insert.
    std::ptrdiff_t const curValuePos = valuePositions[i];
    new ( newPtr + insertPositions[ i ] + nToInsert - i - 1 ) T( values[curValuePos] );
    callBacks.insert( nToInsert - i, curValuePos, insertPositions[i], prevInsertPos );

    prevInsertPos = insertPositions[i];
  }

  // If all the values to insert were pre-calculated we can return.
  if( nToInsert <= MAX_PRE_CALCULATED )
    return nToInsert;

  // Insert the rest of the values.
  std::ptrdiff_t const prevValuePos = valuePositions[MAX_PRE_CALCULATED - 1];
  std::ptrdiff_t nInserted = MAX_PRE_CALCULATED;
  for( std::ptrdiff_t i = prevValuePos - 1; i >= 0; --i )
  {
    // Skip duplicates
    if( values[i] == values[i + 1] )
      continue;

    std::ptrdiff_t const pos = find( newPtr, prevInsertPos, values[i] );

    // If values[i] is already in the array skip it.
    if( pos != prevInsertPos && newPtr[pos] == values[i] )
      continue;

    // Else shift the values up and insert.
    arrayManipulation::shiftUp( newPtr, prevInsertPos, pos, std::ptrdiff_t( nToInsert - nInserted ) );
    new ( newPtr + pos + nToInsert - nInserted - 1 ) T( values[i] );
    callBacks.insert( nToInsert - nInserted, i, pos, prevInsertPos );

    nInserted += 1;
    prevInsertPos = pos;

    // If all the values have been inserted then exit the loop.
    if( nInserted == nToInsert )
      break;
  }

  LVARRAY_ASSERT_MSG( nInserted == nToInsert, nInserted << " " << nToInsert );

  return nToInsert;
}

/**
 * @tparam T the type of values in the array.
 * @brief Insert the given values into the array if they don't already exist.
 * @param [in] ptr pointer to the array, must be sorted under less<T>.
 * @param [in] size the size of the array.
 * @param [in] values the values to insert, must be sorted under less<T>.
 * @param [in] nVals the number of values to insert.
 * @return The number of values inserted.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE inline
std::ptrdiff_t insertSorted( T * const LVARRAY_RESTRICT ptr,
                             std::ptrdiff_t const size,
                             T const * const LVARRAY_RESTRICT values,
                             std::ptrdiff_t const nVals )
{ return insertSorted( ptr, size, values, nVals, CallBacks< T > {} ); }

/**
 * @tparam T the type of values in the array.
 * @tparam CALLBACKS the type of the callBacks class.
 * @brief Insert the given values into the array if they don't already exist.
 * @param [in] ptr pointer to the array, must be sorted under less<T>.
 * @param [in] size the size of the array.
 * @param [in] values the values to insert.
 * @param [in] nVals the number of values to insert.
 * @param [in/out] callBacks class which must define at least a method T * incrementSize(std::ptrdiff_t),
 *                 set(std::ptrdiff_t, std::ptrdiff_t) and insert(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t,
 * std::ptrdiff_t).
 *                 incrementSize is called with the number of values to insert and returns a new pointer to the array.
 *                 After an insert has occurred insert is called with the number of values left to insert, the
 *                 index of the value inserted, the index at which it was inserted and the index at
 *                 which the the previous value was inserted or size if it is the first value inserted.
 *                 set is called when the array is empty and it takes a position in the array and the position of the
 *                 value that will occupy that position.
 * @return The number of values inserted.
 */
DISABLE_HD_WARNING
template< typename T, typename CALLBACKS >
LVARRAY_HOST_DEVICE inline
std::ptrdiff_t insert( T * const LVARRAY_RESTRICT ptr,
                       std::ptrdiff_t const size,
                       T const * const LVARRAY_RESTRICT values,
                       std::ptrdiff_t const nVals,
                       CALLBACKS && callBacks )
{
  LVARRAY_ASSERT( ptr != nullptr || size == 0 );
  LVARRAY_ASSERT( arrayManipulation::isPositive( size ) );
  LVARRAY_ASSERT( values != nullptr || nVals == 0 );
  LVARRAY_ASSERT( nVals >= 0 );

  constexpr std::ptrdiff_t LOCAL_SIZE = 16;
  T localBuffer[LOCAL_SIZE];
  T * const buffer = createTemporaryBuffer( values, nVals, localBuffer );
  makeSorted( buffer, buffer + nVals );

  std::ptrdiff_t const nInserted = insertSorted( ptr, size, buffer, nVals, std::move( callBacks ) );

  freeTemporaryBuffer( buffer, nVals, localBuffer );
  return nInserted;
}

/**
 * @tparam T the type of values in the array.
 * @brief Insert the given values into the array if they don't already exist.
 * @param [in] ptr pointer to the array, must be sorted under less<T>.
 * @param [in] size the size of the array.
 * @param [in] values the values to insert.
 * @param [in] nVals the number of values to insert.
 * @return The number of values inserted.
 */
DISABLE_HD_WARNING
template< typename T, typename CALLBACKS >
LVARRAY_HOST_DEVICE inline
std::ptrdiff_t insert( T * const LVARRAY_RESTRICT ptr,
                       std::ptrdiff_t const size,
                       T const * const LVARRAY_RESTRICT values,
                       std::ptrdiff_t const nVals )
{ return insert( ptr, size, values, nVals, CallBacks< T > {} ); }

} // namespace sortedArrayManipulation
} // namespace LvArray

#endif // SORTEDARRAYMANIPULATION_HPP_
