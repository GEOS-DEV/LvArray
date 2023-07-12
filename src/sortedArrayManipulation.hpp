/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file sortedArrayManipulation.hpp
 * @brief This file contains common sorted array manipulation routines.
 *   Aside from the functions that take a callback every function assumes that
 *   the array has a capacity large enough for the given operation.
 */

#pragma once

// Source includes
#include "Macros.hpp"
#include "arrayManipulation.hpp"
#include "sortedArrayManipulationHelpers.hpp"

// System includes
#include <cstdlib>      // for std::malloc and std::free.
#include <algorithm>    // for std::sort

namespace LvArray
{

/**
 * @brief Contains functions for operating on a contiguous sorted unique array of values.
 * @details Most functions accept a pointer and a size as the first two arguments. Values
 *   in this range are expected to be in a valid state and sorted unique.
 *   Values past the end of the array are expected to be uninitialized.
 */
namespace sortedArrayManipulation
{

/**
 * @enum Description
 * @brief Describes an as some combination of sorted/unsorted and unique/with duplicates.
 */
enum Description
{
  SORTED_UNIQUE,
  UNSORTED_NO_DUPLICATES,
  SORTED_WITH_DUPLICATES,
  UNSORTED_WITH_DUPLICATES
};

/**
 * @return True if @p desc describes an array that is sorted.
 * @param desc The Description to query.
 */
LVARRAY_HOST_DEVICE constexpr inline
bool isSorted( Description const desc )
{ return desc == SORTED_UNIQUE || desc == SORTED_WITH_DUPLICATES; }

/**
 * @return True if @p desc describes an array that contains no duplicates.
 * @param desc the Description to query.
 */
LVARRAY_HOST_DEVICE constexpr inline
bool isUnique( Description const desc )
{ return desc == SORTED_UNIQUE || desc == UNSORTED_NO_DUPLICATES; }

/**
 * @tparam T The type of the values contained in the array.
 * @class CallBacks
 * @brief This class provides a no-op callbacks interface for the ArrayManipulation sorted routines.
 */
template< typename T >
class CallBacks
{
public:

  /**
   * @brief Callback signaling that the size of the array has increased.
   * @param curPtr the current pointer to the array.
   * @param nToAdd the increase in the size of the array.
   * @return a pointer to the array.
   */
  LVARRAY_HOST_DEVICE inline
  T * incrementSize( T * const curPtr,
                     std::ptrdiff_t const nToAdd )
  {
    LVARRAY_UNUSED_VARIABLE( nToAdd );
    return curPtr;
  }

  /**
   * @brief Callback signaling that a value was inserted at the given position.
   * @param pos the position the value was inserted at.
   */
  LVARRAY_HOST_DEVICE inline
  void insert( std::ptrdiff_t const pos )
  { LVARRAY_UNUSED_VARIABLE( pos ); }

  /**
   * @brief Callback signaling that the entry of the array at the first position was set to the value
   *        at the second position.
   * @param pos the position in the array that was set.
   * @param valuePos the position of the value that the entry in the array was set to.
   */
  LVARRAY_HOST_DEVICE inline
  void set( std::ptrdiff_t const pos,
            std::ptrdiff_t const valuePos )
  {
    LVARRAY_UNUSED_VARIABLE( pos );
    LVARRAY_UNUSED_VARIABLE( valuePos );
  }

  /**
   * @brief Callback signaling that the that the given value was inserted at the given position.
   *        Further information is provided in order to make the insertion efficient.
   * @param nLeftToInsert the number of insertions that occur after this one.
   * @param valuePos the position of the value that was inserted.
   * @param pos the position in the array the value was inserted at.
   * @param prevPos the position the previous value was inserted at or the size of the array
   *             if it is the first insertion.
   */
  LVARRAY_HOST_DEVICE inline
  void insert( std::ptrdiff_t const nLeftToInsert,
               std::ptrdiff_t const valuePos,
               std::ptrdiff_t const pos,
               std::ptrdiff_t const prevPos )
  {
    LVARRAY_UNUSED_VARIABLE( nLeftToInsert );
    LVARRAY_UNUSED_VARIABLE( valuePos );
    LVARRAY_UNUSED_VARIABLE( pos );
    LVARRAY_UNUSED_VARIABLE( prevPos );
  }

  /**
   * @brief Callback signaling that an entry was removed from the array at given position.
   * @param pos The position of the entry that was removed.
   */
  LVARRAY_HOST_DEVICE inline
  void remove( std::ptrdiff_t const pos )
  { LVARRAY_UNUSED_VARIABLE( pos ); }

  /**
   * @brief Callback signaling that the given entry was removed from the given position. Further information
   *        is provided in order to make the removal efficient.
   * @param nRemoved the number of entries removed, starts at 1.
   * @param curPos the position in the array the entry was removed at.
   * @param nextPos the position the next entry will be removed at or the original size of the array
   *             if this was the last entry removed.
   */
  LVARRAY_HOST_DEVICE inline
  void remove( std::ptrdiff_t const nRemoved,
               std::ptrdiff_t const curPos,
               std::ptrdiff_t const nextPos )
  {
    LVARRAY_UNUSED_VARIABLE( nRemoved );
    LVARRAY_UNUSED_VARIABLE( curPos );
    LVARRAY_UNUSED_VARIABLE( nextPos );
  }
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
   * @return Return @p lhs < @p rhs.
   * @param lhs The left side of the comparison.
   * @param rhs The right side of the comparison.
   */
  DISABLE_HD_WARNING
  constexpr LVARRAY_HOST_DEVICE inline
  bool operator() ( T const & lhs, T const & rhs ) const
  { return lhs < rhs; }
};

/**
 * @tparam T the type of the values to compare.
 * @class greater
 * @brief This class operates as functor similar to std::greater.
 */
template< typename T >
struct greater
{
  /**
   * @return Return @p lhs > @p rhs.
   * @param lhs The left side of the comparison.
   * @param rhs The right side of the comparison.
   */
  DISABLE_HD_WARNING
  constexpr LVARRAY_HOST_DEVICE inline
  bool operator() ( T const & lhs, T const & rhs ) const
  { return lhs > rhs; }
};

/**
 * @tparam RandomAccessIterator an iterator type that provides random access.
 * @tparam Compare the type of the comparison function.
 * @brief Sort the given values in place using the given comparator.
 * @param first An iterator to the beginning of the values to sort.
 * @param last An iterator to the end of the values to sort.
 * @param comp A function that does the comparison between two objects.
 * @note should be equivalent to std::sort(first, last, comp).
 */
DISABLE_HD_WARNING
template< typename RandomAccessIterator,
          typename Compare=less< typename std::iterator_traits< RandomAccessIterator >::value_type > >
LVARRAY_HOST_DEVICE inline void makeSorted( RandomAccessIterator const first,
                                            RandomAccessIterator const last,
                                            Compare && comp=Compare() )
{
#if defined(LVARRAY_DEVICE_COMPILE)
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
 * @tparam RandomAccessIteratorA an iterator type that provides random access.
 * @tparam RandomAccessIteratorB an iterator type that provides random access.
 * @tparam Compare the type of the comparison function.
 * @brief Sort the given values in place using the given comparator and perform the same operations
 *   on the data array thus preserving the mapping between values[i] and data[i].
 * @param valueFirst An iterator to the beginning of the values to sort.
 * @param valueLast An iterator to the end of the values to sort.
 * @param dataFirst An iterator to the beginning of the data.
 * @param comp a function that does the comparison between two objects.
 */
DISABLE_HD_WARNING
template< typename RandomAccessIteratorA,
          typename RandomAccessIteratorB,
          typename Compare=less< typename std::iterator_traits< RandomAccessIteratorA >::value_type > >
LVARRAY_HOST_DEVICE inline void dualSort( RandomAccessIteratorA valueFirst, RandomAccessIteratorA valueLast,
                                          RandomAccessIteratorB dataFirst, Compare && comp=Compare() )
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
 * @tparam ITER The type of the iterator to the values to check.
 * @tparam Compare The type of the comparison function, defaults to less.
 * @return Return true if the given array is sorted using comp.
 * @param first Iterator to the beginning of the values to check.
 * @param last Iterator to the end of the values to check.
 * @param comp The comparison method to use.
 * @note Should be equivalent to std::is_sorted(first, last, comp).
 */
DISABLE_HD_WARNING
template< typename ITER, typename Compare=less< typename std::iterator_traits< ITER >::value_type > >
LVARRAY_HOST_DEVICE inline
bool isSorted( ITER first, ITER const last, Compare && comp=Compare() )
{
  if( first == last )
  {
    return true;
  }

  ITER next = first;
  ++next;
  while( next != last )
  {
    if( comp( *next, *first ) )
    {
      return false;
    }

    first = next;
    ++next;
  }

  return true;
}

/**
 * @tparam ITER An iterator type.
 * @tparam Compare The type of the comparison function, defaults to less.
 * @brief Remove duplicates from the array, duplicates aren't destroyed but they're moved out of.
 * @param first Iterator to the beginning of the values.
 * @param last Iterator to the end of the values.
 * @param comp The comparison method to use.
 * @return The number of unique elements, such that [first, first + returnValue) contains the unique elements.
 * @note The range [ @p first, @p last ) must be sorted under @p comp.
 * @note If you are using this and would like the duplicates to remain intact at the end of the array
 *       change the std::move to a portable implementation of std::swap.
 */
DISABLE_HD_WARNING
template< typename ITER, typename Compare=less< typename std::iterator_traits< ITER >::value_type > >
LVARRAY_HOST_DEVICE inline
std::ptrdiff_t removeDuplicates( ITER first, ITER const last, Compare && comp=Compare() )
{
  LVARRAY_ASSERT( isSorted( first, last, comp ) );

  if( first == last )
  {
    return 0;
  }

  std::ptrdiff_t numUnique = 1;

  /**
   * The following statement should be
   *   ITER next = first;
   *   ++next;
   * This works host only however with some host compilers (clang 10.0.1 and XL) it breaks on device in release.
   * It does some really strange things, for example `last - next == 0` and they can have identical
   * values but `next != last`. If you print out the array each iteration it works as expected. It's not a big deal
   * since it amounts to just a single extra iteration, but very strange.
   */
  ITER next = first;
  for(; next != last; ++next )
  {
    if( comp( *first, *next ) )
    {
      ++numUnique;
      ++first;
      if( first != next )
      {
        *first = std::move( *next );
      }
    }
  }

  return numUnique;
}

/**
 * @tparam ITER An iterator type.
 * @tparam Compare The type of the comparison function, defaults to less.
 * @brief Sort and remove duplicates from the array, duplicates aren't destroyed but they're moved out of.
 * @param first Iterator to the beginning of the values.
 * @param last Iterator to the end of the values.
 * @param comp The comparison method to use.
 * @return The number of unique elements, such that [first, first + returnValue) contains the unique elements.
 */
DISABLE_HD_WARNING
template< typename ITER, typename Compare=less< typename std::iterator_traits< ITER >::value_type > >
LVARRAY_HOST_DEVICE inline
std::ptrdiff_t makeSortedUnique( ITER const first, ITER const last, Compare && comp=Compare() )
{
  makeSorted( first, last, comp );
  return removeDuplicates( first, last, comp );
}

/**
 * @tparam ITER An iterator type.
 * @tparam Compare The type of the comparison function, defaults to less.
 * @return Return true iff [ @p first, @p last ) is sorted under @p comp and contains no values which compare
 * equal.
 * @param first Iterator to the beginning of the values.
 * @param last Iterator to the end of the values.
 * @param comp The comparison method to use.
 */
DISABLE_HD_WARNING
template< typename ITER, typename Compare=less< typename std::iterator_traits< ITER >::value_type > >
LVARRAY_HOST_DEVICE inline
bool isSortedUnique( ITER first, ITER const last, Compare && comp=Compare() )
{
  if( first == last )
  {
    return true;
  }

  ITER next = first;
  ++next;
  while( next != last )
  {
    if( comp( *next, *first ) || !comp( *first, *next ) )
    {
      return false;
    }

    first = next;
    ++next;
  }

  return true;
}

/**
 * @tparam T the type of values in the array.
 * @tparam Compare the type of the comparison function, defaults to less<T>.
 * @return Return the index of the first value in the array that compares not less
 *   than @p value or @p size if no such element can be found.
 * @param ptr Pointer to the array, must be sorted under comp.
 * @param size The size of the array.
 * @param value The value to find.
 * @param comp The comparison method to use.
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
  LVARRAY_ASSERT( isSorted( ptr, ptr + size, comp ) );

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
 * @return Return true if @p value is contained in the array.
 * @param ptr Pointer to the array, must be sorted under comp.
 * @param size The size of the array.
 * @param value The value to find.
 * @param comp The comparison method to use.
 * @note Should be equivalent to std::lower_bound(ptr, ptr + size, value, comp).
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
 * @param ptr pointer to the array, must be sorted under less<T>.
 * @param size the size of the array.
 * @param value the value to remove.
 * @param callBacks class which must define a method equivalent to CallBacks::remove(std::ptrdiff_t).
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
 * @tparam ITER An iterator type.
 * @tparam CALLBACKS the type of the callBacks class.
 * @brief Remove the given values from the array if they exist.
 * @param ptr pointer to the array, must be sorted under less<T>.
 * @param size the size of the array.
 * @param first An iterator to the first value to insert.
 * @param last An iterator to the end of the values to insert.
 * @param callBacks class which must define methods equivalent to
 *   CallBacks::remove(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t).
 * @note The values to insert [ @p first, @p last ) must be sorted and contain no duplicates.
 * @return The number of values removed.
 */
DISABLE_HD_WARNING
template< typename T, typename ITER, typename CALLBACKS=CallBacks< T > >
LVARRAY_HOST_DEVICE inline
std::ptrdiff_t remove( T * const LVARRAY_RESTRICT ptr,
                       std::ptrdiff_t const size,
                       ITER const first,
                       ITER const last,
                       CALLBACKS && callBacks=CALLBACKS() )
{
  LVARRAY_ASSERT( ptr != nullptr || size == 0 );
  LVARRAY_ASSERT( arrayManipulation::isPositive( size ) );
  LVARRAY_ASSERT( isSortedUnique( ptr, ptr + size ) );

  LVARRAY_ASSERT( isSortedUnique( first, last ) );

  // Find the position of the first value to remove and the position it's at in the array.
  ITER valueToRemove = first;
  std::ptrdiff_t removalPosition = 0;
  for(; valueToRemove != last; ++valueToRemove )
  {
    removalPosition += find( ptr + removalPosition, size - removalPosition, *valueToRemove );

    // If the current value is larger than the largest value in the array we can return.
    if( removalPosition == size )
    {
      return 0;
    }

    if( ptr[ removalPosition ] == *valueToRemove )
    {
      break;
    }
  }

  // If we didn't find a value to remove we can return.
  if( valueToRemove == last )
  {
    return 0;
  }

  // Loop over the values
  std::ptrdiff_t nRemoved = 0;
  for(; valueToRemove != last; )
  {
    // Find the next value to remove
    ITER nextValueToRemove = valueToRemove;
    ++nextValueToRemove;
    std::ptrdiff_t nextRemovalPosition = removalPosition;
    for(; nextValueToRemove != last; ++nextValueToRemove )
    {
      // Find the position
      nextRemovalPosition += find( ptr + nextRemovalPosition, size - nextRemovalPosition, *nextValueToRemove );

      // If it's not in the array then neither are any of the rest of the values.
      if( nextRemovalPosition == size )
      {
        nextValueToRemove = last;
        break;
      }

      // If it's in the array then we can exit this loop.
      if( ptr[ nextRemovalPosition ] == *nextValueToRemove )
      {
        break;
      }
    }

    if( nextValueToRemove == last )
    {
      nextRemovalPosition = size;
    }

    // Shift the values down and call the callback.
    nRemoved += 1;
    arrayManipulation::shiftDown( ptr, nextRemovalPosition, removalPosition + 1, nRemoved );
    callBacks.remove( nRemoved, removalPosition, nextRemovalPosition );

    valueToRemove = nextValueToRemove;
    removalPosition = nextRemovalPosition;

    // If we've reached the end of the array we can exit the loop.
    if( removalPosition == size )
    {
      break;
    }
  }

  // Destroy the values moved out of at the end of the array.
  for( std::ptrdiff_t i = size - nRemoved; i < size; ++i )
  {
    ptr[ i ].~T();
  }

  return nRemoved;
}

/**
 * @tparam T the type of values in the array.
 * @tparam CALLBACKS the type of the callBacks class.
 * @brief Insert the given value into the array if it doesn't already exist.
 * @param ptr pointer to the array, must be sorted under less<T>.
 * @param size the size of the array.
 * @param value the value to insert.
 * @param callBacks class which must define methods similar to
 *   CallBacks::incrementSize(std::ptrdiff_t) and CallBacks::insert(std::ptrdiff_t).
 * @return True iff the value was inserted.
 */
DISABLE_HD_WARNING
template< typename T, typename CALLBACKS=CallBacks< T > >
LVARRAY_HOST_DEVICE inline
bool insert( T * const LVARRAY_RESTRICT ptr,
             std::ptrdiff_t const size,
             T const & value,
             CALLBACKS && callBacks=CALLBACKS() )
{
  LVARRAY_ASSERT( ptr != nullptr || size == 0 );
  LVARRAY_ASSERT( arrayManipulation::isPositive( size ) );
  LVARRAY_ASSERT( isSortedUnique( ptr, ptr + size ) );

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
  arrayManipulation::emplace( newPtr, size, insertPos, value );
  callBacks.insert( insertPos );
  return true;
}

/**
 * @tparam T the type of values in the array.
 * @tparam ITER An iterator class.
 * @tparam CALLBACKS the type of the callBacks class.
 * @brief Insert the given values into the array if they don't already exist.
 * @param ptr pointer to the array, must be sorted under less<T>.
 * @param size the size of the array.
 * @param first An iterator to the first value to insert.
 * @param last An iterator to the end of the values to insert.
 * @param callBacks class which must define methods similar to
 *   CallBacks::incrementSize(std::ptrdiff_t), CallBacks::set(std::ptrdiff_t, std::ptrdiff_t)
 *   and CallBacks::insert(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, * std::ptrdiff_t).
 * @note The values to insert [ @p first, @p last ) must be sorted and contain no duplicates.
 * @return The number of values inserted.
 */
DISABLE_HD_WARNING
template< typename T, typename ITER, typename CALLBACKS=CallBacks< T > >
LVARRAY_HOST_DEVICE inline
std::ptrdiff_t insert( T * const LVARRAY_RESTRICT ptr,
                       std::ptrdiff_t const size,
                       ITER const first,
                       ITER const last,
                       CALLBACKS && callBacks=CALLBACKS() )
{
  LVARRAY_ASSERT( ptr != nullptr || size == 0 );
  LVARRAY_ASSERT( arrayManipulation::isPositive( size ) );
  LVARRAY_ASSERT( isSortedUnique( ptr, ptr + size ) );

  LVARRAY_ASSERT( isSortedUnique( first, last ) );

  // Special case for inserting into an empty array.
  if( size == 0 )
  {
    std::ptrdiff_t numInserted = arrayManipulation::iterDistance( first, last );
    T * const newPtr = callBacks.incrementSize( ptr, numInserted );

    numInserted = 0;
    for( ITER iter = first; iter != last; ++iter )
    {
      new ( newPtr + numInserted ) T( *iter );
      callBacks.set( numInserted, numInserted );
      ++numInserted;
    }

    return numInserted;
  }

  // Count up the number of values that will actually be inserted.
  std::ptrdiff_t nVals = 0;
  std::ptrdiff_t nToInsert = 0;
  std::ptrdiff_t upperBound = size;
  ITER iter = last;
  while( iter != first )
  {
    --iter;
    ++nVals;
    upperBound = find( ptr, upperBound, *iter );
    if( upperBound == size || ptr[upperBound] != *iter )
    {
      ++nToInsert;
    }
  }

  // Call callBack with the number to insert and get a new pointer back.
  T * const newPtr = callBacks.incrementSize( ptr, nToInsert );

  // If there are no values to insert we can return.
  if( nToInsert == 0 )
  {
    return 0;
  }

  // Insert the rest of the values.
  std::ptrdiff_t currentIndex = nVals - 1;
  std::ptrdiff_t nLeftToInsert = nToInsert;
  upperBound = size;
  iter = last;
  while( iter != first )
  {
    --iter;
    std::ptrdiff_t const pos = find( newPtr, upperBound, *iter );

    // If *iter is already in the array skip it.
    if( pos != upperBound && newPtr[pos] == *iter )
    {
      --currentIndex;
      continue;
    }

    // Else shift the values up and insert.
    arrayManipulation::shiftUp( newPtr, upperBound, pos, nLeftToInsert );
    new ( newPtr + pos + nLeftToInsert - 1 ) T( *iter );
    callBacks.insert( nLeftToInsert, currentIndex, pos, upperBound );

    --currentIndex;
    --nLeftToInsert;
    upperBound = pos;
  }

  return nToInsert;
}

} // namespace sortedArrayManipulation
} // namespace LvArray
