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
 * @file ArrayManipulator.hpp
 * This file contains common sorted array manipulation routines.
 * Aside from the functions that take a callback every function assumes that
 * the array has a capacity large enough for the given operation.
 */

#ifndef SORTEDARRAYMANIPULATION_HPP_
#define SORTEDARRAYMANIPULATION_HPP_

#include "arrayManipulation.hpp"
#include "sortedArrayManipulationHelpers.hpp"
#include "Logger.hpp"
#include <cstdlib>      // for malloc / free.

namespace sortedArrayManipulation
{

/**
 * @class less
 * @brief This class operates as functor similar to std::less.
 */
template <class T>
struct less
{
  DISABLE_HD_WARNING
  CONSTEXPRFUNC LVARRAY_HOST_DEVICE inline bool operator() (T const & lhs, T const & rhs) const restrict_this
  { return lhs < rhs; }
};

/**
 * @class less
 * @brief This class operates as functor similar to std::greater.
 */
template <class T>
struct greater
{
  DISABLE_HD_WARNING
  CONSTEXPRFUNC LVARRAY_HOST_DEVICE inline bool operator() (T const & lhs, T const & rhs) const restrict_this
  { return lhs > rhs; }
};

/**
 * @brief Sort the given values in place using the given comparator.
 * @param [in/out] first a RandomAccessIterator to the beginning of the values to sort.
 * @param [in/out] last a RandomAccessIterator to the end of the values to sort.
 * @param [in/out] comp a function that does the comparison between two objects.
 */
DISABLE_HD_WARNING
template<typename RandomAccessIterator, typename Compare>
LVARRAY_HOST_DEVICE inline void makeSorted(RandomAccessIterator first, RandomAccessIterator last, Compare comp)
{
  if (last - first > internal::S_threshold)
  {
    internal::introsortLoop(first, last, comp);
  }

  internal::insertionSort(first, last - first, comp);
}

/**
 * @brief Sort the given values in place from least to greatest.
 * @param [in/out] first a RandomAccessIterator to the beginning of the values to sort.
 * @param [in/out] last a RandomAccessIterator to the end of the values to sort.
 */
DISABLE_HD_WARNING
template<typename RandomAccessIterator>
LVARRAY_HOST_DEVICE inline void makeSorted(RandomAccessIterator first, RandomAccessIterator last)
{ return makeSorted(first, last, less<typename std::remove_reference<decltype(*first)>::type>()); }

DISABLE_HD_WARNING
template<typename RandomAccessIteratorA, class RandomAccessIteratorB, typename Compare>
LVARRAY_HOST_DEVICE inline void dualSort(RandomAccessIteratorA first, RandomAccessIteratorA last,
                                         RandomAccessIteratorB dataFirst, Compare comp)
{
  std::ptrdiff_t const size = last - first;
  internal::DualIterator<RandomAccessIteratorA, RandomAccessIteratorB> dualIter(first, dataFirst);
  return makeSorted(dualIter, dualIter + size, dualIter.createComparator(comp));
}

DISABLE_HD_WARNING
template<typename RandomAccessIteratorA, class RandomAccessIteratorB>
LVARRAY_HOST_DEVICE inline void dualSort(RandomAccessIteratorA first, RandomAccessIteratorA last,
                                         RandomAccessIteratorB dataFirst)
{ return dualSort(first, last, dataFirst, less<typename std::remove_reference<decltype(*first)>::type>()); }

DISABLE_HD_WARNING
template <class T, int N>
LVARRAY_HOST_DEVICE inline T * createTemporaryBuffer(T const * const values, std::ptrdiff_t nVals, T (&localBuffer)[N])
{
  T * buffer = localBuffer;
  if (nVals <= N)
  {
    for (std::ptrdiff_t i = 0; i < nVals; ++i)
    {
      localBuffer[i] = values[i];
    }
  }
  else
  {
    buffer = static_cast<T*>(std::malloc(sizeof(T) * nVals));

    for (std::ptrdiff_t i = 0; i < nVals; ++i)
    {
      new (buffer + i) T(values[i]);
    }
  }

  return buffer;
}

DISABLE_HD_WARNING
template <class T, int N>
LVARRAY_HOST_DEVICE inline void freeTemporaryBuffer(T * const buffer, std::ptrdiff_t nVals, T (&localBuffer)[N])
{
  if (buffer == localBuffer)
  {
    return;
  }

  for (std::ptrdiff_t i = 0; i < nVals; ++i)
  {
    buffer[i].~T();
  }
  std::free(buffer);
}

/**
 * @brief Return true if the given array is sorted from least to greatest.
 * @param [in] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @note Should be equivalent to std::is_sorted(ptr, ptr + size).
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE, class Compare=less<T>>
LVARRAY_HOST_DEVICE inline
INDEX_TYPE isSorted( T const * const ptr, INDEX_TYPE const size, Compare comp=Compare() )
{
  GEOS_ASSERT( ptr != nullptr || size == 0 );
  GEOS_ASSERT( arrayManipulation::isPositive( size ));

  for( INDEX_TYPE i = 0 ; i < size - 1 ; ++i )
  {
    if( comp(ptr[i + 1], ptr[i]) ) 
    {
      return false;
    }
  }

  return true;
}

/**
 * @brief Return the index of the first value in the array greater than or equal to value.
 * @param [in] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] value the value to find.
 * @note Should be equivalent to std::lower_bound(ptr, ptr + size, value).
 * @pre isSorted(ptr, size) must be true.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE, class Compare=less<T>>
LVARRAY_HOST_DEVICE inline
INDEX_TYPE find( T const * const ptr, INDEX_TYPE const size, T const & value, Compare comp=Compare() )
{
  GEOS_ASSERT( ptr != nullptr || size == 0 );
  GEOS_ASSERT( arrayManipulation::isPositive( size ));
  GEOS_ASSERT(isSorted(ptr, size, comp));

  INDEX_TYPE lower = 0;
  INDEX_TYPE upper = size;
  while( lower != upper )
  {
    INDEX_TYPE const guess = (lower + upper) / 2;
    if (comp(ptr[guess], value))
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
 * @brief Return true if the array contains value.
 * @param [in] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] value the value to find.
 * @note Should be equivalent to std::binary_search(ptr, ptr + size, value).
 * @pre isSorted(ptr, size) must be true.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE, class Compare=less<T>>
LVARRAY_HOST_DEVICE inline
bool contains( T const * const ptr, INDEX_TYPE const size, T const & value, Compare comp=Compare() )
{
  INDEX_TYPE const pos = find( ptr, size, value, comp );
  return (pos != size) && (ptr[pos] == value);
}

/**
 * @brief Remove the given value from the array if it exists.
 * @param [in] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] value the value to remove.
 * @param [in/out] callBacks class which must define at least a method remove(INDEX_TYPE).
 * If the value is found it is removed then this method is called with the index the value
 * was found at.
 *
 * @pre isSorted(ptr, size) must be true.
 * @return True iff the value was removed.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE, class CALLBACKS>
LVARRAY_HOST_DEVICE inline
bool remove( T * const ptr, INDEX_TYPE const size, T const & value, CALLBACKS && callBacks )
{
  GEOS_ASSERT( ptr != nullptr || size == 0 );
  GEOS_ASSERT( arrayManipulation::isPositive( size ));

  // Find the position of value.
  INDEX_TYPE const index = find( ptr, size, value );

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
 * @brief Remove the given values from the array if they exist.
 * @param [in] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] values the values to remove.
 * @param [in] nVals the number of values to remove.
 * @param [in/out] callBacks class which must define at least a method remove(INDEX_TYPE, INDEX_TYPE, INDEX_TYPE).
 * If a value is found it is removed then this method is called with the number of values removed so far,
 * the position of the last value removed and the position of the next value to remove or size if there are no more
 * values to remove.
 *
 * @pre isSorted(ptr, size) must be true.
 * @pre isSorted(values, nVals) must be true.
 * @return The number of values removed.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE, class CALLBACKS>
LVARRAY_HOST_DEVICE inline
INDEX_TYPE removeSorted( T * const ptr, INDEX_TYPE const size, T const * const values,
                         INDEX_TYPE const nVals, CALLBACKS && callBacks )
{
  GEOS_ASSERT( ptr != nullptr || size == 0 );
  GEOS_ASSERT( arrayManipulation::isPositive( size ));
  GEOS_ASSERT( values != nullptr || nVals == 0 );
  GEOS_ASSERT( arrayManipulation::isPositive( nVals ));
  GEOS_ASSERT( isSorted( values, nVals ));

  // If there are no values to remove we can return.
  if( nVals == 0 )
    return 0;

  // Find the position of the first value to remove and the position it's at in the array.
  INDEX_TYPE firstValuePos = nVals;
  INDEX_TYPE curPos = size;
  for( INDEX_TYPE i = 0 ; i < nVals ; ++i )
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
  INDEX_TYPE nRemoved = 0;
  for( INDEX_TYPE curValuePos = firstValuePos ; curValuePos < nVals ; )
  {
    // Find the next value to remove
    INDEX_TYPE nextValuePos = nVals;
    INDEX_TYPE nextPos = size;
    for( INDEX_TYPE j = curValuePos + 1 ; j < nVals ; ++j )
    {
      // Skip over duplicate values
      if( values[j] == values[j - 1] )
        continue;

      // Find the position
      INDEX_TYPE const pos = find( ptr + curPos, size - curPos, values[j] ) + curPos;

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
  for( INDEX_TYPE i = size - nRemoved ; i < size ; ++i )
  {
    ptr[i].~T();
  }

  return nRemoved;
}

/**
 * @brief Remove the given values from the array if they exist.
 * @param [in] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] values the values to remove.
 * @param [in] nVals the number of values to remove.
 * @param [in/out] callBacks class which must define at least a method remove(INDEX_TYPE, INDEX_TYPE, INDEX_TYPE).
 * If a value is found it is removed then this method is called with the number of values removed so far,
 * the position of the last value removed and the position of the next value to remove or size if there are no more
 * values to remove.
 *
 * @pre isSorted(ptr, size) must be true.
 * @return The number of values removed.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE, class CALLBACKS>
LVARRAY_HOST_DEVICE inline
INDEX_TYPE remove( T * const ptr, INDEX_TYPE const size, T const * const values,
                   INDEX_TYPE const nVals, CALLBACKS && callBacks )
{
  GEOS_ASSERT( ptr != nullptr || size == 0 );
  GEOS_ASSERT( arrayManipulation::isPositive( size ));
  GEOS_ASSERT( values != nullptr || nVals == 0 );
  GEOS_ASSERT( arrayManipulation::isPositive( nVals ));

  constexpr INDEX_TYPE LOCAL_SIZE = 16;
  T localBuffer[LOCAL_SIZE];
  T * const buffer = createTemporaryBuffer(values, nVals, localBuffer);
  makeSorted(buffer, buffer + nVals);

  INDEX_TYPE const nInserted = removeSorted(ptr, size, buffer, nVals, std::move(callBacks));
    
  freeTemporaryBuffer(buffer, nVals, localBuffer);
  return nInserted;
}

/**
 * @brief Insert the given value into the array if it doesn't already exist.
 * @param [in] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] value the value to insert.
 * @param [in/out] callBacks class which must define at least a method T * incrementSize(INDEX_TYPE)
 * and insert(INDEX_TYPE). incrementSize is called with the number of values to insert and returns a new
 * pointer to the array. If an insert has occurred insert is called with the position in the array
 * at which the insert took place.
 *
 * @pre isSorted(ptr, size) must be true.
 * @return True iff the value was inserted.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE, class CALLBACKS>
LVARRAY_HOST_DEVICE inline
bool insert( T const * const ptr, INDEX_TYPE const size, T const & value, CALLBACKS && callBacks )
{
  GEOS_ASSERT( ptr != nullptr || size == 0 );
  GEOS_ASSERT( arrayManipulation::isPositive( size ));

  // Find the position of the value.
  INDEX_TYPE const index = find( ptr, size, value );

  // If it's in the array we call the incrementSize callback and return false.
  if( index != size && ptr[index] == value )
  {
    callBacks.incrementSize( 0 );
    return false;
  }

  // If it's in the array we call the incrementSize callback, get the new pointer,
  // insert and call the insert callback.
  T * const newPtr = callBacks.incrementSize( 1 );
  arrayManipulation::insert( newPtr, size, index, value );
  callBacks.insert( index );
  return true;
}

/**
 * @brief Insert the given values into the array if they don't already exist.
 * @param [in] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] values the values to insert.
 * @param [in] nVals the number of values to insert.
 * @param [in/out] callBacks class which must define at least a method T * incrementSize(INDEX_TYPE),
 * set(INDEX_TYPE, INDEX_TYPE) and insert(INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE).
 * incrementSize is called with the number of values to insert and returns a new pointer to the array.
 * After an insert has occurred insert is called with the number of values left to insert, the
 * index of the value inserted, the index at which it was inserted and the index at
 * which the the previous value was inserted or size if it is the first value inserted.
 * set is called when the array is empty and it takes a position in the array and the position of the
 * value that will occupy that position.
 *
 * @pre isSorted(ptr, size) must be true.
 * @pre isSorted(values, nVals) must be true.
 * @return The number of values inserted.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE, class CALLBACKS>
LVARRAY_HOST_DEVICE inline
INDEX_TYPE insertSorted( T const * const ptr, INDEX_TYPE const size, T const * const values,
                         INDEX_TYPE const nVals, CALLBACKS && callBacks )
{
  GEOS_ASSERT( ptr != nullptr || size == 0 );
  GEOS_ASSERT( arrayManipulation::isPositive( size ));
  GEOS_ASSERT( values != nullptr || nVals == 0 );
  GEOS_ASSERT( nVals >= 0 );
  GEOS_ASSERT( isSorted( values, nVals ));

  // If there are no values to insert.
  if( nVals == 0 )
  {
    callBacks.incrementSize( 0 );
    return 0;
  }

  INDEX_TYPE nToInsert = 0; // The number of values that will actually be inserted.

  // Special case for inserting into an empty array.
  if( size == 0 )
  {
    // Count up the number of unique values.
    nToInsert = 1;
    for( INDEX_TYPE i = 1 ; i < nVals ; ++i )
    {
      if( values[i] != values[i - 1] )
        ++nToInsert;
    }

    // Call callBack with the number to insert and get a new pointer back.
    T * const newPtr = callBacks.incrementSize( nToInsert );

    // Insert the first value.
    new (&newPtr[0]) T( values[0] );
    callBacks.set( 0, 0 );

    // Insert the remaining values, checking for duplicates.
    INDEX_TYPE curInsertPos = 1;
    for( INDEX_TYPE i = 1 ; i < nVals ; ++i )
    {
      if( values[i] != values[i - 1] )
      {
        new (&newPtr[curInsertPos]) T( values[i] );
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
  INDEX_TYPE valuePositions[MAX_PRE_CALCULATED];  // Positions of the first 32 values to insert.
  INDEX_TYPE insertPositions[MAX_PRE_CALCULATED]; // Position at which to insert the first 32 values.

  // Loop over the values in reverse (from largest to smallest).
  INDEX_TYPE curPos = size;
  for( INDEX_TYPE i = nVals - 1 ; i >= 0 ; --i )
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
  T * const newPtr = callBacks.incrementSize( nToInsert );

  // If there are no values to insert we can return.
  if( nToInsert == 0 )
    return 0;

  //
  INDEX_TYPE const nPreCalculated = (nToInsert < MAX_PRE_CALCULATED) ?
                                    nToInsert : MAX_PRE_CALCULATED;

  // Insert pre-calculated values.
  INDEX_TYPE prevInsertPos = size;
  for( INDEX_TYPE i = 0 ; i < nPreCalculated ; ++i )
  {
    // Shift the values up...
    arrayManipulation::shiftUp( newPtr, prevInsertPos, insertPositions[i], INDEX_TYPE( nToInsert - i ));

    // and insert.
    INDEX_TYPE const curValuePos = valuePositions[i];
    new (&newPtr[insertPositions[i] + nToInsert - i - 1]) T( values[curValuePos] );
    callBacks.insert( nToInsert - i, curValuePos, insertPositions[i], prevInsertPos );

    prevInsertPos = insertPositions[i];
  }

  // If all the values to insert were pre-calculated we can return.
  if( nToInsert <= MAX_PRE_CALCULATED )
    return nToInsert;

  // Insert the rest of the values.
  INDEX_TYPE const prevValuePos = valuePositions[MAX_PRE_CALCULATED - 1];
  INDEX_TYPE nInserted = MAX_PRE_CALCULATED;
  for( INDEX_TYPE i = prevValuePos - 1 ; i >= 0 ; --i )
  {
    // Skip duplicates
    if( values[i] == values[i + 1] )
      continue;

    INDEX_TYPE const pos = find( newPtr, prevInsertPos, values[i] );

    // If values[i] is already in the array skip it.
    if( pos != prevInsertPos && newPtr[pos] == values[i] )
      continue;

    // Else shift the values up and insert.
    arrayManipulation::shiftUp( newPtr, prevInsertPos, pos, INDEX_TYPE( nToInsert - nInserted ));
    new (&newPtr[pos + nToInsert - nInserted - 1]) T( values[i] );
    callBacks.insert( nToInsert - nInserted, i, pos, prevInsertPos );

    nInserted += 1;
    prevInsertPos = pos;

    // If all the values have been inserted then exit the loop.
    if( nInserted == nToInsert )
      break;
  }

  GEOS_ASSERT_MSG( nInserted == nToInsert, nInserted << " " << nToInsert );

  return nToInsert;
}

/**
 * @brief Insert the given values into the array if they don't already exist.
 * @param [in] ptr pointer to the array.
 * @param [in] size the size of the array.
 * @param [in] values the values to insert.
 * @param [in] nVals the number of values to insert.
 * @param [in/out] callBacks class which must define at least a method T * incrementSize(INDEX_TYPE),
 * set(INDEX_TYPE, INDEX_TYPE) and insert(INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE).
 * incrementSize is called with the number of values to insert and returns a new pointer to the array.
 * After an insert has occurred insert is called with the number of values left to insert, the
 * index of the value inserted, the index at which it was inserted and the index at
 * which the the previous value was inserted or size if it is the first value inserted.
 * set is called when the array is empty and it takes a position in the array and the position of the
 * value that will occupy that position.
 *
 * @pre isSorted(ptr, size) must be true.
 * @return The number of values inserted.
 */
DISABLE_HD_WARNING
template <class T, class INDEX_TYPE, class CALLBACKS>
LVARRAY_HOST_DEVICE inline
INDEX_TYPE insert( T const * const ptr, INDEX_TYPE const size, T const * const values,
                   INDEX_TYPE const nVals, CALLBACKS && callBacks )
{
  GEOS_ASSERT( ptr != nullptr || size == 0 );
  GEOS_ASSERT( arrayManipulation::isPositive( size ));
  GEOS_ASSERT( values != nullptr || nVals == 0 );
  GEOS_ASSERT( nVals >= 0 );

  constexpr INDEX_TYPE LOCAL_SIZE = 16;
  T localBuffer[LOCAL_SIZE];
  T * const buffer = createTemporaryBuffer(values, nVals, localBuffer);
  makeSorted(buffer, buffer + nVals);

  INDEX_TYPE const nInserted = insertSorted(ptr, size, buffer, nVals, std::move(callBacks));

  freeTemporaryBuffer(buffer, nVals, localBuffer);
  return nInserted;
}

} // namespace sortedArrayManipulation

#endif // SORTEDARRAYMANIPULATION_HPP_
