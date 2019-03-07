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
 * @file sorting.hpp
 * These routines were ripped from the gcc standard library (https://gcc.gnu.org/svn/gcc/trunk/libstdc++-v3/include/bits/)
 * almost entirely from stl_heap.h and stl_algo.h.
 */

#ifndef SORTING_HPP_
#define SORTING_HPP_

#include "CXX_UtilsConfig.hpp"
#include "Logger.hpp"

#include <utility>

namespace sorting
{

namespace internal
{

constexpr int S_threshold = 16;

DISABLE_HD_WARNING
template <class T>
LVARRAY_HOST_DEVICE inline void exchange(T & a, T & b)
{
  T temp = std::move(a);
  a = std::move(b);
  b = std::move(temp);
}

DISABLE_HD_WARNING
template <class BidirIt1, class BidirIt2>
LVARRAY_HOST_DEVICE inline void moveBackward(BidirIt1 first, BidirIt1 last, BidirIt2 d_last)
{
  while (first != last)
  {
    *(--d_last) = std::move(*(--last));
  }
}

DISABLE_HD_WARNING
template<typename RandomAccessIterator, typename Distance, typename Tp, typename Compare>
LVARRAY_HOST_DEVICE inline void pushHeap(RandomAccessIterator first, Distance holeIndex, Distance topIndex, Tp value, Compare & comp)
{
  Distance parent = (holeIndex - 1) / 2;
  while (holeIndex > topIndex && comp(*(first + parent), value))
  {
    *(first + holeIndex) = std::move(*(first + parent));
    holeIndex = parent;
    parent = (holeIndex - 1) / 2;
  }

  *(first + holeIndex) = std::move(value);
}

DISABLE_HD_WARNING
template<typename RandomAccessIterator, typename Distance, typename Tp, typename Compare>
LVARRAY_HOST_DEVICE inline void adjustHeap(RandomAccessIterator first, Distance holeIndex, Distance len, Tp value, Compare comp)
{
  const Distance topIndex = holeIndex;
  Distance secondChild = holeIndex;
  while (secondChild < (len - 1) / 2)
  {
    secondChild = 2 * (secondChild + 1);
    if (comp(*(first + secondChild), *(first + (secondChild - 1))))
    {
      secondChild--;
    }

    *(first + holeIndex) = std::move(*(first + secondChild));
    holeIndex = secondChild;
  }
  
  if ((len & 1) == 0 && secondChild == (len - 2) / 2)
  {
    secondChild = 2 * (secondChild + 1);
    *(first + holeIndex) = std::move(*(first + (secondChild - 1)));
    holeIndex = secondChild - 1;
  }
  
  pushHeap(first, holeIndex, topIndex, std::move(value), comp);
}

DISABLE_HD_WARNING
template<typename RandomAccessIterator, typename Compare>
LVARRAY_HOST_DEVICE inline void makeHeap(RandomAccessIterator first, RandomAccessIterator last, Compare& comp)
{
  typedef decltype(last - first) DistanceType;

  if (last - first < 2)
  {
    return;
  }

  const DistanceType len = last - first;
  DistanceType parent = (len - 2) / 2;
  while (true)
  {
    auto value = std::move(*(first + parent));
    adjustHeap(first, parent, len, std::move(value), comp);
    if (parent == 0)
    {
      return;
    }
    --parent;
  }
}

DISABLE_HD_WARNING
template<typename RandomAccessIterator, typename Compare>
LVARRAY_HOST_DEVICE inline void popHeap(RandomAccessIterator first, RandomAccessIterator last, RandomAccessIterator result, Compare & comp)
{
  typedef decltype(last - first) DistanceType;

  auto value = std::move(*result);
  *result = std::move(*first);
  adjustHeap(first, DistanceType(0), DistanceType(last - first), std::move(value), comp);
}

DISABLE_HD_WARNING
template<typename RandomAccessIterator, typename Compare>
LVARRAY_HOST_DEVICE inline void heapSelect(RandomAccessIterator first, RandomAccessIterator middle, RandomAccessIterator last, Compare comp)
{
  makeHeap(first, middle, comp);
  for (RandomAccessIterator i = middle; i < last; ++i)
  {
    if (comp(*i, *first))
    {
      popHeap(first, middle, i, comp);
    }
  }
}

DISABLE_HD_WARNING
template<typename RandomAccessIterator, typename Compare>
LVARRAY_HOST_DEVICE inline void sortHeap(RandomAccessIterator first, RandomAccessIterator last, Compare & comp)
{
  while (last - first > 1)
  {
    --last;
    popHeap(first, last, last, comp);
  }
}

DISABLE_HD_WARNING
template<typename RandomAccessIterator, typename Compare>
LVARRAY_HOST_DEVICE inline void partialSort(RandomAccessIterator first, RandomAccessIterator middle, RandomAccessIterator last, Compare comp)
{
  heapSelect(first, middle, last, comp);
  sortHeap(first, middle, comp);
}

DISABLE_HD_WARNING
template<typename Iterator, typename Compare>
LVARRAY_HOST_DEVICE inline void moveMedianToFirst(Iterator result, Iterator a, Iterator b, Iterator c, Compare comp)
{
  if (comp(*a, *b))
  {
    if (comp(*b, *c))
    {
      exchange(*result, *b);
    }
    else if (comp(*a, *c))
    {
      exchange(*result, *c);
    }
    else
    {
      exchange(*result, *a);
    }
  }
  else if (comp(*a, *c))
  {
    exchange(*result, *a);
  }
  else if (comp(*b, *c))
  {
    exchange(*result, *c);
  }
  else
  {
    exchange(*result, *b);
  }
}

DISABLE_HD_WARNING
template<typename RandomAccessIterator, typename Compare>
LVARRAY_HOST_DEVICE inline RandomAccessIterator unguardedPartition(RandomAccessIterator first, RandomAccessIterator last, RandomAccessIterator pivot, Compare comp)
{
  while (true)
  {
    while (comp(*first, *pivot))
    {
      ++first;
    }
  
    --last;
    while (comp(*pivot, *last))
    {
      --last;
    }
  
   if (!(first < last))
   {
      return first;
   }
  
    exchange(*first, *last);
    ++first;
  }
}

DISABLE_HD_WARNING
template<typename RandomAccessIterator, typename Compare>
LVARRAY_HOST_DEVICE inline RandomAccessIterator unguardedPartitionPivot(RandomAccessIterator first, RandomAccessIterator last, Compare comp)
{
  RandomAccessIterator mid = first + (last - first) / 2;
  moveMedianToFirst(first, first + 1, mid, last - 1, comp);
  return unguardedPartition(first + 1, last, first, comp);
}

DISABLE_HD_WARNING
template<typename RandomAccessIterator, typename Compare>
LVARRAY_HOST_DEVICE inline void unguardedLinearInsert(RandomAccessIterator last, Compare comp)
{
  auto val = std::move(*last);
  RandomAccessIterator next = last;
  --next;
  while (comp(val, *next))
  {
    *last = std::move(*next);
    last = next;
    --next;
  }
  *last = std::move(val);
}

DISABLE_HD_WARNING
template<typename RandomAccessIterator, typename Compare>
LVARRAY_HOST_DEVICE inline void insertionSort(RandomAccessIterator first, RandomAccessIterator last, Compare comp)
{
  if (first == last) return;

  for (RandomAccessIterator i = first + 1; i != last; ++i)
  {
    if (comp(*i, *first))
    {
      auto val = std::move(*i);
      moveBackward(first, i, i + 1);
      *first = std::move(val);
    }
    else
    {
      unguardedLinearInsert(i, comp);
    }
  }
}

DISABLE_HD_WARNING
template<typename RandomAccessIterator, typename Compare>
LVARRAY_HOST_DEVICE inline void unguardedInsertionSort(RandomAccessIterator first, RandomAccessIterator last, Compare comp)
{
  for (RandomAccessIterator i = first; i != last; ++i)
  {
    unguardedLinearInsert(i, comp);
  }
}

DISABLE_HD_WARNING
template<typename RandomAccessIterator, typename SIZE, typename Compare>
LVARRAY_HOST_DEVICE inline void introsortLoop(RandomAccessIterator first, RandomAccessIterator last, SIZE depth_limit, Compare comp)
{
  while (last - first > S_threshold)
  {
    if (depth_limit == 0)
    {
      partialSort(first, last, last, comp);
      return;
    }
    
    --depth_limit;
    RandomAccessIterator cut = unguardedPartitionPivot(first, last, comp);
    introsortLoop(cut, last, depth_limit, comp);
    last = cut;
  }
}

DISABLE_HD_WARNING
template<typename RandomAccessIterator, typename Compare>
LVARRAY_HOST_DEVICE inline void finalInsertionSort(RandomAccessIterator first, RandomAccessIterator last, Compare comp)
{
  if (last - first > S_threshold)
  {
    insertionSort(first, first + S_threshold, comp);
    unguardedInsertionSort(first + S_threshold, last, comp);
  }
  else
  {
    insertionSort(first, last, comp);
  }
}

DISABLE_HD_WARNING
template<typename SIZE>
LVARRAY_HOST_DEVICE inline SIZE lg(SIZE n)
{
  GEOS_ASSERT(n > 0);
  SIZE k = 0;
  for (; n != 0; n >>= 1)
  {
    ++k;
  }
 
 return k - 1;
}

} // namespace internal

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
  if (first != last)
  {
    internal::introsortLoop(first, last, internal::lg(last - first) * 2, comp);
    internal::finalInsertionSort(first, last, comp);
  }
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

} // namespace sorting

#endif // SORTING_HPP_
