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
 * @file sortedArrayManipulationHelpers.hpp
 * These routines were ripped from the gcc standard library (https://gcc.gnu.org/svn/gcc/trunk/libstdc++-v3/include/bits/)
 * almost entirely from stl_heap.h and stl_algo.h.
 */

#ifndef SORTEDARRAYMANIPULATIONHELPERS_HPP_
#define SORTEDARRAYMANIPULATIONHELPERS_HPP_

#include "CXX_UtilsConfig.hpp"
#include "Logger.hpp"

#include <utility>

namespace sortedArrayManipulation
{

namespace internal
{

constexpr int S_threshold = 64;

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
  RandomAccessIterator next = last - 1;
  while (comp(val, *next))
  {
    *last = std::move(*next);
    last = next;
    --next;
  }
  *last = std::move(val);
}

DISABLE_HD_WARNING
template <class RandomAccessIterator, class Compare>
LVARRAY_HOST_DEVICE inline void insertionSort(RandomAccessIterator first, std::ptrdiff_t const n, Compare comp)
{
  for (std::ptrdiff_t i = 1; i < n; ++i)
  {
    if (comp(*(first + i), *first))
    {
      auto val = std::move(*(first + i));
      moveBackward(first, first + i, first + i + 1);
      *first = std::move(val);
    }
    else
    {
      unguardedLinearInsert(first + i, comp);
    }
  }
}

DISABLE_HD_WARNING
template<typename RandomAccessIterator, typename Compare>
LVARRAY_HOST_DEVICE inline void introsortLoop(RandomAccessIterator first, RandomAccessIterator last, Compare comp)
{
  constexpr int MAX_RANGES = 31;
  RandomAccessIterator ranges[MAX_RANGES + 1];

  int nRanges = 1;
  ranges[0] = first;
  ranges[1] = last;
  while (nRanges != 0)
  {
    RandomAccessIterator const m_first = ranges[nRanges - 1];

    if (ranges[nRanges] - m_first > S_threshold)
    {
      GEOS_ASSERT(nRanges < MAX_RANGES);
      ranges[nRanges + 1] = ranges[nRanges];
      ranges[nRanges] = unguardedPartitionPivot(m_first, ranges[nRanges], comp);
      ++nRanges;
    }
    else
    {
      --nRanges;
    }
  }
}

} // namespace internal

} // namespace sortedArrayManipulation

#endif // SORTEDARRAYMANIPULATIONHELPERS_HPP_
