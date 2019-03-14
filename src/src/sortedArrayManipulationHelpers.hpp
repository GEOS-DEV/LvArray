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

template <class A, class B>
struct DualIteratorAccessor
{
  struct Temporary
  {
    DISABLE_HD_WARNING
    LVARRAY_HOST_DEVICE inline Temporary(DualIteratorAccessor const & src) :
      m_a(std::move(src.m_a)),
      m_b(std::move(src.m_b)) 
    {}

    DISABLE_HD_WARNING
    inline ~Temporary() = default;

    A m_a;
    B m_b;
  };

  DualIteratorAccessor() = delete;

  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline  DualIteratorAccessor(A & a, B & val) :
    m_a(a),
    m_b(val)
  {}

  DualIteratorAccessor(DualIteratorAccessor const & src) = delete;

  DISABLE_HD_WARNING
  inline DualIteratorAccessor(DualIteratorAccessor && src) = default;

  DualIteratorAccessor & operator=(DualIteratorAccessor const & src) = delete;

  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIteratorAccessor & operator=(DualIteratorAccessor && src) restrict_this
  {
    m_a = std::move(src.m_a);
    m_b = std::move(src.m_b);
    return *this;
  }

  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE DualIteratorAccessor & operator=(Temporary && src) restrict_this
  {
    m_a = std::move(src.m_a);
    m_b = std::move(src.m_b);
    return *this;
  }

  A & m_a;
  B & m_b;
};

template <class A, class B, class Compare>
struct DualIteratorComparator
{
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIteratorComparator(Compare comp) :
    m_compare(comp)
  {}

  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline bool operator()(DualIteratorAccessor<A, B> const & lhs,
                                      DualIteratorAccessor<A, B> const & rhs) const restrict_this
  { return m_compare(lhs.m_a, rhs.m_a); }

  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline bool operator()(typename DualIteratorAccessor<A, B>::Temporary const & lhs,
                                      DualIteratorAccessor<A, B> const & rhs) const restrict_this
  { return m_compare(lhs.m_a, rhs.m_a); }

  Compare m_compare;
};


template <class RandomAccessIteratorA, class RandomAccessIteratorB>
class DualIterator
{
private:
  RandomAccessIteratorA m_itA;
  RandomAccessIteratorB m_itB;

public:
  using A = typename std::remove_reference<decltype(*m_itA)>::type;
  using B = typename std::remove_reference<decltype(*m_itB)>::type;
  
  DISABLE_HD_WARNING
  inline DualIterator() = default;

  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator(RandomAccessIteratorA itToSort, RandomAccessIteratorB valuesIt) :
    m_itA(itToSort),
    m_itB(valuesIt)
  {}

  DISABLE_HD_WARNING
  inline DualIterator(DualIterator const & src) = default;

  DISABLE_HD_WARNING
  inline DualIterator(DualIterator && src) = default;

  DISABLE_HD_WARNING
  inline DualIterator & operator=(DualIterator const & src) = default;

  DISABLE_HD_WARNING
  inline DualIterator & operator=(DualIterator && src) = default;

  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator operator+(std::ptrdiff_t offset) const restrict_this
  { return DualIterator(m_itA + offset, m_itB + offset); }

  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator & operator+=(std::ptrdiff_t offset) restrict_this
  {
    m_itA += offset;
    m_itB += offset;
    return *this;
  }

  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator & operator++() restrict_this
  {
    m_itA++;
    m_itB++;
    return *this;
  }

  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline std::ptrdiff_t operator-(DualIterator const & rhs) const restrict_this
  {
    std::ptrdiff_t const distance = m_itA - rhs.m_itA;
    GEOS_ASSERT(distance == (m_itB - rhs.m_itB));
    return distance; 
  }

  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator operator-(std::ptrdiff_t offset) const restrict_this
  { return *this + (- offset);  }

  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator & operator-=(std::ptrdiff_t offset) restrict_this
  { return *this += (- offset); }

  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator & operator--() restrict_this
  {
    m_itA--;
    m_itB--;
    return *this;
  }

  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline bool operator!=(DualIterator const & rhs) const restrict_this
  { return m_itA != rhs.m_itA; }

  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline bool operator<(DualIterator const & rhs) const restrict_this
  { return m_itA < rhs.m_itA; }

  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIteratorAccessor<A, B> operator*() const restrict_this
  { return DualIteratorAccessor<A, B>(*m_itA, *m_itB); }

  DISABLE_HD_WARNING
  template <class Compare>
  LVARRAY_HOST_DEVICE inline DualIteratorComparator<A, B, Compare> createComparator(Compare comp) const restrict_this
  { return DualIteratorComparator<A, B, Compare>(comp); }
};

DISABLE_HD_WARNING
template <class T>
LVARRAY_HOST_DEVICE inline T && createTemporary(T & a)
{ return std::move(a); }

DISABLE_HD_WARNING
template <class A, class B>
inline LVARRAY_HOST_DEVICE typename DualIteratorAccessor<A, B>::Temporary createTemporary(DualIteratorAccessor<A, B> const & it)
{ return typename DualIteratorAccessor<A, B>::Temporary(it); }

DISABLE_HD_WARNING
template <class Iterator>
LVARRAY_HOST_DEVICE inline void exchange(Iterator a, Iterator b)
{
  auto temp = createTemporary(*a);
  *a = std::move(*b);
  *b = std::move(temp);
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
      exchange(result, b);
    }
    else if (comp(*a, *c))
    {
      exchange(result, c);
    }
    else
    {
      exchange(result, a);
    }
  }
  else if (comp(*a, *c))
  {
    exchange(result, a);
  }
  else if (comp(*b, *c))
  {
    exchange(result, c);
  }
  else
  {
    exchange(result, b);
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
  
    exchange(first, last);
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
  auto val = createTemporary(*last);
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
      auto val = createTemporary(*(first + i));
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
