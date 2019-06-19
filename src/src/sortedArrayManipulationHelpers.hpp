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
 * These routines were ripped from the gcc standard library
 *(https://gcc.gnu.org/svn/gcc/trunk/libstdc++-v3/include/bits/)
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

/// The size above which we continue with the introsortLoop.
constexpr int INTROSORT_THRESHOLD = 64;

/**
 * @tparam A the type of the first value.
 * @tparam B the type of the second value.
 * @class DualIteratorAccessor
 * @brief This class is the type returned by the DualIterator operator* and holds
 *        a reference to a value in both underlying arrays.
 */
template< class A, class B >
struct DualIteratorAccessor
{
  /**
   * @class Temporary
   * @brief A helper class that holds a copy of the values.
   */
  struct Temporary
  {
    /**
     * @brief Deleted default constructor.
     */
    inline Temporary() = delete;

    /**
     * @brief Deleted copy constructor.
     */
    inline Temporary( Temporary const & ) = delete;

    /**
     * @brief Move constructor, moves the values from src..
     * @param [in] src Temporary to move the values from.
     */
    DISABLE_HD_WARNING
    LVARRAY_HOST_DEVICE inline Temporary( Temporary && src ):
      m_a( std::move( src.m_a )),
      m_b( std::move( src.m_b ))
    {}

    /**
     * @brief Move the values pointed at by src into a temporary location.
     * @param [in] src DualIteratorAccessor to the values to move.
     */
    DISABLE_HD_WARNING
    LVARRAY_HOST_DEVICE inline Temporary( DualIteratorAccessor && src ):
      m_a( std::move( src.m_a )),
      m_b( std::move( src.m_b ))
    {}

    /**
     * @brief Default destructor, must be declared default so we can disable host-device warnings.
     */
    DISABLE_HD_WARNING
    inline ~Temporary() = default;

    /**
     * @brief Deleted copy assignment operator.
     */
    inline Temporary & operator=( Temporary const & ) = delete;

    /**
     * @brief Deleted move assignment operator.
     */
    inline Temporary & operator=( Temporary && ) = delete;

    A m_a;
    B m_b;
  };

  /**
   * @brief Delete the default constructor.
   */
  DualIteratorAccessor() = delete;

  /**
   * @brief Constructor to wrap the two given values.
   * @param [in/out] a the first value.
   * @param [in/out] b the second value.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIteratorAccessor( A & a, B & b ):
    m_a( a ),
    m_b( b )
  {}

  /**
   * @brief Delete the copy constructor.
   */
  DualIteratorAccessor( DualIteratorAccessor const & src ) = delete;

  /**
   * @brief Default the move constructor.
   */
  DISABLE_HD_WARNING
  inline DualIteratorAccessor( DualIteratorAccessor && src ) = default;

  /**
   * @brief Delete the default copy assignment operator.
   */
  DualIteratorAccessor & operator=( DualIteratorAccessor const & src ) = delete;

  /**
   * @brief Move assignment operator that moves the values of src into the values of this.
   * @param [in/out] src the DualIteratorAccessor to move from.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIteratorAccessor & operator=( DualIteratorAccessor && src ) restrict_this
  {
    m_a = std::move( src.m_a );
    m_b = std::move( src.m_b );
    return *this;
  }

  /**
   * @brief Move assignment operator that moves the values of src into the values of this.
   * @param [in/out] src the Temporary to move from.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE DualIteratorAccessor & operator=( Temporary && src ) restrict_this
  {
    m_a = std::move( src.m_a );
    m_b = std::move( src.m_b );
    return *this;
  }

  A & m_a;
  B & m_b;
};

/**
 * @tparam A the type of the first value.
 * @tparam B the type of the second value.
 * @tparam Compare the type of the comparison function.
 * @class DualIteratorComparator
 * @brief This class is used to make a comparison method that takes in DualIteratorAccessor<A, B> from
 *        a comparison method that takes in objects of type A.
 */
template< class A, class B, class Compare >
struct DualIteratorComparator
{
  /**
   * @brief Constructor that takes in a comparison method.
   * @param [in/out] comp the comparison method to wrap.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIteratorComparator( Compare comp ):
    m_compare( comp )
  {}

  /**
   * @brief Compare lhs with rhs using their first values.
   * @param [in] lhs a DualIteratorAccessor of the first object to compare.
   * @param [in] rhs a DualIteratorAccessor of the second object to compare.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline bool operator()( DualIteratorAccessor< A, B > const & lhs,
                                              DualIteratorAccessor< A, B > const & rhs ) const restrict_this
  { return m_compare( lhs.m_a, rhs.m_a ); }

  /**
   * @brief Compare lhs with rhs using their first values.
   * @param [in] lhs a DualIteratorAccessor::Temporary of the first object to compare.
   * @param [in] rhs a DualIteratorAccessor of the second object to compare.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline bool operator()( typename DualIteratorAccessor< A, B >::Temporary const & lhs,
                                              DualIteratorAccessor< A, B > const & rhs ) const restrict_this
  { return m_compare( lhs.m_a, rhs.m_a ); }

  Compare m_compare;
};

/**
 * @tparam RandomAccessIteratorA the type of the first iterator.
 * @tparam RandomAccessIteratorB the type of the second iterator.
 * @class DualIterator
 * @brief This class acts as a single iterator while wrapping two iterators.
 */
template< class RandomAccessIteratorA, class RandomAccessIteratorB >
class DualIterator
{
private:
  RandomAccessIteratorA m_itA;
  RandomAccessIteratorB m_itB;

public:
  using A = typename std::remove_reference< decltype(*m_itA) >::type;
  using B = typename std::remove_reference< decltype(*m_itB) >::type;

  /**
   * @brief Default constructor.
   */
  DISABLE_HD_WARNING
  inline DualIterator() = default;

  /**
   * @brief Constructor taking the two iterators to wrap.
   * @param [in] itA the first iterator to wrap.
   * @param [in] itB the second iterator to wrap.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator( RandomAccessIteratorA const itA, RandomAccessIteratorB const itB ):
    m_itA( itA ),
    m_itB( itB )
  {}

  /**
   * @brief Default copy constructor.
   */
  DISABLE_HD_WARNING
  inline DualIterator( DualIterator const & src ) = default;

  /**
   * @brief Default move constructor.
   */
  DISABLE_HD_WARNING
  inline DualIterator( DualIterator && src ) = default;

  /**
   * @brief Default copy assignment operator.
   */
  DISABLE_HD_WARNING
  inline DualIterator & operator=( DualIterator const & src ) = default;

  /**
   * @brief Default move assignment operator.
   */
  DISABLE_HD_WARNING
  inline DualIterator & operator=( DualIterator && src ) = default;

  /**
   * @brief Return a new DualIterator where both of the underlying iterators have been incremented by offset.
   * @param [in] offset the value to increment the iterators by.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator operator+( std::ptrdiff_t const offset ) const restrict_this
  { return DualIterator( m_itA + offset, m_itB + offset ); }

  /**
   * @brief Increment the underlying iterators by offset and return a reference to *this.
   * @param [in] offset the value to increment the iterators by.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator & operator+=( std::ptrdiff_t const offset ) restrict_this
  {
    m_itA += offset;
    m_itB += offset;
    return *this;
  }

  /**
   * @brief Increment the underlying iterators by 1 and return a reference to *this.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator & operator++() restrict_this
  {
    m_itA++;
    m_itB++;
    return *this;
  }

  /**
   * @brief Return the distance between *this and the given DualIterator.
   * @param [in] rhs the DualIterator to find the distance between.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline std::ptrdiff_t operator-( DualIterator const & rhs ) const restrict_this
  {
    std::ptrdiff_t const distance = m_itA - rhs.m_itA;
    GEOS_ASSERT( distance == (m_itB - rhs.m_itB));
    return distance;
  }

  /**
   * @brief Return a new DualIterator where both of the underlying iterators have been decrement by offset.
   * @param [in] offset the value to decrement the iterators by.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator operator-( std::ptrdiff_t const offset ) const restrict_this
  { return *this + (-offset); }

  /**
   * @brief Decrement the underlying iterators by offset and return a reference to *this.
   * @param [in] offset the value to decrement the iterators by.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator & operator-=( std::ptrdiff_t const offset ) restrict_this
  { return *this += (-offset); }

  /**
   * @brief Decrement the underlying iterators by 1 and return a reference to *this.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator & operator--() restrict_this
  {
    m_itA--;
    m_itB--;
    return *this;
  }

  /**
   * @brief Return true if rhs does not point to the same values.
   * @param [in] rhs the DualIterator to compare against.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline bool operator!=( DualIterator const & rhs ) const restrict_this
  { return m_itA != rhs.m_itA; }

  /**
   * @brief Return true if is less than this.
   * @param [in] rhs the DualIterator to compare against.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline bool operator<( DualIterator const & rhs ) const restrict_this
  { return m_itA < rhs.m_itA; }

  /**
   * @brief Return a DualIteratorAccessor to the two values at which m_itA and m_itB point at.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIteratorAccessor< A, B > operator*() const restrict_this
  { return DualIteratorAccessor< A, B >( *m_itA, *m_itB ); }

  /**
   * @tparam Compare the type of the comparison method.
   * @brief Return a new Comparator which will compare two DualIteratorAccessors using the given comparison.
   * @param [in] comp the comparison to wrap.
   */
  DISABLE_HD_WARNING
  template< class Compare >
  LVARRAY_HOST_DEVICE inline DualIteratorComparator< A, B, Compare > createComparator( Compare comp ) const restrict_this
  { return DualIteratorComparator< A, B, Compare >( comp ); }
};

/**
 * @tparam T the type of the value to create.
 * @brief Return a temporary object holding the value of a.
 * @param [in/out] a the value to move.
 */
DISABLE_HD_WARNING
template< class T >
LVARRAY_HOST_DEVICE inline T && createTemporary( T & a )
{
  return std::move( a );
}

/**
 * @tparam A the type of the first value in the DualIteratorAccessor.
 * @tparam B the type of the second value in the DualIteratorAccessor.
 * @brief Return a temporary object holding the values held in the DualIteratorAccessor.
 * @param [in/out] it the DualIteratorAccessor to move from.
 */
DISABLE_HD_WARNING
template< class A, class B >
inline LVARRAY_HOST_DEVICE typename DualIteratorAccessor< A, B >::Temporary createTemporary( DualIteratorAccessor< A, B > && it )
{ return typename DualIteratorAccessor< A, B >::Temporary( std::move( it )); }

/**
 * @tparam Iterator an iterator type.
 * @brief Exchange the values pointed at by the two iterators.
 * @param [in/out] a the first iterator.
 * @param [in/out] b the second iterator.
 */
DISABLE_HD_WARNING
template< class Iterator >
LVARRAY_HOST_DEVICE inline void exchange( Iterator a, Iterator b )
{
  auto temp = createTemporary( *a );
  *a = std::move( *b );
  *b = std::move( temp );
}

/**
 * @tparam BidirIt1 a bidirectional iterator type.
 * @tparam BidirIt2 a bidirectional iterator type.
 * @brief Move the elements in [first, last) to another range ending in d_last. The elements are moved in reverse order.
 * @param [in/out] first iterator to the beginning of the values to move.
 * @param [in/out] last iterator to the end of the values to move.
 * @param [in/out] d_last iterator to the end of the values to move to.
 *
 * @note This should be equivalent to std::move_backward and this implementation was gotten from
 *       https://en.cppreference.com/w/cpp/algorithm/move_backward.
 */
DISABLE_HD_WARNING
template< class BidirIt1, class BidirIt2 >
LVARRAY_HOST_DEVICE inline void moveBackward( BidirIt1 first, BidirIt1 last, BidirIt2 d_last )
{
  while( first != last )
  {
    *(--d_last) = std::move( *(--last));
  }
}

/**
 * @tparam Iterator an iterator type.
 * @tparam Compare the type of the comparison method.
 * @brief Compute the median of *a, *b and *c and place it in *result.
 * @param [out] result iterator to the place to store the result.
 * @param [in] a iterator to the first value to compare
 * @param [in] b iterator to the first value to compare.
 * @param [in] c iterator to the first value to compare.
 * @param [in/out] comp the comparison method to use.
 *
 * @note This method was adapted from the gcc standard libarary header stl_algo.h
 *       which can be found at https://gcc.gnu.org/svn/gcc/trunk/libstdc++-v3/include/bits/.
 */
DISABLE_HD_WARNING
template< typename Iterator, typename Compare >
LVARRAY_HOST_DEVICE inline void computeMedian( Iterator result, Iterator a, Iterator b, Iterator c, Compare comp )
{
  if( comp( *a, *b ))
  {
    if( comp( *b, *c ))
    {
      exchange( result, b );
    }
    else if( comp( *a, *c ))
    {
      exchange( result, c );
    }
    else
    {
      exchange( result, a );
    }
  }
  else if( comp( *a, *c ))
  {
    exchange( result, a );
  }
  else if( comp( *b, *c ))
  {
    exchange( result, c );
  }
  else
  {
    exchange( result, b );
  }
}

/**
 * @tparam RandomAccessIterator a random access iterator type.
 * @tparam Compare the type of the comparison method.
 * @brief Partition the range [first, last) with regards to *pivot.
 * @param [in/out] first iterator to the beginning of the range to partition.
 * @param [in/out] last iterator to the end of the range to partition.
 * @param [in] pivot iterator to the value to use as a pivot.
 * @param [in/out] comp the comparison method to use.
 * @return a RandomAccessIterator.
 *
 * @note This method was adapted from the gcc standard libarary header stl_algo.h
 *       which can be found at https://gcc.gnu.org/svn/gcc/trunk/libstdc++-v3/include/bits/.
 */
DISABLE_HD_WARNING
template< typename RandomAccessIterator, typename Compare >
LVARRAY_HOST_DEVICE inline RandomAccessIterator unguardedPartition( RandomAccessIterator first, RandomAccessIterator last,
                                                                    RandomAccessIterator pivot, Compare comp )
{
  while( true )
  {
    while( comp( *first, *pivot ))
    {
      ++first;
    }

    --last;
    while( comp( *pivot, *last ))
    {
      --last;
    }

    if( !(first < last))
    {
      return first;
    }

    exchange( first, last );
    ++first;
  }
}

/**
 * @tparam RandomAccessIterator a random access iterator type.
 * @tparam Compare the type of the comparison method.
 * @brief Partition the range [first, last).
 * @param [in/out] first iterator to the beginning of the range to partition.
 * @param [in/out] last iterator to the end of the range to partition.
 * @param [in/out] comp the comparison method to use.
 *
 * @note This method was adapted from the gcc standard libarary header stl_algo.h
 *       which can be found at https://gcc.gnu.org/svn/gcc/trunk/libstdc++-v3/include/bits/.
 */
DISABLE_HD_WARNING
template< typename RandomAccessIterator, typename Compare >
LVARRAY_HOST_DEVICE inline RandomAccessIterator unguardedPartitionPivot( RandomAccessIterator first, RandomAccessIterator last, Compare comp )
{
  RandomAccessIterator mid = first + (last - first) / 2;
  computeMedian( first, first + 1, mid, last - 1, comp );
  return unguardedPartition( first + 1, last, first, comp );
}

/**
 * @tparam RandomAccessIterator a random access iterator type.
 * @tparam Compare the type of the comparison method.
 * @param [in/out] last iterator to the end of the range.
 * @param [in/out] comp the comparison method to use.
 *
 * @note This method was adapted from the gcc standard libarary header stl_algo.h
 *       which can be found at https://gcc.gnu.org/svn/gcc/trunk/libstdc++-v3/include/bits/.
 */
DISABLE_HD_WARNING
template< typename RandomAccessIterator, typename Compare >
LVARRAY_HOST_DEVICE inline void unguardedLinearInsert( RandomAccessIterator last, Compare comp )
{
  auto val = createTemporary( *last );
  RandomAccessIterator next = last - 1;
  while( comp( val, *next ))
  {
    *last = std::move( *next );
    last = next;
    --next;
  }
  *last = std::move( val );
}

/**
 * @tparam RandomAccessIterator a random access iterator type.
 * @tparam Compare the type of the comparison method.
 * @brief sort the range [first, first + n) under comp using insertions sort.
 * @param [in/out] first iterator to the beginning of the range.
 * @param [in] n the size of the range.
 * @param [in/out] comp the comparison method to use.
 *
 * @note This method was adapted from the gcc standard libarary header stl_algo.h
 *       which can be found at https://gcc.gnu.org/svn/gcc/trunk/libstdc++-v3/include/bits/.
 */
DISABLE_HD_WARNING
template< class RandomAccessIterator, class Compare >
LVARRAY_HOST_DEVICE inline void insertionSort( RandomAccessIterator first, std::ptrdiff_t const n, Compare comp )
{
  for( std::ptrdiff_t i = 1 ; i < n ; ++i )
  {
    if( comp( *(first + i), *first ))
    {
      auto val = createTemporary( *(first + i));
      moveBackward( first, first + i, first + i + 1 );
      *first = std::move( val );
    }
    else
    {
      unguardedLinearInsert( first + i, comp );
    }
  }
}

/**
 * @tparam RandomAccessIterator a random access iterator type.
 * @tparam Compare the type of the comparison method.
 * @brief Partially sort the range [first, last) under comp.
 * @param [in/out] first iterator to the beginning of the range.
 * @param [in/out] last iterator to the end of the range.
 * @param [in/out] comp the comparison method to use.
 *
 * @note This method was heavily modified to get rid of recursion. As a result
 *       it no-longer does a heap-sort when the recursion depth reaches a limit.
 *       On randomly generated datasets it performs as well or better than std::sort.
 *       However because it no longer does heap-sort there may be certain cases where
 *       std::sort performs better. If this becomes an issue try adding heap-sort back in.
 * @note This method was adapted from the gcc standard libarary header stl_algo.h
 *       which can be found at https://gcc.gnu.org/svn/gcc/trunk/libstdc++-v3/include/bits/.
 */
DISABLE_HD_WARNING
template< typename RandomAccessIterator, typename Compare >
LVARRAY_HOST_DEVICE inline void introsortLoop( RandomAccessIterator first, RandomAccessIterator last, Compare comp )
{
  constexpr int MAX_RANGES = 31;
  RandomAccessIterator ranges[MAX_RANGES + 1];

  int nRanges = 1;
  ranges[0] = first;
  ranges[1] = last;
  while( nRanges != 0 )
  {
    RandomAccessIterator const m_first = ranges[nRanges - 1];

    if( ranges[nRanges] - m_first > INTROSORT_THRESHOLD )
    {
      GEOS_ASSERT( nRanges < MAX_RANGES );
      ranges[nRanges + 1] = ranges[nRanges];
      ranges[nRanges] = unguardedPartitionPivot( m_first, ranges[nRanges], comp );
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
