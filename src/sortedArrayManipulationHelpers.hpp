/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file sortedArrayManipulationHelpers.hpp
 * @brief Contains helper functions for the sortedArrayManipulation routines.
 *   Much of this was adapted from stl_heap.h and stl_algo.h of the gcc standard library
 *   See https://gcc.gnu.org/svn/gcc/trunk/libstdc++-v3/include/bits/
 */

#pragma once

#include "LvArrayConfig.hpp"
#include "Macros.hpp"

#include <utility>

namespace LvArray
{
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
     * @param src Temporary to move the values from.
     */
    DISABLE_HD_WARNING
    LVARRAY_HOST_DEVICE inline Temporary( Temporary && src ):
      m_a( std::move( src.m_a )),
      m_b( std::move( src.m_b ))
    {}

    /**
     * @brief Move the values pointed at by src into a temporary location.
     * @param src DualIteratorAccessor to the values to move.
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
     * @return *this.
     */
    inline Temporary & operator=( Temporary const & ) = delete;

    /**
     * @brief Deleted move assignment operator.
     * @return *this.
     */
    inline Temporary & operator=( Temporary && ) = delete;

    /// A copy of type A.
    A m_a;

    /// A copy of type B.
    B m_b;
  };

  /**
   * @brief Delete the default constructor.
   */
  DualIteratorAccessor() = delete;

  /**
   * @brief Constructor to wrap the two given values.
   * @param a the first value.
   * @param b the second value.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIteratorAccessor( A & a, B & b ):
    m_a( a ),
    m_b( b )
  {}

  /**
   * @brief Delete the copy constructor.
   */
  DualIteratorAccessor( DualIteratorAccessor const & ) = delete;

  /**
   * @brief Default the move constructor.
   */
  DISABLE_HD_WARNING
  inline DualIteratorAccessor( DualIteratorAccessor && ) = default;

  /**
   * @brief Delete the default copy assignment operator.
   * @return *this.
   */
  DualIteratorAccessor & operator=( DualIteratorAccessor const & ) = delete;

  /**
   * @brief Move assignment operator that moves the values of src into the values of this.
   * @param src the DualIteratorAccessor to move from.
   * @return *this.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIteratorAccessor & operator=( DualIteratorAccessor && src )
  {
    m_a = std::move( src.m_a );
    m_b = std::move( src.m_b );
    return *this;
  }

  /**
   * @brief Move assignment operator that moves the values of src into the values of this.
   * @param src the Temporary to move from.
   * @return *this.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE DualIteratorAccessor & operator=( Temporary && src )
  {
    m_a = std::move( src.m_a );
    m_b = std::move( src.m_b );
    return *this;
  }

  /// A reference to the first value.
  A & m_a;

  /// A reference to the second value.
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
template< typename A, typename B, typename Compare >
struct DualIteratorComparator
{
  /**
   * @brief Constructor that takes in a comparison method.
   * @param comp the comparison method to wrap.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIteratorComparator( Compare comp ):
    m_compare( comp )
  {}

  /**
   * @brief Compare @p lhs with @p rhs using their first values.
   * @param lhs a DualIteratorAccessor of the first object to compare.
   * @param rhs a DualIteratorAccessor of the second object to compare.
   * @return The result of the comparison of @p lhs with @p rhs using their first values.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline bool operator()( DualIteratorAccessor< A, B > const & lhs,
                                              DualIteratorAccessor< A, B > const & rhs ) const
  { return m_compare( lhs.m_a, rhs.m_a ); }

  /**
   * @brief Compare @p lhs with @p rhs using their first values.
   * @param lhs a DualIteratorAccessor::Temporary of the first object to compare.
   * @param rhs a DualIteratorAccessor of the second object to compare.
   * @return The result of the comparison of @p lhs with @p rhs using their first values.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline bool operator()( typename DualIteratorAccessor< A, B >::Temporary const & lhs,
                                              DualIteratorAccessor< A, B > const & rhs ) const
  { return m_compare( lhs.m_a, rhs.m_a ); }

private:
  /// The comparator that compares objects of type A.
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
public:
  /// An alias for the type pointed to by RandomAccessIteratorA.
  using A = typename std::remove_reference< decltype( *std::declval< RandomAccessIteratorA >() ) >::type;

  /// An alias for the type pointed to by RandomAccessIteratorB.
  using B = typename std::remove_reference< decltype( *std::declval< RandomAccessIteratorB >() ) >::type;

  /**
   * @brief Default constructor.
   */
  DISABLE_HD_WARNING
  inline DualIterator() = default;

  /**
   * @brief Constructor taking the two iterators to wrap.
   * @param itA the first iterator to wrap.
   * @param itB the second iterator to wrap.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator( RandomAccessIteratorA const itA, RandomAccessIteratorB const itB ):
    m_itA( itA ),
    m_itB( itB )
  {}

  /**
   * @brief Default assignment operator constructor.
   * @param src The DualIterator to copy.
   */
  DISABLE_HD_WARNING
  inline DualIterator( DualIterator const & src ) = default;

  /**
   * @brief Default move constructor.
   * @param src The DualIterator to move from.
   */
  DISABLE_HD_WARNING
  inline DualIterator( DualIterator && src ) = default;

  /**
   * @brief Default copy assignment operator.
   * @param src The DualIterator to copy.
   * @return *this.
   */
  DISABLE_HD_WARNING
  inline DualIterator & operator=( DualIterator const & src ) = default;

  /**
   * @brief Default move assignment operator.
   * @param src The DualIterator to move from.
   * @return *this.
   * */
  DISABLE_HD_WARNING
  inline DualIterator & operator=( DualIterator && src ) = default;

  /**
   * @brief Return a new DualIterator where both of the underlying iterators have been incremented by offset.
   * @param offset the value to increment the iterators by.
   * @return A new DualIterator where both of the underlying iterators have been incremented by offset.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator operator+( std::ptrdiff_t const offset ) const
  { return DualIterator( m_itA + offset, m_itB + offset ); }

  /**
   * @brief Increment the underlying iterators by offset and return a reference to *this.
   * @param offset the value to increment the iterators by.
   * @return *this.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator & operator+=( std::ptrdiff_t const offset )
  {
    m_itA += offset;
    m_itB += offset;
    return *this;
  }

  /**
   * @brief Increment the underlying iterators by 1 and return a reference to *this.
   * @return *this.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator & operator++()
  {
    m_itA++;
    m_itB++;
    return *this;
  }

  /**
   * @brief Return the distance between *this and @p rhs.
   * @param rhs the DualIterator to find the distance between.
   * @return The distance between *this and the @p rhs.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline std::ptrdiff_t operator-( DualIterator const & rhs ) const
  {
    std::ptrdiff_t const distance = m_itA - rhs.m_itA;
    LVARRAY_ASSERT_EQ( distance, m_itB - rhs.m_itB );
    return distance;
  }

  /**
   * @brief Return a new DualIterator where both of the underlying iterators have been decrement by @p offset.
   * @param offset the value to decrement the iterators by.
   * @return A new DualIterator where both of the underlying iterators have been decrement by @p offset.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator operator-( std::ptrdiff_t const offset ) const
  { return *this + (-offset); }

  /**
   * @brief Decrement the underlying iterators by @p offset and return a reference to *this.
   * @param offset the value to decrement the iterators by.
   * @return *this.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator & operator-=( std::ptrdiff_t const offset )
  { return *this += (-offset); }

  /**
   * @brief Decrement the underlying iterators by 1 and return a reference to *this.
   * @return *this.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIterator & operator--()
  {
    m_itA--;
    m_itB--;
    return *this;
  }

  /**
   * @brief Return true if @p rhs does not point to the same values.
   * @param rhs the DualIterator to compare against.
   * @return True if @p rhs does not point to the same values.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline bool operator!=( DualIterator const & rhs ) const
  { return m_itA != rhs.m_itA; }

  /**
   * @brief Return true if this less than @p rhs.
   * @param rhs the DualIterator to compare against.
   * @return True if this less than @p rhs.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline bool operator<( DualIterator const & rhs ) const
  { return m_itA < rhs.m_itA; }

  /**
   * @brief Return a DualIteratorAccessor to the two values at which m_itA and m_itB point at.
   * @return A DualIteratorAccessor to the two values at which m_itA and m_itB point at.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline DualIteratorAccessor< A, B > operator*() const
  { return DualIteratorAccessor< A, B >( *m_itA, *m_itB ); }

  /**
   * @brief Return a new Comparator which will compare two DualIteratorAccessors using the given comparison.
   * @tparam Compare the type of the comparison method.
   * @param comp the comparison to wrap.
   * @return A new Comparator which will compare two DualIteratorAccessors using the given comparison.
   */
  DISABLE_HD_WARNING
  template< class Compare >
  LVARRAY_HOST_DEVICE inline DualIteratorComparator< A, B, Compare > createComparator( Compare && comp ) const
  { return DualIteratorComparator< A, B, Compare >( std::forward< Compare >( comp ) ); }

private:
  /// The first iterator.
  RandomAccessIteratorA m_itA;

  /// The second iterator.
  RandomAccessIteratorB m_itB;

};

/**
 * @tparam T the type of the value to create.
 * @brief Return a temporary object holding the value of a.
 * @param a the value to move.
 * @return Return a temporary object holding the value of a.
 */
DISABLE_HD_WARNING
template< class T >
LVARRAY_HOST_DEVICE inline T && createTemporary( T & a )
{
  return std::move( a );
}

/**
 * @brief Return a temporary object holding the values held in the DualIteratorAccessor.
 * @tparam A the type of the first value in the DualIteratorAccessor.
 * @tparam B the type of the second value in the DualIteratorAccessor.
 * @param it the DualIteratorAccessor to move from.
 * @return Return a temporary object holding the values held in the DualIteratorAccessor.
 */
DISABLE_HD_WARNING
template< class A, class B >
inline LVARRAY_HOST_DEVICE typename DualIteratorAccessor< A, B >::Temporary createTemporary( DualIteratorAccessor< A, B > && it )
{ return typename DualIteratorAccessor< A, B >::Temporary( std::move( it )); }

/**
 * @tparam Iterator an iterator type.
 * @brief Exchange the values pointed at by the two iterators.
 * @param a the first iterator.
 * @param b the second iterator.
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
 * @param first iterator to the beginning of the values to move.
 * @param last iterator to the end of the values to move.
 * @param d_last iterator to the end of the values to move to.
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
 * @param result iterator to the place to store the result.
 * @param a iterator to the first value to compare
 * @param b iterator to the first value to compare.
 * @param c iterator to the first value to compare.
 * @param comp the comparison method to use.
 *
 * @note This method was adapted from the gcc standard libarary header stl_algo.h
 *       which can be found at https://gcc.gnu.org/svn/gcc/trunk/libstdc++-v3/include/bits/.
 */
DISABLE_HD_WARNING
template< typename Iterator, typename Compare >
LVARRAY_HOST_DEVICE inline void computeMedian( Iterator result, Iterator a, Iterator b, Iterator c, Compare && comp )
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
 * @param first iterator to the beginning of the range to partition.
 * @param last iterator to the end of the range to partition.
 * @param pivot iterator to the value to use as a pivot.
 * @param comp the comparison method to use.
 * @return An iterator to pivot.
 * @note This method was adapted from the gcc standard libarary header stl_algo.h
 *       which can be found at https://gcc.gnu.org/svn/gcc/trunk/libstdc++-v3/include/bits/.
 */
DISABLE_HD_WARNING
template< typename RandomAccessIterator, typename Compare >
LVARRAY_HOST_DEVICE inline RandomAccessIterator unguardedPartition( RandomAccessIterator first, RandomAccessIterator last,
                                                                    RandomAccessIterator pivot, Compare && comp )
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
 * @brief Partition the range [first, last).
 * @tparam RandomAccessIterator a random access iterator type.
 * @tparam Compare the type of the comparison method.
 * @param first iterator to the beginning of the range to partition.
 * @param last iterator to the end of the range to partition.
 * @param comp the comparison method to use.
 * @return An iterator to pivot.
 * @note This method was adapted from the gcc standard libarary header stl_algo.h
 *       which can be found at https://gcc.gnu.org/svn/gcc/trunk/libstdc++-v3/include/bits/.
 */
DISABLE_HD_WARNING
template< typename RandomAccessIterator, typename Compare >
LVARRAY_HOST_DEVICE inline RandomAccessIterator unguardedPartitionPivot( RandomAccessIterator first, RandomAccessIterator last, Compare && comp )
{
  RandomAccessIterator mid = first + (last - first) / 2;
  computeMedian( first, first + 1, mid, last - 1, comp );
  return unguardedPartition( first + 1, last, first, comp );
}

/**
 * @tparam RandomAccessIterator a random access iterator type.
 * @tparam Compare the type of the comparison method.
 * @param last iterator to the end of the range.
 * @param comp the comparison method to use.
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
 * @param first iterator to the beginning of the range.
 * @param n the size of the range.
 * @param comp the comparison method to use.
 *
 * @note This method was adapted from the gcc standard libarary header stl_algo.h
 *       which can be found at https://gcc.gnu.org/svn/gcc/trunk/libstdc++-v3/include/bits/.
 */
DISABLE_HD_WARNING
template< class RandomAccessIterator, class Compare >
LVARRAY_HOST_DEVICE inline void insertionSort( RandomAccessIterator first, std::ptrdiff_t const n, Compare comp )
{
  for( std::ptrdiff_t i = 1; i < n; ++i )
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
 * @param first iterator to the beginning of the range.
 * @param last iterator to the end of the range.
 * @param comp the comparison method to use.
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
LVARRAY_HOST_DEVICE inline void introsortLoop( RandomAccessIterator first, RandomAccessIterator last, Compare && comp )
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
      LVARRAY_ASSERT( nRanges < MAX_RANGES );
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
} // namespace LvArray
