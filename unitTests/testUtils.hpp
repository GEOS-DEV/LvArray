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

#ifndef TEST_UTILS_HPP_
#define TEST_UTILS_HPP_

// Source includes
#include "LvArrayConfig.hpp"
#include "Array.hpp"
#include "SortedArray.hpp"
#include "ArrayOfArrays.hpp"
#include "ArrayOfSets.hpp"
#include "SparsityPattern.hpp"
#include "CRSMatrix.hpp"
#include "Macros.hpp"

#if defined(USE_CHAI)
#include "NewChaiBuffer.hpp"
#else
#include "MallocBuffer.hpp"
#endif


// TPL includes
#include <RAJA/RAJA.hpp>
#include <gtest/gtest.h>

// System includes
#include <string>
#include <ostream>

namespace LvArray
{
namespace testing
{

template< typename >
struct RAJAHelper
{};

using serialPolicy = RAJA::loop_exec;

template<>
struct RAJAHelper< serialPolicy >
{
  using ReducePolicy = RAJA::seq_reduce;
  using AtomicPolicy = RAJA::seq_atomic;
  static constexpr MemorySpace space = MemorySpace::CPU;
};

#if defined(USE_OPENMP)

using parallelHostPolicy = RAJA::omp_parallel_for_exec;

template<>
struct RAJAHelper< parallelHostPolicy >
{
  using ReducePolicy = RAJA::omp_reduce;
  using AtomicPolicy = RAJA::builtin_atomic;
  static constexpr MemorySpace space = MemorySpace::CPU;
};

#endif

#if defined(USE_CUDA)

template< unsigned long THREADS_PER_BLOCK >
using parallelDevicePolicy = RAJA::cuda_exec< THREADS_PER_BLOCK >;

template< unsigned long N >
struct RAJAHelper< RAJA::cuda_exec< N > >
{
  using ReducePolicy = RAJA::cuda_reduce;
  using AtomicPolicy = RAJA::cuda_atomic;
  static constexpr MemorySpace space = MemorySpace::GPU;
};

#endif

template< typename POLICY, typename INDEX_TYPE, typename LAMBDA >
inline void forall( INDEX_TYPE const max, LAMBDA && body )
{
  RAJA::forall< POLICY >( RAJA::TypedRangeSegment< INDEX_TYPE >( 0, max ), std::forward< LAMBDA >( body ) );
}

#ifndef __CUDA_ARCH__
#define PORTABLE_EXPECT_EQ( L, R ) EXPECT_EQ( L, R )
#else
#define PORTABLE_EXPECT_EQ( L, R ) LVARRAY_ERROR_IF_NE( L, R )
#endif

// Comparator that compares a std::pair by it's first object.
template< class A, class B, class COMP=std::less< A > >
struct PairComp
{
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline
  constexpr bool operator()( const std::pair< A, B > & lhs, const std::pair< A, B > & rhs ) const
  {
    return COMP()( lhs.first, rhs.first );
  }
};

#if defined(USE_CHAI)
template< typename T >
using DEFAULT_BUFFER = NewChaiBuffer< T >;
#else
template< typename T >
using DEFAULT_BUFFER = MallocBuffer< T >;
#endif

template< typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
struct ArrayConversion
{
  template< typename T, int NDIM, typename PERMUTATION >
  using Array = Array< T, NDIM, PERMUTATION, INDEX_TYPE, BUFFER_TYPE >;

  template< typename T, int NDIM, int USD >
  using ArrayView = ArrayView< T, NDIM, USD, INDEX_TYPE, BUFFER_TYPE >;

  template< typename T, int NDIM, int USD >
  using ArraySlice = ArraySlice< T, NDIM, USD, INDEX_TYPE >;

  template< typename T >
  using SortedArray = SortedArray< T, INDEX_TYPE, BUFFER_TYPE >;

  template< typename T >
  using SortedArrayView = SortedArrayView< T, INDEX_TYPE const, BUFFER_TYPE >;

  template< typename T >
  using ArrayOfArrays = ArrayOfArrays< T, INDEX_TYPE, BUFFER_TYPE >;

  template< typename T, bool CONST_SIZES >
  using ArrayOfArraysView = ArrayOfArraysView< T, INDEX_TYPE const, CONST_SIZES, BUFFER_TYPE >;

  template< typename T >
  using ArrayOfSets = ArrayOfSets< T, INDEX_TYPE, BUFFER_TYPE >;

  template< typename T >
  using ArrayOfSetsView = ArrayOfSetsView< T, INDEX_TYPE const, BUFFER_TYPE >;

  template< typename COL_TYPE >
  using SparsityPattern = SparsityPattern< COL_TYPE, INDEX_TYPE, BUFFER_TYPE >;

  template< typename COL_TYPE >
  using SparsityPatternView = SparsityPatternView< COL_TYPE, INDEX_TYPE const, BUFFER_TYPE >;

  template< typename T, typename COL_TYPE >
  using CRSMatrix = CRSMatrix< T, COL_TYPE, INDEX_TYPE, BUFFER_TYPE >;

  template< typename T, typename COL_TYPE >
  using CRSMatrixView = CRSMatrixView< T, COL_TYPE, INDEX_TYPE const, BUFFER_TYPE >;
};

template< typename >
struct ArrayConverter;

template< typename T,
          int NDIM,
          typename PERMUTATION,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
struct ArrayConverter< Array< T, NDIM, PERMUTATION, INDEX_TYPE, BUFFER_TYPE > > :
  public ArrayConversion< INDEX_TYPE, BUFFER_TYPE >
{};

template< typename T,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
struct ArrayConverter< ArrayOfArrays< T, INDEX_TYPE, BUFFER_TYPE > > :
  public ArrayConversion< INDEX_TYPE, BUFFER_TYPE >
{};

template< typename T,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
struct ArrayConverter< ArrayOfSets< T, INDEX_TYPE, BUFFER_TYPE > > :
  public ArrayConversion< INDEX_TYPE, BUFFER_TYPE >
{};

template< typename COL_TYPE,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
struct ArrayConverter< SparsityPattern< COL_TYPE, INDEX_TYPE, BUFFER_TYPE > > :
  public ArrayConversion< INDEX_TYPE, BUFFER_TYPE >
{};

template< typename T,
          typename COL_TYPE,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
struct ArrayConverter< CRSMatrix< T, COL_TYPE, INDEX_TYPE, BUFFER_TYPE > > :
  public ArrayConversion< INDEX_TYPE, BUFFER_TYPE >
{};



/**
 * @class TestString
 * @brief A wrapper around std::string that adds a constructor that takes a number
 *        and converts it to a string. Used for testing purposes.
 * * @note The default constructors, destructor, and operator are implemented here
 *       because otherwise nvcc will complain about host-device errors.
 */
class TestString
{
public:

  TestString():
    TestString( 0 )
  {}

  template< class T >
  TestString( T val ):
    m_string( std::to_string( val ) +
              std::string( " The rest of this is to avoid any small string optimizations. " ) +
              std::to_string( 2 * val ) )
  {}

  TestString( TestString const & src ):
    m_string( src.m_string )
  {}

  TestString( TestString && src ):
    m_string( std::move( src.m_string ) )
  {}

  ~TestString()
  {}

  void operator=( TestString const & src )
  {
    m_string = src.m_string;
  }

  void operator=( TestString && src )
  {
    m_string = std::move( src.m_string );
  }

  void operator+=( TestString const & other )
  {
    long newVal = getValue() + other.getValue();
    *this = TestString( newVal );
  }

  bool operator==( TestString const & rhs ) const
  { return m_string == rhs.m_string; }

  bool operator!=( TestString const & rhs ) const
  { return m_string != rhs.m_string; }

  bool operator<( TestString const & rhs ) const
  { return getValue() < rhs.getValue(); }

  bool operator<=( TestString const & rhs ) const
  { return getValue() <= rhs.getValue(); }

  bool operator>( TestString const & rhs ) const
  { return getValue() > rhs.getValue(); }

  bool operator>=( TestString const & rhs ) const
  { return getValue() >= rhs.getValue(); }

  friend std::ostream & operator<<( std::ostream & os, TestString const & testString )
  { return os << testString.m_string; }

private:

  long getValue() const
  { return std::stol( m_string ); }

  std::string m_string;
};

/**
 * @class Tensor
 * @brief A simple struct containing three doubles used for testing purposes.
 * Can be used on the device.
 */
struct Tensor
{
  LVARRAY_HOST_DEVICE Tensor():
    Tensor( 0 )
  {}

  template< class T >
  LVARRAY_HOST_DEVICE explicit Tensor( T val ):
    m_x( 3 * val ), m_y( 3 * val + 1 ), m_z( 3 * val + 2 )
  {}

  LVARRAY_HOST_DEVICE Tensor( Tensor const & src ):
    m_x( src.m_x ),
    m_y( src.m_y ),
    m_z( src.m_z )
  {}

  LVARRAY_HOST_DEVICE Tensor & operator=( Tensor const & src )
  {
    m_x = src.m_x;
    m_y = src.m_y;
    m_z = src.m_z;
    return *this;
  }

  LVARRAY_HOST_DEVICE Tensor & operator+=( Tensor const & rhs )
  {
    m_x += rhs.m_x;
    m_y += rhs.m_y;
    m_z += rhs.m_z;
    return *this;
  }

  LVARRAY_HOST_DEVICE bool operator==( Tensor const & rhs ) const
  {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
    return m_x == rhs.m_x && m_y == rhs.m_y && m_z == rhs.m_z;
#pragma GCC diagnostic pop
  }

  LVARRAY_HOST_DEVICE bool operator<( Tensor const & rhs ) const
  { return m_x < rhs.m_x; }

  LVARRAY_HOST_DEVICE bool operator>( Tensor const & rhs ) const
  { return m_x > rhs.m_x; }

  LVARRAY_HOST_DEVICE bool operator !=( Tensor const & rhs ) const
  { return !(*this == rhs); }

  friend std::ostream & operator<<( std::ostream & stream, Tensor const & t )
  {
    stream << "(" << t.m_x << ", " << t.m_y << ", " << t.m_z << ")";
    return stream;
  }

private:
  double m_x, m_y, m_z;
};

} // namespace testing
} // namespace LvArray

#endif // TEST_UTILS_HPP_
