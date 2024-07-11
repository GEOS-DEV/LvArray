/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

#pragma once

// Source includes
#include "LvArrayConfig.hpp"
#include "Macros.hpp"

#include "MallocBuffer.hpp"

#if defined(LVARRAY_USE_CHAI)
  #include "ChaiBuffer.hpp"
#endif


// TPL includes
#include <RAJA/RAJA.hpp>
#include <umpire/strategy/QuickPool.hpp>
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

using serialPolicy = RAJA::seq_exec;

template<>
struct RAJAHelper< serialPolicy >
{
  using ReducePolicy = RAJA::seq_reduce;
  using AtomicPolicy = RAJA::seq_atomic;
  static constexpr MemorySpace space = MemorySpace::host;
};

#if defined(RAJA_ENABLE_OPENMP)

using parallelHostPolicy = RAJA::omp_parallel_for_exec;

template<>
struct RAJAHelper< parallelHostPolicy >
{
  using ReducePolicy = RAJA::omp_reduce;
  using AtomicPolicy = RAJA::omp_atomic;
  static constexpr MemorySpace space = MemorySpace::host;
};

#endif

#if defined(LVARRAY_USE_CUDA)

template< unsigned long THREADS_PER_BLOCK >
using parallelDevicePolicy = RAJA::cuda_exec< THREADS_PER_BLOCK >;


template< typename X, typename Y, size_t BLOCK_SIZE, bool ASYNC >
struct RAJAHelper< RAJA::policy::cuda::cuda_exec_explicit< X, Y, BLOCK_SIZE, ASYNC > >
{
  using ReducePolicy = RAJA::cuda_reduce;
  using AtomicPolicy = RAJA::cuda_atomic;
  static constexpr MemorySpace space = MemorySpace::cuda;
};

#endif

template< typename POLICY, typename INDEX_TYPE, typename LAMBDA >
inline void forall( INDEX_TYPE const max, LAMBDA && body )
{
  RAJA::forall< POLICY >( RAJA::TypedRangeSegment< INDEX_TYPE >( 0, max ), std::forward< LAMBDA >( body ) );
}

template< typename T, typename LAYOUT >
LVARRAY_HOST_DEVICE inline constexpr
T * getRAJAViewData( RAJA::View< T, LAYOUT > const & view )
{
#if RAJA_VERSION_MAJOR <= 0 && RAJA_VERSION_MINOR <= 13
  return view.data;
#else
  return view.get_data();
#endif
}

template< typename T, typename LAYOUT >
LVARRAY_HOST_DEVICE inline constexpr
LAYOUT const & getRAJAViewLayout( RAJA::View< T, LAYOUT > const & view )
{
#if RAJA_VERSION_MAJOR <= 0 && RAJA_VERSION_MINOR <= 13
  return view.layout;
#else
  return view.get_layout();
#endif
}


#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define PORTABLE_EXPECT_EQ( L, R ) LVARRAY_ERROR_IF_NE( L, R )
#define PORTABLE_EXPECT_NEAR( L, R, EPSILON ) LVARRAY_ERROR_IF_GE_MSG( math::abs( ( L ) -( R ) ), EPSILON, \
                                                                       STRINGIZE( L ) " = " << ( L ) << "\n" << STRINGIZE( R ) " = " << ( R ) );
#else
#define PORTABLE_EXPECT_EQ( L, R ) EXPECT_EQ( L, R )
#define PORTABLE_EXPECT_NEAR( L, R, EPSILON ) EXPECT_LE( math::abs( ( L ) -( R ) ), EPSILON ) << \
    STRINGIZE( L ) " = " << ( L ) << "\n" << STRINGIZE( R ) " = " << ( R );
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

#if defined(LVARRAY_USE_CHAI)
template< typename T >
using DEFAULT_BUFFER = ChaiBuffer< T >;
#else
template< typename T >
using DEFAULT_BUFFER = MallocBuffer< T >;
#endif

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
  explicit TestString( T val ):
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
  Tensor() = default;

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
