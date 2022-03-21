/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "StackBuffer.hpp"
#include "Array.hpp"
#include "testUtils.hpp"

// TPL includes
#include <gtest/gtest.h>

namespace LvArray
{
namespace testing
{

using INDEX_TYPE = std::ptrdiff_t;

LVARRAY_HOST_DEVICE
INDEX_TYPE toLinearIndex( INDEX_TYPE const i )
{ return i; }

LVARRAY_HOST_DEVICE
INDEX_TYPE toLinearIndex( INDEX_TYPE const i, INDEX_TYPE const j )
{ return i + 10 * j; }

LVARRAY_HOST_DEVICE
INDEX_TYPE toLinearIndex( INDEX_TYPE const i, INDEX_TYPE const j, INDEX_TYPE const k )
{ return i + 10 * j + 100 * k; }

constexpr INDEX_TYPE pow( INDEX_TYPE const base, INDEX_TYPE const exponent )
{ return exponent == 0 ? 1 : base * pow( base, exponent - 1 ); }

template< typename PERMUTATION >
class StackArrayTest : public ::testing::Test
{
public:
  static constexpr int NDIM = typeManipulation::getDimension< PERMUTATION >;

  void resize()
  {
    init();

    m_array.resize( 4 );

    forValuesInSliceWithIndices( m_array.toSlice(), []( int const value, auto const ... indices )
    {
      EXPECT_EQ( value, toLinearIndex( indices ... ) );
    } );

    INDEX_TYPE dims[ NDIM ];
    for( int i = 0; i < NDIM; ++i )
    { dims[ i ] = i + 1; }

    m_array.resize( NDIM, dims );

    for( int i = 0; i < NDIM; ++i )
    { EXPECT_EQ( m_array.size( i ), i + 1 ); }

    forValuesInSliceWithIndices( m_array.toSlice(), []( int & value, auto const ... indices )
    {
      value = toLinearIndex( indices ... );
    } );

    ASSERT_DEATH_IF_SUPPORTED( m_array.resize( 2 * CAPACITY ), "" );
  }

protected:

  void init()
  {
    INDEX_TYPE dims[ NDIM ];

    for( int i = 0; i < NDIM; ++i )
    { dims[ i ] = 8; }

    m_array.resize( NDIM, dims );

    for( int i = 0; i < NDIM; ++i )
    { EXPECT_EQ( m_array.size( i ), 8 ); }

    forValuesInSliceWithIndices( m_array.toSlice(), []( int & value, auto const ... indices )
    {
      value = toLinearIndex( indices ... );
    } );
  }

  static constexpr INDEX_TYPE CAPACITY = pow( 8, NDIM );
  StackArray< int, NDIM, PERMUTATION, INDEX_TYPE, CAPACITY > m_array;
};

using StackArrayTestTypes = ::testing::Types<
  RAJA::PERM_I
  , RAJA::PERM_IJ
  , RAJA::PERM_JI
  , RAJA::PERM_IJK
  , RAJA::PERM_IKJ
  , RAJA::PERM_JIK
  , RAJA::PERM_JKI
  , RAJA::PERM_KIJ
  , RAJA::PERM_KJI
  >;

TYPED_TEST_SUITE( StackArrayTest, StackArrayTestTypes, );

TYPED_TEST( StackArrayTest, resize )
{
  this->resize();
}


template< typename PERMUTATION_POLICY_PAIR >
class StackArrayCaptureTest : public StackArrayTest< typename PERMUTATION_POLICY_PAIR::first_type >
{
public:
  using PERMUTATION = typename PERMUTATION_POLICY_PAIR::first_type;
  using POLICY = typename PERMUTATION_POLICY_PAIR::second_type;

  using ParentClass = StackArrayTest< PERMUTATION >;
  using ParentClass::NDIM;
  using ParentClass::CAPACITY;

  void captureInLambda()
  {
    this->init();

    StackArray< int, NDIM, PERMUTATION, INDEX_TYPE, CAPACITY > const & array = this->m_array;
    forall< POLICY >( 10, [array] LVARRAY_HOST_DEVICE ( int const )
        {
          forValuesInSliceWithIndices( array.toSlice(), [] ( int & value, auto const ... indices )
      {
        INDEX_TYPE const index = toLinearIndex( indices ... );
        PORTABLE_EXPECT_EQ( value, index );
      } );
        } );
  }

  static void createInLambda()
  {
    INDEX_TYPE const capacity = CAPACITY;
    forall< POLICY >( 10, [capacity] LVARRAY_HOST_DEVICE ( INDEX_TYPE )
        {
          StackArray< int, NDIM, PERMUTATION, INDEX_TYPE, CAPACITY > const array;
          PORTABLE_EXPECT_EQ( array.size(), 0 );
          PORTABLE_EXPECT_EQ( array.capacity(), capacity );
        } );
  }

  static void resizeInLambda()
  {
    INDEX_TYPE dims[ NDIM ];
    for( int i = 0; i < NDIM; ++i )
    { dims[ i ] = 8; }

    INDEX_TYPE const capacity = CAPACITY;
    forall< POLICY >( 10, [dims, capacity] LVARRAY_HOST_DEVICE ( int )
        {
          StackArray< int, NDIM, PERMUTATION, INDEX_TYPE, CAPACITY > array;
          PORTABLE_EXPECT_EQ( array.size(), 0 );
          PORTABLE_EXPECT_EQ( array.capacity(), capacity );

          array.resize( NDIM, dims );

          for( int i = 0; i < NDIM; ++i )
          { PORTABLE_EXPECT_EQ( array.size( i ), 8 ); }

          PORTABLE_EXPECT_EQ( array.size(), array.capacity() );
        } );
  }

  /// This needs to use the parallelDevice policy because you can't nest host-device lambdas.
  static void resizeMultipleInLambda()
  {
    INDEX_TYPE dims[ NDIM ];
    for( int i = 0; i < NDIM; ++i )
    { dims[ i ] = 8; }

    INDEX_TYPE const capacity = CAPACITY;
    forall< POLICY >( 10, [dims, capacity] LVARRAY_DEVICE ( int )
        {
          StackArray< int, NDIM, PERMUTATION, INDEX_TYPE, CAPACITY > array;
          PORTABLE_EXPECT_EQ( array.size(), 0 );
          PORTABLE_EXPECT_EQ( array.capacity(), capacity );

          array.resize( NDIM, dims );

          for( int i = 0; i < NDIM; ++i )
          { PORTABLE_EXPECT_EQ( array.size( i ), 8 ); }

          PORTABLE_EXPECT_EQ( array.size(), capacity );

          forValuesInSliceWithIndices( array.toSlice(), SetValue() );

          array.resize( 2 );

          PORTABLE_EXPECT_EQ( array.size( 0 ), 2 );
          for( int i = 1; i < NDIM; ++i )
          { PORTABLE_EXPECT_EQ( array.size( i ), 8 ); }

          PORTABLE_EXPECT_EQ( array.size(), array.capacity() / 4 );

          forValuesInSliceWithIndices( array.toSlice(), CheckValue() );
        } );
  }

  template< typename _PERMUTATION=PERMUTATION >
  static std::enable_if_t< typeManipulation::getDimension< _PERMUTATION > == 1 >
  sizedConstructorInLambda()
  {
    INDEX_TYPE const capacity = CAPACITY;
    forall< POLICY >( 10, [capacity] LVARRAY_DEVICE ( int )
        {
          StackArray< int, NDIM, PERMUTATION, INDEX_TYPE, CAPACITY > array( CAPACITY );
          PORTABLE_EXPECT_EQ( array.capacity(), capacity );
          PORTABLE_EXPECT_EQ( array.size(), capacity );
          PORTABLE_EXPECT_EQ( array.size( 0 ), capacity );
        } );
  }

  template< typename _PERMUTATION=PERMUTATION >
  static std::enable_if_t< typeManipulation::getDimension< _PERMUTATION > == 2 >
  sizedConstructorInLambda()
  {
    INDEX_TYPE const capacity = CAPACITY;
    int const size = 8;
    forall< POLICY >( 10, [capacity, size] LVARRAY_DEVICE ( int )
        {
          StackArray< int, NDIM, PERMUTATION, INDEX_TYPE, CAPACITY > array( size - 1, size );
          PORTABLE_EXPECT_EQ( array.capacity(), capacity );
          PORTABLE_EXPECT_EQ( array.size(), ( size - 1 ) * size );
          PORTABLE_EXPECT_EQ( array.size( 0 ), size - 1 );
          PORTABLE_EXPECT_EQ( array.size( 1 ), size );
        } );
  }

  template< typename _PERMUTATION=PERMUTATION >
  static std::enable_if_t< typeManipulation::getDimension< _PERMUTATION > == 3 >
  sizedConstructorInLambda()
  {
    INDEX_TYPE const capacity = CAPACITY;
    int const size = 8;
    forall< POLICY >( 10, [capacity, size] LVARRAY_DEVICE ( int )
        {
          StackArray< int, NDIM, PERMUTATION, INDEX_TYPE, CAPACITY > array( size - 2, size - 1, size );
          PORTABLE_EXPECT_EQ( array.capacity(), capacity );
          PORTABLE_EXPECT_EQ( array.size(), ( size - 2 ) * ( size - 1 ) * size );
          PORTABLE_EXPECT_EQ( array.size( 0 ), size - 2 );
          PORTABLE_EXPECT_EQ( array.size( 1 ), size - 1 );
          PORTABLE_EXPECT_EQ( array.size( 2 ), size );
        } );
  }

private:
  struct SetValue
  {
    template< typename ... INDICES >
    LVARRAY_HOST_DEVICE void operator() ( int & value, INDICES const ... indices )
    { value = toLinearIndex( indices ... ); }
  };

  struct CheckValue
  {
    template< typename ... INDICES >
    LVARRAY_HOST_DEVICE void operator() ( int & value, INDICES const ... indices )
    { PORTABLE_EXPECT_EQ( value, toLinearIndex( indices ... ) ); }
  };
};

using StackArrayCaptureTestTypes = ::testing::Types<
  std::pair< RAJA::PERM_I, serialPolicy >
  , std::pair< RAJA::PERM_IJ, serialPolicy >
  , std::pair< RAJA::PERM_JI, serialPolicy >
  , std::pair< RAJA::PERM_IJK, serialPolicy >
  , std::pair< RAJA::PERM_IKJ, serialPolicy >
  , std::pair< RAJA::PERM_JIK, serialPolicy >
  , std::pair< RAJA::PERM_JKI, serialPolicy >
  , std::pair< RAJA::PERM_KIJ, serialPolicy >
  , std::pair< RAJA::PERM_KJI, serialPolicy >

#if defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP)
  , std::pair< RAJA::PERM_I, parallelDevicePolicy< 32 > >
  , std::pair< RAJA::PERM_IJ, parallelDevicePolicy< 32 > >
  , std::pair< RAJA::PERM_JI, parallelDevicePolicy< 32 > >
  , std::pair< RAJA::PERM_IJK, parallelDevicePolicy< 32 > >
  , std::pair< RAJA::PERM_IKJ, parallelDevicePolicy< 32 > >
  , std::pair< RAJA::PERM_JIK, parallelDevicePolicy< 32 > >
  , std::pair< RAJA::PERM_JKI, parallelDevicePolicy< 32 > >
  , std::pair< RAJA::PERM_KIJ, parallelDevicePolicy< 32 > >
  , std::pair< RAJA::PERM_KJI, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( StackArrayCaptureTest, StackArrayCaptureTestTypes, );

TYPED_TEST( StackArrayCaptureTest, captureInLambda )
{
  this->captureInLambda();
}

TYPED_TEST( StackArrayCaptureTest, createInLambda )
{
  this->createInLambda();
}

TYPED_TEST( StackArrayCaptureTest, resizeInLambda )
{
  this->resizeInLambda();
}

TYPED_TEST( StackArrayCaptureTest, resizeMultipleInLambda )
{
  this->resizeMultipleInLambda();
}

TYPED_TEST( StackArrayCaptureTest, sizedConstructorInLambda )
{
  this->sizedConstructorInLambda();
}

} // namespace testing
} // namespace LvArray

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
