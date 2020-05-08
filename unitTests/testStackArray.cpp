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

// Source includes
#include "StackBuffer.hpp"
#include "Array.hpp"
#include "testUtils.hpp"

// TPL includes
#include <gtest/gtest.h>

// System includes
#include <cmath>

namespace LvArray
{
namespace testing
{

template< typename T, int NDIM, int MAX_SIZE >
using stackArray = StackArray< T, NDIM, camp::make_idx_seq_t< NDIM >, int, MAX_SIZE >;

LVARRAY_HOST_DEVICE
int toLinearIndex( int const i )
{ return i; }

LVARRAY_HOST_DEVICE
int toLinearIndex( int const i, int const j )
{ return i + 10 * j; }

LVARRAY_HOST_DEVICE
int toLinearIndex( int const i, int const j, int const k )
{ return i + 10 * j + 100 * k; }

constexpr int pow( int const base, int const exponent )
{ return exponent == 0 ? 1 : base * pow( base, exponent - 1 ); }

template< typename NDIM_T >
class StackArrayTest : public ::testing::Test
{
public:
  static constexpr int NDIM = NDIM_T::value;

  void resize()
  {

    init();

    m_array.resize( 4 );

    forValuesInSliceWithIndices( m_array.toSlice(), []( int const value, auto const ... indices )
    {
      EXPECT_EQ( value, toLinearIndex( indices ... ) );
    } );

    int dims[ NDIM ];
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
    int dims[ NDIM ];

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

  static constexpr int CAPACITY = pow( 8, NDIM );
  stackArray< int, NDIM, CAPACITY > m_array;
};

using StackArrayTestTypes = ::testing::Types<
  std::integral_constant< int, 1 >
  , std::integral_constant< int, 2 >
  , std::integral_constant< int, 3 >
  >;

TYPED_TEST_SUITE( StackArrayTest, StackArrayTestTypes, );

TYPED_TEST( StackArrayTest, resize )
{
  this->resize();
}


template< typename NDIM_POLICY_PAIR >
class StackArrayCaptureTest : public StackArrayTest< typename NDIM_POLICY_PAIR::first_type >
{
public:
  using StackArrayTest< typename NDIM_POLICY_PAIR::first_type >::NDIM;
  using StackArrayTest< typename NDIM_POLICY_PAIR::first_type >::CAPACITY;
  using POLICY = typename NDIM_POLICY_PAIR::second_type;

  void captureInLambda()
  {
    this->init();

    stackArray< int, NDIM, CAPACITY > const & array = this->m_array;
    forall< POLICY >( 10, [array] LVARRAY_HOST_DEVICE ( int const )
        {
          forValuesInSliceWithIndices( array.toSlice(), [] ( int & value, auto const ... indices )
      {
        int const index = toLinearIndex( indices ... );
        PORTABLE_EXPECT_EQ( value, index );
      } );
        } );
  }

  static void createInLambda()
  {
    int const capacity = CAPACITY;
    forall< POLICY >( 10, [capacity] LVARRAY_HOST_DEVICE ( int )
        {
          stackArray< int, NDIM, CAPACITY > const array;
          PORTABLE_EXPECT_EQ( array.size(), 0 );
          PORTABLE_EXPECT_EQ( array.capacity(), capacity );

        } );
  }

  static void resizeInLambda()
  {
    int dims[ NDIM ];
    for( int i = 0; i < NDIM; ++i )
    { dims[ i ] = 8; }

    int const capacity = CAPACITY;
    forall< POLICY >( 10, [dims, capacity] LVARRAY_HOST_DEVICE ( int )
        {
          stackArray< int, NDIM, CAPACITY > array;
          PORTABLE_EXPECT_EQ( array.size(), 0 );
          PORTABLE_EXPECT_EQ( array.capacity(), capacity );

          array.resize( NDIM, dims );

          for( int i = 0; i < NDIM; ++i )
          { PORTABLE_EXPECT_EQ( array.size( i ), 8 ); }

          PORTABLE_EXPECT_EQ( array.size(), array.capacity() );
        } );
  }

#if defined(USE_CUDA)
  /// This needs to use the parallelDevice policy because you can't next host-device lambdas.
  static void resizeMultipleInLambda()
  {
    int dims[ NDIM ];
    for( int i = 0; i < NDIM; ++i )
    { dims[ i ] = 8; }

    int const capacity = CAPACITY;
    forall< parallelDevicePolicy >( 10, [dims, capacity] LVARRAY_DEVICE ( int )
        {
          stackArray< int, NDIM, CAPACITY > array;
          PORTABLE_EXPECT_EQ( array.size(), 0 );
          PORTABLE_EXPECT_EQ( array.capacity(), capacity );

          array.resize( NDIM, dims );

          for( int i = 0; i < NDIM; ++i )
          { PORTABLE_EXPECT_EQ( array.size( i ), 8 ); }

          PORTABLE_EXPECT_EQ( array.size(), capacity );

          forValuesInSliceWithIndices( array.toSlice(), [] LVARRAY_DEVICE ( int & value, auto const ... indices )
          {
            value = toLinearIndex( indices ... );
          } );

          array.resize( 2 );

          PORTABLE_EXPECT_EQ( array.size( 0 ), 2 );
          for( int i = 1; i < NDIM; ++i )
          { PORTABLE_EXPECT_EQ( array.size( i ), 8 ); }

          PORTABLE_EXPECT_EQ( array.size(), array.capacity() / 4 );

          forValuesInSliceWithIndices( array.toSlice(), [] LVARRAY_DEVICE ( int & value, auto const ... indices )
          {
            PORTABLE_EXPECT_EQ( value, toLinearIndex( indices ... ) );
          } );
        } );
  }
#endif
};

#if defined(USE_CUDA)
void sizedConstructorInLambda1D()
{
  int const SIZE = 8;
  constexpr int CAPACITY = SIZE;
  forall< parallelDevicePolicy >( 10, [] LVARRAY_DEVICE ( int )
      {
        stackArray< int, 1, CAPACITY > array( SIZE );
        PORTABLE_EXPECT_EQ( array.capacity(), CAPACITY );
        PORTABLE_EXPECT_EQ( array.size(), SIZE );
        PORTABLE_EXPECT_EQ( array.size( 0 ), SIZE );
      } );
}

void sizedConstructorInLambda2D()
{
  constexpr int SIZE = 8;
  constexpr int CAPACITY = SIZE * SIZE;
  forall< parallelDevicePolicy >( 10, [] LVARRAY_DEVICE ( int )
      {
        stackArray< int, 2, CAPACITY > array( SIZE - 1, SIZE );
        PORTABLE_EXPECT_EQ( array.capacity(), CAPACITY );
        PORTABLE_EXPECT_EQ( array.size(), ( SIZE - 1 ) * SIZE );
        PORTABLE_EXPECT_EQ( array.size( 0 ), SIZE - 1 );
        PORTABLE_EXPECT_EQ( array.size( 1 ), SIZE );
      } );
}

void sizedConstructorInLambda3D()
{
  constexpr int SIZE = 8;
  constexpr int CAPACITY = SIZE * SIZE * SIZE;
  forall< parallelDevicePolicy >( 10, [] LVARRAY_DEVICE ( int )
      {
        stackArray< int, 3, CAPACITY > array( SIZE - 2, SIZE - 1, SIZE );
        PORTABLE_EXPECT_EQ( array.capacity(), CAPACITY );
        PORTABLE_EXPECT_EQ( array.size(), ( SIZE - 2 ) * ( SIZE - 1 ) * SIZE );
        PORTABLE_EXPECT_EQ( array.size( 0 ), SIZE - 2 );
        PORTABLE_EXPECT_EQ( array.size( 1 ), SIZE - 1 );
        PORTABLE_EXPECT_EQ( array.size( 2 ), SIZE );
      } );
}
#endif

using StackArrayCaptureTestTypes = ::testing::Types<
  std::pair< std::integral_constant< int, 1 >, serialPolicy >
  , std::pair< std::integral_constant< int, 2 >, serialPolicy >
  , std::pair< std::integral_constant< int, 3 >, serialPolicy >
#if defined(USE_CUDA)
  , std::pair< std::integral_constant< int, 1 >, parallelDevicePolicy >
  , std::pair< std::integral_constant< int, 2 >, parallelDevicePolicy >
  , std::pair< std::integral_constant< int, 3 >, parallelDevicePolicy >
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

#ifdef USE_CUDA
TEST( StackArrayCaptureTest, resizeMultipleInLambda1D )
{
  StackArrayCaptureTest< std::pair< std::integral_constant< int, 1 >, parallelDevicePolicy > >::resizeMultipleInLambda();
}

TEST( StackArrayCaptureTest, resizeMultipleInLambda2D )
{
  StackArrayCaptureTest< std::pair< std::integral_constant< int, 2 >, parallelDevicePolicy > >::resizeMultipleInLambda();
}

TEST( StackArrayCaptureTest, resizeMultipleInLambda3D )
{
  StackArrayCaptureTest< std::pair< std::integral_constant< int, 3 >, parallelDevicePolicy > >::resizeMultipleInLambda();
}

TEST( StackArrayCaptureTest, sizedConstructorInLambda1D )
{
  sizedConstructorInLambda1D();
}

TEST( StackArrayCaptureTest, sizedConstructorInLambda2D )
{
  sizedConstructorInLambda2D();
}

TEST( StackArrayCaptureTest, sizedConstructorInLambda3D )
{
  sizedConstructorInLambda3D();
}
#endif

} // namespace testing
} // namespace LvArray

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
