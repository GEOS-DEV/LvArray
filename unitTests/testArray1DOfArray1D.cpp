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
#include "Array.hpp"
#include "streamIO.hpp"
#include "testUtils.hpp"
#include "MallocBuffer.hpp"

// TPL includes
#include <gtest/gtest.h>

// System includes
#include <random>

namespace LvArray
{
namespace testing
{

using INDEX_TYPE = std::ptrdiff_t;

inline INDEX_TYPE randomInteger( INDEX_TYPE const min, INDEX_TYPE const max )
{
  static std::mt19937_64 gen;
  return std::uniform_int_distribution< INDEX_TYPE >( min, max )( gen );
}

template< typename ARRAY1D_POLICY_PAIR >
class Array1DOfArray1DTest : public ::testing::Test
{
public:
  using ARRAY1D = typename ARRAY1D_POLICY_PAIR::first_type;
  using POLICY = typename ARRAY1D_POLICY_PAIR::second_type;

  using T = typename ARRAY1D::value_type;

  template< typename U >
  using Array1D = typename ArrayConverter< ARRAY1D >::template Array< U, 1, RAJA::PERM_I >;

  template< typename U >
  using ArrayView1D = typename ArrayConverter< ARRAY1D >::template ArrayView< U, 1, 0 >;

  void modifyInKernel()
  {
    Array1D< Array1D< T > > array;
    init( array );

    ArrayView1D< ArrayView1D< T > const > const & view = array.toView();
    forall< POLICY >( array.size(), [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < view[ i ].size(); ++j )
          {
            view[ i ][ j ] += view[ i ][ j ];
          }
        } );

    forall< serialPolicy >( array.size(), [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < view[ i ].size(); ++j )
          {
            T val( MAX_SIZE * i + j );
            val += val;
            PORTABLE_EXPECT_EQ( view[ i ][ j ], val );
          }
        } );
  }

  void readInKernel()
  {
    Array1D< Array1D< T > > array;
    init( array );

    // Create a shallow copy of the inner arrays that we can modify later.
    Array1D< ArrayView1D< T > > copy( array.size() );
    for( INDEX_TYPE i = 0; i < array.size(); ++i )
    { copy[ i ] = array[ i ].toView(); }

    // Create a const view and launch a kernel checking the values.
    ArrayView1D< ArrayView1D< T const > const > const & viewConst = array.toViewConst();
    forall< POLICY >( array.size(), [viewConst] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < viewConst[ i ].size(); ++j )
          {
            PORTABLE_EXPECT_EQ( viewConst[ i ][ j ], T( MAX_SIZE * i + j ) );
          }
        } );

    // Modify the copy. We can't directly modify the array since the inner arrays may be on device.
    for( INDEX_TYPE i = 0; i < copy.size(); ++i )
    {
      for( INDEX_TYPE j = 0; j < copy[ i ].size(); ++j )
      {
        copy[ i ][ j ] += copy[ i ][ j ];
      }
    }

    // Check that the modifications are present in the original array after launching a host kernel.
    forall< serialPolicy >( array.size(), [viewConst] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < viewConst[ i ].size(); ++j )
          {
            T val( MAX_SIZE * i + j );
            val += val;
            PORTABLE_EXPECT_EQ( viewConst[ i ][ j ], val );
          }
        } );
  }

  void move()
  {
    Array1D< Array1D< T > > array;
    init( array );

    array.move( RAJAHelper< POLICY >::space );
    Array1D< T > const * dataPointer = array.data();
    forall< POLICY >( array.size(), [dataPointer] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < dataPointer[ i ].size(); ++j )
          {
            dataPointer[ i ][ j ] += dataPointer[ i ][ j ];
          }
        } );

    array.move( MemorySpace::CPU );
    dataPointer = array.data();
    forall< serialPolicy >( array.size(), [dataPointer] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < dataPointer[ i ].size(); ++j )
          {
            T val( MAX_SIZE * i + j );
            val += val;
            PORTABLE_EXPECT_EQ( dataPointer[ i ][ j ], val );
          }
        } );
  }

  void moveNoTouch()
  {
    Array1D< Array1D< T > > array;
    init( array );

    // Create a shallow copy of the inner arrays that we can modify later.
    Array1D< ArrayView1D< T > > copy( array.size() );
    for( INDEX_TYPE i = 0; i < array.size(); ++i )
    { copy[ i ] = array[ i ].toView(); }

    // Create a const view and launch a kernel checking the values.
    array.move( RAJAHelper< POLICY >::space, false );
    Array1D< T > const * dataPointer = array.data();
    forall< POLICY >( array.size(), [dataPointer] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < dataPointer[ i ].size(); ++j )
          {
            PORTABLE_EXPECT_EQ( dataPointer[ i ][ j ], T( MAX_SIZE * i + j ) );
          }
        } );

    // Modify the copy. We can't directly modify the array since the inner arrays may be on device.
    for( INDEX_TYPE i = 0; i < copy.size(); ++i )
    {
      for( INDEX_TYPE j = 0; j < copy[ i ].size(); ++j )
      {
        copy[ i ][ j ] += copy[ i ][ j ];
      }
    }

    // Check that the modifications are present in the original array after launching a host kernel.
    array.move( MemorySpace::CPU );
    dataPointer = array.data();
    forall< serialPolicy >( array.size(), [dataPointer] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < dataPointer[ i ].size(); ++j )
          {
            T val( MAX_SIZE * i + j );
            val += val;
            PORTABLE_EXPECT_EQ( dataPointer[ i ][ j ], val );
          }
        } );
  }

  void emptyMove()
  {
    Array1D< Array1D< T > > array( 10 );
    array.move( RAJAHelper< POLICY >::space );
    array.move( MemorySpace::CPU );
  }

private:
  void init( Array1D< Array1D< T > > & array )
  {
    array.resize( randomInteger( 1, MAX_SIZE ) );

    for( INDEX_TYPE i = 0; i < array.size(); ++i )
    {
      array[ i ].resize( randomInteger( 0, MAX_SIZE ) );

      for( INDEX_TYPE j = 0; j < array[ i ].size(); ++j )
      {
        array[ i ][ j ] = T( MAX_SIZE * i + j );
      }
    }
  }

  static constexpr INDEX_TYPE MAX_SIZE = 20;
};

template< typename T, template< typename > class BUFFER_TYPE >
using Array1D = Array< T, 1, RAJA::PERM_I, INDEX_TYPE, BUFFER_TYPE >;

using Array1DOfArray1DTestTypes = ::testing::Types<
  std::pair< Array1D< int, MallocBuffer >, serialPolicy >
  , std::pair< Array1D< Tensor, MallocBuffer >, serialPolicy >
  , std::pair< Array1D< TestString, MallocBuffer >, serialPolicy >
#if defined(USE_CHAI)
  , std::pair< Array1D< int, NewChaiBuffer >, serialPolicy >
  , std::pair< Array1D< Tensor, NewChaiBuffer >, serialPolicy >
  , std::pair< Array1D< TestString, NewChaiBuffer >, serialPolicy >
#endif
#if defined(USE_CUDA) && defined(USE_CHAI)
  , std::pair< Array1D< int, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array1D< Tensor, NewChaiBuffer >, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( Array1DOfArray1DTest, Array1DOfArray1DTestTypes, );

TYPED_TEST( Array1DOfArray1DTest, modifyInKernel )
{
  this->modifyInKernel();
}

TYPED_TEST( Array1DOfArray1DTest, readInKernel )
{
  this->readInKernel();
}

TYPED_TEST( Array1DOfArray1DTest, move )
{
  this->move();
}

TYPED_TEST( Array1DOfArray1DTest, moveNoTouch )
{
  this->moveNoTouch();
}

TYPED_TEST( Array1DOfArray1DTest, emptyMove )
{
  this->emptyMove();
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
