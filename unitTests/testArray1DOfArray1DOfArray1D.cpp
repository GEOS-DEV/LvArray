/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "Array.hpp"
#include "output.hpp"
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

template< typename U, typename T >
struct ToArray1D
{};

template< typename U, typename T, int NDIM, typename PERM, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
struct ToArray1D< U, Array< T, NDIM, PERM, INDEX_TYPE, BUFFER_TYPE > >
{
  using array = Array< U, 1, RAJA::PERM_I, INDEX_TYPE, BUFFER_TYPE >;
  using view = ArrayView< U, 1, 0, INDEX_TYPE, BUFFER_TYPE >;
};

template< typename ARRAY1D_POLICY_PAIR >
class Array1DOfArray1DOfArray1DTest : public ::testing::Test
{
public:
  using ARRAY1D = typename ARRAY1D_POLICY_PAIR::first_type;
  using POLICY = typename ARRAY1D_POLICY_PAIR::second_type;

  using T = typename ARRAY1D::ValueType;
  using IndexType = typename ARRAY1D::IndexType;

  template< typename U >
  using Array1D = typename ToArray1D< U, ARRAY1D >::array;

  template< typename U >
  using ArrayView1D = typename ToArray1D< U, ARRAY1D >::view;

  void modifyInKernel()
  {
    Array1D< Array1D< Array1D< T > > > array;
    init( array );

    ArrayView1D< ArrayView1D< ArrayView1D< T > const > const > const & view = array.toNestedView();
    forall< POLICY >( array.size(), [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < view[ i ].size(); ++j )
          {
            for( INDEX_TYPE k = 0; k < view[ i ][ j ].size(); ++k )
            {
              view[ i ][ j ][ k ] += view[ i ][ j ][ k ];
            }
          }
        } );

    forall< serialPolicy >( array.size(), [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < view[ i ].size(); ++j )
          {
            for( INDEX_TYPE k = 0; k < view[ i ][ j ].size(); ++k )
            {
              T val( MAX_SIZE * MAX_SIZE * i + MAX_SIZE * j + k );
              val += val;
              PORTABLE_EXPECT_EQ( view[ i ][ j ][ k ], val );
            }
          }
        } );
  }

  void readInKernel()
  {
    Array1D< Array1D< Array1D< T > > > array;
    init( array );

    // Create a shallow copy of the inner arrays that we can modify later.
    Array1D< Array1D< ArrayView1D< T > > > copy( array.size() );
    for( INDEX_TYPE i = 0; i < array.size(); ++i )
    {
      copy[ i ].resize( array[ i ].size() );
      for( INDEX_TYPE j = 0; j < array[ i ].size(); ++j )
      {
        copy[ i ][ j ] = array[ i ][ j ].toView();
      }
    }

    // Create a const view and launch a kernel checking the values.
    ArrayView1D< ArrayView1D< ArrayView1D< T const > const > const > const & viewConst = array.toNestedViewConst();
    forall< POLICY >( array.size(), [viewConst] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < viewConst[ i ].size(); ++j )
          {
            for( INDEX_TYPE k = 0; k < viewConst[ i ][ j ].size(); ++k )
            {
              PORTABLE_EXPECT_EQ( viewConst[ i ][ j ][ k ], T( MAX_SIZE * MAX_SIZE * i + MAX_SIZE * j + k ) );
            }
          }
        } );

    // Modify the copy. We can't directly modify the array since the inner arrays may be on device.
    for( INDEX_TYPE i = 0; i < copy.size(); ++i )
    {
      for( INDEX_TYPE j = 0; j < copy[ i ].size(); ++j )
      {
        for( INDEX_TYPE k = 0; k < copy[ i ][ j ].size(); ++k )
        {
          copy[ i ][ j ][ k ] += copy[ i ][ j ][ k ];
        }
      }
    }

    // Check that the modifications are present in the original array after launching a host kernel.
    forall< serialPolicy >( array.size(), [viewConst] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < viewConst[ i ].size(); ++j )
          {
            for( INDEX_TYPE k = 0; k < viewConst[ i ][ j ].size(); ++k )
            {
              T val( MAX_SIZE * MAX_SIZE * i + MAX_SIZE * j + k );
              val += val;
              PORTABLE_EXPECT_EQ( viewConst[ i ][ j ][ k ], val );
            }
          }
        } );
  }

  void move()
  {
    Array1D< Array1D< Array1D< T > > > array;
    init( array );

    array.move( RAJAHelper< POLICY >::space );
    Array1D< Array1D< T > > const * dataPointer = array.data();
    forall< POLICY >( array.size(), [dataPointer] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < dataPointer[ i ].size(); ++j )
          {
            for( INDEX_TYPE k = 0; k < dataPointer[ i ][ j ].size(); ++k )
            {
              dataPointer[ i ][ j ][ k ] += dataPointer[ i ][ j ][ k ];
            }
          }
        } );

    array.move( MemorySpace::host );
    dataPointer = array.data();
    forall< serialPolicy >( array.size(), [dataPointer] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < dataPointer[ i ].size(); ++j )
          {
            for( INDEX_TYPE k = 0; k < dataPointer[ i ][ j ].size(); ++k )
            {
              T val( MAX_SIZE * MAX_SIZE * i + MAX_SIZE * j + k );
              val += val;
              PORTABLE_EXPECT_EQ( dataPointer[ i ][ j ][ k ], val );
            }
          }
        } );
  }

  void moveNoTouch()
  {
    Array1D< Array1D< Array1D< T > > > array;
    init( array );

    // Create a shallow copy of the inner arrays that we can modify later.
    Array1D< Array1D< ArrayView1D< T > > > copy( array.size() );
    for( INDEX_TYPE i = 0; i < array.size(); ++i )
    {
      copy[ i ].resize( array[ i ].size() );
      for( INDEX_TYPE j = 0; j < array[ i ].size(); ++j )
      {
        copy[ i ][ j ] = array[ i ][ j ].toView();
      }
    }

    // Create a const view and launch a kernel checking the values.
    array.move( RAJAHelper< POLICY >::space, false );
    Array1D< Array1D< T > > const * dataPointer = array.data();
    forall< POLICY >( array.size(), [dataPointer] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < dataPointer[ i ].size(); ++j )
          {
            for( INDEX_TYPE k = 0; k < dataPointer[ i ][ j ].size(); ++k )
            {
              PORTABLE_EXPECT_EQ( dataPointer[ i ][ j ][ k ], T( MAX_SIZE * MAX_SIZE * i + MAX_SIZE * j + k ) );
            }
          }
        } );

    // Modify the copy. We can't directly modify the array since the inner arrays may be on device.
    for( INDEX_TYPE i = 0; i < copy.size(); ++i )
    {
      for( INDEX_TYPE j = 0; j < copy[ i ].size(); ++j )
      {
        for( INDEX_TYPE k = 0; k < copy[ i ][ j ].size(); ++k )
        {
          copy[ i ][ j ][ k ] += copy[ i ][ j ][ k ];
        }
      }
    }

    // Check that the modifications are present in the original array after launching a host kernel.
    array.move( MemorySpace::host, false );
    dataPointer = array.data();
    forall< serialPolicy >( array.size(), [dataPointer] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < dataPointer[ i ].size(); ++j )
          {
            for( INDEX_TYPE k = 0; k < dataPointer[ i ][ j ].size(); ++k )
            {
              T val( MAX_SIZE * MAX_SIZE * i + MAX_SIZE * j + k );
              val += val;
              PORTABLE_EXPECT_EQ( dataPointer[ i ][ j ][ k ], val );
            }
          }
        } );
  }

private:
  void init( Array1D< Array1D< Array1D< T > > > & array )
  {
    array.resize( randomInteger( 1, MAX_SIZE ) );

    for( INDEX_TYPE i = 0; i < array.size(); ++i )
    {
      array[ i ].resize( randomInteger( 0, MAX_SIZE ) );

      for( INDEX_TYPE j = 0; j < array[ i ].size(); ++j )
      {
        array[ i ][ j ].resize( randomInteger( 0, MAX_SIZE ) );
        for( INDEX_TYPE k = 0; k < array[ i ][ j ].size(); ++k )
        {
          array[ i ][ j ][ k ] = T( MAX_SIZE * MAX_SIZE * i + MAX_SIZE * j + k );
        }
      }
    }
  }

  static constexpr INDEX_TYPE MAX_SIZE = 20;
};

template< typename T, template< typename > class BUFFER_TYPE >
using Array1D = Array< T, 1, RAJA::PERM_I, INDEX_TYPE, BUFFER_TYPE >;

using Array1DOfArray1DOfArray1DTestTypes = ::testing::Types<
  std::pair< Array1D< int, MallocBuffer >, serialPolicy >
  , std::pair< Array1D< Tensor, MallocBuffer >, serialPolicy >
  , std::pair< Array1D< TestString, MallocBuffer >, serialPolicy >
#if defined(LVARRAY_USE_CHAI)
  , std::pair< Array1D< int, ChaiBuffer >, serialPolicy >
  , std::pair< Array1D< Tensor, ChaiBuffer >, serialPolicy >
  , std::pair< Array1D< TestString, ChaiBuffer >, serialPolicy >
#endif
#if ( defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP ) ) && defined(LVARRAY_USE_CHAI)
  , std::pair< Array1D< int, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< Array1D< Tensor, ChaiBuffer >, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( Array1DOfArray1DOfArray1DTest, Array1DOfArray1DOfArray1DTestTypes, );

TYPED_TEST( Array1DOfArray1DOfArray1DTest, modifyInKernel )
{
  this->modifyInKernel();
}

TYPED_TEST( Array1DOfArray1DOfArray1DTest, readInKernel )
{
  this->readInKernel();
}

TYPED_TEST( Array1DOfArray1DOfArray1DTest, move )
{
  this->move();
}

TYPED_TEST( Array1DOfArray1DOfArray1DTest, moveNoTouch )
{
  this->moveNoTouch();
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
