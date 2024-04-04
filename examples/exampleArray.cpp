/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "Array.hpp"
#include "MallocBuffer.hpp"
#if defined(LVARRAY_USE_CHAI)
  #include "ChaiBuffer.hpp"
#endif

// TPL includes
#include <RAJA/RAJA.hpp>
#include <gtest/gtest.h>

// system includes
#include <string>

// You can't define a nvcc extended host-device or device lambda in a TEST statement.
#define CUDA_TEST( X, Y )                 \
  static void cuda_test_ ## X ## _ ## Y();    \
  TEST( X, Y ) { cuda_test_ ## X ## _ ## Y(); } \
  static void cuda_test_ ## X ## _ ## Y()

// Sphinx start after constructors
TEST( Array, constructors )
{
  {
    // Create an empty 2D array of integers.
    LvArray::Array< int,
                    2,
                    camp::idx_seq< 0, 1 >,
                    std::ptrdiff_t,
                    LvArray::MallocBuffer > array;
    EXPECT_TRUE( array.empty() );
    EXPECT_EQ( array.size(), 0 );
    EXPECT_EQ( array.size( 0 ), 0 );
    EXPECT_EQ( array.size( 1 ), 0 );
  }

  {
    // Create a 3D array of std::string of size 3 x 4 x 5.
    LvArray::Array< std::string,
                    3,
                    camp::idx_seq< 0, 1, 2 >,
                    std::ptrdiff_t,
                    LvArray::MallocBuffer > array( 3, 4, 5 );
    EXPECT_FALSE( array.empty() );
    EXPECT_EQ( array.size(), 3 * 4 * 5 );
    EXPECT_EQ( array.size( 0 ), 3 );
    EXPECT_EQ( array.size( 1 ), 4 );
    EXPECT_EQ( array.size( 2 ), 5 );

    // The values are default initialized.
    std::string const * const values = array.data();
    for( std::ptrdiff_t i = 0; i < array.size(); ++i )
    {
      EXPECT_EQ( values[ i ], std::string() );
    }
  }
}
// Sphinx end before constructors

// Sphinx start after accessors
TEST( Array, accessors )
{
  // Create a 2D array of integers.
  LvArray::Array< int,
                  2,
                  camp::idx_seq< 0, 1 >,
                  std::ptrdiff_t,
                  LvArray::MallocBuffer > array( 3, 4 );

  // Access using operator().
  for( std::ptrdiff_t i = 0; i < array.size( 0 ); ++i )
  {
    for( std::ptrdiff_t j = 0; j < array.size( 1 ); ++j )
    {
      array( i, j ) = array.size( 1 ) * i + j;
    }
  }

  // Access using operator[].
  for( std::ptrdiff_t i = 0; i < array.size( 0 ); ++i )
  {
    for( std::ptrdiff_t j = 0; j < array.size( 1 ); ++j )
    {
      EXPECT_EQ( array[ i ][ j ], array.size( 1 ) * i + j );
    }
  }
}
// Sphinx end before accessors

// Sphinx start after permutations
TEST( Array, permutations )
{
  {
    // Create a 3D array of doubles in the standard layout.
    LvArray::Array< int,
                    3,
                    camp::idx_seq< 0, 1, 2 >,
                    std::ptrdiff_t,
                    LvArray::MallocBuffer > array( 3, 4, 5 );

    // Index 0 has the largest stride while index 2 has unit stride.
    EXPECT_EQ( array.strides()[ 0 ], array.size( 2 ) * array.size( 1 ) );
    EXPECT_EQ( array.strides()[ 1 ], array.size( 2 ) );
    EXPECT_EQ( array.strides()[ 2 ], 1 );

    int const * const pointer = array.data();
    for( std::ptrdiff_t i = 0; i < array.size( 0 ); ++i )
    {
      for( std::ptrdiff_t j = 0; j < array.size( 1 ); ++j )
      {
        for( std::ptrdiff_t k = 0; k < array.size( 2 ); ++k )
        {
          std::ptrdiff_t const offset = array.size( 2 ) * array.size( 1 ) * i +
                                        array.size( 2 ) * j + k;
          EXPECT_EQ( &array( i, j, k ), pointer + offset );
        }
      }
    }
  }

  {
    // Create a 3D array of doubles in a flipped layout.
    LvArray::Array< int,
                    3,
                    camp::idx_seq< 2, 1, 0 >,
                    std::ptrdiff_t,
                    LvArray::MallocBuffer > array( 3, 4, 5 );

    // Index 0 has the unit stride while index 2 has the largest stride.
    EXPECT_EQ( array.strides()[ 0 ], 1 );
    EXPECT_EQ( array.strides()[ 1 ], array.size( 0 ) );
    EXPECT_EQ( array.strides()[ 2 ], array.size( 0 ) * array.size( 1 ) );

    int const * const pointer = array.data();
    for( std::ptrdiff_t i = 0; i < array.size( 0 ); ++i )
    {
      for( std::ptrdiff_t j = 0; j < array.size( 1 ); ++j )
      {
        for( std::ptrdiff_t k = 0; k < array.size( 2 ); ++k )
        {
          std::ptrdiff_t const offset = i + array.size( 0 ) * j +
                                        array.size( 0 ) * array.size( 1 ) * k;
          EXPECT_EQ( &array[ i ][ j ][ k ], pointer + offset );
        }
      }
    }
  }
}
// Sphinx end before permutations

// Sphinx start after resize
TEST( Array, resize )
{
  LvArray::Array< int,
                  3,
                  camp::idx_seq< 0, 1, 2 >,
                  std::ptrdiff_t,
                  LvArray::MallocBuffer > array;

  // Resize using a pointer
  std::ptrdiff_t const sizes[ 3 ] = { 2, 5, 6 };
  array.resize( 3, sizes );
  EXPECT_EQ( array.size(), 2 * 5 * 6 );
  EXPECT_EQ( array.size( 0 ), 2 );
  EXPECT_EQ( array.size( 1 ), 5 );
  EXPECT_EQ( array.size( 2 ), 6 );

  // Resizing using a variadic parameter pack.
  array.resize( 3, 4, 2 );
  EXPECT_EQ( array.size(), 3 * 4 * 2 );
  EXPECT_EQ( array.size( 0 ), 3 );
  EXPECT_EQ( array.size( 1 ), 4 );
  EXPECT_EQ( array.size( 2 ), 2 );

  // Resize the second and third dimensions
  array.resizeDimension< 1, 2 >( 3, 6 );
  EXPECT_EQ( array.size(), 3 * 3 * 6 );
  EXPECT_EQ( array.size( 0 ), 3 );
  EXPECT_EQ( array.size( 1 ), 3 );
  EXPECT_EQ( array.size( 2 ), 6 );
}
// Sphinx end before resize

// Sphinx start after resizeSingleDimension
TEST( Array, resizeSingleDimension )
{
  LvArray::Array< int,
                  2,
                  camp::idx_seq< 1, 0 >,
                  std::ptrdiff_t,
                  LvArray::MallocBuffer > array( 5, 6 );

  for( std::ptrdiff_t i = 0; i < array.size( 0 ); ++i )
  {
    for( std::ptrdiff_t j = 0; j < array.size( 1 ); ++j )
    {
      array( i, j ) = 6 * i + j;
    }
  }

  // Grow the first dimension from 5 to 8.
  array.resize( 8 );
  for( std::ptrdiff_t i = 0; i < array.size( 0 ); ++i )
  {
    for( std::ptrdiff_t j = 0; j < array.size( 1 ); ++j )
    {
      if( i < 5 )
      {
        EXPECT_EQ( array( i, j ), 6 * i + j );
      }
      else
      {
        EXPECT_EQ( array( i, j ), 0 );
      }
    }
  }
}
// Sphinx end before resizeSingleDimension

// Sphinx start after arrayView
TEST( Array, arrayView )
{
  LvArray::Array< int,
                  2,
                  camp::idx_seq< 1, 0 >,
                  std::ptrdiff_t,
                  LvArray::MallocBuffer > array( 5, 6 );

  // Create a view.
  LvArray::ArrayView< int,
                      2,
                      0,
                      std::ptrdiff_t,
                      LvArray::MallocBuffer > const view = array;
  EXPECT_EQ( view.data(), array.data() );

  // Create a view with const values.
  LvArray::ArrayView< int const,
                      2,
                      0,
                      std::ptrdiff_t,
                      LvArray::MallocBuffer > const viewConst = array.toViewConst();
  EXPECT_EQ( viewConst.data(), array.data() );

  // Copy a view.
  LvArray::ArrayView< int,
                      2,
                      0,
                      std::ptrdiff_t,
                      LvArray::MallocBuffer > const viewCopy = view;
  EXPECT_EQ( viewCopy.data(), array.data() );
}
// Sphinx end before arrayView

// Sphinx start after arraySlice
TEST( Array, arraySlice )
{
  {
    LvArray::Array< int,
                    2,
                    camp::idx_seq< 0, 1 >,
                    std::ptrdiff_t,
                    LvArray::MallocBuffer > array( 5, 6 );

    // The unit stride dimension of array is 1 so when we slice off
    // the first dimension the unit stride dimension of the slice is 0.
    LvArray::ArraySlice< int,
                         1,
                         0,
                         std::ptrdiff_t > const slice = array[ 2 ];
    EXPECT_TRUE( slice.isContiguous() );
    EXPECT_EQ( slice.size(), 6 );
    EXPECT_EQ( slice.size( 0 ), 6 );
    slice[ 3 ] = 1;
  }

  {
    LvArray::Array< int,
                    3,
                    camp::idx_seq< 2, 1, 0 >,
                    std::ptrdiff_t,
                    LvArray::MallocBuffer > array( 3, 5, 6 );

    // The unit stride dimension of array is 0 so when we slice off
    // the first dimension the unit stride dimension of the slice is -1.
    LvArray::ArraySlice< int,
                         2,
                         -1,
                         std::ptrdiff_t > const slice = array[ 2 ];
    EXPECT_FALSE( slice.isContiguous() );
    EXPECT_EQ( slice.size(), 5 * 6 );
    EXPECT_EQ( slice.size( 0 ), 5 );
    EXPECT_EQ( slice.size( 1 ), 6 );
    slice( 3, 4 ) = 1;
  }
}
// Sphinx end before arraySlice

#if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
// Sphinx start after chaiBuffer
CUDA_TEST( Array, chaiBuffer )
{
  LvArray::Array< int,
                  2,
                  camp::idx_seq< 1, 0 >,
                  std::ptrdiff_t,
                  LvArray::ChaiBuffer > array( 5, 6 );

  // Move the array to the device.
  array.move( LvArray::MemorySpace::cuda );
  int * const devicePointer = array.data();

  RAJA::forall< RAJA::cuda_exec< 32 > >(
    RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, array.size() ),
    [devicePointer] __device__ ( std::ptrdiff_t const i )
  {
    devicePointer[ i ] = i;
  }
    );

  LvArray::ArrayView< int,
                      2,
                      0,
                      std::ptrdiff_t,
                      LvArray::ChaiBuffer > const & view = array;

  // Capture the view in a host kernel which moves the data back to the host.
  RAJA::forall< RAJA::seq_exec >(
    RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, view.size() ),
    [view] ( std::ptrdiff_t const i )
  {
    EXPECT_EQ( view.data()[ i ], i );
  }
    );
}
// Sphinx end before chaiBuffer

// Sphinx start after setName
TEST( Array, setName )
{
  LvArray::Array< int,
                  2,
                  camp::idx_seq< 1, 0 >,
                  std::ptrdiff_t,
                  LvArray::ChaiBuffer > array( 1024, 1024 );

  // Move the array to the device.
  array.move( LvArray::MemorySpace::cuda );

  // Provide a name and move the array to the host.
  array.setName( "my_array" );
  array.move( LvArray::MemorySpace::host );
}
// Sphinx end before setName

#endif

// Sphinx start after sum int
template< int NDIM, int USD >
int sum( LvArray::ArraySlice< int const, NDIM, USD, std::ptrdiff_t > const slice )
{
  int value = 0;
  for( int const val : slice )
  {
    value += val;
  }

  return value;
}
// Sphinx end before sum int

// Sphinx start after sum double
template< int NDIM, int USD >
double sum( LvArray::ArraySlice< double const, NDIM, USD, std::ptrdiff_t > const slice )
{
  double value = 0;
  LvArray::forValuesInSlice( slice, [&value] ( double const val )
  {
    value += val;
  } );

  return value;
}
// Sphinx end before sum double

// Sphinx start after copy
template< int NDIM, int DST_USD, int SRC_USD >
void copy( LvArray::ArraySlice< int, NDIM, DST_USD, std::ptrdiff_t > const dst,
           LvArray::ArraySlice< int const, NDIM, SRC_USD, std::ptrdiff_t > const src )
{
  for( int dim = 0; dim < NDIM; ++dim )
  {
    LVARRAY_ERROR_IF_NE( dst.size( dim ), src.size( dim ) );
  }

  LvArray::forValuesInSliceWithIndices( dst,
                                        [src] ( int & val, auto const ... indices )
  {
    val = src( indices ... );
  }
                                        );
}
// Sphinx end before copy

// Sphinx start after bounds check
TEST( Array, boundsCheck )
{
#if defined(ARRAY_USE_BOUNDS_CHECK)
  LvArray::Array< int, 3, camp::idx_seq< 0, 1, 2 >, std::ptrdiff_t, LvArray::MallocBuffer > x( 3, 4, 5 );

  // Out of bounds access aborts the program.
  EXPECT_DEATH_IF_SUPPORTED( x( 2, 3, 4 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( x( -1, 4, 6 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( x[ 0 ][ 10 ][ 2 ], "" );

  // Out of bounds emplace
  LvArray::Array< int, 1, camp::idx_seq< 0 >, std::ptrdiff_t, LvArray::MallocBuffer > x( 10 );
  EXPECT_DEATH_IF_SUPPORTED( x.emplace( -1, 5 ) );
#endif
}
// Sphinx end before bounds check

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
