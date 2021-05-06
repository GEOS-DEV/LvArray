/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "Array.hpp"
#include "tensorOps.hpp"
#include "MallocBuffer.hpp"
#if defined(LVARRAY_USE_CHAI)
  #include "ChaiBuffer.hpp"
#endif

// TPL includes
#include <RAJA/RAJA.hpp>
#include <gtest/gtest.h>

// You can't define a nvcc extended host-device or device lambda in a TEST statement.
#define CUDA_TEST( X, Y )                 \
  static void cuda_test_ ## X ## _ ## Y();    \
  TEST( X, Y ) { cuda_test_ ## X ## _ ## Y(); } \
  static void cuda_test_ ## X ## _ ## Y()

// Sphinx start after inner product
TEST( tensorOps, AiBi )
{
  double const x[ 3 ] = { 0, 1, 2 };
  double const y[ 3 ] = { 1, 2, 3 };

  // Works with c-arrays
  EXPECT_EQ( LvArray::tensorOps::AiBi< 3 >( x, y ), 8 );

  LvArray::Array< double,
                  1,
                  camp::idx_seq< 0 >,
                  std::ptrdiff_t,
                  LvArray::MallocBuffer > xArray( 3 );

  xArray( 0 ) = 0;
  xArray( 1 ) = 1;
  xArray( 2 ) = 2;

  // Works with LvArray::Array.
  EXPECT_EQ( LvArray::tensorOps::AiBi< 3 >( xArray, y ), 8 );

  // Works with LvArray::ArrayView.
  EXPECT_EQ( LvArray::tensorOps::AiBi< 3 >( xArray.toView(), y ), 8 );
  EXPECT_EQ( LvArray::tensorOps::AiBi< 3 >( xArray.toViewConst(), y ), 8 );

  // Create a 2D array with fortran layout.
  LvArray::Array< double,
                  2,
                  camp::idx_seq< 1, 0 >,
                  std::ptrdiff_t,
                  LvArray::MallocBuffer > yArray( 1, 3 );

  yArray( 0, 0 ) = 1;
  yArray( 0, 1 ) = 2;
  yArray( 0, 2 ) = 3;

  // Works with non contiguous slices.
  EXPECT_EQ( LvArray::tensorOps::AiBi< 3 >( x, yArray[ 0 ] ), 8 );
  EXPECT_EQ( LvArray::tensorOps::AiBi< 3 >( xArray, yArray[ 0 ] ), 8 );
}
// Sphinx end before inner product

#if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
// Sphinx start after device
CUDA_TEST( tensorOps, device )
{
  // Create an array of 5 3x3 symmetric matrices.
  LvArray::Array< int,
                  2,
                  camp::idx_seq< 1, 0 >,
                  std::ptrdiff_t,
                  LvArray::ChaiBuffer > symmetricMatrices( 5, 6 );

  int offset = 0;
  for( int & value : symmetricMatrices )
  {
    value = offset++;
  }

  LvArray::ArrayView< int const,
                      2,
                      0,
                      std::ptrdiff_t,
                      LvArray::ChaiBuffer > symmetricMatricesView = symmetricMatrices.toViewConst();

  // The tensorOps methods work on device.
  RAJA::forall< RAJA::cuda_exec< 32 > >(
    RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, symmetricMatricesView.size( 0 ) ),
    [symmetricMatricesView] __device__ ( std::ptrdiff_t const i )
  {
    double result[ 3 ];
    float x[ 3 ] = { 1.3f, 2.2f, 5.3f };

    // You can mix value types.
    LvArray::tensorOps::Ri_eq_symAijBj< 3 >( result, symmetricMatricesView[ i ], x );

    LVARRAY_ERROR_IF_NE( result[ 0 ], x[ 0 ] * symmetricMatricesView( i, 0 ) +
                         x[ 1 ] * symmetricMatricesView( i, 5 ) +
                         x[ 2 ] * symmetricMatricesView( i, 4 ) );
  }
    );
}
// Sphinx end before device
#endif

// Sphinx start after bounds check
TEST( tensorOps, boundsCheck )
{
  double x[ 3 ] = { 0, 1, 2 };
  LvArray::tensorOps::normalize< 3 >( x );

  // This would fail to compile since x is not length 4.
  // LvArray::tensorOps::normalize< 4 >( x );

  LvArray::Array< double,
                  1,
                  camp::idx_seq< 0 >,
                  std::ptrdiff_t,
                  LvArray::MallocBuffer > xArray( 8 );

  xArray( 0 ) = 10;

#if defined(LVARRAY_BOUNDS_CHECK)
  // This will fail at runtime.
  EXPECT_DEATH_IF_SUPPORTED( LvArray::tensorOps::normalize< 3 >( xArray ), "" );
#endif

  int matrix[ 2 ][ 2 ] = { { 1, 0 }, { 0, 1 } };
  LvArray::tensorOps::normalize< 2 >( matrix[ 0 ] );

  // This would fail to compile since normalize expects a vector
  // LvArray::tensorOps::normalize< 4 >( matrix );

  LvArray::Array< double,
                  2,
                  camp::idx_seq< 0, 1 >,
                  std::ptrdiff_t,
                  LvArray::MallocBuffer > matrixArray( 2, 2 );

  matrixArray( 0, 0 ) = 1;
  matrixArray( 0, 1 ) = 1;

  LvArray::tensorOps::normalize< 2 >( matrixArray[ 0 ] );

  // This will also fail to compile
  // LvArray::tensorOps::normalize< 2 >( matrixArray );
}
// Sphinx end before bounds check

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
