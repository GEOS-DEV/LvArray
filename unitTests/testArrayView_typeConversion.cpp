/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "Array.hpp"
#include "testUtils.hpp"

// TPL includes
#include <gtest/gtest.h>

namespace LvArray
{
namespace testing
{

template< template< typename > class BUFFER_TYPE >
void testTypeConversion()
{
  {
    Array< int, 1, RAJA::PERM_I, int, BUFFER_TYPE > array( 20 );
    ArrayView< int[ 4 ], 1, 0, int, BUFFER_TYPE > const view4( array );

    EXPECT_EQ( view4.size(), array.size() / 4 );
    EXPECT_EQ( reinterpret_cast< int * >( view4.data() ), array.data() );

    ArrayView< int[ 2 ], 1, 0, int, BUFFER_TYPE > const view2( view4 );
    EXPECT_EQ( view2.size(), array.size() / 2 );
    EXPECT_EQ( reinterpret_cast< int * >( view2.data() ), array.data() );
  }

  {
    Array< int, 2, RAJA::PERM_IJ, int, BUFFER_TYPE > array( 10, 20 );
    ArrayView< int[ 4 ], 2, 1, int, BUFFER_TYPE > const view4( array );
    EXPECT_EQ( view4.size(), array.size() / 4 );
    EXPECT_EQ( view4.size( 0 ), array.size( 0 ) );
    EXPECT_EQ( view4.size( 1 ), array.size( 1 ) / 4 );

    for( int i = 0; i < view4.size( 0 ); ++i )
    {
      for( int j = 0; j < view4.size( 1 ); ++j )
      {
        for( int k = 0; k < 4; ++k )
        {
          EXPECT_EQ( &view4( i, j )[ k ], &array( i, 4 * j + k ) );
        }
      }
    }

    ArrayView< int[ 2 ], 2, 1, int, BUFFER_TYPE > view2( view4 );
    EXPECT_EQ( view2.size(), array.size() / 2 );
    EXPECT_EQ( view2.size( 0 ), array.size( 0 ) );
    EXPECT_EQ( view2.size( 1 ), array.size( 1 ) / 2 );

    for( int i = 0; i < view2.size( 0 ); ++i )
    {
      for( int j = 0; j < view2.size( 1 ); ++j )
      {
        for( int k = 0; k < 2; ++k )
        {
          EXPECT_EQ( &view2( i, j )[ k ], &array( i, 2 * j + k ) );
        }
      }
    }
  }

  {
    Array< int, 3, RAJA::PERM_KJI, int, BUFFER_TYPE > array( 8, 10, 11 );
    ArrayView< int[ 4 ], 3, 0, int, BUFFER_TYPE > const view4( array );
    EXPECT_EQ( view4.size(), array.size() / 4 );
    EXPECT_EQ( view4.size( 0 ), array.size( 0 ) / 4 );
    EXPECT_EQ( view4.size( 1 ), array.size( 1 ) );
    EXPECT_EQ( view4.size( 2 ), array.size( 2 ) );

    for( int i = 0; i < view4.size( 0 ); ++i )
    {
      for( int j = 0; j < view4.size( 1 ); ++j )
      {
        for( int k = 0; k < view4.size( 2 ); ++k )
        {
          for( int l = 0; l < 4; ++l )
          {
            EXPECT_EQ( &view4( i, j, k )[ l ], &array( 4 * i + l, j, k ) );
          }
        }
      }
    }

    ArrayView< int[ 2 ], 3, 0, int, BUFFER_TYPE > const view2( view4 );
    EXPECT_EQ( view2.size(), array.size() / 2 );
    EXPECT_EQ( view2.size( 0 ), array.size( 0 ) / 2 );
    EXPECT_EQ( view2.size( 1 ), array.size( 1 ) );
    EXPECT_EQ( view2.size( 2 ), array.size( 2 ) );

    for( int i = 0; i < view2.size( 0 ); ++i )
    {
      for( int j = 0; j < view2.size( 1 ); ++j )
      {
        for( int k = 0; k < view2.size( 2 ); ++k )
        {
          for( int l = 0; l < 2; ++l )
          {
            EXPECT_EQ( &view2( i, j, k )[ l ], &array( 2 * i + l, j, k ) );
          }
        }
      }
    }
  }
}

TEST( ArrayView, MallocBuffer_TypeConversion )
{
  testTypeConversion< MallocBuffer >();
}

#if defined(LVARRAY_USE_CHAI)

TEST( ArrayView, ChaiBuffer_TypeConversion )
{
  testTypeConversion< ChaiBuffer >();
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
