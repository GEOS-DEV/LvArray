/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "SortedArray.hpp"
#include "MallocBuffer.hpp"
#if defined(LVARRAY_USE_CHAI)
  #include "ChaiBuffer.hpp"
#endif

// TPL includes
#include <RAJA/RAJA.hpp>
#include <gtest/gtest.h>

// system includes
#include <string>
#include <random>

// You can't define a nvcc extended host-device or device lambda in a TEST statement.
#define CUDA_TEST( X, Y )                 \
  static void cuda_test_ ## X ## _ ## Y();    \
  TEST( X, Y ) { cuda_test_ ## X ## _ ## Y(); } \
  static void cuda_test_ ## X ## _ ## Y()

// Sphinx start after construction
TEST( SortedArray, construction )
{
  // Construct an empty set.
  LvArray::SortedArray< std::string, std::ptrdiff_t, LvArray::MallocBuffer > set;
  EXPECT_TRUE( set.empty() );

  // Insert two objects one at a time.
  EXPECT_TRUE( set.insert( "zebra" ) );
  EXPECT_TRUE( set.insert( "aardvark" ) );

  // "zebra" is already in the set so it won't be inserted again.
  EXPECT_FALSE( set.insert( "zebra" ) );

  // Query the contents of the set.
  EXPECT_EQ( set.size(), 2 );
  EXPECT_TRUE( set.contains( "zebra" ) );
  EXPECT_FALSE( set.contains( "whale" ) );

  // Insert two objects at once.
  std::string const moreAnimals[ 2 ] = { "cat", "dog" };
  EXPECT_EQ( set.insert( moreAnimals, moreAnimals + 2 ), 2 );

  // Remove a single object.
  set.remove( "aardvark" );

  EXPECT_EQ( set.size(), 3 );
  EXPECT_EQ( set[ 0 ], "cat" );
  EXPECT_EQ( set[ 1 ], "dog" );
  EXPECT_EQ( set[ 2 ], "zebra" );
}
// Sphinx end before construction

#if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
// Sphinx start after ChaiBuffer
CUDA_TEST( SortedArray, ChaiBuffer )
{
  // Construct an empty set consisting of the even numbers { 0, 2, 4 }.
  LvArray::SortedArray< int, std::ptrdiff_t, LvArray::ChaiBuffer > set;
  int const values[ 4 ] = { 0, 2, 4 };
  EXPECT_EQ( set.insert( values, values + 3 ), 3 );
  EXPECT_EQ( set.size(), 3 );

  // Create a view and capture it on device, this will copy the data to the device.
  RAJA::forall< RAJA::cuda_exec< 32 > >(
    RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, set.size() ),
    [view = set.toView()] __device__ ( std::ptrdiff_t const i )
  {
    LVARRAY_ERROR_IF_NE( view[ i ], 2 * i );
    LVARRAY_ERROR_IF( view.contains( 2 * i + 1 ), "The set should only contain odd numbers!" );
  }
    );

  // Move the set back to the CPU and modify it.
  set.move( LvArray::MemorySpace::host );
  set.insert( 6 );

  // Verify that the modification is seen on device.
  RAJA::forall< RAJA::cuda_exec< 32 > >(
    RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, 1 ),
    [view = set.toView()] __device__ ( std::ptrdiff_t const )
  {
    LVARRAY_ERROR_IF( !view.contains( 6 ), "The set should contain 6!" );
  }
    );
}
// Sphinx end before ChaiBuffer
#endif

#if defined(LVARRAY_BOUNDS_CHECK)
// Sphinx start after bounds check
TEST( SortedArray, boundsCheck )
{
  // Create a set containing {2, 4}
  LvArray::SortedArray< int, std::ptrdiff_t, LvArray::MallocBuffer > set;
  set.insert( 4 );
  set.insert( 2 );

  // Invalid access.
  EXPECT_DEATH_IF_SUPPORTED( set[ 5 ], "" );

  // Attempt to insert unsorted values.
  int const unsortedInsert[ 2 ] = { 4, 0 };
  EXPECT_DEATH_IF_SUPPORTED( set.insert( unsortedInsert, unsortedInsert + 2 ), "" );

  // Attempt to insert nonUnique values.
  int const notUnique[ 2 ] = { 5, 5 };
  EXPECT_DEATH_IF_SUPPORTED( set.insert( notUnique, notUnique + 2 ), "" );

  // Attempt to remove unsorted values.
  int const unsortedRemove[ 2 ] = { 4, 2 };
  EXPECT_DEATH_IF_SUPPORTED( set.remove( unsortedRemove, unsortedRemove + 2 ), "" );
}
// Sphinx end before bounds check
#endif

// Sphinx start after fast construction
TEST( SortedArray, fastConstruction )
{
  LvArray::SortedArray< int, std::ptrdiff_t, LvArray::MallocBuffer > set;

  // Create a temporary list of 100 random numbers between 0 and 99.
  std::vector< int > temporarySpace( 100 );
  std::mt19937 gen;
  std::uniform_int_distribution< int > dis( 0, 99 );
  for( int i = 0; i < 100; ++i )
  {
    temporarySpace[ i ] = dis( gen );
  }

  // Sort the random numbers and move any duplicates to the end.
  std::ptrdiff_t const numUniqueValues = LvArray::sortedArrayManipulation::makeSortedUnique( temporarySpace.begin(),
                                                                                             temporarySpace.end() );

  // Insert into the set.
  set.insert( temporarySpace.begin(), temporarySpace.begin() + numUniqueValues );
}
// Sphinx end before fast construction

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
