/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "ArrayOfSets.hpp"
#include "ArrayOfArrays.hpp"
#include "MallocBuffer.hpp"

// TPL includes
#include <RAJA/RAJA.hpp>
#include <gtest/gtest.h>

// system includes
#include <string>

// Sphinx start after examples
TEST( ArrayOfSets, examples )
{
  LvArray::ArrayOfSets< std::string, std::ptrdiff_t, LvArray::MallocBuffer > arrayOfSets;

  // Append a set with capacity 2.
  arrayOfSets.appendSet( 2 );
  arrayOfSets.insertIntoSet( 0, "oh" );
  arrayOfSets.insertIntoSet( 0, "my" );

  // Insert a set at the beginning with capacity 3.
  arrayOfSets.insertSet( 0, 3 );
  arrayOfSets.insertIntoSet( 0, "lions" );
  arrayOfSets.insertIntoSet( 0, "tigers" );
  arrayOfSets.insertIntoSet( 0, "bears" );

  // "tigers" is already in the set.
  EXPECT_FALSE( arrayOfSets.insertIntoSet( 0, "tigers" ) );

  EXPECT_EQ( arrayOfSets( 0, 0 ), "bears" );
  EXPECT_EQ( arrayOfSets( 0, 1 ), "lions" );
  EXPECT_EQ( arrayOfSets( 0, 2 ), "tigers" );

  EXPECT_EQ( arrayOfSets[ 1 ][ 0 ], "my" );
  EXPECT_EQ( arrayOfSets[ 1 ][ 1 ], "oh" );
}
// Sphinx end before examples

// Sphinx start after assimilate
TEST( ArrayOfSets, assimilate )
{
  LvArray::ArrayOfSets< int, std::ptrdiff_t, LvArray::MallocBuffer > arrayOfSets;

  // Create an ArrayOfArrays and populate the inner arrays with sorted unique values.
  LvArray::ArrayOfArrays< int, std::ptrdiff_t, LvArray::MallocBuffer > arrayOfArrays( 3 );

  // The first array is empty, the second is {0} and the third is {0, 1}.
  arrayOfArrays.emplaceBack( 1, 0 );
  arrayOfArrays.emplaceBack( 2, 0 );
  arrayOfArrays.emplaceBack( 2, 1 );

  // Assimilate arrayOfArrays into arrayOfSets.
  arrayOfSets.assimilate< RAJA::loop_exec >( std::move( arrayOfArrays ),
                                             LvArray::sortedArrayManipulation::Description::SORTED_UNIQUE );

  // After being assimilated arrayOfArrays is empty.
  EXPECT_EQ( arrayOfArrays.size(), 0 );

  // arrayOfSets now contains the values.
  EXPECT_EQ( arrayOfSets.size(), 3 );
  EXPECT_EQ( arrayOfSets.sizeOfSet( 0 ), 0 );
  EXPECT_EQ( arrayOfSets.sizeOfSet( 1 ), 1 );
  EXPECT_EQ( arrayOfSets.sizeOfSet( 2 ), 2 );
  EXPECT_EQ( arrayOfSets( 1, 0 ), 0 );
  EXPECT_EQ( arrayOfSets( 2, 0 ), 0 );
  EXPECT_EQ( arrayOfSets( 2, 1 ), 1 );

  // Resize arrayOfArrays and populate it the inner arrays with values that are neither sorted nor unique.
  arrayOfArrays.resize( 2 );

  // The first array is {4, -1} and the second is {4, 4}.
  arrayOfArrays.emplaceBack( 0, 3 );
  arrayOfArrays.emplaceBack( 0, -1 );
  arrayOfArrays.emplaceBack( 1, 4 );
  arrayOfArrays.emplaceBack( 1, 4 );

  // Assimilate the arrayOfArrays yet again.
  arrayOfSets.assimilate< RAJA::loop_exec >( std::move( arrayOfArrays ),
                                             LvArray::sortedArrayManipulation::Description::UNSORTED_WITH_DUPLICATES );

  EXPECT_EQ( arrayOfSets.size(), 2 );
  EXPECT_EQ( arrayOfSets.sizeOfSet( 0 ), 2 );
  EXPECT_EQ( arrayOfSets.sizeOfSet( 1 ), 1 );
  EXPECT_EQ( arrayOfSets( 0, 0 ), -1 );
  EXPECT_EQ( arrayOfSets( 0, 1 ), 3 );
  EXPECT_EQ( arrayOfSets( 1, 0 ), 4 );
}
// Sphinx end before assimilate

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
