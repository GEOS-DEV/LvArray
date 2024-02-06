/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "ArrayOfArrays.hpp"
#include "MallocBuffer.hpp"
#if defined(LVARRAY_USE_CHAI)
  #include "ChaiBuffer.hpp"
#endif
#include "sortedArrayManipulation.hpp"

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

// Sphinx start after examples
TEST( ArrayOfArrays, construction )
{
  // Create an empty ArrayOfArrays.
  LvArray::ArrayOfArrays< std::string, std::ptrdiff_t, LvArray::MallocBuffer > arrayOfArrays;
  EXPECT_EQ( arrayOfArrays.size(), 0 );

  // Append an array of length 2.
  arrayOfArrays.appendArray( 2 );
  EXPECT_EQ( arrayOfArrays.size(), 1 );
  EXPECT_EQ( arrayOfArrays.sizeOfArray( 0 ), 2 );
  EXPECT_EQ( arrayOfArrays.capacityOfArray( 0 ), 2 );
  arrayOfArrays( 0, 0 ) = "First array, first entry.";
  arrayOfArrays( 0, 1 ) = "First array, second entry.";

  // Append another array of length 3.
  arrayOfArrays.appendArray( 3 );
  EXPECT_EQ( arrayOfArrays.size(), 2 );
  EXPECT_EQ( arrayOfArrays.sizeOfArray( 1 ), 3 );
  EXPECT_EQ( arrayOfArrays.capacityOfArray( 1 ), 3 );
  arrayOfArrays( 1, 0 ) = "Second array, first entry.";
  arrayOfArrays( 1, 2 ) = "Second array, third entry.";

  EXPECT_EQ( arrayOfArrays[ 0 ][ 1 ], "First array, second entry." );
  EXPECT_EQ( arrayOfArrays[ 1 ][ 2 ], "Second array, third entry." );

  // Values are default initialized.
  EXPECT_EQ( arrayOfArrays[ 1 ][ 1 ], std::string() );
}

TEST( ArrayOfArrays, modification )
{
  LvArray::ArrayOfArrays< std::string, std::ptrdiff_t, LvArray::MallocBuffer > arrayOfArrays;

  // Append an array.
  arrayOfArrays.appendArray( 3 );
  arrayOfArrays( 0, 0 ) = "First array, first entry.";
  arrayOfArrays( 0, 1 ) = "First array, second entry.";
  arrayOfArrays( 0, 2 ) = "First array, third entry.";

  // Insert a new array at the beginning.
  std::array< std::string, 2 > newFirstArray = { "New first array, first entry.",
                                                 "New first array, second entry." };
  arrayOfArrays.insertArray( 0, newFirstArray.begin(), newFirstArray.end() );

  EXPECT_EQ( arrayOfArrays.size(), 2 );
  EXPECT_EQ( arrayOfArrays.sizeOfArray( 0 ), 2 );
  EXPECT_EQ( arrayOfArrays.sizeOfArray( 1 ), 3 );

  EXPECT_EQ( arrayOfArrays( 0, 1 ), "New first array, second entry." );
  EXPECT_EQ( arrayOfArrays[ 1 ][ 1 ], "First array, second entry." );

  // Erase the values from what is now the second array.
  arrayOfArrays.clearArray( 1 );
  EXPECT_EQ( arrayOfArrays.sizeOfArray( 1 ), 0 );

  // Append a value to the end of each array.
  arrayOfArrays.emplaceBack( 1, "Second array, first entry." );
  arrayOfArrays.emplaceBack( 0, "New first array, third entry." );

  EXPECT_EQ( arrayOfArrays.sizeOfArray( 0 ), 3 );
  EXPECT_EQ( arrayOfArrays.sizeOfArray( 1 ), 1 );

  EXPECT_EQ( arrayOfArrays[ 1 ][ 0 ], "Second array, first entry." );
  EXPECT_EQ( arrayOfArrays( 0, 2 ), "New first array, third entry." );
}
// Sphinx end before examples

#if defined(RAJA_ENABLE_OPENMP)
// Sphinx start after view
TEST( ArrayOfArrays, view )
{
  // Create an ArrayOfArrays with 10 inner arrays each with capacity 9.
  LvArray::ArrayOfArrays< int, std::ptrdiff_t, LvArray::MallocBuffer > arrayOfArrays( 10, 9 );

  {
    // Then create a view.
    LvArray::ArrayOfArraysView< int,
                                std::ptrdiff_t const,
                                false,
                                LvArray::MallocBuffer > const view = arrayOfArrays.toView();
    EXPECT_EQ( view.size(), 10 );
    EXPECT_EQ( view.capacityOfArray( 7 ), 9 );

    // Modify every inner array in parallel
    RAJA::forall< RAJA::omp_parallel_for_exec >(
      RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, view.size() ),
      [view] ( std::ptrdiff_t const i )
    {
      for( std::ptrdiff_t j = 0; j < i; ++j )
      {
        view.emplaceBack( i, 10 * i + j );
      }
    }
      );

    EXPECT_EQ( view.sizeOfArray( 9 ), view.capacityOfArray( 9 ) );

    // The last array is at capacity. Therefore any attempt at increasing its size through
    // a view is invalid and may lead to undefined behavior!
    // view.emplaceBack( 9, 0 );
  }

  {
    // Create a view which cannot modify the sizes of the inner arrays.
    LvArray::ArrayOfArraysView< int,
                                std::ptrdiff_t const,
                                true,
                                LvArray::MallocBuffer > const viewConstSizes = arrayOfArrays.toViewConstSizes();
    for( std::ptrdiff_t i = 0; i < viewConstSizes.size(); ++i )
    {
      for( int & value : viewConstSizes[ i ] )
      {
        value *= 2;
      }

      // This would be a compilation error
      // viewConstSizes.emplaceBack( i, 0 );
    }
  }

  {
    // Create a view which has read only access.
    LvArray::ArrayOfArraysView< int const,
                                std::ptrdiff_t const,
                                true,
                                LvArray::MallocBuffer > const viewConst = arrayOfArrays.toViewConst();

    for( std::ptrdiff_t i = 0; i < viewConst.size(); ++i )
    {
      for( std::ptrdiff_t j = 0; j < viewConst.sizeOfArray( i ); ++j )
      {
        EXPECT_EQ( viewConst( i, j ), 2 * ( 10 * i + j ) );

        // This would be a compilation error
        // viewConst( i, j ) = 0;
      }
    }
  }

  // Verify that all the modifications are present in the parent ArrayOfArrays.
  EXPECT_EQ( arrayOfArrays.size(), 10 );
  for( std::ptrdiff_t i = 0; i < arrayOfArrays.size(); ++i )
  {
    EXPECT_EQ( arrayOfArrays.sizeOfArray( i ), i );
    for( std::ptrdiff_t j = 0; j < arrayOfArrays.sizeOfArray( i ); ++j )
    {
      EXPECT_EQ( arrayOfArrays( i, j ), 2 * ( 10 * i + j ) );
    }
  }
}
// Sphinx end before view

// Sphinx start after atomic
TEST( ArrayOfArrays, atomic )
{
  // Create an ArrayOfArrays with 1 array with a capacity of 100.
  LvArray::ArrayOfArrays< int, std::ptrdiff_t, LvArray::MallocBuffer > arrayOfArrays( 1, 100 );
  EXPECT_EQ( arrayOfArrays.sizeOfArray( 0 ), 0 );

  // Then create a view.
  LvArray::ArrayOfArraysView< int,
                              std::ptrdiff_t const,
                              false,
                              LvArray::MallocBuffer > const view = arrayOfArrays.toView();

  // Append to the single inner array in parallel
  RAJA::forall< RAJA::omp_parallel_for_exec >(
    RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, view.capacityOfArray( 0 ) ),
    [view] ( std::ptrdiff_t const i )
  {
    view.emplaceBackAtomic< RAJA::builtin_atomic >( 0, i );
  }
    );

  EXPECT_EQ( arrayOfArrays.sizeOfArray( 0 ), 100 );

  // Sort the entries in the array since they were appended in an arbitrary order
  LvArray::sortedArrayManipulation::makeSorted( arrayOfArrays[ 0 ].begin(), arrayOfArrays[ 0 ].end() );

  for( std::ptrdiff_t i = 0; i < arrayOfArrays.sizeOfArray( 0 ); ++i )
  {
    EXPECT_EQ( arrayOfArrays( 0, i ), i );
  }
}
// Sphinx end before atomic
#endif

// Sphinx start after compress
TEST( ArrayOfArrays, compress )
{
  // Create an ArrayOfArrays with three inner arrays each with capacity 5.
  LvArray::ArrayOfArrays< int, std::ptrdiff_t, LvArray::MallocBuffer > arrayOfArrays( 3, 5 );
  EXPECT_EQ( arrayOfArrays.sizeOfArray( 1 ), 0 );
  EXPECT_EQ( arrayOfArrays.capacityOfArray( 1 ), 5 );

  // Append values to the inner arrays.
  std::array< int, 5 > values = { 0, 1, 2, 3, 4 };
  arrayOfArrays.appendToArray( 0, values.begin(), values.begin() + 3 );
  arrayOfArrays.appendToArray( 1, values.begin(), values.begin() + 4 );
  arrayOfArrays.appendToArray( 2, values.begin(), values.begin() + 5 );

  EXPECT_EQ( arrayOfArrays.sizeOfArray( 0 ), 3 );
  EXPECT_EQ( arrayOfArrays.capacityOfArray( 0 ), 5 );

  // Since the first array has some extra capacity it is not contiguous in memory
  // with the second array.
  EXPECT_NE( &arrayOfArrays( 0, 2 ) + 1, &arrayOfArrays( 1, 0 ) );

  // Compress the ArrayOfArrays. This doesn't change the sizes or values of the
  // inner arrays, only their capacities.
  arrayOfArrays.compress();

  EXPECT_EQ( arrayOfArrays.sizeOfArray( 0 ), 3 );
  EXPECT_EQ( arrayOfArrays.capacityOfArray( 0 ), 3 );

  // The values are preserved.
  EXPECT_EQ( arrayOfArrays( 2, 4 ), 4 );

  // And the inner arrays are now contiguous.
  EXPECT_EQ( &arrayOfArrays( 0, 2 ) + 1, &arrayOfArrays( 1, 0 ) );
}
// Sphinx end before compress

// Sphinx start after resizeFromCapacities
TEST( ArrayOfArrays, resizeFromCapacities )
{
  // Create an ArrayOfArrays with two inner arrays
  LvArray::ArrayOfArrays< int, std::ptrdiff_t, LvArray::MallocBuffer > arrayOfArrays( 2 );

  // Append values to the inner arrays.
  std::array< int, 5 > values = { 0, 1, 2, 3, 4 };
  arrayOfArrays.appendToArray( 0, values.begin(), values.begin() + 3 );
  arrayOfArrays.appendToArray( 1, values.begin(), values.begin() + 4 );

  EXPECT_EQ( arrayOfArrays.sizeOfArray( 0 ), 3 );

  // Resize the ArrayOfArrays from a new list of capacities.
  std::array< std::ptrdiff_t, 3 > newCapacities = { 3, 5, 2 };
  arrayOfArrays.resizeFromCapacities< RAJA::seq_exec >( newCapacities.size(), newCapacities.data() );

  // This will clear any existing arrays.
  EXPECT_EQ( arrayOfArrays.size(), 3 );
  EXPECT_EQ( arrayOfArrays.sizeOfArray( 1 ), 0 );
  EXPECT_EQ( arrayOfArrays.capacityOfArray( 1 ), newCapacities[ 1 ] );
}
// Sphinx end before resizeFromCapacities

#if defined(LVARRAY_USE_CUDA) && defined(LVARRAY_USE_CHAI)
// Sphinx start after ChaiBuffer
CUDA_TEST( ArrayOfArrays, ChaiBuffer )
{
  LvArray::ArrayOfArrays< int, std::ptrdiff_t, LvArray::ChaiBuffer > arrayOfArrays( 10, 9 );

  {
    // Create a view.
    LvArray::ArrayOfArraysView< int,
                                std::ptrdiff_t const,
                                false,
                                LvArray::ChaiBuffer > const view = arrayOfArrays.toView();

    // Capture the view on device. This will copy the values, sizes and offsets.
    // The values and sizes will be touched.
    RAJA::forall< RAJA::cuda_exec< 32 > >(
      RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, view.size() ),
      [view] __device__ ( std::ptrdiff_t const i )
    {
      for( std::ptrdiff_t j = 0; j < i; ++j )
      {
        view.emplace( i, 0, 10 * i + j );
      }
    }
      );
  }

  {
    // Create a view which cannot modify the sizes of the inner arrays.
    LvArray::ArrayOfArraysView< int,
                                std::ptrdiff_t const,
                                true,
                                LvArray::ChaiBuffer > const viewConstSizes = arrayOfArrays.toViewConstSizes();

    // Capture the view on the host. This will copy back the values and sizes since they were previously touched
    // on device. It will only touch the values on host.
    RAJA::forall< RAJA::seq_exec >(
      RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, viewConstSizes.size() ),
      [viewConstSizes] ( std::ptrdiff_t const i )
    {
      for( int & value : viewConstSizes[ i ] )
      {
        value *= 2;
      }
    }
      );
  }

  {
    // Create a view which has read only access.
    LvArray::ArrayOfArraysView< int const,
                                std::ptrdiff_t const,
                                true,
                                LvArray::ChaiBuffer > const viewConst = arrayOfArrays.toViewConst();

    // Capture the view on device. Since the values were previously touched on host it will copy them over.
    // Both the sizes and offsets are current on device so they are not copied over. Nothing is touched.
    RAJA::forall< RAJA::seq_exec >(
      RAJA::TypedRangeSegment< std::ptrdiff_t >( 0, viewConst.size() ),
      [viewConst] ( std::ptrdiff_t const i )
    {
      for( std::ptrdiff_t j = 0; j < viewConst.sizeOfArray( i ); ++j )
      {
        LVARRAY_ERROR_IF_NE( viewConst( i, j ), 2 * ( 10 * i + i - j - 1 ) );
      }
    }
      );
  }

  // This won't copy any data since everything is current on host. It will however touch the values,
  // sizes and offsets.
  arrayOfArrays.move( LvArray::MemorySpace::host );

  // Verify that all the modifications are present in the parent ArrayOfArrays.
  EXPECT_EQ( arrayOfArrays.size(), 10 );
  for( std::ptrdiff_t i = 0; i < arrayOfArrays.size(); ++i )
  {
    EXPECT_EQ( arrayOfArrays.sizeOfArray( i ), i );
    for( std::ptrdiff_t j = 0; j < arrayOfArrays.sizeOfArray( i ); ++j )
    {
      EXPECT_EQ( arrayOfArrays( i, j ), 2 * ( 10 * i + i - j - 1 ) );
    }
  }
}
// Sphinx end before ChaiBuffer
#endif

// Sphinx start after bounds check
TEST( ArrayOfArrays, boundsCheck )
{
#if defined(LVARRAY_BOUNDS_CHECK)
  LvArray::ArrayOfArrays< int, std::ptrdiff_t, LvArray::MallocBuffer > arrayOfArrays;

  // Append an array.
  std::array< int, 5 > values = { 0, 1, 2, 3, 4 };
  arrayOfArrays.appendArray( values.begin(), values.end() );

  EXPECT_EQ( arrayOfArrays.size(), 1 );
  EXPECT_EQ( arrayOfArrays.sizeOfArray( 0 ), 5 );

  // Out of bounds access aborts the program.
  EXPECT_DEATH_IF_SUPPORTED( arrayOfArrays( 0, -1 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( arrayOfArrays[ 0 ][ 6 ], "" );
  EXPECT_DEATH_IF_SUPPORTED( arrayOfArrays[ 1 ][ 5 ], "" );

  EXPECT_DEATH_IF_SUPPORTED( arrayOfArrays.capacityOfArray( 5 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( arrayOfArrays.insertArray( 5, values.begin(), values.end() ), "" );
  EXPECT_DEATH_IF_SUPPORTED( arrayOfArrays.emplace( 0, 44, 4 ), "" );
  EXPECT_DEATH_IF_SUPPORTED( arrayOfArrays.emplace( 1, 44, 4 ), "" );
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
