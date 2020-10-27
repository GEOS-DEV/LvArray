/*
 * Copyright (c) 2020, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "Array.hpp"
#include "SortedArray.hpp"
#include "ArrayOfArrays.hpp"
#include "ArrayOfSets.hpp"
#include "SparsityPattern.hpp"
#include "CRSMatrix.hpp"
#include "testUtils.hpp"

// TPL includes
#include <gtest/gtest.h>

/*
 * When ALLOW_INVALID_OPERATIONS is defined each of the tests below should have a compilation error from
 * attempting to perform an operation forbidden by LvArray.
 */
// #define ALLOW_INVALID_OPERATIONS

namespace LvArray
{
namespace testing
{

TEST( Array, toView )
{
  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, MallocBuffer > array;
  auto const view = array.toView();
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidView = std::move( array ).toView();
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
}

TEST( Array, toViewConst )
{
  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, MallocBuffer > array;
  auto const view = array.toViewConst();
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidView = std::move( array ).toViewConst();
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
}

TEST( Array, toNestedView )
{
  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, MallocBuffer > array;
  auto const & view = array.toNestedView();
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const & invalidView = std::move( array ).toNestedView();
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
}

TEST( Array, toNestedViewConst )
{
  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, MallocBuffer > array;
  auto const & view = array.toNestedViewConst();
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const & invalidView = std::move( array ).toNestedViewConst();
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
}

TEST( Array, toViewConstUDC )
{
  Array< int, 1, RAJA::PERM_I, std::ptrdiff_t, MallocBuffer > array;
  ArrayView< int const, 1, 0, std::ptrdiff_t, MallocBuffer > const view = array;
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  ArrayView< int const, 1, 0, std::ptrdiff_t, MallocBuffer > const invalidView = std::move( array );
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
}

TEST( Array, accessOperator )
{
  Array< int, 2, RAJA::PERM_IJ, std::ptrdiff_t, MallocBuffer > array( 5, 5 );
  auto const slice = array[ 0 ];
  LVARRAY_UNUSED_VARIABLE( slice );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidSlice = std::move( array )[ 0 ];
  LVARRAY_UNUSED_VARIABLE( invalidSlice );
#endif
}

TEST( ArrayView, toSlice )
{
  ArrayView< int, 1, 0, std::ptrdiff_t, MallocBuffer > view;
  auto const slice = view.toSlice();
  LVARRAY_UNUSED_VARIABLE( slice );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidSlice = std::move( view ).toSlice();
  LVARRAY_UNUSED_VARIABLE( invalidSlice );
#endif
}

TEST( ArrayView, toSliceConst )
{
  ArrayView< int, 1, 0, std::ptrdiff_t, MallocBuffer > view;
  auto const slice = view.toSliceConst();
  LVARRAY_UNUSED_VARIABLE( slice );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidSlice = std::move( view ).toSliceConst();
  LVARRAY_UNUSED_VARIABLE( invalidSlice );
#endif
}

TEST( ArrayView, toSliceUDC )
{
  ArrayView< int, 1, 0, std::ptrdiff_t, MallocBuffer > view;
  ArraySlice< int, 1, 0, std::ptrdiff_t > const slice = view;
  LVARRAY_UNUSED_VARIABLE( slice );

#if defined(ALLOW_INVALID_OPERATIONS)
  ArraySlice< int, 1, 0, std::ptrdiff_t > const invalidSlice = std::move( view );
  LVARRAY_UNUSED_VARIABLE( invalidSlice );
#endif
}

TEST( ArrayView, toSliceConstUDC )
{
  ArrayView< int, 1, 0, std::ptrdiff_t, MallocBuffer > view;
  ArraySlice< int const, 1, 0, std::ptrdiff_t > const slice = view;
  LVARRAY_UNUSED_VARIABLE( slice );

#if defined(ALLOW_INVALID_OPERATIONS)
  ArraySlice< int const, 1, 0, std::ptrdiff_t > const invalidSlice = std::move( view );
  LVARRAY_UNUSED_VARIABLE( invalidSlice );
#endif
}

TEST( SortedArray, toView )
{
  SortedArray< int, std::ptrdiff_t, MallocBuffer > array;
  auto const view = array.toView();
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidView = std::move( array ).toView();
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
}

TEST( SortedArray, toViewConst )
{
  SortedArray< int, std::ptrdiff_t, MallocBuffer > array;
  auto const view = array.toViewConst();
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidView = std::move( array ).toViewConst();
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
}

TEST( ArrayOfArrays, toView )
{
  ArrayOfArrays< int, std::ptrdiff_t, MallocBuffer > array;
  auto const view = array.toView();
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidView = std::move( array ).toView();
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
}

TEST( ArrayOfArrays, toViewConstSizes )
{
  ArrayOfArrays< int, std::ptrdiff_t, MallocBuffer > array;
  auto const view = array.toViewConstSizes();
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidView = std::move( array ).toViewConstSizes();
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
}

TEST( ArrayOfArrays, toViewConst )
{
  ArrayOfArrays< int, std::ptrdiff_t, MallocBuffer > array;
  auto const view = array.toViewConst();
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidView = std::move( array ).toViewConst();
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
}

TEST( ArrayOfSets, toView )
{
  ArrayOfSets< int, std::ptrdiff_t, MallocBuffer > array;
  auto const view = array.toView();
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidView = std::move( array ).toView();
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
}

TEST( ArrayOfSets, toViewConst )
{
  ArrayOfSets< int, std::ptrdiff_t, MallocBuffer > array;
  auto const view = array.toViewConst();
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidView = std::move( array ).toViewConst();
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
}

TEST( ArrayOfSets, toArrayOfArraysView )
{
  ArrayOfSets< int, std::ptrdiff_t, MallocBuffer > array;
  auto const view = array.toArrayOfArraysView();
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidView = std::move( array ).toArrayOfArraysView();
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
}

TEST( SparsityPattern, toView )
{
  SparsityPattern< int, std::ptrdiff_t, MallocBuffer > sparsity;
  auto const view = sparsity.toView();
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidView = std::move( sparsity ).toView();
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
}

TEST( SparsityPattern, toViewConst )
{
  SparsityPattern< int, std::ptrdiff_t, MallocBuffer > sparsity;
  auto const view = sparsity.toViewConst();
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidView = std::move( sparsity ).toViewConst();
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
}

TEST( CRSMatrix, toView )
{
  CRSMatrix< int, int, std::ptrdiff_t, MallocBuffer > matrix;
  auto const view = matrix.toView();
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidView = std::move( matrix ).toView();
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
}

TEST( CRSMatrix, toViewConstSizes )
{
  CRSMatrix< int, int, std::ptrdiff_t, MallocBuffer > matrix;
  auto const view = matrix.toViewConstSizes();
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidView = std::move( matrix ).toViewConstSizes();
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
}

TEST( CRSMatrix, toViewConst )
{
  CRSMatrix< int, int, std::ptrdiff_t, MallocBuffer > matrix;
  auto const view = matrix.toViewConst();
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidView = std::move( matrix ).toViewConst();
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
}

TEST( CRSMatrix, toSparsityPatternView )
{
  CRSMatrix< int, int, std::ptrdiff_t, MallocBuffer > matrix;
  auto const view = matrix.toSparsityPatternView();
  LVARRAY_UNUSED_VARIABLE( view );

#if defined(ALLOW_INVALID_OPERATIONS)
  auto const invalidView = std::move( matrix ).toSparsityPatternView();
  LVARRAY_UNUSED_VARIABLE( invalidView );
  FAIL();
#endif
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
