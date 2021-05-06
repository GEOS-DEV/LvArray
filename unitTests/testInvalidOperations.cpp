/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
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

#define HAS_RVALUE_MEMBER_FUNCTION_NO_RTYPE( NAME, ... ) \
  IS_VALID_EXPRESSION( HasRvalueMemberFunction_ ## NAME, CLASS, std::declval< CLASS && >().NAME( __VA_ARGS__ ) )

namespace LvArray
{
namespace testing
{

HAS_RVALUE_MEMBER_FUNCTION_NO_RTYPE( toView, );
HAS_RVALUE_MEMBER_FUNCTION_NO_RTYPE( toViewConst, );
HAS_RVALUE_MEMBER_FUNCTION_NO_RTYPE( toNestedView, );
HAS_RVALUE_MEMBER_FUNCTION_NO_RTYPE( toNestedViewConst, );
HAS_RVALUE_MEMBER_FUNCTION_NO_RTYPE( toSlice, );
HAS_RVALUE_MEMBER_FUNCTION_NO_RTYPE( toSliceConst, );
HAS_RVALUE_MEMBER_FUNCTION_NO_RTYPE( toViewConstSizes, );
HAS_RVALUE_MEMBER_FUNCTION_NO_RTYPE( toArrayOfArraysView, );
HAS_RVALUE_MEMBER_FUNCTION_NO_RTYPE( toSparsityPatternView, );

IS_VALID_EXPRESSION( HasRvalueSubscriptOperator, CLASS, std::declval< CLASS && >()[ 0 ] );

template< typename T >
using ArrayT = Array< T, 1, RAJA::PERM_I, std::ptrdiff_t, MallocBuffer >;

template< typename T >
using ArrayT2D = Array< T, 2, RAJA::PERM_IJ, std::ptrdiff_t, MallocBuffer >;

template< typename T >
using ArrayViewT = ArrayView< T, 1, 0, std::ptrdiff_t, MallocBuffer >;

template< typename T >
using ArraySliceT = ArraySlice< T, 1, 0, std::ptrdiff_t >;

// Array
static_assert( !HasRvalueMemberFunction_toView< ArrayT< int > >,
               "Cannot call toView on an Array rvalue." );
static_assert( !HasRvalueMemberFunction_toViewConst< ArrayT< int > >,
               "Cannot call toViewConst on an Array rvalue." );
static_assert( !HasRvalueMemberFunction_toNestedView< ArrayT< int > >,
               "Cannot call toNestedView on an Array rvalue." );
static_assert( !HasRvalueMemberFunction_toNestedViewConst< ArrayT< int > >,
               "Cannot call toNestedViewConst on an Array rvalue." );
static_assert( !std::is_convertible< ArrayT< int > &&, ArrayViewT< int const > >::value,
               "The conversion from an Array< T > rvalue to ArrayView< T const > is not allowed." );

// ArrayView
static_assert( !HasRvalueMemberFunction_toSlice< ArrayViewT< int > >,
               "Cannot call toSlice on an ArrayView rvalue." );
static_assert( !HasRvalueMemberFunction_toSliceConst< ArrayViewT< int > >,
               "Cannot call toSliceConst on an ArrayView rvalue." );
static_assert( !std::is_convertible< ArrayViewT< int > &&, ArraySliceT< int > >::value,
               "The conversion from an ArrayView< T > rvalue to an ArraySlice< T > is not allowed." );
static_assert( !std::is_convertible< ArrayViewT< int > &&, ArraySliceT< int const > >::value,
               "The conversion from an ArrayView< T > rvalue to an ArraySlice< T const > is not allowed." );
static_assert( !HasRvalueSubscriptOperator< ArrayView< int, 2, 1, std::ptrdiff_t, MallocBuffer > >,
               "The subscript operator on a multidimensional ArrayView rvalue is not allowed." );

// SortedArrayView
static_assert( !HasRvalueMemberFunction_toView< SortedArray< int, std::ptrdiff_t, MallocBuffer > >,
               "Cannot call toView on a SortedArray rvalue." );
static_assert( !HasRvalueMemberFunction_toViewConst< SortedArray< int, std::ptrdiff_t, MallocBuffer > >,
               "Cannot call toViewConst on a SortedArray rvalue." );

// ArrayOfArrays
static_assert( !HasRvalueMemberFunction_toView< ArrayOfArrays< int, std::ptrdiff_t, MallocBuffer > >,
               "Cannot call toView on an ArrayOfArrays rvalue." );
static_assert( !HasRvalueMemberFunction_toViewConst< ArrayOfArrays< int, std::ptrdiff_t, MallocBuffer > >,
               "Cannot call toViewConst on an ArrayOfArrays rvalue." );
static_assert( !HasRvalueMemberFunction_toViewConstSizes< ArrayOfArrays< int, std::ptrdiff_t, MallocBuffer > >,
               "Cannot call toViewConstSizes on an ArrayOfArrays rvalue." );

// ArrayOfSets
static_assert( !HasRvalueMemberFunction_toView< ArrayOfSets< int, std::ptrdiff_t, MallocBuffer > >,
               "Cannot call toView on an ArrayOfSets rvalue." );
static_assert( !HasRvalueMemberFunction_toViewConst< ArrayOfSets< int, std::ptrdiff_t, MallocBuffer > >,
               "Cannot call toViewConst on an ArrayOfSets rvalue." );
static_assert( !HasRvalueMemberFunction_toArrayOfArraysView< ArrayOfSets< int, std::ptrdiff_t, MallocBuffer > >,
               "Cannot call toArrayOfArraysView on an ArrayOfSets rvalue." );

// SparsityPattern
static_assert( !HasRvalueMemberFunction_toView< SparsityPattern< int, std::ptrdiff_t, MallocBuffer > >,
               "Cannot call toView on a SparsityPattern rvalue." );
static_assert( !HasRvalueMemberFunction_toViewConst< SparsityPattern< int, std::ptrdiff_t, MallocBuffer > >,
               "Cannot call toViewConst on a SparsityPattern rvalue." );

// CRSMatrix
static_assert( !HasRvalueMemberFunction_toView< CRSMatrix< int, int, std::ptrdiff_t, MallocBuffer > >,
               "Cannot call toView on a CRSMatrix rvalue." );
static_assert( !HasRvalueMemberFunction_toViewConst< CRSMatrix< int, int, std::ptrdiff_t, MallocBuffer > >,
               "Cannot call toViewConst on a CRSMatrix rvalue." );
static_assert( !HasRvalueMemberFunction_toViewConstSizes< CRSMatrix< int, int, std::ptrdiff_t, MallocBuffer > >,
               "Cannot call toViewConstSizes on a CRSMatrix rvalue." );
static_assert( !HasRvalueMemberFunction_toSparsityPatternView< CRSMatrix< int, int, std::ptrdiff_t, MallocBuffer > >,
               "Cannot call toSparsityPatternView on a CRSMatrix rvalue." );

} // namespace testing
} // namespace LvArray

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
