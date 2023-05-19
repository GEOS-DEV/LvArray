/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "Layout.hpp"

// TPL includes
#include <gtest/gtest.h>

namespace LvArray
{

static_assert( std::is_same_v< DynamicExtent< 0, int >, Extent<> > );
static_assert( std::is_same_v< DynamicExtent< 1, int >, Extent< int > > );
static_assert( std::is_same_v< DynamicExtent< 2, int >, Extent< int, int > > );
static_assert( std::is_same_v< DynamicExtent< 3, int >, Extent< int, int, int > > );

static_assert( std::is_same_v< Stride< _1 >,         CompactStride< Extent< _0 >, LayoutLeft  > > );
static_assert( std::is_same_v< Stride< _1 >,         CompactStride< Extent< _1 >, LayoutLeft > > );
static_assert( std::is_same_v< Stride< _1, _2 >,     CompactStride< Extent< _2, _3 >, LayoutLeft > > );
static_assert( std::is_same_v< Stride< _1, _2 >,     CompactStride< Extent< _2, _3 >, LayoutLeft, camp::idx_seq< 0, 1 > > > );
static_assert( std::is_same_v< Stride< _3, _1 >,     CompactStride< Extent< _2, _3 >, LayoutLeft, camp::idx_seq< 1, 0 > > > );
static_assert( std::is_same_v< Stride< _1, _2, _2 >, CompactStride< Extent< _2, _1, _3 >, LayoutLeft, camp::idx_seq< 0, 1, 2 > > > );
static_assert( std::is_same_v< Stride< _3, _6, _1 >, CompactStride< Extent< _2, _1, _3 >, LayoutLeft, camp::idx_seq< 2, 0, 1 > > > );
static_assert( std::is_same_v< Stride< _3, _3, _1 >, CompactStride< Extent< _2, _1, _3 >, LayoutLeft, camp::idx_seq< 2, 1, 0 > > > );
static_assert( std::is_same_v< Stride< _3, _1, _1 >, CompactStride< Extent< _2, _1, _3 >, LayoutLeft, camp::idx_seq< 1, 2, 0 > > > );

static_assert( std::is_same_v< Stride< _1 >,         CompactStride< Extent< _0 >, LayoutRight > > );
static_assert( std::is_same_v< Stride< _1 >,         CompactStride< Extent< _1 >, LayoutRight > > );
static_assert( std::is_same_v< Stride< _3, _1 >,     CompactStride< Extent< _2, _3 >, LayoutRight > > );
static_assert( std::is_same_v< Stride< _3, _1 >,     CompactStride< Extent< _2, _3 >, LayoutRight, camp::idx_seq< 0, 1 > > > );
static_assert( std::is_same_v< Stride< _1, _2 >,     CompactStride< Extent< _2, _3 >, LayoutRight, camp::idx_seq< 1, 0 > > > );
static_assert( std::is_same_v< Stride< _3, _3, _1 >, CompactStride< Extent< _2, _1, _3 >, LayoutRight, camp::idx_seq< 0, 1, 2 > > > );
static_assert( std::is_same_v< Stride< _1, _1, _2 >, CompactStride< Extent< _2, _1, _3 >, LayoutRight, camp::idx_seq< 2, 0, 1 > > > );
static_assert( std::is_same_v< Stride< _1, _2, _2 >, CompactStride< Extent< _2, _1, _3 >, LayoutRight, camp::idx_seq< 2, 1, 0 > > > );
static_assert( std::is_same_v< Stride< _1, _6, _2 >, CompactStride< Extent< _2, _1, _3 >, LayoutRight, camp::idx_seq< 1, 2, 0 > > > );

static_assert( std::is_same_v< Layout< Extent< _4 >, Stride< _1 > >,                 decltype( makeLayout( makeExtent( _4{} ) ) ) > );
static_assert( std::is_same_v< Layout< Extent< _4, _2 >, Stride< _2, _1 > >,         decltype( makeLayout( makeExtent( _4{}, _2{} ) ) ) > );
static_assert( std::is_same_v< Layout< Extent< _4, _0 >, Stride< _0, _1 > >,         decltype( makeLayout( makeExtent( _4{}, _0{} ) ) ) > );
static_assert( std::is_same_v< Layout< Extent< _1, _2, _3 >, Stride< _6, _3, _1 > >, decltype( makeLayout( makeExtent( _1{}, _2{}, _3{} ) ) ) > );
static_assert( std::is_same_v< Layout< Extent< _4, int >, Stride< int, _1 > >,       decltype( makeLayout( makeExtent( _4{}, 4 ) ) ) > );
static_assert( std::is_same_v< Layout< Extent< int, _4 >, Stride< _4, _1 > >,        decltype( makeLayout( makeExtent(  4,  _4{} ) ) ) > );

using TestLayout = Layout< Extent< _1, _2, _3 >, CompactStride< Extent< _1, _2, _3 >, LayoutRight > >;
static_assert( std::is_same_v< Layout< Extent< _1, _2, _3 >, Stride< _6, _3, _1 > >, TestLayout > );

static_assert( std::is_same_v< _0, decltype( TestLayout{}( _0{}, _0{}, _0{} ) ) > );
static_assert( std::is_same_v< _1, decltype( TestLayout{}( _0{}, _0{}, _1{} ) ) > );
static_assert( std::is_same_v< _3, decltype( TestLayout{}( _0{}, _1{}, _0{} ) ) > );
static_assert( std::is_same_v< _6, decltype( TestLayout{}( _1{}, _0{}, _0{} ) ) > );
static_assert( std::is_same_v< _5, decltype( TestLayout{}( _0{}, _1{}, _2{} ) ) > );
static_assert( std::is_same_v< _3, decltype( TestLayout{}( _0{}, _1{},  _{} ) ) > );
static_assert( std::is_same_v< _6, decltype( TestLayout{}( _1{},  _{},  _{} ) ) > );
static_assert( std::is_same_v< _6, decltype( TestLayout{}(  _{}, _2{},  _{} ) ) > );
static_assert( std::is_same_v< _3, decltype( TestLayout{}(  _{},  _{}, _3{} ) ) > );
static_assert( std::is_same_v< _0, decltype( TestLayout{}(  _{},  _{},  _{} ) ) > );
static_assert( std::is_same_v< _6, decltype( TestLayout{}( _1{} ) ) > );
static_assert( std::is_same_v< _9, decltype( TestLayout{}( _1{}, _1{} ) ) > );

static_assert( std::is_same_v< Layout< Extent<>, Stride<> >,                 decltype( TestLayout{}.slice( _1{}, _1{}, _2{} ) ) > );
static_assert( std::is_same_v< Layout< Extent< _3 >, Stride< _1 > >,         decltype( TestLayout{}.slice( _1{}, _1{},  _{} ) ) > );
static_assert( std::is_same_v< Layout< Extent< _2 >, Stride< _3 > >,         decltype( TestLayout{}.slice( _1{},  _{}, _1{} ) ) > );
static_assert( std::is_same_v< Layout< Extent< _1 >, Stride< _6 > >,         decltype( TestLayout{}.slice(  _{}, _1{}, _1{} ) ) > );
static_assert( std::is_same_v< Layout< Extent< _1, _2 >, Stride< _6, _3 > >, decltype( TestLayout{}.slice(  _{},  _{}, _1{} ) ) > );
static_assert( std::is_same_v< Layout< Extent< _1, _3 >, Stride< _6, _1 > >, decltype( TestLayout{}.slice(  _{}, _1{},  _{} ) ) > );
static_assert( std::is_same_v< Layout< Extent< _2, _3 >, Stride< _3, _1 > >, decltype( TestLayout{}.slice( _1{},  _{},  _{} ) ) > );
static_assert( std::is_same_v< TestLayout,                                   decltype( TestLayout{}.slice(  _{},  _{},  _{} ) ) > );
static_assert( std::is_same_v< Layout< Extent< _2, _3 >, Stride< _3, _1 > >, decltype( TestLayout{}.slice( _1{} ) ) > );
static_assert( std::is_same_v< Layout< Extent< _3 >, Stride< _1 > >,         decltype( TestLayout{}.slice( _1{}, _1{} ) ) > );

static_assert( std::is_same_v< Layout< Extent< _8 >, Stride< _1 > >, decltype( Layout< Extent< _2 >, Stride< _1 > >{}.downcast< 4 >() ) > );
static_assert( std::is_same_v< Layout< Extent< _3, _8 >, Stride< _8, _1 > >, decltype( Layout< Extent< _3, _2 >, Stride< _2, _1 > >{}.downcast< 4 >() ) > );
static_assert( std::is_same_v< TestLayout,                                                            decltype( TestLayout{}.downcast< 1 >() ) > );
static_assert( std::is_same_v< Layout< Extent< _1, _2, _6 >, Stride< _12, _6, _1 > >,                 decltype( TestLayout{}.downcast< 2 >() ) > );
static_assert( std::is_same_v< Layout< Extent< _1, _2, _9 >, Stride< Constant< int, 18 >, _9, _1 > >, decltype( TestLayout{}.downcast< 3 >() ) > );

static_assert( 0 == Layout< Extent<>, Stride<> >::NDIM );
static_assert( 1 == Layout< Extent< _2 >, Stride< _1 > >::NDIM );
static_assert( 2 == Layout< Extent< _2, _3 >, Stride< _1, _2 > >::NDIM );
static_assert( 2 == Layout< Extent< int, int >, Stride< int, _1 > >::NDIM );

static_assert( -1 == Layout< Extent<>, Stride<> >::USD );
static_assert(  0 == Layout< Extent< _2 >, Stride< _1 > >::USD );
static_assert(  0 == Layout< Extent< _2, _3 >, Stride< _1, _2 > >::USD );
static_assert(  0 == Layout< Extent< int, int >, Stride< _1, int > >::USD );
static_assert(  1 == Layout< Extent< int, int >, Stride< int, _1 > >::USD );
static_assert( -1 == Layout< Extent< int, int >, Stride< int, int > >::USD );
static_assert(  1 == Layout< Extent< int, int, int >, Stride< int, _1, int > >::USD );

static_assert( std::is_assignable_v< TestLayout, TestLayout > );

using TestLayout2 = Layout< Extent< int, int, int >, CompactStride< Extent< int, int, int >, LayoutRight > >;
static_assert( std::is_nothrow_assignable_v< TestLayout, TestLayout2 > );
static_assert( std::is_nothrow_assignable_v< TestLayout2, TestLayout > );
static_assert( std::is_nothrow_constructible_v< TestLayout, TestLayout2 > );
static_assert( std::is_nothrow_constructible_v< TestLayout2, TestLayout > );

using TestLayout3 = Layout< Extent< int, int >, CompactStride< Extent< int, int >, LayoutRight > >;
static_assert( !std::is_nothrow_assignable_v< TestLayout, TestLayout3> );
static_assert( !std::is_nothrow_assignable_v< TestLayout3, TestLayout > );
static_assert( !std::is_nothrow_constructible_v< TestLayout, TestLayout3 > );
static_assert( !std::is_nothrow_constructible_v< TestLayout3, TestLayout > );

} // namespace LvArray

int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
