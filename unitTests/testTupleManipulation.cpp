/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "Constant.hpp"
#include "tupleManipulation.hpp"

// TPL includes
#include <gtest/gtest.h>

namespace LvArray
{

namespace tupleManipulation
{

static_assert( std::is_same_v< camp::tuple<>,         decltype( repeat< 0 >( _1{} ) ) > );
static_assert( std::is_same_v< camp::tuple< _1 >,     decltype( repeat< 1 >( _1{} ) ) > );
static_assert( std::is_same_v< camp::tuple< _1, _1 >, decltype( repeat< 2 >( _1{} ) ) > );

static_assert( std::is_same_v< camp::tuple< _0 >,      decltype( replace< 0 >( makeTuple( _1{} ), _0{} ) ) > );
static_assert( std::is_same_v< camp::tuple< _0, _2 >,  decltype( replace< 0 >( makeTuple( _1{}, _2{} ), _0{} ) ) > );
static_assert( std::is_same_v< camp::tuple< _1, _0 >,  decltype( replace< 1 >( makeTuple( _1{}, _2{} ), _0{} ) ) > );
static_assert( std::is_same_v< camp::tuple< int, _2 >, decltype( replace< 0 >( makeTuple( _1{}, _2{} ), 0 ) ) > );
static_assert( std::is_same_v< camp::tuple< _1, int >, decltype( replace< 1 >( makeTuple( _1{}, _2{} ), 0 ) ) > );

static_assert( std::is_same_v< camp::tuple< _1 >,         decltype( append( makeTuple(), _1{} ) ) > );
static_assert( std::is_same_v< camp::tuple< _1, _1, _1 >, decltype( append<3>( makeTuple(), _1{} ) ) > );
static_assert( std::is_same_v< camp::tuple< _1, _2, _3 >, decltype( append( makeTuple( _1{}, _2{} ), _3{} ) ) > );

static_assert( std::is_same_v< camp::tuple< _1 >,         decltype( prepend( makeTuple(), _1{} ) ) > );
static_assert( std::is_same_v< camp::tuple< _1, _1, _1 >, decltype( prepend<3>( makeTuple(), _1{} ) ) > );
static_assert( std::is_same_v< camp::tuple< _3, _1, _2 >, decltype( prepend( makeTuple( _1{}, _2{} ), _3{} ) ) > );

static_assert( std::is_same_v< camp::tuple< _1, _1 >,         decltype( insert<0,2>( makeTuple(), _1{} ) ) > );
static_assert( std::is_same_v< camp::tuple< _0, _1, _1, _0 >, decltype( insert<1,2>( makeTuple( _0{}, _0{} ), _1{} ) ) > );

static_assert( std::is_same_v< camp::tuple<>,         decltype( remove<0,2>( makeTuple( _1{}, _1{} ) ) ) > );
static_assert( std::is_same_v< camp::tuple< _0, _0 >, decltype( remove<1,2>( makeTuple( _0{}, _1{}, _1{}, _0{} ) ) ) > );

static_assert( std::is_same_v< camp::tuple<>,         decltype( select( makeTuple( _1{}, _2{}, _3{} ), camp::idx_seq<>{} ) ) > );
static_assert( std::is_same_v< camp::tuple< _1, _3 >, decltype( select( makeTuple( _1{}, _2{}, _3{} ), camp::idx_seq<0,2>{} ) ) > );

static_assert( std::is_same_v< camp::tuple<>,         decltype( take<1,1>( makeTuple( _1{}, _2{}, _3{} ) ) ) > );
static_assert( std::is_same_v< camp::tuple< _2, _3 >, decltype( take<1,3>( makeTuple( _1{}, _2{}, _3{} ) ) ) > );

static_assert( std::is_same_v< camp::tuple<>,                                               decltype( zip2( makeTuple(), makeTuple() ) ) > );
static_assert( std::is_same_v< camp::tuple< camp::tuple<>, camp::tuple<> >,                 decltype( unzip2( makeTuple() ) ) > );
static_assert( std::is_same_v< camp::tuple< camp::tuple< _1, _2 > >,                        decltype( zip2( makeTuple( _1{} ), makeTuple( _2{} ) ) ) > );
static_assert( std::is_same_v< camp::tuple< camp::tuple< _1 >, camp::tuple< _2 > >,         decltype( unzip2( make_tuple( makeTuple( _1{}, _2{} ) ) ) ) > );
static_assert( std::is_same_v< camp::tuple< camp::tuple< _1, _2 >, camp::tuple< _3, _4 > >, decltype( zip2( makeTuple( _1{}, _3{} ), makeTuple( _2{}, _4{} ) ) ) > );
static_assert( std::is_same_v< camp::tuple< camp::tuple< _1, _3 >, camp::tuple< _2, _4 > >, decltype( unzip2( makeTuple(  makeTuple( _1{}, _2{} ), makeTuple( _3{}, _4{} ) ) ) ) > );

static_assert( std::is_same_v< _1, decltype( foldLeft( makeTuple(), _1{}, op_plus{} ) ) > );
static_assert( std::is_same_v< _6, decltype( foldLeft( makeTuple( _1{}, _2{}, _3{} ), _1{}, op_mult{} ) ) > );
static_assert( std::is_same_v< _0, decltype( foldLeft( makeTuple( _1{}, _2{}, _3{} ), _6{}, op_minus{} ) ) > );

static_assert( std::is_same_v< _1, decltype( foldRight( makeTuple(), _1{}, op_plus{} ) ) > );
static_assert( std::is_same_v< _6, decltype( foldRight( makeTuple( _1{}, _2{}, _3{} ), _1{}, op_mult{} ) ) > );
static_assert( std::is_same_v< _0, decltype( foldRight( makeTuple( _1{}, _2{}, _3{} ), _2{}, op_minus{} ) ) > );

static_assert( std::is_same_v< _t, decltype( allOf2( makeTuple( _1{}, _2{} ), makeTuple( _2{}, _3{} ), cmp_lt{} ) ) > );
static_assert( std::is_same_v< _t, decltype( allOf2( makeTuple( _1{}, _2{} ), makeTuple( _2{}, _3{} ), cmp_le{} ) ) > );
static_assert( std::is_same_v< _f, decltype( allOf2( makeTuple( _1{}, _2{} ), makeTuple( _2{}, _3{} ), cmp_gt{} ) ) > );
static_assert( std::is_same_v< _f, decltype( allOf2( makeTuple( _1{}, _2{} ), makeTuple( _2{}, _3{} ), cmp_ge{} ) ) > );
static_assert( std::is_same_v< _t, decltype( allOf2( makeTuple( _1{}, _2{} ), makeTuple( _1{}, _2{} ), cmp_eq{} ) ) > );
static_assert( std::is_same_v< _f, decltype( allOf2( makeTuple( _1{}, _2{} ), makeTuple( _1{}, _3{} ), cmp_eq{} ) ) > );
static_assert( std::is_same_v< _f, decltype( anyOf2( makeTuple( _1{}, _2{} ), makeTuple( _1{}, _2{} ), cmp_ne{} ) ) > );
static_assert( std::is_same_v< _t, decltype( anyOf2( makeTuple( _1{}, _2{} ), makeTuple( _1{}, _3{} ), cmp_ne{} ) ) > );

} // namespace tupleManipulation

} // namespace LvArray

int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
