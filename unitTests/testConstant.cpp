/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "Constant.hpp"

// TPL includes
#include <gtest/gtest.h>

namespace LvArray
{

static_assert( !is_constant_v< int > );
static_assert( is_constant_v< _0 > );
static_assert( is_constant_v< _1 > );
static_assert( is_constant_v< _t > );
static_assert( is_constant_v< _f > );

static_assert( is_constant_value_v< 0, _0 > );
static_assert( is_constant_value_v< 1, _1 > );
static_assert( is_constant_value_v< 345, Constant< int, 345 > > );
static_assert( is_constant_value_v< true,  _t > );
static_assert( is_constant_value_v< false, _f > );

static_assert( std::is_same_v< decltype( _0{} + _0{} ), _0 > );
static_assert( std::is_same_v< decltype( _0{} + _1{} ), _1 > );
static_assert( std::is_same_v< decltype( _1{} + _0{} ), _1 > );
static_assert( std::is_same_v< decltype( _1{} + _1{} ), _2 > );
static_assert( std::is_same_v< decltype( _2{} + _1{} ), _3 > );
static_assert( std::is_same_v< decltype( _1{} + _1{} + _1{} ), _3 > );

static_assert( std::is_same_v< decltype( _0{} + 0 ), int > );
static_assert( std::is_same_v< decltype( 0 + _0{} ), int > );
static_assert( std::is_same_v< decltype( _1{} + 0 ), int > );
static_assert( std::is_same_v< decltype( 0 + _1{} ), int > );

static_assert( std::is_same_v< decltype( _0{} - _0{} ), _0 > );
static_assert( std::is_same_v< decltype( _1{} - _0{} ), _1 > );
static_assert( std::is_same_v< decltype( _0{} - _1{} ), Constant< int, -1 > > );
static_assert( std::is_same_v< decltype( _1{} - _1{} ), _0 > );

static_assert( std::is_same_v< decltype( _0{} * _0{} ), _0 > );
static_assert( std::is_same_v< decltype( _0{} * _1{} ), _0 > );
static_assert( std::is_same_v< decltype( _1{} * _0{} ), _0 > );
static_assert( std::is_same_v< decltype( _1{} * _1{} ), _1 > );
static_assert( std::is_same_v< decltype( _0{} *    1 ), _0 > );
static_assert( std::is_same_v< decltype(    1 * _0{} ), _0 > );

static_assert( std::is_same_v< decltype( _0{} / _1{} ), _0 > );
static_assert( std::is_same_v< decltype( _1{} / _1{} ), _1 > );
static_assert( std::is_same_v< decltype( _2{} / _1{} ), _2 > );
static_assert( std::is_same_v< decltype( _2{} / _3{} ), _0 > );
static_assert( std::is_same_v< decltype( _4{} / _3{} ), _1 > );
static_assert( std::is_same_v< decltype( _0{} /    1 ), _0 > );

static_assert( std::is_same_v< decltype( _0{} % _1{} ), _0 > );
static_assert( std::is_same_v< decltype( _1{} % _1{} ), _0 > );
static_assert( std::is_same_v< decltype( _2{} % _1{} ), _0 > );
static_assert( std::is_same_v< decltype( _2{} % _3{} ), _2 > );
static_assert( std::is_same_v< decltype( _4{} % _3{} ), _1 > );
static_assert( std::is_same_v< decltype( _4{} % _2{} ), _0 > );
static_assert( std::is_same_v< decltype( _0{} %    1 ), _0 > );

static_assert( std::is_same_v< decltype( _0{} == _0{} ), _t > );
static_assert( std::is_same_v< decltype( _0{} != _0{} ), _f > );
static_assert( std::is_same_v< decltype( _0{} == _1{} ), _f > );
static_assert( std::is_same_v< decltype( _0{} != _1{} ), _t > );
static_assert( std::is_same_v< decltype( _0{} <  _1{} ), _t > );
static_assert( std::is_same_v< decltype( _0{} <= _1{} ), _t > );
static_assert( std::is_same_v< decltype( _1{} <= _1{} ), _t > );
static_assert( std::is_same_v< decltype( _1{} <  _0{} ), _f > );
static_assert( std::is_same_v< decltype( _1{} <= _0{} ), _f > );
static_assert( std::is_same_v< decltype( _1{} >  _0{} ), _t > );
static_assert( std::is_same_v< decltype( _1{} >= _0{} ), _t > );
static_assert( std::is_same_v< decltype( _0{} >  _0{} ), _f > );
static_assert( std::is_same_v< decltype( _0{} >= _1{} ), _f > );

static_assert( std::is_nothrow_assignable_v< Constant< int, 1 >, int > );
static_assert( std::is_nothrow_assignable_v< int&, Constant< int, 1 > > );

using namespace literals;

static_assert( std::is_same_v< decltype( 0_Ci ),   Constant< int, 0 > > );
static_assert( std::is_same_v< decltype( 1_Ci ),   Constant< int, 1 > > );
static_assert( std::is_same_v< decltype( 345_Ci ), Constant< int, 345 > > );

static_assert( std::is_same_v< decltype( 0_Cull ),   Constant< unsigned long long, 0 > > );
static_assert( std::is_same_v< decltype( 1_Cull ),   Constant< unsigned long long, 1 > > );
static_assert( std::is_same_v< decltype( 345_Cull ), Constant< unsigned long long, 345 > > );

} // namespace LvArray

int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
