/*
 * Copyright (c) 2020, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */


#include <iostream>
#include "Macros.hpp"
#include "system.hpp"
#include "system.hpp"
#include "testFloatingPointExceptionsHelpers.hpp"

namespace testFloatingPointExceptionsHelpers
{

void func3( double divisor )
{
  double a = 1.0 / divisor;
  std::cout << "1.0/0.0 didn't kill program, result is " << a;
}

void func2( double divisor )
{
  func3( divisor );
}

void func1( double divisor )
{
  func2( divisor );
}

void func0( double divisor )
{
  func1( divisor );
}

void testStackTrace( double divisor )
{
  LvArray::system::setSignalHandling( []( int const signal ) { LvArray::system::stackTraceHandler( signal, true ); } );
  func0( divisor );
}

//TEST(testStackTrace_DeathTest, stackTrace)
//{
//   EXPECT_DEATH_IF_SUPPORTED(testStackTrace(0), IGNORE_OUTPUT);
//}



void show_fe_exceptions( void )
{
  printf( "exceptions raised:" );
  if( fetestexcept( FE_DIVBYZERO ))
    printf( " FE_DIVBYZERO" );
  if( fetestexcept( FE_INEXACT ))
    printf( " FE_INEXACT" );
  if( fetestexcept( FE_INVALID ))
    printf( " FE_INVALID" );
  if( fetestexcept( FE_OVERFLOW ))
    printf( " FE_OVERFLOW" );
  if( fetestexcept( FE_UNDERFLOW ))
    printf( " FE_UNDERFLOW" );
  feclearexcept( FE_ALL_EXCEPT );
  printf( "\n" );
}

double uf_test( double x, double denominator )
{
  return x / denominator;
}

double of_test( double x, double y )
{
  return x * y;
}

double invalid_test( double LVARRAY_UNUSED_ARG( x ) )
{
  return std::acos( 2.0 );
}

}
