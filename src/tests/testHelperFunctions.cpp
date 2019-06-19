/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2019, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-746361
 *
 * All rights reserved. See COPYRIGHT for details.
 *
 * This file is part of the GEOSX Simulation Framework.
 *
 * GEOSX is a free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */


#include <iostream>
#include "SetFPE.hpp"
#include "SetSignalHandling.hpp"
#include "stackTrace.hpp"
#include "testHelperFunctions.hpp"

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
  cxx_utilities::setSignalHandling( cxx_utilities::handler1 );
  func0( divisor );
}

//TEST(testStackTrace_DeathTest, stackTrace)
//{
//   EXPECT_DEATH_IF_SUPPORTED(testStackTrace(0), IGNORE_OUTPUT);
//}



#if !defined(__APPLE__) && defined(__clang__) && !defined(USE_CUDA)
#pragma STDC FENV_ACCESS ON
#endif
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
  return x/denominator;
}

double of_test( double x, double y )
{
  return x*y;
}

double invalid_test( double x )
{
  return std::acos( 2.0 );
}

}
