/*
 * Copyright (c) 2020, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */



#include <fenv.h>
#include <cmath>
#include <float.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace testFloatingPointExceptionsHelpers
{
void func3( double divisor );

void func2( double divisor );

void func1( double divisor );

void func0( double divisor );

void testStackTrace( double divisor );

void show_fe_exceptions( void );

double uf_test( double x, double denominator );

double of_test( double x, double y );

double invalid_test( double x );
}
