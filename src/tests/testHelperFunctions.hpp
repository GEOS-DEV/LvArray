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



#include <fenv.h>
#include <cmath>
#include <float.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace testFloatingPointExceptionsHelpers
{
void func3(double divisor);

void func2(double divisor);

void func1(double divisor);

void func0(double divisor);

void testStackTrace(double divisor);

void show_fe_exceptions(void);

double uf_test(double x, double denominator);

double of_test( double x, double y );

double invalid_test( double x );
}

