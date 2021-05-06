/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */


#include "Macros.hpp"
#include "system.hpp"
#include "system.hpp"
#include "testFloatingPointExceptionsHelpers.hpp"

#include <fenv.h>
#include <cmath>
#include <float.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#include <iostream>
#endif

namespace testFloatingPointExceptionsHelpers
{

double multiply( double const x, double const y )
{ return x * y; }

double divide( double const x, double const y )
{ return x / y; }

double invalid()
{ return std::acos( 2.0 ); }

}
