/*
 * Copyright (c) 2020, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "testFloatingPointExceptionsHelpers.hpp"
#include "system.hpp"
#include "system.hpp"

// TPL includes
#include <gtest/gtest.h>

// System includes
#include <fenv.h>
#include <cmath>
#include <float.h>


#if defined(__x86_64__)

using namespace testFloatingPointExceptionsHelpers;

const char IGNORE_OUTPUT[] = ".*";

TEST( TestFloatingPointEnvironment, test_FE_UNDERFLOW_flush )
{
  LvArray::system::enableFloatingPointExceptions( FE_UNDERFLOW );
  EXPECT_DEATH_IF_SUPPORTED( uf_test( DBL_MIN, 2 ), IGNORE_OUTPUT );
  LvArray::system::disableFloatingPointExceptions( FE_UNDERFLOW );

  LvArray::system::setFPE();
  double fpnum = uf_test( DBL_MIN, 2 );
  int fpclassification = std::fpclassify( fpnum );
  EXPECT_NE( fpclassification, FP_SUBNORMAL );
}

TEST( TestFloatingPointEnvironment, test_FE_DIVBYZERO )
{
  LvArray::system::setFPE();
  EXPECT_DEATH_IF_SUPPORTED( func3( 0.0 ), IGNORE_OUTPUT );
}


TEST( TestFloatingPointEnvironment, test_FE_OVERFLOW )
{
  LvArray::system::setFPE();
  EXPECT_DEATH_IF_SUPPORTED( of_test( 2, DBL_MAX ), IGNORE_OUTPUT );
}

TEST( TestFloatingPointEnvironment, test_FE_INVALID )
{
  LvArray::system::setFPE();
  EXPECT_DEATH_IF_SUPPORTED( invalid_test( 0.0 ), IGNORE_OUTPUT );
}

#endif
