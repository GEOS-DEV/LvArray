/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
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

namespace LvArray
{
namespace testing
{

TEST( TestFloatingPointEnvironment, Underflow )
{
  system::enableFloatingPointExceptions( FE_UNDERFLOW );
  EXPECT_DEATH_IF_SUPPORTED( divide( DBL_MIN, 2 ), IGNORE_OUTPUT );
  system::disableFloatingPointExceptions( FE_UNDERFLOW );

  system::setFPE();
  double fpnum = divide( DBL_MIN, 2 );
  int fpclassification = std::fpclassify( fpnum );
  EXPECT_NE( fpclassification, FP_SUBNORMAL );
}

TEST( TestFloatingPointEnvironment, DivideByZero )
{
  system::setFPE();
  EXPECT_DEATH_IF_SUPPORTED( divide( 1, 0 ), IGNORE_OUTPUT );
}

TEST( TestFloatingPointEnvironment, Overlow )
{
  system::setFPE();
  EXPECT_DEATH_IF_SUPPORTED( multiply( DBL_MAX, 2 ), IGNORE_OUTPUT );
}

TEST( TestFloatingPointEnvironment, Invalid )
{
  system::setFPE();
  EXPECT_DEATH_IF_SUPPORTED( invalid(), IGNORE_OUTPUT );
}

TEST( TestFloatingPointEnvironment, FloatingPointExceptionGuard )
{
  system::setFPE();

  {
    system::FloatingPointExceptionGuard guard( FE_UNDERFLOW );
    divide( DBL_MIN, 2 );
    EXPECT_DEATH_IF_SUPPORTED( multiply( DBL_MAX, 2 ), IGNORE_OUTPUT );
  }
}

} // namespace testing
} // namespace LvArray

#endif

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
