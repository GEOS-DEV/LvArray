/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "testFloatingPointExceptionsHelpers.hpp"
#include "system.hpp"

// TPL includes
#include <gtest/gtest.h>

// System includes
#include <fenv.h>
#include <cmath>
#include <float.h>

using namespace testFloatingPointExceptionsHelpers;

#if defined(__APPLE__) && defined(__MACH__) && defined(__aarch64__)
char const DIVIDE_BY_ZERO_REGEX[] =
  R"(((floating divide by zero)|(possible floating-point trap; subtype unavailable on this platform))(.|\n)*StackTrace)";
char const OVERFLOW_REGEX[] =
  R"(((floating overflow)|(possible floating-point trap; subtype unavailable on this platform))(.|\n)*StackTrace)";
char const INVALID_REGEX[] =
  R"(((floating invalid operation)|(possible floating-point trap; subtype unavailable on this platform))(.|\n)*StackTrace)";
#else
char const DIVIDE_BY_ZERO_REGEX[] = R"((floating divide by zero)(.|\n)*StackTrace)";
char const OVERFLOW_REGEX[] = R"((floating overflow)(.|\n)*StackTrace)";
char const INVALID_REGEX[] = R"((floating invalid operation)(.|\n)*StackTrace)";
#endif

namespace LvArray
{
namespace testing
{

TEST( TestFloatingPointEnvironment, Underflow )
{
  system::setFPE();
  double fpnum = divide( DBL_MIN, 2 );
  EXPECT_DOUBLE_EQ( fpnum, 0.0 );
}

TEST( TestFloatingPointEnvironment, DivideByZero )
{
  system::setFPE();
  EXPECT_DEATH_IF_SUPPORTED( divide( 1, 0 ), DIVIDE_BY_ZERO_REGEX );
}

TEST( TestFloatingPointEnvironment, Overflow )
{
  system::setFPE();
  EXPECT_DEATH_IF_SUPPORTED( multiply( DBL_MAX, 2 ), OVERFLOW_REGEX );
}

TEST( TestFloatingPointEnvironment, Invalid )
{
  system::setFPE();
  EXPECT_DEATH_IF_SUPPORTED( invalid(), INVALID_REGEX );
}

TEST( TestFloatingPointEnvironment, FloatingPointExceptionGuard )
{
  system::setFPE();

  {
    system::FloatingPointExceptionGuard guard( FE_UNDERFLOW );
    divide( DBL_MIN, 2 );
    EXPECT_DEATH_IF_SUPPORTED( multiply( DBL_MAX, 2 ), OVERFLOW_REGEX );
  }
}

} // namespace testing
} // namespace LvArray


// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{

  LvArray::system::setSignalHandling();

  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
