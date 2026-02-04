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

using namespace testFloatingPointExceptionsHelpers;

const char IGNORE_OUTPUT[] = ".*";

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
  EXPECT_DEATH_IF_SUPPORTED( divide( 1, 0 ), R"((floating divide by zero)(.|\n)*StackTrace)" );
}

TEST( TestFloatingPointEnvironment, Overlow )
{
  system::setFPE();
  EXPECT_DEATH_IF_SUPPORTED( multiply( DBL_MAX, 2 ), R"((floating overflow)(.|\n)*StackTrace)" );
}

TEST( TestFloatingPointEnvironment, Invalid )
{
  system::setFPE();
  EXPECT_DEATH_IF_SUPPORTED( invalid(), R"((floating invalid operation)(.|\n)*StackTrace)" );
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
