/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "system.hpp"
#include "Macros.hpp"

// TPL includes
#include <RAJA/RAJA.hpp>
#include <gtest/gtest.h>


namespace LvArray
{
namespace testing
{

std::string __attribute__((noinline)) baz()
{
  return system::stackTrace( true );
}

std::string __attribute__((noinline)) bar()
{
  return baz();
}

std::string __attribute__((noinline)) foo()
{
  bar();
  std::string const result = bar();
  bar();
  return result;
}

#if defined( RAJA_ENABLE_OPENMP )

TEST( stackTrace, OpenMP )
{
  RAJA::forall< RAJA::omp_parallel_for_exec >( RAJA::TypedRangeSegment< int >( 0, 100 ), []( int ){} );
  LVARRAY_LOG( foo() );
}

#endif

TEST( stackTrace, normal )
{
  LVARRAY_LOG( foo() );
}

} // namespace testing
} // namespace LvArray

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
