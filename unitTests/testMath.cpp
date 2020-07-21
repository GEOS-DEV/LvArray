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

// Source includes
#include "math.hpp"
#include "testUtils.hpp"
#include "limits.hpp"

// TPL includes
#include <gtest/gtest.h>

namespace LvArray
{
namespace testing
{

template< typename T_POLICY_PAIR >
struct TestMath : public ::testing::Test
{
  using T = typename T_POLICY_PAIR::first_type;
  using POLICY = typename T_POLICY_PAIR::second_type;

  void maxAndMin()
  {
    forall< POLICY >( 1, [] LVARRAY_HOST_DEVICE ( int )
        {
          T a = 7;
          T b = -55;
          PORTABLE_EXPECT_EQ( math::max( a, b ), a );
          PORTABLE_EXPECT_EQ( math::min( a, b ), b );

          a = 120;
          b = 240;
          PORTABLE_EXPECT_EQ( math::max( a, b ), b );
          PORTABLE_EXPECT_EQ( math::min( a, b ), a );
        } );
  }

  void abs()
  {
    forall< POLICY >( 1, [] LVARRAY_HOST_DEVICE ( int )
        {
          T a = 7;
          PORTABLE_EXPECT_EQ( math::abs( a ), a );

          a = -55;
          PORTABLE_EXPECT_EQ( math::abs( a ), -a );
        } );
  }

  void sqrtAndInvSqrt()
  {
    using FloatingPoint = decltype( math::sqrt( T() ) );
    FloatingPoint const epsilon = NumericLimits< FloatingPoint >::epsilon;
    forall< POLICY >( 1, [epsilon] LVARRAY_HOST_DEVICE ( int )
        {

          T a = 5 * 5;
          PORTABLE_EXPECT_EQ( math::sqrt( a ), 5.0 );
          PORTABLE_EXPECT_NEAR( math::invSqrt( a ), 1.0 / 5.0, epsilon );

          a = 12 * 12;
          PORTABLE_EXPECT_EQ( math::sqrt( a ), 12.0 );
          PORTABLE_EXPECT_NEAR( math::invSqrt( a ), 1.0 / 12.0, epsilon );
        } );
  }
};


using TestMathTypes = ::testing::Types<
  std::pair< int, serialPolicy >
  , std::pair< long int, serialPolicy >
  , std::pair< long long int, serialPolicy >
  , std::pair< float, serialPolicy >
  , std::pair< double, serialPolicy >
#if defined(USE_CUDA)
  , std::pair< int, parallelDevicePolicy< 32 > >
  , std::pair< long int, parallelDevicePolicy< 32 > >
  , std::pair< long long int, parallelDevicePolicy< 32 > >
  , std::pair< float, parallelDevicePolicy< 32 > >
  , std::pair< double, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( TestMath, TestMathTypes, );

TYPED_TEST( TestMath, maxAndMin )
{
  this->maxAndMin();
}

TYPED_TEST( TestMath, abs )
{
  this->abs();
}

TYPED_TEST( TestMath, sqrtAndInvSqrt )
{
  this->sqrtAndInvSqrt();
}


template< typename FLOATING_POINT_POLICY_PAIR >
struct TestMathFloatingPointOnly : public ::testing::Test
{
  using FloatingPoint = typename FLOATING_POINT_POLICY_PAIR::first_type;
  using POLICY = typename FLOATING_POINT_POLICY_PAIR::second_type;

  void trig()
  {
    FloatingPoint const epsilon = NumericLimits< FloatingPoint >::epsilon;
    forall< POLICY >( 1, [epsilon] LVARRAY_HOST_DEVICE ( int )
        {
          FloatingPoint coords[ 2 ][ 3 ] = { { 1, 2, 1.10714871779409050301 },
            { 4, -1, -0.24497866312686415417 } };

          for( int i = 0; i < 2; ++i )
          {
            FloatingPoint const x = coords[ i ][ 0 ];
            FloatingPoint const y = coords[ i ][ 1 ];
            FloatingPoint const theta = coords[ i ][ 2 ];

            FloatingPoint const sinExpected = y / math::sqrt( x * x + y * y );
            FloatingPoint const cosExpected = x / math::sqrt( x * x + y * y );
            FloatingPoint const tanExpected = y / x;

            PORTABLE_EXPECT_NEAR( math::sin( theta ), sinExpected, epsilon );
            PORTABLE_EXPECT_NEAR( math::cos( theta ), cosExpected, epsilon );
            PORTABLE_EXPECT_NEAR( math::tan( theta ), tanExpected, 2.1 * epsilon );

            FloatingPoint sinTheta;
            FloatingPoint cosTheta;
            math::sincos( theta, sinTheta, cosTheta );
            PORTABLE_EXPECT_NEAR( math::cos( theta ), cosTheta, epsilon );
            PORTABLE_EXPECT_NEAR( math::sin( theta ), sinTheta, epsilon );

            PORTABLE_EXPECT_NEAR( math::abs( math::asin( sinTheta ) ), math::abs( theta ), 1.1 * epsilon );
            PORTABLE_EXPECT_NEAR( math::abs( math::acos( cosTheta ) ), math::abs( theta ), 1.1 * epsilon );
            PORTABLE_EXPECT_NEAR( math::atan2( y, x ), theta, epsilon );
          }
        } );
  }
};

using TestMathFloatingPointOnlyTypes = ::testing::Types<
  std::pair< float, serialPolicy >
  , std::pair< double, serialPolicy >
#if defined(USE_CUDA)
  , std::pair< float, parallelDevicePolicy< 32 > >
  , std::pair< double, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( TestMathFloatingPointOnly, TestMathFloatingPointOnlyTypes, );

TYPED_TEST( TestMathFloatingPointOnly, trig )
{
  this->trig();
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
