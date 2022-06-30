/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "math.hpp"
#include "testUtils.hpp"
#include "limits.hpp"
#include "output.hpp"

// TPL includes
#include <gtest/gtest.h>

namespace LvArray
{
LVARRAY_DEVICE
double diffModulo2PI( double result, double expected )
{
  double diff = LvArray::math::abs( fmod( result - expected, 2*M_PI ) );
  if( diff > M_PI )
    diff = LvArray::math::abs( diff - 2 * M_PI );
  return diff;
}

namespace testing
{

template< typename T_POLICY_PAIR >
struct TestMath : public ::testing::Test
{
  using T = typename T_POLICY_PAIR::first_type;
  using POLICY = typename T_POLICY_PAIR::second_type;

  void numValues()
  {
    constexpr int N = math::numValues< T >();
    static_assert( N == 1, "Should be one." );
  }

  void convert()
  {
    forall< POLICY >( 1, [] LVARRAY_HOST_DEVICE ( int )
        {
          T const x = math::convert< T >( 5 );
          PORTABLE_EXPECT_EQ( x, T( 5 ) );
        } );
  }

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

  void square()
  {
    forall< POLICY >( 1, [] LVARRAY_HOST_DEVICE ( int )
        {
          PORTABLE_EXPECT_EQ( math::square( T( 3 ) ), T( 9 ) );
        } );
  }

  void sqrtAndInvSqrt()
  {
    using FloatingPoint = decltype( math::sqrt( T() ) );
    forall< POLICY >( 1, [] LVARRAY_HOST_DEVICE ( int )
        {
          FloatingPoint const epsilon = NumericLimitsNC< FloatingPoint >{}.epsilon;

          T a = 5 * 5;
          PORTABLE_EXPECT_EQ( math::sqrt( a ), FloatingPoint( 5.0 ) );
          PORTABLE_EXPECT_NEAR( math::invSqrt( a ), FloatingPoint( 1.0 / 5.0 ), epsilon );

          a = 12 * 12;
          PORTABLE_EXPECT_EQ( math::sqrt( a ), FloatingPoint( 12.0 ) );
          PORTABLE_EXPECT_NEAR( math::invSqrt( a ), FloatingPoint( 1.0 / 12.0 ), epsilon );
        } );
  }

  void trig()
  {
    using FloatingPoint = decltype( math::sin( T() ) );
    forall< POLICY >( 1, [] LVARRAY_HOST_DEVICE ( int )
        {
          FloatingPoint const epsilon = NumericLimitsNC< FloatingPoint >{}.epsilon;

          PORTABLE_EXPECT_NEAR( math::sin( T( 10 ) ), FloatingPoint( ::sin( 10.0 ) ), epsilon );
          PORTABLE_EXPECT_NEAR( math::cos( T( 10 ) ), FloatingPoint( ::cos( 10.0 ) ), epsilon );

          FloatingPoint s, c;
          math::sincos( T( 10 ), s, c );
          PORTABLE_EXPECT_NEAR( s, FloatingPoint( ::sin( 10.0 ) ), epsilon );
          PORTABLE_EXPECT_NEAR( c, FloatingPoint( ::cos( 10.0 ) ), epsilon );

          PORTABLE_EXPECT_NEAR( math::tan( T( 10 ) ), FloatingPoint( ::tan( 10.0 ) ), epsilon );

          PORTABLE_EXPECT_NEAR( math::asin( T( 1 ) ), FloatingPoint( ::asin( 1.0 ) ), epsilon );
          PORTABLE_EXPECT_NEAR( math::acos( T( 0 ) ), FloatingPoint( ::acos( 0.0 ) ), epsilon );
          PORTABLE_EXPECT_NEAR( math::atan2( T( 1 ), T( 2 ) ), FloatingPoint( ::atan2( 1.0, 2.0 ) ), epsilon );
        } );
  }

  void exponential()
  {
    using FloatingPoint = decltype( math::exp( T() ) );
    forall< POLICY >( 1, [] LVARRAY_HOST_DEVICE ( int )
        {
          FloatingPoint const epsilon = NumericLimitsNC< FloatingPoint >{}.epsilon;

          PORTABLE_EXPECT_NEAR( math::exp( T( 2 ) ), FloatingPoint( ::exp( 2.0 ) ), epsilon );
          PORTABLE_EXPECT_NEAR( math::log( T( 2 ) ), FloatingPoint( ::log( 2.0 ) ), epsilon );
        } );
  }
};


using TestMathTypes = ::testing::Types<
  std::pair< int, serialPolicy >
  , std::pair< long int, serialPolicy >
  , std::pair< long long int, serialPolicy >
  , std::pair< float, serialPolicy >
  , std::pair< double, serialPolicy >
#if defined( LVARRAY_USE_CUDA ) || defined( LVARRAY_USE_HIP )
  , std::pair< int, parallelDevicePolicy< 32 > >
  , std::pair< long int, parallelDevicePolicy< 32 > >
  , std::pair< long long int, parallelDevicePolicy< 32 > >
  , std::pair< float, parallelDevicePolicy< 32 > >
  , std::pair< double, parallelDevicePolicy< 32 > >
#endif
#if defined( LVARRAY_USE_CUDA )
  , std::pair< __half, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( TestMath, TestMathTypes, );

TYPED_TEST( TestMath, numValues )
{
  this->numValues();
}

TYPED_TEST( TestMath, convert )
{
  this->convert();
}

TYPED_TEST( TestMath, maxAndMin )
{
  this->maxAndMin();
}

TYPED_TEST( TestMath, abs )
{
  this->abs();
}

TYPED_TEST( TestMath, square )
{
  this->square();
}

TYPED_TEST( TestMath, sqrtAndInvSqrt )
{
  this->sqrtAndInvSqrt();
}

TYPED_TEST( TestMath, trig )
{
  this->trig();
}

TYPED_TEST( TestMath, exponential )
{
  this->exponential();
}


template< typename T_POLICY_PAIR >
struct TestMath2 : public ::testing::Test
{
  using T = typename T_POLICY_PAIR::first_type;
  using POLICY = typename T_POLICY_PAIR::second_type;
  using SingleT = math::SingleType< T >;

  void numValues()
  {
    constexpr int N = math::numValues< T >();
    static_assert( N == 2, "Should be two." );
  }

  void convert()
  {
    forall< POLICY >( 1, [] LVARRAY_HOST_DEVICE ( int )
        {
          T x = math::convert< T >( 5 );
          PORTABLE_EXPECT_EQ( math::getFirst( x ), SingleT( 5 ) );
          PORTABLE_EXPECT_EQ( math::getSecond( x ), SingleT( 5 ) );

          x = math::convert< T >( SingleT( 3 ) );
          PORTABLE_EXPECT_EQ( math::getFirst( x ), SingleT( 3 ) );
          PORTABLE_EXPECT_EQ( math::getSecond( x ), SingleT( 3 ) );

          x = math::convert< T >( SingleT( 1 ), SingleT( 2 ) );
          PORTABLE_EXPECT_EQ( math::getFirst( x ), SingleT( 1 ) );
          PORTABLE_EXPECT_EQ( math::getSecond( x ), SingleT( 2 ) );

          x = math::convert< T >( 5, 6 );
          PORTABLE_EXPECT_EQ( math::getFirst( x ), SingleT( 5 ) );
          PORTABLE_EXPECT_EQ( math::getSecond( x ), SingleT( 6 ) );

          SingleT const * const xPtr = reinterpret_cast< SingleT const * >( &x );
          PORTABLE_EXPECT_EQ( xPtr[ 0 ], SingleT( 5 ) );
          PORTABLE_EXPECT_EQ( xPtr[ 1 ], SingleT( 6 ) );
        } );
  }

  void maxAndMin()
  {
    forall< POLICY >( 1, [] LVARRAY_HOST_DEVICE ( int )
        {
          T a = math::convert< T >( -60, 10 );
          T b = math::convert< T >( 64, 13 );
          PORTABLE_EXPECT_EQ( math::max( a, b ), b );
          PORTABLE_EXPECT_EQ( math::min( a, b ), a );

          a = math::convert< T >( 7, 100 );
          b = math::convert< T >( -55, 110 );
          PORTABLE_EXPECT_EQ( math::max( a, b ), math::convert< T >( 7, 110 ) );
          PORTABLE_EXPECT_EQ( math::min( a, b ), math::convert< T >( -55, 100 ) );
        } );
  }

  void abs()
  {
    forall< POLICY >( 1, [] LVARRAY_HOST_DEVICE ( int )
        {
          T a = math::convert< T >( 7, 8 );
          PORTABLE_EXPECT_EQ( math::abs( a ), a );

          a = math::convert< T >( -101, -4 );
          PORTABLE_EXPECT_EQ( math::abs( a ), -a );

          a = math::convert< T >( 4, -8 );
          PORTABLE_EXPECT_EQ( math::abs( a ), math::convert< T >( 4, 8 ) );
        } );
  }

  void square()
  {
    forall< POLICY >( 1, [] LVARRAY_HOST_DEVICE ( int )
        {
          T const a = math::convert< T >( 7, 8 );
          PORTABLE_EXPECT_EQ( math::square( a ), math::convert< T >( 49, 64 ) );
        } );
  }

  void sqrtAndInvSqrt()
  {
    forall< POLICY >( 1, [] LVARRAY_HOST_DEVICE ( int )
        {
          T const epsilon = math::convert< T >( NumericLimitsNC< SingleT >{}.epsilon );

          T a = math::convert< T >( 5 * 5, 12 * 12 );
          PORTABLE_EXPECT_EQ( math::sqrt( a ), math::convert< T >( 5.0, 12 ) );
          PORTABLE_EXPECT_NEAR( math::invSqrt( a ), math::convert< T >( 1.0 / 5.0, 1.0 / 12.0 ), epsilon );
        } );
  }

  void trig()
  {
    forall< POLICY >( 1, [] LVARRAY_HOST_DEVICE ( int )
        {
          T const epsilon = math::convert< T >( NumericLimitsNC< SingleT >{}.epsilon );

          PORTABLE_EXPECT_NEAR( math::sin( math::convert< T >( 1, 2 ) ),
                                math::convert< T >( ::sin( 1.0 ), ::sin( 2.0 ) ), epsilon );
          PORTABLE_EXPECT_NEAR( math::cos( math::convert< T >( 1, 2 ) ),
                                math::convert< T >( ::cos( 1.0 ), ::cos( 2.0 ) ), epsilon );

          T s, c;
          math::sincos( math::convert< T >( 1, 2 ), s, c );
          PORTABLE_EXPECT_NEAR( s, math::convert< T >( ::sin( 1.0 ), ::sin( 2.0 ) ), epsilon );
          PORTABLE_EXPECT_NEAR( c, math::convert< T >( ::cos( 1.0 ), ::cos( 2.0 ) ), epsilon );

          T expected = math::convert< T >( ::tan( 1.0 ), ::tan( 2.0 ) );
          PORTABLE_EXPECT_NEAR( math::tan( math::convert< T >( 1, 2 ) ), expected, expected * epsilon );

          PORTABLE_EXPECT_NEAR( math::asin( math::convert< T >( 0.25, 0.75 ) ),
                                math::convert< T >( ::asin( 0.25 ), ::asin( 0.75 ) ), epsilon );

          PORTABLE_EXPECT_NEAR( math::acos( math::convert< T >( 0.25, 0.75 ) ),
                                math::convert< T >( ::acos( 0.25 ), ::acos( 0.75 ) ), epsilon );

          PORTABLE_EXPECT_NEAR( math::atan2( math::convert< T >( 1 ), math::convert< T >( 2, -3 ) ),
                                math::convert< T >( ::atan2( 1.0, 2.0 ), ::atan2( 1.0, -3.0 ) ), epsilon );
        } );
  }

  void exponential()
  {
    forall< POLICY >( 1, [] LVARRAY_HOST_DEVICE ( int )
        {
          T const epsilon = math::convert< T >( NumericLimitsNC< SingleT >{}.epsilon );

          PORTABLE_EXPECT_NEAR( math::exp( math::convert< T >( 2, 3 ) ), math::convert< T >( 7.38905609893, 20.0855369232 ), epsilon );
          PORTABLE_EXPECT_NEAR( math::log( math::convert< T >( 2, 3 ) ), math::convert< T >( 0.69314718056, 1.09861228867 ), epsilon );
        } );
  }
};

#if defined( LVARRAY_USE_CUDA ) || defined( LVARRAY_USE_HIP )

using TestMath2Types = ::testing::Types<
  std::pair< __half2, parallelDevicePolicy< 32 > >
  >;

TYPED_TEST_SUITE( TestMath2, TestMath2Types, );

TYPED_TEST( TestMath2, numValues )
{
  this->numValues();
}

TYPED_TEST( TestMath2, convert )
{
  this->convert();
}

TYPED_TEST( TestMath2, maxAndMin )
{
  this->maxAndMin();
}

TYPED_TEST( TestMath2, abs )
{
  this->abs();
}

TYPED_TEST( TestMath2, square )
{
  this->square();
}

TYPED_TEST( TestMath2, sqrtAndInvSqrt )
{
  this->sqrtAndInvSqrt();
}

TYPED_TEST( TestMath2, trig )
{
  this->trig();
}

TYPED_TEST( TestMath2, exponential )
{
  this->exponential();
}

template< typename LAMBDA >
void forAllHalvesinMinus1to1( bool const include1, LAMBDA && lambda )
{
  forall< parallelDevicePolicy< 32 > >( 1, [include1, lambda] LVARRAY_DEVICE ( int )
      {
        __half limit( 1.0 / ( 1 << 13 ) );
        __half increment( 1.0 / ( 1 << 24 ) );

        for( __half x( 0 ); x < __half( 1 ); limit *= 2, increment *= 2 )
        {
          for(; x < limit; x += increment )
          {
            lambda( -x );
            lambda( +x );
          }
        }

        if( include1 )
        {
          lambda( __half( -1 ) );
          lambda( __half( +1 ) );
        }
      } );
}
#endif
#if defined(LVARRAY_USE_CUDA)
void asinHalfAccuracy()
{
  RAJA::ReduceMax< RAJA::cuda_reduce, double > maxDiff( 0 );
  RAJA::ReduceMax< RAJA::cuda_reduce, double > maxRDiff( 0 );

  forAllHalvesinMinus1to1( true, [maxDiff, maxRDiff] LVARRAY_DEVICE ( __half const x )
      {
        double const result = math::asin( x );
        double const expected = math::asin( double( x ) );

        double const diff = LvArray::math::abs( result - expected );
        double const rdiff = math::abs( diff / expected );

        maxDiff.max( diff );
        maxRDiff.max( rdiff );
      } );

  EXPECT_LT( maxDiff.get(), 1.5 * float( NumericLimitsNC< __half >{}.epsilon ) );
  EXPECT_LT( maxRDiff.get(), 8 * float( NumericLimitsNC< __half >{}.epsilon ) );

  printf( "Max difference          is %.8e\n", maxDiff.get() );
  printf( "Max relative difference is %.8e\n", maxRDiff.get() );
}

void asinHalf2Accuracy()
{
  RAJA::ReduceMax< RAJA::cuda_reduce, double > maxDiff( 0 );
  RAJA::ReduceMax< RAJA::cuda_reduce, double > maxRDiff( 0 );

  forAllHalvesinMinus1to1( true, [maxDiff, maxRDiff] LVARRAY_DEVICE ( __half const x )
      {
        __half2 const result = math::asin( math::convert< __half2 >( x, -x ) );
        double const expected0 = math::asin( double( x ) );
        double const expected1 = math::asin( double( -x ) );

        {
          double const diff0 = LvArray::math::abs( double( math::getFirst( result ) ) - expected0 );
          double const rdiff0 = math::abs( diff0 / expected0 );
          maxDiff.max( diff0 );
          maxRDiff.max( rdiff0 );
        }

        {
          double const diff1 = LvArray::math::abs( double( math::getSecond( result ) ) - expected1 );
          double const rdiff1 = math::abs( diff1 / expected1 );
          maxDiff.max( diff1 );
          maxRDiff.max( rdiff1 );
        }
      } );

  EXPECT_LT( maxDiff.get(), 1.5 * float( NumericLimitsNC< __half >{}.epsilon ) );
  EXPECT_LT( maxRDiff.get(), 8 * float( NumericLimitsNC< __half >{}.epsilon ) );

  printf( "Max difference          is %.8e\n", maxDiff.get() );
  printf( "Max relative difference is %.8e\n", maxRDiff.get() );
}

void acosHalfAccuracy()
{
  RAJA::ReduceMax< RAJA::cuda_reduce, double > maxDiff( 0 );
  RAJA::ReduceMax< RAJA::cuda_reduce, double > maxRDiff( 0 );

  // CUDA acos( +/-1 ) produces a NaN :(
  forAllHalvesinMinus1to1( false, [maxDiff, maxRDiff] LVARRAY_DEVICE ( __half const x )
      {
        double const result = math::acos( x );
        double const expected = math::acos( double( x ) );

        double const diff = LvArray::math::abs( result - expected );
        double const rdiff = abs( diff / expected );

        maxDiff.max( diff );
        maxRDiff.max( rdiff );
      } );

  EXPECT_LT( maxDiff.get(), 3 * float( NumericLimitsNC< __half >{}.epsilon ) );
  EXPECT_LT( maxRDiff.get(), 1.5 * float( NumericLimitsNC< __half >{}.epsilon ) );

  printf( "Max difference          is %.8e\n", maxDiff.get() );
  printf( "Max relative difference is %.8e\n", maxRDiff.get() );
}

void acosHalf2Accuracy()
{
  RAJA::ReduceMax< RAJA::cuda_reduce, double > maxDiff( 0 );
  RAJA::ReduceMax< RAJA::cuda_reduce, double > maxRDiff( 0 );

  forAllHalvesinMinus1to1( false, [maxDiff, maxRDiff] LVARRAY_DEVICE ( __half const x )
      {
        __half2 const result = math::acos( math::convert< __half2 >( x, -x ) );
        double const expected0 = math::acos( double( x ) );
        double const expected1 = math::acos( double( -x ) );

        {
          double const diff0 = LvArray::math::abs( double( math::getFirst( result ) ) - expected0 );
          double const rdiff0 = math::abs( diff0 / expected0 );
          maxDiff.max( diff0 );
          maxRDiff.max( rdiff0 );
        }

        {
          double const diff1 = LvArray::math::abs( double( math::getSecond( result ) ) - expected1 );
          double const rdiff1 = math::abs( diff1 / expected1 );
          maxDiff.max( diff1 );
          maxRDiff.max( rdiff1 );
        }
      } );

  EXPECT_LT( maxDiff.get(), 3 * float( NumericLimitsNC< __half >{}.epsilon ) );
  EXPECT_LT( maxRDiff.get(), 1.5 * float( NumericLimitsNC< __half >{}.epsilon ) );

  printf( "Max difference          is %.8e\n", maxDiff.get() );
  printf( "Max relative difference is %.8e\n", maxRDiff.get() );
}

void atan2HalfAccuracy()
{
  RAJA::ReduceMax< RAJA::cuda_reduce, double > maxDiff( 0 );
  RAJA::ReduceMax< RAJA::cuda_reduce, double > maxRDiff( 0 );

  forAllHalvesinMinus1to1( true, [maxDiff, maxRDiff] LVARRAY_DEVICE ( __half const x )
      {
        // atan2( x, 1 )
        {
          double const result = math::atan2( x, __half( 1.0 ) );
          double const expected = math::atan2( double( x ), 1.0 );

          double const diff = diffModulo2PI( result, expected );
          double const rdiff = abs( diff / expected );

          maxDiff.max( diff );
          maxRDiff.max( rdiff );
        }

        // atan2( x, -1 )
        {
          double const result = math::atan2( x, __half( -1.0 ) );
          double const expected = math::atan2( double( x ), -1.0 );

          double const diff = diffModulo2PI( result, expected );
          double const rdiff = abs( diff / expected );

          maxDiff.max( diff );
          maxRDiff.max( rdiff );
        }

        // atan2( 1, x )
        {
          double const result = math::atan2( __half( 1.0 ), x );
          double const expected = math::atan2( 1.0, double( x ) );

          double const diff = diffModulo2PI( result, expected );
          double const rdiff = abs( diff / expected );

          maxDiff.max( diff );
          maxRDiff.max( rdiff );
        }

        // atan2( -1, x )
        {
          double const result = math::atan2( __half( -1.0 ), x );
          double const expected = math::atan2( -1.0, double( x ) );

          double const diff = diffModulo2PI( result, expected );
          double const rdiff = abs( diff / expected );

          maxDiff.max( diff );
          maxRDiff.max( rdiff );
        }
      } );

  EXPECT_LT( maxDiff.get(), 3 * float( NumericLimitsNC< __half >{}.epsilon ) );
  EXPECT_LT( maxRDiff.get(), 2 * float( NumericLimitsNC< __half >{}.epsilon ) );

  printf( "Max difference          is %.8e\n", maxDiff.get() );
  printf( "Max relative difference is %.8e\n", maxRDiff.get() );
}

void atan2Half2Accuracy()
{
  RAJA::ReduceMax< RAJA::cuda_reduce, double > maxDiff( 0 );
  RAJA::ReduceMax< RAJA::cuda_reduce, double > maxRDiff( 0 );

  forAllHalvesinMinus1to1( true, [maxDiff, maxRDiff] LVARRAY_DEVICE ( __half const x )
      {
        // atan2( x, -1 ), atan2( 1, x );
        {
          __half2 const result = math::atan2( math::convert< __half2 >( x, 1 ), math::convert< __half2 >( -1, x ) );
          double const expected0 = math::atan2( double( x ), -1 );
          double const expected1 = math::atan2( 1, double( x ) );

          {
            double const diff0 = diffModulo2PI( double( math::getFirst( result ) ), expected0 );
            double const rdiff0 = math::abs( diff0 / expected0 );
            maxDiff.max( diff0 );
            maxRDiff.max( rdiff0 );
          }

          {
            double const diff1 = diffModulo2PI( double( math::getSecond( result ) ), expected1 );
            double const rdiff1 = math::abs( diff1 / expected1 );
            maxDiff.max( diff1 );
            maxRDiff.max( rdiff1 );
          }
        }

        // atan2( x, 1 ), atan2( -1, x );
        {
          __half2 const result = math::atan2( math::convert< __half2 >( x, -1 ), math::convert< __half2 >( 1, x ) );
          double const expected0 = math::atan2( double( x ), 1 );
          double const expected1 = math::atan2( -1, double( x ) );

          {
            double const diff0 = diffModulo2PI( double( math::getFirst( result ) ), expected0 );
            double const rdiff0 = math::abs( diff0 / expected0 );
            maxDiff.max( diff0 );
            maxRDiff.max( rdiff0 );
          }

          {
            double const diff1 = diffModulo2PI( double( math::getSecond( result ) ), expected1 );
            double const rdiff1 = math::abs( diff1 / expected1 );
            maxDiff.max( diff1 );
            maxRDiff.max( rdiff1 );
          }
        }

      } );

  EXPECT_LT( maxDiff.get(), 3 * float( NumericLimitsNC< __half >{}.epsilon ) );
  EXPECT_LT( maxRDiff.get(), 2 * float( NumericLimitsNC< __half >{}.epsilon ) );

  printf( "Max difference          is %.8e\n", maxDiff.get() );
  printf( "Max relative difference is %.8e\n", maxRDiff.get() );
}

// These tests take 100x longer in debug mode.
#if defined( NDEBUG )

TEST( TestHalfMath, asinHalfAccuracy )
{
  asinHalfAccuracy();
}

TEST( TestHalfMath, asinHalf2Accuracy )
{
  asinHalf2Accuracy();
}

TEST( TestHalfMath, acosHalfAccuracy )
{
  acosHalfAccuracy();
}

TEST( TestHalfMath, acosHalf2Accuracy )
{
  acosHalf2Accuracy();
}

TEST( TestHalfMath, atan2HalfAccuracy )
{
  atan2HalfAccuracy();
}

TEST( TestHalfMath, atan2Half2Accuracy )
{
  atan2Half2Accuracy();
}

#endif

#endif


} // namespace testing
} // namespace LvArray

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
