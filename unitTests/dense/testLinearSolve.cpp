/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "dense/linearSolve.hpp"

#include "../testUtils.hpp"

#include "output.hpp"

#if defined( LVARRAY_USE_MAGMA )
  #include <magma.h>
#endif

#define EXPECT_COMPLEX_NEAR( z1, z2, absError ) \
  EXPECT_NEAR( std::real( z1 ), std::real( z2 ), absError ); \
  EXPECT_NEAR( std::imag( z1 ), std::imag( z2 ), absError )

namespace LvArray
{
namespace testing
{

using namespace dense;

template< typename T >
using Array1d = Array< T, 1, RAJA::PERM_I, DenseInt, DEFAULT_BUFFER >;

template< typename T, typename PERM >
using Array2d = Array< T, 2, PERM, DenseInt, DEFAULT_BUFFER >;


template< typename T >
struct GESV_Test : public ::testing::Test
{
  void test( BuiltInBackends const backend, DenseInt const N, DenseInt const nrhs )
  {
    Array2d< T, RAJA::PERM_JI > A( N, N );
    Array2d< T, RAJA::PERM_JI > B( N, nrhs ) ;
    Array1d< DenseInt > pivots( N );

    for( DenseInt row = 0; row < N; ++row )
    {
      for( DenseInt col = 0; col < N; ++col )
      {
        A( row, col ) = randomNumber();
      }

      for( DenseInt col = 0; col < nrhs; ++col )
      {
        B( row, col ) = randomNumber();
      }
    }

    Array2d< T, RAJA::PERM_JI > ACopy( A );
    Array2d< T, RAJA::PERM_JI > X( B );
    gesv( backend, ACopy.toView(), X.toView(), pivots );

    // TODO(corbett5): replace this with matrix matrix multiplication
    X.move( MemorySpace::host, true );
    for( DenseInt i = 0; i < N; ++i )
    {
      for( DenseInt j = 0; j < nrhs; ++j )
      {
        T dot = 0;
        for( DenseInt k = 0; k < N; ++k )
        {
          dot += A( i, k ) * X( k, j );
        }

        EXPECT_COMPLEX_NEAR( dot, B( i, j ), 10 * N * std::numeric_limits< RealVersion< T > >::epsilon() );
      }
    }
  }

private:
  
  template< typename _T=T >
  std::enable_if_t< !IsComplex< _T >, T >
  randomNumber()
  { return m_dist( m_gen ); }

  template< typename _T=T >
  std::enable_if_t< IsComplex< _T >, T >
  randomNumber()
  { return { m_dist( m_gen ), m_dist( m_gen ) }; }

  std::mt19937_64 m_gen;
  std::uniform_real_distribution< RealVersion< T > > m_dist;
};

using GESV_Test_types = ::testing::Types<
  float,
  double,
  std::complex< float >,
  std::complex< double >
  >;
TYPED_TEST_SUITE( GESV_Test, GESV_Test_types, );

TYPED_TEST( GESV_Test, LAPACK_2x2 )
{
  this->test( BuiltInBackends::LAPACK, 2, 1 );
  this->test( BuiltInBackends::LAPACK, 2, 2 );
}

TYPED_TEST( GESV_Test, LAPACK_10x10 )
{
  this->test( BuiltInBackends::LAPACK, 10, 1 );
  this->test( BuiltInBackends::LAPACK, 10, 3 );
}

TYPED_TEST( GESV_Test, LAPACK_100x100 )
{
  this->test( BuiltInBackends::LAPACK, 100, 1 );
  this->test( BuiltInBackends::LAPACK, 100, 10 );
}

TYPED_TEST( GESV_Test, LAPACK_1000x1000 )
{
  this->test( BuiltInBackends::LAPACK, 100, 1 );
  this->test( BuiltInBackends::LAPACK, 100, 10 );
}

#if defined( LVARRAY_USE_MAGMA )

TYPED_TEST( GESV_Test, MAGMA_2x2 )
{
  this->test( BuiltInBackends::MAGMA, 2, 1 );
  this->test( BuiltInBackends::MAGMA, 2, 2 );
}

TYPED_TEST( GESV_Test, MAGMA_10x10 )
{
  this->test( BuiltInBackends::MAGMA, 10, 1 );
  this->test( BuiltInBackends::MAGMA, 10, 3 );
}

TYPED_TEST( GESV_Test, MAGMA_100x100 )
{
  this->test( BuiltInBackends::MAGMA, 100, 1 );
  this->test( BuiltInBackends::MAGMA, 100, 10 );
}

TYPED_TEST( GESV_Test, MAGMA_1000x1000 )
{
  this->test( BuiltInBackends::MAGMA, 100, 1 );
  this->test( BuiltInBackends::MAGMA, 100, 10 );
}

TYPED_TEST( GESV_Test, MAGMA_GPU_2x2 )
{
  this->test( BuiltInBackends::MAGMA_GPU, 2, 1 );
  this->test( BuiltInBackends::MAGMA_GPU, 2, 2 );
}

TYPED_TEST( GESV_Test, MAGMA_GPU_10x10 )
{
  this->test( BuiltInBackends::MAGMA_GPU, 10, 1 );
  this->test( BuiltInBackends::MAGMA_GPU, 10, 3 );
}

TYPED_TEST( GESV_Test, MAGMA_GPU_100x100 )
{
  this->test( BuiltInBackends::MAGMA_GPU, 100, 1 );
  this->test( BuiltInBackends::MAGMA_GPU, 100, 10 );
}

TYPED_TEST( GESV_Test, MAGMA_GPU_1000x1000 )
{
  this->test( BuiltInBackends::MAGMA_GPU, 100, 1 );
  this->test( BuiltInBackends::MAGMA_GPU, 100, 10 );
}

#endif

} // namespace testing
} // namespace LvArray

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
#if defined( LVARRAY_USE_MAGMA )
  magma_init();
#endif

  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();

#if defined( LVARRAY_USE_MAGMA )
  magma_finalize();
#endif

  return result;
}
