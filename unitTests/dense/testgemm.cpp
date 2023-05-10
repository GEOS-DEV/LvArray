/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "dense/dense.hpp"
#include "dense/BlasLapackInterface.hpp"

#include "../testUtils.hpp"

#include <random>

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

// This should probably go in a common place
template< typename T, typename PERM >
using Array2d = Array< T, 2, PERM, std::ptrdiff_t, DEFAULT_BUFFER >;

template< typename T >
std::enable_if_t< std::is_floating_point< T >::value, T >
randomValue( std::mt19937 & gen )
{ return std::uniform_real_distribution< T >{ -1, 1 }( gen ); }

template< typename T >
std::enable_if_t< dense::IsComplex< T >, T >
randomValue( std::mt19937 & gen )
{ 
  return { std::uniform_real_distribution< dense::RealVersion< T > >{ -1, 1 }( gen ),
           std::uniform_real_distribution< dense::RealVersion< T > >{ -1, 1 }( gen ) }; 
}

template< typename T, typename PERM >
Array2d< T, PERM > randomMatrix( std::ptrdiff_t const N, std::ptrdiff_t const M )
{
  std::mt19937 gen( std::random_device{}() );

  Array2d< T, PERM > const ret( N, M );

  for( std::ptrdiff_t r = 0; r < N; ++r )
  {
    for( std::ptrdiff_t c = 0; c < M; ++c )
    {
      ret( r, c ) = T{10} * randomValue< T >( gen );
    }
  }

  return ret; 
}

template< typename T, typename PERM >
std::enable_if_t< std::is_floating_point< T >::value >
checkEqual( Array2d< T, PERM > const & lhs, Array2d< T, PERM > const & rhs, double rTol )
{
  ASSERT_EQ( lhs.size( 0 ), rhs.size( 0 ) );
  ASSERT_EQ( lhs.size( 1 ), rhs.size( 1 ) );

  for( std::ptrdiff_t i = 0; i < lhs.size(); ++i )
  {
    EXPECT_NEAR( lhs.data()[ i ], rhs.data()[ i ], std::abs( lhs.data()[ i ] ) * rTol );
  }
}

template< typename T, typename PERM >
std::enable_if_t< dense::IsComplex< T > >
checkEqual( Array2d< T, PERM > const & lhs, Array2d< T, PERM > const & rhs, double rTol )
{
  ASSERT_EQ( lhs.size( 0 ), rhs.size( 0 ) );
  ASSERT_EQ( lhs.size( 1 ), rhs.size( 1 ) );

  for( std::ptrdiff_t i = 0; i < lhs.size(); ++i )
  {
    EXPECT_COMPLEX_NEAR( lhs.data()[ i ], rhs.data()[ i ], std::abs( lhs.data()[ i ] ) * rTol );
  }
}

template< typename INTERFACE, typename T, typename PERM_A, typename PERM_B >
struct GemmTest
{
  std::mt19937 gen();

  void Rij_eq_AikBkj()
  {
    std::mt19937 gen( std::random_device{}() );

    int const N = std::uniform_int_distribution< std::ptrdiff_t >{ 0, 20 }( gen );
    int const M = std::uniform_int_distribution< std::ptrdiff_t >{ 0, 20 }( gen );
    int const K = std::uniform_int_distribution< std::ptrdiff_t >{ 0, 20 }( gen );
    
    T const alpha = T{10} * randomValue< T >( gen );
    T const beta = T{10} * randomValue< T >( gen );
    
    Array2d< T, PERM_A > const A = randomMatrix< T, PERM_A >( N, K );
    Array2d< T, PERM_B > const B = randomMatrix< T, PERM_B >( K, M );
    Array2d< T, RAJA::PERM_JI > const C = randomMatrix< T, RAJA::PERM_JI >( N, M );
    
    Array2d< T, PERM_A > const Acopy = A;
    Array2d< T, PERM_B > const Bcopy = B;
    Array2d< T, RAJA::PERM_JI > const Ccopy = C;

    dense::gemm< INTERFACE >( dense::Operation::NO_OP, dense::Operation::NO_OP, alpha, A, B, beta, C );

    A.move( MemorySpace::host, false );
    B.move( MemorySpace::host, false );
    C.move( MemorySpace::host, false );

    for( std::ptrdiff_t i = 0; i < N; ++i )
    {
      for( std::ptrdiff_t j = 0; j < M; ++j )
      {
        T dot = 0;
        for( std::ptrdiff_t k = 0; k < K; ++k )
        {
          dot += Acopy( i, k ) * Bcopy( k, j );
        }

        Ccopy( i, j ) = alpha * dot + beta * Ccopy( i, j );
      }
    }

    checkEqual( A, Acopy, 0 );
    checkEqual( B, Bcopy, 0 );
    checkEqual( C, Ccopy, 1e3 * std::numeric_limits< dense::RealVersion< T > >::epsilon() );
  }
};

TEST( LapackInterface_float, Rij_eq_AikBkj )
{
  GemmTest< dense::BlasLapackInterface< float >, float, RAJA::PERM_JI, RAJA::PERM_JI >().Rij_eq_AikBkj();
}

TEST( LapackInterface_double, Rij_eq_AikBkj )
{
  GemmTest< dense::BlasLapackInterface< double >, double, RAJA::PERM_JI, RAJA::PERM_JI >().Rij_eq_AikBkj();
}

TEST( LapackInterface_complex_float, Rij_eq_AikBkj )
{
  GemmTest< dense::BlasLapackInterface< std::complex< float > >, std::complex< float >, RAJA::PERM_JI, RAJA::PERM_JI >().Rij_eq_AikBkj();
}

TEST( LapackInterface_complex_double, Rij_eq_AikBkj )
{
  GemmTest< dense::BlasLapackInterface< std::complex< double > >, std::complex< double >, RAJA::PERM_JI, RAJA::PERM_JI >().Rij_eq_AikBkj();
}

TEST( LapackInterface_float, Rij_eq_AikBkj_foo )
{
  GemmTest< dense::BlasLapackInterface< float >, float, RAJA::PERM_IJ, RAJA::PERM_JI >().Rij_eq_AikBkj();
}

TEST( LapackInterface_double, Rij_eq_AikBkj_foo )
{
  GemmTest< dense::BlasLapackInterface< double >, double, RAJA::PERM_IJ, RAJA::PERM_JI >().Rij_eq_AikBkj();
}

TEST( LapackInterface_complex_float, Rij_eq_AikBkj_foo )
{
  GemmTest< dense::BlasLapackInterface< std::complex< float > >, std::complex< float >, RAJA::PERM_IJ, RAJA::PERM_JI >().Rij_eq_AikBkj();
}

TEST( LapackInterface_complex_double, Rij_eq_AikBkj_foo )
{
  GemmTest< dense::BlasLapackInterface< std::complex< double > >, std::complex< double >, RAJA::PERM_IJ, RAJA::PERM_JI >().Rij_eq_AikBkj();
}


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
