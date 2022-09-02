/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "dense/eigenDecomposition.hpp"

#include "../testUtils.hpp"

namespace LvArray
{
namespace testing
{

using namespace dense;

template< typename T >
using Array1d = Array< T, 1, RAJA::PERM_I, std::ptrdiff_t, DEFAULT_BUFFER >;

template< typename T, typename PERM >
using Array2d = Array< T, 2, PERM, std::ptrdiff_t, DEFAULT_BUFFER >;

template< typename T >
struct HEEVR_TEST
{
  HEEVR_TEST( BuiltInBackends const backend ):
    m_backend( backend )
  {}
  
  void threeByThreeEigenvalues()
  {
    resize( 3, 3, 0 );

    m_matrix( 1, 1 ) = 2;
    m_matrix( 0, 0 ) = 3;
    m_matrix( 2, 2 ) = -4;

    SymmetricMatrixStorageType storageType = SymmetricMatrixStorageType::UPPER_TRIANGULAR;

    heevr(
      m_backend,
      EigenDecompositionOptions( EigenDecompositionOptions::EIGENVALUES ),
      m_matrix.toView(),
      m_eigenvalues.toView(),
      m_eigenvectors.toView(),
      m_support,
      m_workspace,
      storageType );

    EXPECT_DOUBLE_EQ( m_eigenvalues[ 0 ], -4 );
    EXPECT_DOUBLE_EQ( m_eigenvalues[ 1 ], 2 );
    EXPECT_DOUBLE_EQ( m_eigenvalues[ 2 ], 3 );
  }

private:
  void resize( DenseInt const n, DenseInt const nvals, DenseInt const nvec )
  {
    m_matrix.resize( n, n );
    m_eigenvalues.resize( nvals );
    m_eigenvectors.resize( n, nvec );;
    m_support.resize( 2 * n );
  }

  BuiltInBackends const m_backend;
  Array2d< std::complex< T >, RAJA::PERM_JI > m_matrix;
  Array1d< T > m_eigenvalues;
  Array2d< std::complex< T >, RAJA::PERM_JI > m_eigenvectors;
  Array1d< int > m_support;
  ArrayWorkspace< std::complex< T >, ChaiBuffer > m_workspace;
};

TEST( eigenvalues_float, lapack )
{
  HEEVR_TEST< float > test( BuiltInBackends::LAPACK );

  test.threeByThreeEigenvalues();
}

TEST( eigenvalues_double, lapack )
{
  HEEVR_TEST< double > test( BuiltInBackends::LAPACK );

  test.threeByThreeEigenvalues();
}

TEST( eigenvalues_float, magma )
{
  HEEVR_TEST< float > test( BuiltInBackends::MAGMA );

  test.threeByThreeEigenvalues();
}

TEST( eigenvalues_double, magma )
{
  HEEVR_TEST< double > test( BuiltInBackends::MAGMA );

  test.threeByThreeEigenvalues();
}

TEST( eigenvalues_float, magma_gpu )
{
  HEEVR_TEST< float > test( BuiltInBackends::MAGMA_GPU );

  test.threeByThreeEigenvalues();
}

TEST( eigenvalues_double, magma_gpu )
{
  HEEVR_TEST< double > test( BuiltInBackends::MAGMA_GPU );

  test.threeByThreeEigenvalues();
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
