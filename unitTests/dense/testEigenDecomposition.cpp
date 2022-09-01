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

template< typename T >
using Array1d = Array< T, 1, RAJA::PERM_I, std::ptrdiff_t, DEFAULT_BUFFER >;

template< typename T, typename PERM >
using Array2d = Array< T, 2, PERM, std::ptrdiff_t, DEFAULT_BUFFER >;


TEST( heevr, allEigenvalues )
{
  Array2d< std::complex< double >, RAJA::PERM_JI > matrix( 3, 3 );
  matrix( 1, 1 ) = 2;
  matrix( 0, 0 ) = 3;
  matrix( 2, 2 ) = -4;

  Array1d< double > eigenvalues( 3 );
  Array2d< std::complex< double >, RAJA::PERM_JI > eigenvectors;
  Array1d< int > support( 6 );
  dense::ArrayWorkspace< std::complex< double >, ChaiBuffer > workspace;
  dense::SymmetricMatrixStorageType storageType = dense::SymmetricMatrixStorageType::UPPER_TRIANGULAR;

  dense::heevr< double >(
    MemorySpace::host,
    dense::EigenDecompositionOptions( dense::EigenDecompositionOptions::Type::EIGENVALUES ),
    matrix.toView(),
    eigenvalues.toView(),
    eigenvectors.toView(),
    support,
    workspace,
    storageType );

  EXPECT_DOUBLE_EQ( eigenvalues[ 0 ], -4 );
  EXPECT_DOUBLE_EQ( eigenvalues[ 1 ], 2 );
  EXPECT_DOUBLE_EQ( eigenvalues[ 2 ], 3 );
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
