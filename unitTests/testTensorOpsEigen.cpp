/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "tensorOps.hpp"
#include "Array.hpp"
#include "testUtils.hpp"
#include "output.hpp"
#include "testTensorOpsCommon.hpp"

// TPL includes
#include <gtest/gtest.h>

// System includes
#include <random>

namespace LvArray
{
namespace testing
{


template< typename T_FLOAT_M_POLICY_TUPLE >
class TestEigendecomposition : public ::testing::Test
{
public:

  using T = std::tuple_element_t< 0, T_FLOAT_M_POLICY_TUPLE >;
  using FLOAT = std::tuple_element_t< 1, T_FLOAT_M_POLICY_TUPLE >;
  static constexpr std::ptrdiff_t M = std::tuple_element_t< 2, T_FLOAT_M_POLICY_TUPLE > {};
  using POLICY = std::tuple_element_t< 3, T_FLOAT_M_POLICY_TUPLE >;

  static constexpr INDEX_TYPE NUM_MATRICES = 100;

  void test()
  {
    m_matrices.setName( "matrices" );
    m_eigenvalues.setName( "eigenvalues" );
    m_expectedEigenvalues.setName( "expectedEigenvalues" );
    m_eigenvectors.setName( "eigenvectors" );

    double const begin = std::is_integral< T >::value ? 10 : math::max( 1e-25, double( NumericLimits< T >::min ) );
    double const end = math::min( 1e35, double( NumericLimits< T >::max ) );
    double const scalingFactor = std::is_integral< T >::value ? 1e5 : 1e10;

    for( double scale = begin; scale < 10 * end; scale *= scalingFactor )
    {
      T const maxValue = math::min( scale, end );

      // Fill in the matrices
      m_matrices.move( MemorySpace::host );
      for( T & value : m_matrices )
      { value = randomValue( maxValue, m_gen ); }

      // Compute the decompositions.
      { SCOPED_TRACE( "Initial decompositions" ); computeDecompositions( maxValue ); }

      // Verify the systems.
      { SCOPED_TRACE( "Initial decompositions" ); verifyDecompositions( 0, maxValue ); }

      // The method of duplicating eigenvalues doesn't work when the matrix being
      // decomposed has integral values.
      if( !std::is_integral< T >::value )
      {
        for( int i = 0; i < M - 1; ++i )
        {
          duplicateEigenvalues();
          { SCOPED_TRACE( "Decomposition " + std::to_string( i + 1 ) ); computeDecompositions( maxValue ); }
          { SCOPED_TRACE( "Decomposition " + std::to_string( i + 1 ) ); verifyDecompositions( i + 1, maxValue ); }
        }
      }
    }
  }

  void computeDecompositions( T const scale ) const
  {
    FLOAT const tol = tolerance();

    ArrayViewT< T const, 2, 1 > const matrices = m_matrices.toViewConst();
    ArrayViewT< FLOAT, 2, 1 > const eigenvalues = m_eigenvalues.toView();
    ArrayViewT< FLOAT, 3, 2 > const eigenvectors = m_eigenvectors.toView();
    forall< POLICY >( matrices.size( 0 ), [=] LVARRAY_HOST_DEVICE ( std::ptrdiff_t const i )
        {
          FLOAT eigenvaluesOnly[ M ];
          tensorOps::symEigenvalues< M >( eigenvaluesOnly, matrices[ i ] );
          tensorOps::symEigenvectors< M >( eigenvalues[ i ], eigenvectors[ i ], matrices[ i ] );

          // Verify that both routines produce the same eigenvalues.
          for( INDEX_TYPE j = 0; j < M; ++j )
          { PORTABLE_EXPECT_NEAR( eigenvaluesOnly[ j ], eigenvalues( i, j ), 4 * scale * tol ); }
        } );
  }

  void verifyDecompositions( int const iteration, T const scale ) const
  {
    FLOAT const tol = ( iteration == 0 ) ? tolerance() : reducedTolerance();
    std::vector< double > relativeDiffs;

    ArrayViewT< T const, 2, 1 > const matrices = m_matrices.toViewConst();
    ArrayViewT< FLOAT const, 2, 1 > const eigenvalues = m_eigenvalues.toViewConst();
    ArrayViewT< FLOAT, 2, 1 > const expectedEigenvalues = m_expectedEigenvalues.toView();
    ArrayViewT< FLOAT const, 3, 2 > const eigenvectors = m_eigenvectors.toViewConst();
    forall< serialPolicy >( matrices.size( 0 ), [=, &relativeDiffs] ( std::ptrdiff_t const i )
    {
      if( iteration > 0 )
      {
        // Sort the expected eigenvalues
        std::sort( expectedEigenvalues[ i ].begin(), expectedEigenvalues[ i ].end() );

        for( int j = 0; j < M; ++j )
        { PORTABLE_EXPECT_NEAR( eigenvalues( i, j ), expectedEigenvalues( i, j ), scale * tol ); }
      }

      // Check that the eigenvalues are sorted.
      for( int j = 0; j < M - 1; ++j )
      { EXPECT_LT( eigenvalues( i, j ), eigenvalues( i, j + 1 ) + scale * tolerance() ); }

      // Check that the eigenvectors are orthogonal to each other.
      for( int j = 0; j < M; ++j )
      {
        for( int k = j + 1; k < M; ++k )
        { EXPECT_LT( tensorOps::AiBi< M >( eigenvectors[ i ][ j ], eigenvectors[ i ][ k ] ), tol ); }
      }

      for( int j = 0; j < M; ++j )
      {
        // Check that the eigenvector has unit-norm.
        PORTABLE_EXPECT_NEAR( tensorOps::l2Norm< M >( eigenvectors[ i ][ j ] ), 1, tol );

        // Check that the matrix scales the eigenvector by the eigenvalue.
        FLOAT output[ M ];
        tensorOps::Ri_eq_symAijBj< M >( output, matrices[ i ], eigenvectors[ i ][ j ] );

        double diff = 0;
        for( int k = 0; k < M; ++k )
        {
          double const coordDiff = math::abs( output[ k ] - eigenvalues( i, j ) * eigenvectors( i, j, k ) );
          diff += coordDiff * coordDiff;
          PORTABLE_EXPECT_NEAR( output[ k ], eigenvalues( i, j ) * eigenvectors( i, j, k ), scale * tol );
        }

        relativeDiffs.push_back( math::sqrt( diff ) / scale );
      }
    } );

    FLOAT maxDiff = 0;
    FLOAT meanDiff = 0;
    for( FLOAT const diff : relativeDiffs )
    {
      maxDiff = std::max( maxDiff, diff );
      meanDiff += diff;
    }

    meanDiff /= relativeDiffs.size();
    FLOAT variance = 0;
    for( FLOAT const diff : relativeDiffs )
    { variance += ( diff - meanDiff ) * ( diff - meanDiff ); }

    LVARRAY_LOG( "scale = " << scale << ", iteration = " << iteration << "\n" <<
                 "\tmaxDiff = " << maxDiff << "\n" <<
                 "\tmeanDiff = " << meanDiff << "\n" <<
                 "\tstandard deviation = " << math::sqrt( variance ) << std::endl );
  }

  // Construct new matrices by reducing the number of unique eigenvalues.
  // This is done by A = Q \lambda Q^T where Q is the matrix whose columns are the eigenvectors.
  void duplicateEigenvalues()
  {
    ArrayViewT< T, 2, 1 > const matrices = m_matrices.toView();
    ArrayViewT< FLOAT, 2, 1 > const eigenvalues = m_eigenvalues.toView();
    ArrayViewT< FLOAT, 2, 1 > const expectedEigenvalues = m_expectedEigenvalues.toView();
    ArrayViewT< FLOAT, 3, 2 > const eigenvectors = m_eigenvectors.toView();
    forall< serialPolicy >( matrices.size( 0 ), [=] ( std::ptrdiff_t const i )
    {
      // Since we're constructing the matrix we know the eigenvalues beforehand.
      for( INDEX_TYPE j = 0; j < M; ++j )
      {
        expectedEigenvalues( i, j ) = eigenvalues( i, j );
        eigenvalues( i, j ) = 0;
      }

      // Equate two randomly chosen eigenvalues
      std::pair< int, int > const components = randomComponents();
      expectedEigenvalues( i, components.first ) = expectedEigenvalues( i, components.second );

      // Construct a diagonal matrix of the eigenvalues
      FLOAT lambda[ tensorOps::SYM_SIZE< M > ];

      for( int j = 0; j < tensorOps::SYM_SIZE< M >; ++j )
      { lambda[ j ] = 0; }

      for( int j = 0; j < M; ++j )
      { lambda[ j ] = expectedEigenvalues( i, j ); }

      // Construct Q from the eigenvectors.
      FLOAT Q[ M ][ M ];
      tensorOps::transpose< M, M >( Q, eigenvectors[ i ] );
      tensorOps::Rij_eq_AikSymBklAjl< M >( matrices[ i ], Q, lambda );
    } );
  }

  std::pair< int, int > randomComponents()
  {
    int components[ M ];
    for( int i = 0; i < M; ++i )
    { components[ i ] = i; }

    std::shuffle( components, components + M, m_gen );
    return { components[ 0 ], components[ 1 ] };
  }

  FLOAT tolerance() const
  { return 800 * NumericLimits< FLOAT >::epsilon; }

  // The algorithm gets less accurate the closer the eigenvalues are together
  // The reductions decrease the gap between the eigenvalues and hence require a
  // lower tolerance.
  template< typename _FLOAT=FLOAT >
  std::enable_if_t< std::is_same< _FLOAT, float >::value, FLOAT >
  reducedTolerance() const
  { return 8e-4; }

  template< typename _FLOAT=FLOAT >
  std::enable_if_t< std::is_same< _FLOAT, double >::value, FLOAT >
  reducedTolerance() const
  { return 3e-8; }

  std::mt19937_64 m_gen;
  ArrayT< T, RAJA::PERM_IJ > m_matrices { NUM_MATRICES, tensorOps::SYM_SIZE< M > };
  ArrayT< FLOAT, RAJA::PERM_IJ > m_eigenvalues { NUM_MATRICES, M };
  ArrayT< FLOAT, RAJA::PERM_IJ > m_expectedEigenvalues { NUM_MATRICES, M };
  ArrayT< FLOAT, RAJA::PERM_IJK > m_eigenvectors { NUM_MATRICES, M, M };
};

using TestEigendecompositionTypes = ::testing::Types<
  std::tuple< double, double, std::integral_constant< int, 2 >, serialPolicy >
  , std::tuple< std::int64_t, double, std::integral_constant< int, 2 >, serialPolicy >
  , std::tuple< float, float, std::integral_constant< int, 2 >, serialPolicy >
  , std::tuple< double, double, std::integral_constant< int, 3 >, serialPolicy >
  , std::tuple< std::int64_t, double, std::integral_constant< int, 3 >, serialPolicy >
  , std::tuple< float, float, std::integral_constant< int, 3 >, serialPolicy >
#if ( defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP) ) && defined(LVARRAY_USE_CHAI)
  , std::tuple< double, double, std::integral_constant< int, 2 >, parallelDevicePolicy< 32 > >
  , std::tuple< std::int64_t, double, std::integral_constant< int, 2 >, parallelDevicePolicy< 32 > >
  , std::tuple< float, float, std::integral_constant< int, 2 >, parallelDevicePolicy< 32 > >
  , std::tuple< double, double, std::integral_constant< int, 3 >, parallelDevicePolicy< 32 > >
  , std::tuple< std::int64_t, double, std::integral_constant< int, 3 >, parallelDevicePolicy< 32 > >
  , std::tuple< float, float, std::integral_constant< int, 3 >, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( TestEigendecomposition, TestEigendecompositionTypes, );

TYPED_TEST( TestEigendecomposition, test )
{
  this->test();
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
