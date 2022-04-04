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

namespace LvArray
{
namespace testing
{

template< int M, typename MATRIX, typename SYM_MATRIX >
LVARRAY_HOST_DEVICE inline
std::enable_if_t< M == 2 >
checkSymmetricToDense( MATRIX const & matrix, SYM_MATRIX const symMatrix )
{
  PORTABLE_EXPECT_EQ( matrix[ 0 ][ 0 ], symMatrix[ 0 ] );
  PORTABLE_EXPECT_EQ( matrix[ 1 ][ 1 ], symMatrix[ 1 ] );

  PORTABLE_EXPECT_EQ( matrix[ 0 ][ 1 ], symMatrix[ 2 ] );
  PORTABLE_EXPECT_EQ( matrix[ 1 ][ 0 ], symMatrix[ 2 ] );
}

template< int M, typename MATRIX, typename SYM_MATRIX >
LVARRAY_HOST_DEVICE inline
std::enable_if_t< M == 2 >
checkDenseToSymmetric( MATRIX const & matrix, SYM_MATRIX const symMatrix )
{
  PORTABLE_EXPECT_EQ( matrix[ 0 ][ 0 ], symMatrix[ 0 ] );
  PORTABLE_EXPECT_EQ( matrix[ 1 ][ 1 ], symMatrix[ 1 ] );

  PORTABLE_EXPECT_EQ( matrix[ 0 ][ 1 ], symMatrix[ 2 ] );
}

template< int M, typename MATRIX, typename SYM_MATRIX >
LVARRAY_HOST_DEVICE inline
std::enable_if_t< M == 3 >
checkSymmetricToDense( MATRIX const & matrix, SYM_MATRIX const symMatrix )
{
  PORTABLE_EXPECT_EQ( matrix[ 0 ][ 0 ], symMatrix[ 0 ] );
  PORTABLE_EXPECT_EQ( matrix[ 1 ][ 1 ], symMatrix[ 1 ] );
  PORTABLE_EXPECT_EQ( matrix[ 2 ][ 2 ], symMatrix[ 2 ] );

  PORTABLE_EXPECT_EQ( matrix[ 1 ][ 2 ], symMatrix[ 3 ] );
  PORTABLE_EXPECT_EQ( matrix[ 2 ][ 1 ], symMatrix[ 3 ] );

  PORTABLE_EXPECT_EQ( matrix[ 0 ][ 2 ], symMatrix[ 4 ] );
  PORTABLE_EXPECT_EQ( matrix[ 2 ][ 0 ], symMatrix[ 4 ] );

  PORTABLE_EXPECT_EQ( matrix[ 0 ][ 1 ], symMatrix[ 5 ] );
  PORTABLE_EXPECT_EQ( matrix[ 1 ][ 0 ], symMatrix[ 5 ] );
}

template< int M, typename MATRIX, typename SYM_MATRIX >
LVARRAY_HOST_DEVICE inline
std::enable_if_t< M == 3 >
checkDenseToSymmetric( MATRIX const & matrix, SYM_MATRIX const symMatrix )
{
  PORTABLE_EXPECT_EQ( matrix[ 0 ][ 0 ], symMatrix[ 0 ] );
  PORTABLE_EXPECT_EQ( matrix[ 1 ][ 1 ], symMatrix[ 1 ] );
  PORTABLE_EXPECT_EQ( matrix[ 2 ][ 2 ], symMatrix[ 2 ] );

  PORTABLE_EXPECT_EQ( matrix[ 1 ][ 2 ], symMatrix[ 3 ] );

  PORTABLE_EXPECT_EQ( matrix[ 0 ][ 2 ], symMatrix[ 4 ] );

  PORTABLE_EXPECT_EQ( matrix[ 0 ][ 1 ], symMatrix[ 5 ] );
}

template< typename T_N_POLICY_TUPLE >
class FixedSizeSquareMatrixTest : public ::testing::Test
{
public:
  using T = std::tuple_element_t< 0, T_N_POLICY_TUPLE >;
  static constexpr std::ptrdiff_t N = std::tuple_element_t< 1, T_N_POLICY_TUPLE > {};
  using POLICY = std::tuple_element_t< 2, T_N_POLICY_TUPLE >;

  static_assert( N == 2 || N == 3, "Other sizes not supported." );
  static constexpr std::ptrdiff_t SYM_SIZE = ( N == 2 ) ? 3 : 6;

  void SetUp() override
  {
    fill( m_vectorA_IJ.toSlice(), m_seedVectorA );
    fill( m_vectorA_JI.toSlice(), m_seedVectorA );
    fill( m_vectorA_local, m_seedVectorA );

    fill( m_vectorB_IJ.toSlice(), m_seedVectorB );
    fill( m_vectorB_JI.toSlice(), m_seedVectorB );
    fill( m_vectorB_local, m_seedVectorB );

    fill( m_matrixA_IJK.toSlice(), m_seedMatrixA );
    fill( m_matrixA_IKJ.toSlice(), m_seedMatrixA );
    fill( m_matrixA_KJI.toSlice(), m_seedMatrixA );
    fill( m_matrixA_local, m_seedMatrixA );

    fill( m_matrixB_IJK.toSlice(), m_seedMatrixB );
    fill( m_matrixB_IKJ.toSlice(), m_seedMatrixB );
    fill( m_matrixB_KJI.toSlice(), m_seedMatrixB );
    fill( m_matrixB_local, m_seedMatrixB );

    fill( m_symMatrixA_IJ.toSlice(), m_seedSymMatrixA );
    fill( m_symMatrixA_JI.toSlice(), m_seedSymMatrixA );
    fill( m_symMatrixA_local, m_seedSymMatrixA );

    fill( m_symMatrixB_IJ.toSlice(), m_seedSymMatrixB );
    fill( m_symMatrixB_JI.toSlice(), m_seedSymMatrixB );
    fill( m_symMatrixB_local, m_seedSymMatrixB );
  }

  void symAijBj()
  {
    T result[ N ];
    {
      T denseSymA[ N ][ N ];
      tensorOps::symmetricToDense< N >( denseSymA, m_symMatrixA_local );
      tensorOps::Ri_eq_AijBj< N, N >( result, denseSymA, m_vectorB_local );
    }

    ArrayViewT< T, 2, 1 > const vectorA_IJ = m_vectorA_IJ.toView();
    ArrayViewT< T, 2, 0 > const vectorA_JI = m_vectorA_JI.toView();

    ArrayViewT< T const, 2, 1 > const symMatrixA_IJ = m_symMatrixA_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const symMatrixA_JI = m_symMatrixA_JI.toViewConst();
    T const ( &symMatrixA_local )[ SYM_SIZE ] = m_symMatrixA_local;

    ArrayViewT< T const, 2, 1 > const vectorB_IJ = m_vectorB_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const vectorB_JI = m_vectorB_JI.toViewConst();
    T const ( &vectorB_local )[ N ] = m_vectorB_local;

    std::ptrdiff_t const vectorASeed = m_seedVectorA;

    forall< POLICY >( 1, [result, vectorA_IJ, vectorA_JI, symMatrixA_IJ, symMatrixA_JI, symMatrixA_local, vectorB_IJ, vectorB_JI, vectorB_local, vectorASeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( vectorA, symMatrix, vectorB ) \
            fill( vectorA, vectorASeed ); \
            tensorOps::Ri_eq_symAijBj< N >( vectorA, symMatrix, vectorB ); \
            CHECK_EQUALITY_1D( N, vectorA, result )

          #define _TEST_PERMS( vectorA, symMatrix, vectorB0, vectorB1, vectorB2 ) \
            _TEST( vectorA, symMatrix, vectorB0 ); \
            _TEST( vectorA, symMatrix, vectorB1 ); \
            _TEST( vectorA, symMatrix, vectorB2 )

          T vectorA_local[ N ];

          _TEST_PERMS( vectorA_IJ[ 0 ], symMatrixA_IJ[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_IJ[ 0 ], symMatrixA_JI[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_IJ[ 0 ], symMatrixA_local, vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );

          _TEST_PERMS( vectorA_JI[ 0 ], symMatrixA_IJ[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_JI[ 0 ], symMatrixA_JI[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_JI[ 0 ], symMatrixA_local, vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );

          _TEST_PERMS( vectorA_local, symMatrixA_IJ[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_local, symMatrixA_JI[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_local, symMatrixA_local, vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void Ri_add_symAijBj()
  {
    T result[ N ];
    tensorOps::copy< N >( result, m_vectorA_local );
    {
      T denseSymA[ N ][ N ];
      tensorOps::symmetricToDense< N >( denseSymA, m_symMatrixA_local );
      tensorOps::Ri_add_AijBj< N, N >( result, denseSymA, m_vectorB_local );
    }

    ArrayViewT< T, 2, 1 > const vectorA_IJ = m_vectorA_IJ.toView();
    ArrayViewT< T, 2, 0 > const vectorA_JI = m_vectorA_JI.toView();

    ArrayViewT< T const, 2, 1 > const symMatrixA_IJ = m_symMatrixA_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const symMatrixA_JI = m_symMatrixA_JI.toViewConst();
    T const ( &symMatrixA_local )[ SYM_SIZE ] = m_symMatrixA_local;

    ArrayViewT< T const, 2, 1 > const vectorB_IJ = m_vectorB_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const vectorB_JI = m_vectorB_JI.toViewConst();
    T const ( &vectorB_local )[ N ] = m_vectorB_local;

    std::ptrdiff_t const vectorASeed = m_seedVectorA;

    forall< POLICY >( 1, [result, vectorA_IJ, vectorA_JI, symMatrixA_IJ, symMatrixA_JI, symMatrixA_local, vectorB_IJ, vectorB_JI, vectorB_local, vectorASeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( vectorA, symMatrix, vectorB ) \
            fill( vectorA, vectorASeed ); \
            tensorOps::Ri_add_symAijBj< N >( vectorA, symMatrix, vectorB ); \
            CHECK_EQUALITY_1D( N, vectorA, result )

          #define _TEST_PERMS( vectorA, symMatrix, vectorB0, vectorB1, vectorB2 ) \
            _TEST( vectorA, symMatrix, vectorB0 ); \
            _TEST( vectorA, symMatrix, vectorB1 ); \
            _TEST( vectorA, symMatrix, vectorB2 )

          T vectorA_local[ N ];

          _TEST_PERMS( vectorA_IJ[ 0 ], symMatrixA_IJ[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_IJ[ 0 ], symMatrixA_JI[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_IJ[ 0 ], symMatrixA_local, vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );

          _TEST_PERMS( vectorA_JI[ 0 ], symMatrixA_IJ[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_JI[ 0 ], symMatrixA_JI[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_JI[ 0 ], symMatrixA_local, vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );

          _TEST_PERMS( vectorA_local, symMatrixA_IJ[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_local, symMatrixA_JI[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_local, symMatrixA_local, vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void symAikBjk()
  {
    T result[ N ][ N ];
    {
      T denseSymA[ N ][ N ];
      tensorOps::symmetricToDense< N >( denseSymA, m_symMatrixA_local );
      tensorOps::Rij_eq_AikBjk< N, N, N >( result, denseSymA, m_matrixB_local );
    }

    ArrayViewT< T, 3, 2 > const matrixA_IJK = m_matrixA_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixA_KJI = m_matrixA_KJI.toView();

    ArrayViewT< T const, 2, 1 > const symMatrixA_IJ = m_symMatrixA_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const symMatrixA_JI = m_symMatrixA_JI.toViewConst();
    T const ( &symMatrixA_local )[ SYM_SIZE ] = m_symMatrixA_local;

    ArrayViewT< T const, 3, 2 > const matrixB_IJK = m_matrixB_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixB_IKJ = m_matrixB_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixB_KJI = m_matrixB_KJI.toViewConst();
    T const ( &matrixB_local )[ N ][ N ] = m_matrixB_local;

    std::ptrdiff_t const matrixASeed = m_seedMatrixA;

    forall< POLICY >( 1,
                      [result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, symMatrixA_local, symMatrixA_IJ, symMatrixA_JI, matrixB_IJK, matrixB_IKJ, matrixB_KJI,
                       matrixB_local,
                       matrixASeed] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( matrixA, symMatrix, matrixB ) \
            fill( matrixA, matrixASeed ); \
            tensorOps::Rij_eq_symAikBjk< N >( matrixA, symMatrix, matrixB ); \
            CHECK_EQUALITY_2D( N, N, matrixA, result )

          #define _TEST_PERMS( matrixA, symMatrix, matrixB0, matrixB1, matrixB2, matrixB3 ) \
            _TEST( matrixA, symMatrix, matrixB0 ); \
            _TEST( matrixA, symMatrix, matrixB1 ); \
            _TEST( matrixA, symMatrix, matrixB2 ); \
            _TEST( matrixA, symMatrix, matrixB3 )

          T matrixA_local[ N ][ N ];

          _TEST_PERMS( matrixA_IJK[ 0 ], symMatrixA_IJ[ 0 ], matrixB_IJK[ 0 ], matrixB_IKJ[ 0 ], matrixB_KJI[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], symMatrixA_JI[ 0 ], matrixB_IJK[ 0 ], matrixB_IKJ[ 0 ], matrixB_KJI[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], symMatrixA_local, matrixB_IJK[ 0 ], matrixB_IKJ[ 0 ], matrixB_KJI[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], symMatrixA_IJ[ 0 ], matrixB_IJK[ 0 ], matrixB_IKJ[ 0 ], matrixB_KJI[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], symMatrixA_JI[ 0 ], matrixB_IJK[ 0 ], matrixB_IKJ[ 0 ], matrixB_KJI[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], symMatrixA_local, matrixB_IJK[ 0 ], matrixB_IKJ[ 0 ], matrixB_KJI[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], symMatrixA_IJ[ 0 ], matrixB_IJK[ 0 ], matrixB_IKJ[ 0 ], matrixB_KJI[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], symMatrixA_JI[ 0 ], matrixB_IJK[ 0 ], matrixB_IKJ[ 0 ], matrixB_KJI[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], symMatrixA_local, matrixB_IJK[ 0 ], matrixB_IKJ[ 0 ], matrixB_KJI[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_local, symMatrixA_IJ[ 0 ], matrixB_IJK[ 0 ], matrixB_IKJ[ 0 ], matrixB_KJI[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_local, symMatrixA_JI[ 0 ], matrixB_IJK[ 0 ], matrixB_IKJ[ 0 ], matrixB_KJI[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_local, symMatrixA_local, matrixB_IJK[ 0 ], matrixB_IKJ[ 0 ], matrixB_KJI[ 0 ], matrixB_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void Rij_eq_AikSymBklAjl()
  {
    T result[ SYM_SIZE ];
    {
      T denseSymB[ N ][ N ];
      tensorOps::symmetricToDense< N >( denseSymB, m_symMatrixB_local );

      T temp[ N ][ N ];
      tensorOps::Rij_eq_AikBjk< N, N, N >( temp, denseSymB, m_matrixA_local );

      T denseResult[ N ][ N ];
      tensorOps::Rij_eq_AikBkj< N, N, N >( denseResult, m_matrixA_local, temp );
      tensorOps::denseToSymmetric< N >( result, denseResult );
    }

    ArrayViewT< T, 2, 1 > const symMatrixA_IJ = m_symMatrixA_IJ.toView();
    ArrayViewT< T, 2, 0 > const symMatrixA_JI = m_symMatrixA_JI.toView();

    ArrayViewT< T const, 3, 2 > const matrixA_IJK = m_matrixA_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixA_IKJ = m_matrixA_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixA_KJI = m_matrixA_KJI.toViewConst();
    T const ( &matrixA_local )[ N ][ N ] = m_matrixA_local;

    ArrayViewT< T const, 2, 1 > const symMatrixB_IJ = m_symMatrixB_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const symMatrixB_JI = m_symMatrixB_JI.toViewConst();
    T const ( &symMatrixB_local )[ SYM_SIZE ] = m_symMatrixB_local;

    std::ptrdiff_t const matrixASeed = m_seedMatrixA;

    forall< POLICY >( 1,
                      [result, symMatrixA_IJ, symMatrixA_JI, matrixA_IJK, matrixA_IKJ, matrixA_KJI, matrixA_local, symMatrixB_IJ, symMatrixB_JI,
                       symMatrixB_local, matrixASeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( symMatrixA, matrixA, symMatrixB ) \
            fill( symMatrixA, matrixASeed ); \
            tensorOps::Rij_eq_AikSymBklAjl< N >( symMatrixA, matrixA, symMatrixB ); \
            CHECK_EQUALITY_1D( SYM_SIZE, symMatrixA, result )

          #define _TEST_PERMS( matrixA, symMatrix, symMatrixB0, symMatrixB1, symMatrixB2 ) \
            _TEST( matrixA, symMatrix, symMatrixB0 ); \
            _TEST( matrixA, symMatrix, symMatrixB1 ); \
            _TEST( matrixA, symMatrix, symMatrixB2 )

          T symMatrixA_local[ SYM_SIZE ];

          _TEST_PERMS( symMatrixA_IJ[ 0 ], matrixA_IJK[ 0 ], symMatrixB_IJ[ 0 ], symMatrixB_JI[ 0 ], symMatrixB_local );
          _TEST_PERMS( symMatrixA_IJ[ 0 ], matrixA_IKJ[ 0 ], symMatrixB_IJ[ 0 ], symMatrixB_JI[ 0 ], symMatrixB_local );
          _TEST_PERMS( symMatrixA_IJ[ 0 ], matrixA_KJI[ 0 ], symMatrixB_IJ[ 0 ], symMatrixB_JI[ 0 ], symMatrixB_local );
          _TEST_PERMS( symMatrixA_IJ[ 0 ], matrixA_local, symMatrixB_IJ[ 0 ], symMatrixB_JI[ 0 ], symMatrixB_local );
          _TEST_PERMS( symMatrixA_JI[ 0 ], matrixA_IJK[ 0 ], symMatrixB_IJ[ 0 ], symMatrixB_JI[ 0 ], symMatrixB_local );
          _TEST_PERMS( symMatrixA_JI[ 0 ], matrixA_IKJ[ 0 ], symMatrixB_IJ[ 0 ], symMatrixB_JI[ 0 ], symMatrixB_local );
          _TEST_PERMS( symMatrixA_JI[ 0 ], matrixA_KJI[ 0 ], symMatrixB_IJ[ 0 ], symMatrixB_JI[ 0 ], symMatrixB_local );
          _TEST_PERMS( symMatrixA_JI[ 0 ], matrixA_local, symMatrixB_IJ[ 0 ], symMatrixB_JI[ 0 ], symMatrixB_local );
          _TEST_PERMS( symMatrixA_local, matrixA_IJK[ 0 ], symMatrixB_IJ[ 0 ], symMatrixB_JI[ 0 ], symMatrixB_local );
          _TEST_PERMS( symMatrixA_local, matrixA_IKJ[ 0 ], symMatrixB_IJ[ 0 ], symMatrixB_JI[ 0 ], symMatrixB_local );
          _TEST_PERMS( symMatrixA_local, matrixA_KJI[ 0 ], symMatrixB_IJ[ 0 ], symMatrixB_JI[ 0 ], symMatrixB_local );
          _TEST_PERMS( symMatrixA_local, matrixA_local, symMatrixB_IJ[ 0 ], symMatrixB_JI[ 0 ], symMatrixB_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testSymRij_eq_AiAj()
  {
    T result[ N ][ N ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    {
      for( std::ptrdiff_t j = 0; j < N; ++j )
      {
        result[ i ][ j ] = m_vectorA_local[ i ] * m_vectorA_local[ j ];
      }
    }

    T symResult[ SYM_SIZE ];
    tensorOps::denseToSymmetric< N >( symResult, result );

    ArrayViewT< T, 2, 1 > const symMatrixA_IJ = m_symMatrixA_IJ.toView();
    ArrayViewT< T, 2, 0 > const symMatrixA_JI = m_symMatrixA_JI.toView();

    ArrayViewT< T const, 2, 1 > const vectorA_IJ = m_vectorA_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const vectorA_JI = m_vectorA_JI.toViewConst();
    T const ( &vectorA_local )[ N ] = m_vectorA_local;

    std::ptrdiff_t const matrixSeed = m_seedSymMatrixA;

    forall< POLICY >( 1, [symResult, symMatrixA_IJ, symMatrixA_JI, vectorA_IJ, vectorA_JI, vectorA_local, matrixSeed]
                      LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( symMatrix, vectorN ) \
            fill( symMatrix, matrixSeed ); \
            tensorOps::symRij_eq_AiAj< N >( symMatrix, vectorN ); \
            CHECK_EQUALITY_1D( SYM_SIZE, symMatrix, symResult )

          #define _TEST_PERMS( symMatrix, vectorN0, vectorN1, vectorN2 ) \
            _TEST( symMatrix, vectorN0 ); \
            _TEST( symMatrix, vectorN1 ); \
            _TEST( symMatrix, vectorN2 )

          T symMatrix_local[ SYM_SIZE ];

          _TEST_PERMS( symMatrixA_IJ[ 0 ], vectorA_IJ[ 0 ], vectorA_JI[ 0 ], vectorA_local );
          _TEST_PERMS( symMatrixA_JI[ 0 ], vectorA_IJ[ 0 ], vectorA_JI[ 0 ], vectorA_local );
          _TEST_PERMS( symMatrix_local, vectorA_IJ[ 0 ], vectorA_JI[ 0 ], vectorA_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testSymRij_eq_AiBj_plus_AjBi()
  {
    T result[ N ][ N ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    {
      for( std::ptrdiff_t j = 0; j < N; ++j )
      {
        result[ i ][ j ] = m_vectorA_local[ i ] * m_vectorB_local[ j ] + m_vectorA_local[ j ] * m_vectorB_local[ i ];
      }
    }

    T symResult[ SYM_SIZE ];
    tensorOps::denseToSymmetric< N >( symResult, result );

    ArrayViewT< T, 2, 1 > const symMatrixA_IJ = m_symMatrixA_IJ.toView();
    ArrayViewT< T, 2, 0 > const symMatrixA_JI = m_symMatrixA_JI.toView();

    ArrayViewT< T const, 2, 1 > const vectorA_IJ = m_vectorA_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const vectorA_JI = m_vectorA_JI.toViewConst();
    T const ( &vectorA_local )[ N ] = m_vectorA_local;

    ArrayViewT< T const, 2, 1 > const vectorB_IJ = m_vectorB_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const vectorB_JI = m_vectorB_JI.toViewConst();
    T const ( &vectorB_local )[ N ] = m_vectorB_local;

    std::ptrdiff_t const matrixSeed = m_seedSymMatrixA;

    forall< POLICY >( 1, [symResult, symMatrixA_IJ, symMatrixA_JI, vectorA_IJ, vectorA_JI, vectorB_IJ, vectorB_JI, vectorA_local, vectorB_local, matrixSeed]
                      LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( symMatrix, vectorN, vectorM ) \
            fill( symMatrix, matrixSeed ); \
            tensorOps::symRij_eq_AiBj_plus_AjBi< N >( symMatrix, vectorN, vectorM ); \
            CHECK_EQUALITY_1D( SYM_SIZE, symMatrix, symResult )

          #define _TEST_PERMS( symMatrix, vectorN, vectorM0, vectorM1, vectorM2 ) \
            _TEST( symMatrix, vectorN, vectorM0 ); \
            _TEST( symMatrix, vectorN, vectorM1 ); \
            _TEST( symMatrix, vectorN, vectorM2 )

          T symMatrix_local[ SYM_SIZE ];

          _TEST_PERMS( symMatrixA_IJ[ 0 ], vectorA_IJ[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( symMatrixA_JI[ 0 ], vectorA_IJ[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );

          _TEST_PERMS( symMatrixA_IJ[ 0 ], vectorA_JI[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( symMatrixA_JI[ 0 ], vectorA_JI[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );

          _TEST_PERMS( symMatrixA_IJ[ 0 ], vectorA_local, vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( symMatrixA_JI[ 0 ], vectorA_local, vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );

          _TEST_PERMS( symMatrix_local, vectorA_IJ[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( symMatrix_local, vectorA_JI[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( symMatrix_local, vectorA_local, vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );

         #undef _TEST_PERMS
         #undef _TEST
        } );
  }

  void symmetricToDense()
  {
    ArrayViewT< T const, 2, 1 > const symMatrixA_IJ = m_symMatrixA_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const symMatrixA_JI = m_symMatrixA_JI.toViewConst();
    T const ( &symMatrixA_local )[ SYM_SIZE ] = m_symMatrixA_local;

    ArrayViewT< T, 3, 2 > const matrixA_IJK = m_matrixA_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixA_KJI = m_matrixA_KJI.toView();

    std::ptrdiff_t const matrixASeed = m_seedMatrixA;

    forall< POLICY >( 1, [=] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( symMatrix, matrix ) \
            fill( matrix, matrixASeed ); \
            tensorOps::symmetricToDense< N >( matrix, symMatrix ); \
            checkSymmetricToDense< N >( matrix, symMatrix );

          #define _TEST_PERMS( symMatrix, matrix0, matrix1, matrix2, matrix3 ) \
            _TEST( symMatrix, matrix0 ) \
            _TEST( symMatrix, matrix1 ) \
            _TEST( symMatrix, matrix2 ) \
            _TEST( symMatrix, matrix3 )

          T matrixA_local[ N ][ N ];

          _TEST_PERMS( symMatrixA_IJ[ 0 ], matrixA_IJK[ 0 ], matrixA_IKJ[ 0 ], matrixA_KJI[ 0 ], matrixA_local );
          _TEST_PERMS( symMatrixA_JI[ 0 ], matrixA_IJK[ 0 ], matrixA_IKJ[ 0 ], matrixA_KJI[ 0 ], matrixA_local );
          _TEST_PERMS( symMatrixA_local, matrixA_IJK[ 0 ], matrixA_IKJ[ 0 ], matrixA_KJI[ 0 ], matrixA_local );

      #undef _TEST_PERMS
      #undef _TEST
        } );
  }

  void denseToSymmetric()
  {
    ArrayViewT< T, 2, 1 > const symMatrixA_IJ = m_symMatrixA_IJ.toView();
    ArrayViewT< T, 2, 0 > const symMatrixA_JI = m_symMatrixA_JI.toView();

    ArrayViewT< T const, 3, 2 > const matrixA_IJK = m_matrixA_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixA_IKJ = m_matrixA_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixA_KJI = m_matrixA_KJI.toViewConst();
    T const ( &matrixA_local )[ N ][ N ] = m_matrixA_local;

    std::ptrdiff_t const symMatrixASeed = m_seedSymMatrixA;

    forall< POLICY >( 1, [=] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( symMatrix, matrix ) \
            fill( symMatrix, symMatrixASeed ); \
            tensorOps::denseToSymmetric< N >( symMatrix, matrix ); \
            checkDenseToSymmetric< N >( matrix, symMatrix );

          #define _TEST_PERMS( symMatrix, matrix0, matrix1, matrix2, matrix3 ) \
            _TEST( symMatrix, matrix0 ) \
            _TEST( symMatrix, matrix1 ) \
            _TEST( symMatrix, matrix2 ) \
            _TEST( symMatrix, matrix3 )

          T symMatrixA_local[ SYM_SIZE ];

          _TEST_PERMS( symMatrixA_IJ[ 0 ], matrixA_IJK[ 0 ], matrixA_IKJ[ 0 ], matrixA_KJI[ 0 ], matrixA_local );
          _TEST_PERMS( symMatrixA_JI[ 0 ], matrixA_IJK[ 0 ], matrixA_IKJ[ 0 ], matrixA_KJI[ 0 ], matrixA_local );
          _TEST_PERMS( symMatrixA_local, matrixA_IJK[ 0 ], matrixA_IKJ[ 0 ], matrixA_KJI[ 0 ], matrixA_local );

      #undef _TEST_PERMS
      #undef _TEST
        } );
  }

private:

  std::ptrdiff_t const m_seedVectorA = 0;
  ArrayT< T, RAJA::PERM_IJ > m_vectorA_IJ { 1, N };
  ArrayT< T, RAJA::PERM_JI > m_vectorA_JI { 1, N };
  T m_vectorA_local[ N ];

  std::ptrdiff_t const m_seedVectorB = m_seedVectorA + N;
  ArrayT< T, RAJA::PERM_IJ > m_vectorB_IJ { 1, N };
  ArrayT< T, RAJA::PERM_JI > m_vectorB_JI { 1, N };
  T m_vectorB_local[ N ];

  std::ptrdiff_t const m_seedMatrixA = m_seedVectorB + N;
  ArrayT< T, RAJA::PERM_IJK > m_matrixA_IJK { 1, N, N };
  ArrayT< T, RAJA::PERM_IKJ > m_matrixA_IKJ { 1, N, N };
  ArrayT< T, RAJA::PERM_KJI > m_matrixA_KJI { 1, N, N };
  T m_matrixA_local[ N ][ N ];

  std::ptrdiff_t const m_seedMatrixB = m_seedMatrixA + N * N;
  ArrayT< T, RAJA::PERM_IJK > m_matrixB_IJK { 1, N, N };
  ArrayT< T, RAJA::PERM_IKJ > m_matrixB_IKJ { 1, N, N };
  ArrayT< T, RAJA::PERM_KJI > m_matrixB_KJI { 1, N, N };
  T m_matrixB_local[ N ][ N ];

  std::ptrdiff_t const m_seedSymMatrixA = 0;
  ArrayT< T, RAJA::PERM_IJ > m_symMatrixA_IJ { 1, SYM_SIZE };
  ArrayT< T, RAJA::PERM_JI > m_symMatrixA_JI { 1, SYM_SIZE };
  T m_symMatrixA_local[ SYM_SIZE ];

  std::ptrdiff_t const m_seedSymMatrixB = m_seedSymMatrixA + N;
  ArrayT< T, RAJA::PERM_IJ > m_symMatrixB_IJ { 1, SYM_SIZE};
  ArrayT< T, RAJA::PERM_JI > m_symMatrixB_JI { 1, SYM_SIZE };
  T m_symMatrixB_local[ SYM_SIZE ];
};

using FixedSizeSquareMatrixTestTypes = ::testing::Types<
  std::tuple< double, std::integral_constant< int, 2 >, serialPolicy >
  , std::tuple< double, std::integral_constant< int, 3 >, serialPolicy >

#if ( defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP) ) && defined(LVARRAY_USE_CHAI)
  , std::tuple< double, std::integral_constant< int, 2 >, parallelDevicePolicy< 32 > >
  , std::tuple< double, std::integral_constant< int, 3 >, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( FixedSizeSquareMatrixTest, FixedSizeSquareMatrixTestTypes, );

TYPED_TEST( FixedSizeSquareMatrixTest, symAijBj )
{
  this->symAijBj();
}

TYPED_TEST( FixedSizeSquareMatrixTest, Ri_add_symAijBj )
{
  this->Ri_add_symAijBj();
}

TYPED_TEST( FixedSizeSquareMatrixTest, symAikBjk )
{
  this->symAikBjk();
}

TYPED_TEST( FixedSizeSquareMatrixTest, Rij_eq_AikSymBklAjl )
{
  this->Rij_eq_AikSymBklAjl();
}

TYPED_TEST( FixedSizeSquareMatrixTest, symRij_eq_AiAj )
{
  this->testSymRij_eq_AiAj();
}

TYPED_TEST( FixedSizeSquareMatrixTest, symRij_eq_AiBj_plus_AjBi )
{
  this->testSymRij_eq_AiBj_plus_AjBi();
}

TYPED_TEST( FixedSizeSquareMatrixTest, symmetricToDense )
{
  this->symmetricToDense();
}

TYPED_TEST( FixedSizeSquareMatrixTest, denseToSymmetric )
{
  this->denseToSymmetric();
}
} // namespace testing
} // namespace LvArray
