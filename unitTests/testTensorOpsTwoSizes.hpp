/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

#pragma once

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

template< typename T_N_M_POLICY_TUPLE >
class TwoSizesTest : public ::testing::Test
{
public:
  using T = std::tuple_element_t< 0, T_N_M_POLICY_TUPLE >;
  static constexpr std::ptrdiff_t N = std::tuple_element_t< 1, T_N_M_POLICY_TUPLE > {};
  static constexpr std::ptrdiff_t M = std::tuple_element_t< 2, T_N_M_POLICY_TUPLE > {};
  using POLICY = std::tuple_element_t< 3, T_N_M_POLICY_TUPLE >;

  void SetUp() override
  {
    fill( m_matrixA_IJK.toSlice(), m_matrixASeed );
    fill( m_matrixA_IKJ.toSlice(), m_matrixASeed );
    fill( m_matrixA_KJI.toSlice(), m_matrixASeed );
    fill( m_matrixA_local, m_matrixASeed );

    fill( m_matrixB_IJK.toSlice(), m_matrixBSeed );
    fill( m_matrixB_IKJ.toSlice(), m_matrixBSeed );
    fill( m_matrixB_KJI.toSlice(), m_matrixBSeed );
    fill( m_matrixB_local, m_matrixBSeed );

    fill( m_matrixNN_IJK.toSlice(), m_matrixNNSeed );
    fill( m_matrixNN_IKJ.toSlice(), m_matrixNNSeed );
    fill( m_matrixNN_KJI.toSlice(), m_matrixNNSeed );
    fill( m_matrixNN_local, m_matrixNNSeed );

    fill( m_matrixMN_IJK.toSlice(), m_matrixMNSeed );
    fill( m_matrixMN_IKJ.toSlice(), m_matrixMNSeed );
    fill( m_matrixMN_KJI.toSlice(), m_matrixMNSeed );
    fill( m_matrixMN_local, m_matrixMNSeed );

    fill( m_vectorN_IJ.toSlice(), m_vectorNSeed );
    fill( m_vectorN_JI.toSlice(), m_vectorNSeed );
    fill( m_vectorN_local, m_vectorNSeed );

    fill( m_vectorM_IJ.toSlice(), m_vectorMSeed );
    fill( m_vectorM_JI.toSlice(), m_vectorMSeed );
    fill( m_vectorM_local, m_vectorMSeed );
  }

  void testScale()
  {
    T scale = T( 3.14 );
    T result[ N ][ M ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    {
      for( std::ptrdiff_t j = 0; j < M; ++j )
      {
        result[ i ][ j ] = m_matrixA_local[ i ][ j ] * scale;
      }
    }

    ArrayViewT< T, 3, 2 > const matrixA_IJK = m_matrixA_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixA_KJI = m_matrixA_KJI.toView();

    std::ptrdiff_t const aSeed = m_matrixASeed;
    forall< POLICY >( 1, [scale, result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, aSeed] LVARRAY_HOST_DEVICE ( int )
        {
          tensorOps::scale< N, M >( matrixA_IJK[ 0 ], scale );
          CHECK_EQUALITY_2D( N, M, matrixA_IJK[ 0 ], result );

          tensorOps::scale< N, M >( matrixA_IKJ[ 0 ], scale );
          CHECK_EQUALITY_2D( N, M, matrixA_IKJ[ 0 ], result );

          tensorOps::scale< N, M >( matrixA_KJI[ 0 ], scale );
          CHECK_EQUALITY_2D( N, M, matrixA_KJI[ 0 ], result );

          T matrix_local[ N ][ M ];
          fill( matrix_local, aSeed );
          tensorOps::scale< N, M >( matrix_local, scale );
          CHECK_EQUALITY_2D( N, M, matrix_local, result );
        } );
  }

  void testFill()
  {
    ArrayViewT< T, 3, 2 > const matrixA_IJK = m_matrixA_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixA_KJI = m_matrixA_KJI.toView();

    forall< POLICY >( 1, [matrixA_IJK, matrixA_IKJ, matrixA_KJI] LVARRAY_HOST_DEVICE ( int )
        {
          for( int i = 0; i < 3; ++i )
          {
            T const value = 3.14 * i;
            tensorOps::fill< N, M >( matrixA_IJK[ 0 ], value );
            for( std::ptrdiff_t j = 0; j < N; ++j )
            {
              for( std::ptrdiff_t k = 0; k < M; ++k )
              {
                PORTABLE_EXPECT_EQ( matrixA_IJK( 0, j, k ), value );
              }
            }

            tensorOps::fill< N, M >( matrixA_IKJ[ 0 ], value );
            for( std::ptrdiff_t j = 0; j < N; ++j )
            {
              for( std::ptrdiff_t k = 0; k < M; ++k )
              {
                PORTABLE_EXPECT_EQ( matrixA_IKJ( 0, j, k ), value );
              }
            }

            tensorOps::fill< N, M >( matrixA_KJI[ 0 ], value );
            for( std::ptrdiff_t j = 0; j < N; ++j )
            {
              for( std::ptrdiff_t k = 0; k < M; ++k )
              {
                PORTABLE_EXPECT_EQ( matrixA_KJI( 0, j, k ), value );
              }
            }

            T matrix_local[ N ][ M ];
            tensorOps::fill< N, M >( matrix_local, value );
            for( std::ptrdiff_t j = 0; j < N; ++j )
            {
              for( std::ptrdiff_t k = 0; k < M; ++k )
              {
                PORTABLE_EXPECT_EQ( matrix_local[ j ][ k ], value );
              }
            }
          }
        } );
  }

  void testAiBj()
  {
    T result[ N ][ M ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    {
      for( std::ptrdiff_t j = 0; j < M; ++j )
      {
        result[ i ][ j ] = m_vectorN_local[ i ] * m_vectorM_local[ j ];
      }
    }

    ArrayViewT< T, 3, 2 > const matrixA_IJK = m_matrixA_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixA_KJI = m_matrixA_KJI.toView();

    ArrayViewT< T const, 2, 1 > const vectorN_IJ = m_vectorN_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const vectorN_JI = m_vectorN_JI.toViewConst();
    T const ( &vectorN_local )[ N ] = m_vectorN_local;

    ArrayViewT< T const, 2, 1 > const vectorM_IJ = m_vectorM_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const vectorM_JI = m_vectorM_JI.toViewConst();
    T const ( &vectorM_local )[ M ] = m_vectorM_local;

    std::ptrdiff_t const matrixSeed = m_matrixASeed;

    forall< POLICY >( 1, [result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, vectorN_IJ, vectorN_JI, vectorN_local,
                          vectorM_IJ, vectorM_JI, vectorM_local, matrixSeed] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( matrix, vectorN, vectorM ) \
            fill( matrix, matrixSeed ); \
            tensorOps::Rij_eq_AiBj< N, M >( matrix, vectorN, vectorM ); \
            CHECK_EQUALITY_2D( N, M, matrix, result )

          #define _TEST_PERMS( matrix, vectorN, vectorM0, vectorM1, vectorM2 ) \
            _TEST( matrix, vectorN, vectorM0 ); \
            _TEST( matrix, vectorN, vectorM1 ); \
            _TEST( matrix, vectorN, vectorM2 )

          T matrix_local[ N ][ M ];

          _TEST_PERMS( matrixA_IJK[ 0 ], vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrix_local, vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrix_local, vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrix_local, vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testPlusAiBj()
  {
    T result[ N ][ M ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    {
      for( std::ptrdiff_t j = 0; j < M; ++j )
      {
        result[ i ][ j ] = m_matrixA_local[ i ][ j ] + m_vectorN_local[ i ] * m_vectorM_local[ j ];
      }
    }

    ArrayViewT< T, 3, 2 > const matrixA_IJK = m_matrixA_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixA_KJI = m_matrixA_KJI.toView();

    ArrayViewT< T const, 2, 1 > const vectorN_IJ = m_vectorN_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const vectorN_JI = m_vectorN_JI.toViewConst();
    T const ( &vectorN_local )[ N ] = m_vectorN_local;

    ArrayViewT< T const, 2, 1 > const vectorM_IJ = m_vectorM_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const vectorM_JI = m_vectorM_JI.toViewConst();
    T const ( &vectorM_local )[ M ] = m_vectorM_local;

    std::ptrdiff_t const matrixSeed = m_matrixASeed;

    forall< POLICY >( 1,
                      [result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, vectorN_IJ, vectorN_JI, vectorN_local, vectorM_IJ, vectorM_JI, vectorM_local, matrixSeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( matrix, vectorN, vectorM ) \
            fill( matrix, matrixSeed ); \
            tensorOps::Rij_add_AiBj< N, M >( matrix, vectorN, vectorM ); \
            CHECK_EQUALITY_2D( N, M, matrix, result )

          #define _TEST_PERMS( matrix, vectorN, vectorM0, vectorM1, vectorM2 ) \
            _TEST( matrix, vectorN, vectorM0 ); \
            _TEST( matrix, vectorN, vectorM1 ); \
            _TEST( matrix, vectorN, vectorM2 )

          T matrix_local[ N ][ M ];

          _TEST_PERMS( matrixA_IJK[ 0 ], vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrix_local, vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrix_local, vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrix_local, vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testAijBj()
  {
    T result[ N ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    {
      T dot = 0;
      for( std::ptrdiff_t j = 0; j < M; ++j )
      {
        dot += m_matrixA_local[ i ][ j ] * m_vectorM_local[ j ];
      }
      result[ i ] = dot;
    }

    ArrayViewT< T const, 3, 2 > const matrixA_IJK = m_matrixA_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixA_IKJ = m_matrixA_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixA_KJI = m_matrixA_KJI.toViewConst();
    T const ( &matrix_local )[ N ][ M ] = m_matrixA_local;

    ArrayViewT< T, 2, 1 > const vectorN_IJ = m_vectorN_IJ.toView();
    ArrayViewT< T, 2, 0 > const vectorN_JI = m_vectorN_JI.toView();

    ArrayViewT< T const, 2, 1 > const vectorM_IJ = m_vectorM_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const vectorM_JI = m_vectorM_JI.toViewConst();
    T const ( &vectorM_local )[ M ] = m_vectorM_local;

    std::ptrdiff_t const vectorNSeed = m_vectorNSeed;

    forall< POLICY >( 1,
                      [result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, matrix_local, vectorN_IJ, vectorN_JI, vectorM_IJ, vectorM_JI, vectorM_local, vectorNSeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( matrix, vectorN, vectorM ) \
            fill( vectorN, vectorNSeed ); \
            tensorOps::Ri_eq_AijBj< N, M >( vectorN, matrix, vectorM ); \
            CHECK_EQUALITY_1D( N, vectorN, result )

          #define _TEST_PERMS( matrix, vectorN, vectorM0, vectorM1, vectorM2 ) \
            _TEST( matrix, vectorN, vectorM0 ); \
            _TEST( matrix, vectorN, vectorM1 ); \
            _TEST( matrix, vectorN, vectorM2 )

          T vectorN_local[ N ];

          _TEST_PERMS( matrixA_IJK[ 0 ], vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrix_local, vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrix_local, vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrix_local, vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testPlusAijBj()
  {
    T result[ N ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    {
      T dot = 0;
      for( std::ptrdiff_t j = 0; j < M; ++j )
      {
        dot += m_matrixA_local[ i ][ j ] * m_vectorM_local[ j ];
      }
      result[ i ] = m_vectorN_local[ i ] + dot;
    }

    ArrayViewT< T const, 3, 2 > const matrixA_IJK = m_matrixA_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixA_IKJ = m_matrixA_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixA_KJI = m_matrixA_KJI.toViewConst();
    T const ( &matrix_local )[ N ][ M ] = m_matrixA_local;

    ArrayViewT< T, 2, 1 > const vectorN_IJ = m_vectorN_IJ.toView();
    ArrayViewT< T, 2, 0 > const vectorN_JI = m_vectorN_JI.toView();

    ArrayViewT< T const, 2, 1 > const vectorM_IJ = m_vectorM_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const vectorM_JI = m_vectorM_JI.toViewConst();
    T const ( &vectorM_local )[ M ] = m_vectorM_local;

    std::ptrdiff_t const vectorNSeed = m_vectorNSeed;

    forall< POLICY >( 1,
                      [result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, matrix_local, vectorN_IJ, vectorN_JI, vectorM_IJ, vectorM_JI, vectorM_local, vectorNSeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( matrix, vectorN, vectorM ) \
            fill( vectorN, vectorNSeed ); \
            tensorOps::Ri_add_AijBj< N, M >( vectorN, matrix, vectorM ); \
            CHECK_EQUALITY_1D( N, vectorN, result )

          #define _TEST_PERMS( matrix, vectorN, vectorM0, vectorM1, vectorM2 ) \
            _TEST( matrix, vectorN, vectorM0 ); \
            _TEST( matrix, vectorN, vectorM1 ); \
            _TEST( matrix, vectorN, vectorM2 )

          T vectorN_local[ N ];

          _TEST_PERMS( matrixA_IJK[ 0 ], vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrix_local, vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrix_local, vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrix_local, vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testAjiBj()
  {
    T result[ M ];
    for( std::ptrdiff_t i = 0; i < M; ++i )
    {
      T dot = 0;
      for( std::ptrdiff_t j = 0; j < N; ++j )
      {
        dot += m_matrixA_local[ j ][ i ] * m_vectorN_local[ j ];
      }
      result[ i ] = dot;
    }

    ArrayViewT< T const, 3, 2 > const matrixA_IJK = m_matrixA_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixA_IKJ = m_matrixA_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixA_KJI = m_matrixA_KJI.toViewConst();
    T const ( &matrix_local )[ N ][ M ] = m_matrixA_local;

    ArrayViewT< T const, 2, 1 > const vectorN_IJ = m_vectorN_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const vectorN_JI = m_vectorN_JI.toViewConst();
    T const ( &vectorN_local )[ N ] = m_vectorN_local;

    ArrayViewT< T, 2, 1 > const vectorM_IJ = m_vectorM_IJ.toView();
    ArrayViewT< T, 2, 0 > const vectorM_JI = m_vectorM_JI.toView();

    std::ptrdiff_t const vectorMSeed = m_vectorMSeed;

    forall< POLICY >( 1,
                      [result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, matrix_local, vectorN_IJ, vectorN_JI, vectorN_local, vectorM_IJ, vectorM_JI, vectorMSeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( matrix, vectorN, vectorM ) \
            fill( vectorM, vectorMSeed ); \
            tensorOps::Ri_eq_AjiBj< M, N >( vectorM, matrix, vectorN ); \
            CHECK_EQUALITY_1D( M, vectorM, result )

          #define _TEST_PERMS( matrix, vectorN, vectorM0, vectorM1, vectorM2 ) \
            _TEST( matrix, vectorN, vectorM0 ); \
            _TEST( matrix, vectorN, vectorM1 ); \
            _TEST( matrix, vectorN, vectorM2 )

          T vectorM_local[ M ];

          _TEST_PERMS( matrixA_IJK[ 0 ], vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrix_local, vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrix_local, vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrix_local, vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testPlusAjiBj()
  {
    T result[ M ];
    for( std::ptrdiff_t i = 0; i < M; ++i )
    {
      T dot = 0;
      for( std::ptrdiff_t j = 0; j < N; ++j )
      {
        dot += m_matrixA_local[ j ][ i ] * m_vectorN_local[ j ];
      }
      result[ i ] = m_vectorM_local[ i ] + dot;
    }

    ArrayViewT< T const, 3, 2 > const matrixA_IJK = m_matrixA_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixA_IKJ = m_matrixA_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixA_KJI = m_matrixA_KJI.toViewConst();
    T const ( &matrix_local )[ N ][ M ] = m_matrixA_local;

    ArrayViewT< T const, 2, 1 > const vectorN_IJ = m_vectorN_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const vectorN_JI = m_vectorN_JI.toViewConst();
    T const ( &vectorN_local )[ N ] = m_vectorN_local;

    ArrayViewT< T, 2, 1 > const vectorM_IJ = m_vectorM_IJ.toView();
    ArrayViewT< T, 2, 0 > const vectorM_JI = m_vectorM_JI.toView();

    std::ptrdiff_t const vectorMSeed = m_vectorMSeed;

    forall< POLICY >( 1,
                      [result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, matrix_local, vectorN_IJ, vectorN_JI, vectorN_local, vectorM_IJ, vectorM_JI, vectorMSeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( matrix, vectorN, vectorM ) \
            fill( vectorM, vectorMSeed ); \
            tensorOps::Ri_add_AjiBj< M, N >( vectorM, matrix, vectorN ); \
            CHECK_EQUALITY_1D( M, vectorM, result )

          #define _TEST_PERMS( matrix, vectorN, vectorM0, vectorM1, vectorM2 ) \
            _TEST( matrix, vectorN, vectorM0 ); \
            _TEST( matrix, vectorN, vectorM1 ); \
            _TEST( matrix, vectorN, vectorM2 )

          T vectorM_local[ M ];

          _TEST_PERMS( matrixA_IJK[ 0 ], vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrix_local, vectorN_IJ[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrix_local, vectorN_JI[ 0 ], vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );
          _TEST_PERMS( matrix_local, vectorN_local, vectorM_IJ[ 0 ], vectorM_JI[ 0 ], vectorM_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testCopy()
  {
    ArrayViewT< T, 3, 2 > const matrixA_IJK = m_matrixA_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixA_KJI = m_matrixA_KJI.toView();

    ArrayViewT< T const, 3, 2 > const matrixB_IJK_view = m_matrixB_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixB_IKJ_view = m_matrixB_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixB_KJI_view = m_matrixB_KJI.toViewConst();
    T const ( &matrixB_local )[ N ][ M ] = m_matrixB_local;

    std::ptrdiff_t const matrixSeed = m_matrixASeed;

    forall< POLICY >( 1, [matrixA_IJK, matrixA_IKJ, matrixA_KJI, matrixB_IJK_view, matrixB_IKJ_view, matrixB_KJI_view, matrixB_local, matrixSeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( dstMatrix, srcMatrix ) \
            fill( dstMatrix, matrixSeed ); \
            tensorOps::copy< N, M >( dstMatrix, srcMatrix ); \
            CHECK_EQUALITY_2D( N, M, dstMatrix, srcMatrix )

          #define _TEST_PERMS( dstMatrix, srcMatrix0, srcMatrix1, srcMatrix2, srcMatrix3 ) \
            _TEST( dstMatrix, srcMatrix0 ); \
            _TEST( dstMatrix, srcMatrix1 ); \
            _TEST( dstMatrix, srcMatrix2 ); \
            _TEST( dstMatrix, srcMatrix3 )

          T matrix_local[ N ][ M ];

          _TEST_PERMS( matrixA_IJK[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrix_local, matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrix_local, matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrix_local, matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testScaledCopy()
  {
    T scale = T( 3.14 );
    T result[ N ][ M ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    {
      for( std::ptrdiff_t j = 0; j < M; ++j )
      {
        result[ i ][ j ] = scale * m_matrixB_local[ i ][ j ];
      }
    }

    ArrayViewT< T, 3, 2 > const matrixA_IJK = m_matrixA_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixA_KJI = m_matrixA_KJI.toView();

    ArrayViewT< T const, 3, 2 > const matrixB_IJK_view = m_matrixB_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixB_IKJ_view = m_matrixB_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixB_KJI_view = m_matrixB_KJI.toViewConst();
    T const ( &matrixB_local )[ N ][ M ] = m_matrixB_local;

    std::ptrdiff_t const matrixSeed = m_matrixASeed;

    forall< POLICY >( 1, [scale, result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, matrixB_IJK_view, matrixB_IKJ_view, matrixB_KJI_view, matrixB_local, matrixSeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( dstMatrix, srcMatrix ) \
            fill( dstMatrix, matrixSeed ); \
            tensorOps::scaledCopy< N, M >( dstMatrix, srcMatrix, scale ); \
            CHECK_EQUALITY_2D( N, M, dstMatrix, result )

          #define _TEST_PERMS( dstMatrix, srcMatrix0, srcMatrix1, srcMatrix2, srcMatrix3 ) \
            _TEST( dstMatrix, srcMatrix0 ); \
            _TEST( dstMatrix, srcMatrix1 ); \
            _TEST( dstMatrix, srcMatrix2 ); \
            _TEST( dstMatrix, srcMatrix3 )

          T matrix_local[ N ][ M ];

          _TEST_PERMS( matrixA_IJK[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrix_local, matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrix_local, matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrix_local, matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testAdd()
  {
    T result[ N ][ M ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    {
      for( std::ptrdiff_t j = 0; j < M; ++j )
      {
        result[ i ][ j ] = m_matrixA_local[ i ][ j ] + m_matrixB_local[ i ][ j ];
      }
    }

    ArrayViewT< T, 3, 2 > const matrixA_IJK = m_matrixA_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixA_KJI = m_matrixA_KJI.toView();

    ArrayViewT< T const, 3, 2 > const matrixB_IJK_view = m_matrixB_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixB_IKJ_view = m_matrixB_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixB_KJI_view = m_matrixB_KJI.toViewConst();
    T const ( &matrixB_local )[ N ][ M ] = m_matrixB_local;

    std::ptrdiff_t const matrixSeed = m_matrixASeed;

    forall< POLICY >( 1, [result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, matrixB_IJK_view, matrixB_IKJ_view, matrixB_KJI_view, matrixB_local, matrixSeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( dstMatrix, srcMatrix ) \
            fill( dstMatrix, matrixSeed ); \
            tensorOps::add< N, M >( dstMatrix, srcMatrix ); \
            CHECK_EQUALITY_2D( N, M, dstMatrix, result )

          #define _TEST_PERMS( dstMatrix, srcMatrix0, srcMatrix1, srcMatrix2, srcMatrix3 ) \
            _TEST( dstMatrix, srcMatrix0 ); \
            _TEST( dstMatrix, srcMatrix1 ); \
            _TEST( dstMatrix, srcMatrix2 ); \
            _TEST( dstMatrix, srcMatrix3 )

          T matrix_local[ N ][ M ];

          _TEST_PERMS( matrixA_IJK[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrix_local, matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrix_local, matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrix_local, matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testScaledAdd()
  {
    ArrayViewT< T, 3, 2 > const matrixA_IJK = m_matrixA_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixA_KJI = m_matrixA_KJI.toView();

    ArrayViewT< T const, 3, 2 > const matrixB_IJK_view = m_matrixB_IJK.toView();
    ArrayViewT< T const, 3, 1 > const matrixB_IKJ_view = m_matrixB_IKJ.toView();
    ArrayViewT< T const, 3, 0 > const matrixB_KJI_view = m_matrixB_KJI.toView();

    T const ( &matrixB_local )[ N ][ M ] = m_matrixB_local;

    std::ptrdiff_t const matrixSeed = m_matrixASeed;

    forall< POLICY >( 1, [matrixA_IJK, matrixA_IKJ, matrixA_KJI, matrixB_IJK_view, matrixB_IKJ_view, matrixB_KJI_view, matrixB_local, matrixSeed ]
                      LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( dstMatrix, srcMatrix ) \
            fill( dstMatrix, matrixSeed ); \
            tensorOps::scaledAdd< N, M >( dstMatrix, srcMatrix, scale ); \
            CHECK_EQUALITY_2D( N, M, dstMatrix, result ); \

          #define _TEST_PERMS( dstMatrix, srcMatrix0, srcMatrix1, srcMatrix2, srcMatrix3 ) \
            _TEST( dstMatrix, srcMatrix0 ); \
            _TEST( dstMatrix, srcMatrix1 ); \
            _TEST( dstMatrix, srcMatrix2 ); \
            _TEST( dstMatrix, srcMatrix3 )

          T matrixA_local[ N ][ M ];
          fill( matrixA_local, matrixSeed );

          T const scale = T( 3.14 );
          T result[ N ][ M ];
          for( std::ptrdiff_t i = 0; i < N; ++i )
          {
            for( std::ptrdiff_t j = 0; j < M; ++j )
            {
              result[ i ][ j ] = matrixA_local[ i ][ j ] + scale * matrixB_local[ i ][ j ];
            }
          }

          _TEST_PERMS( matrixA_IJK[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_local, matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_local, matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );
          _TEST_PERMS( matrixA_local, matrixB_IJK_view[ 0 ], matrixB_IKJ_view[ 0 ], matrixB_KJI_view[ 0 ], matrixB_local );

            #undef _TEST_PERMS
            #undef _TEST
        } );
  }

  void testAkiAkj()
  {
    T result[ N ][ N ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    {
      for( std::ptrdiff_t j = 0; j < N; ++j )
      {
        T dot = 0;
        for( std::ptrdiff_t k = 0; k < M; ++k )
        {
          dot += m_matrixMN_local[ k ][ i ] * m_matrixMN_local[ k ][ j ];
        }
        result[ i ][ j ] = dot;
      }
    }

    ArrayViewT< T const, 3, 2 > const matrixMN_IJK = m_matrixMN_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixMN_IKJ = m_matrixMN_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixMN_KJI = m_matrixMN_KJI.toViewConst();
    T const ( &matrixMN_local )[ M ][ N ] = m_matrixMN_local;

    ArrayViewT< T, 3, 2 > const matrixNN_IJK = m_matrixNN_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixNN_IKJ = m_matrixNN_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixNN_KJI = m_matrixNN_KJI.toView();

    std::ptrdiff_t const matrixNNSeed = m_matrixNNSeed;

    forall< POLICY >( 1,
                      [result, matrixMN_IJK, matrixMN_IKJ, matrixMN_KJI, matrixMN_local, matrixNN_IJK,
                       matrixNN_IKJ, matrixNN_KJI, matrixNNSeed ] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( matrixNN, matrixMN ) \
            fill( matrixNN, matrixNNSeed ); \
            tensorOps::Rij_eq_AkiAkj< N, M >( matrixNN, matrixMN ); \
            CHECK_EQUALITY_2D( N, N, matrixNN, result )

          #define _TEST_PERMS( matrixNN, matrixMN0, matrixMN1, matrixMN2, matrixMN3 ) \
            _TEST( matrixNN, matrixMN0 ); \
            _TEST( matrixNN, matrixMN1 ); \
            _TEST( matrixNN, matrixMN2 ); \
            _TEST( matrixNN, matrixMN3 )

          T matrixNN_local[ N ][ N ];

          _TEST_PERMS( matrixNN_IJK[ 0 ], matrixMN_IJK[ 0 ], matrixMN_IKJ[ 0 ], matrixMN_KJI[ 0 ], matrixMN_local );
          _TEST_PERMS( matrixNN_IKJ[ 0 ], matrixMN_IJK[ 0 ], matrixMN_IKJ[ 0 ], matrixMN_KJI[ 0 ], matrixMN_local );
          _TEST_PERMS( matrixNN_KJI[ 0 ], matrixMN_IJK[ 0 ], matrixMN_IKJ[ 0 ], matrixMN_KJI[ 0 ], matrixMN_local );
          _TEST_PERMS( matrixNN_local, matrixMN_IJK[ 0 ], matrixMN_IKJ[ 0 ], matrixMN_KJI[ 0 ], matrixMN_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testPlusAikAjk()
  {
    T result[ N ][ N ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    {
      for( std::ptrdiff_t j = 0; j < N; ++j )
      {
        T dot = 0;
        for( std::ptrdiff_t k = 0; k < M; ++k )
        {
          dot += m_matrixA_local[ i ][ k ] * m_matrixA_local[ j ][ k ];
        }
        result[ i ][ j ] = m_matrixNN_local[ i ][ j ] + dot;
      }
    }

    ArrayViewT< T const, 3, 2 > const matrixA_IJK = m_matrixA_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixA_IKJ = m_matrixA_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixA_KJI = m_matrixA_KJI.toViewConst();
    T const ( &matrixA_local )[ N ][ M ] = m_matrixA_local;

    ArrayViewT< T, 3, 2 > const matrixNN_IJK = m_matrixNN_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixNN_IKJ = m_matrixNN_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixNN_KJI = m_matrixNN_KJI.toView();

    std::ptrdiff_t const matrixNNSeed = m_matrixNNSeed;

    forall< POLICY >( 1,
                      [result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, matrixA_local, matrixNN_IJK,
                       matrixNN_IKJ, matrixNN_KJI, matrixNNSeed ] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( matrixNN, matrixA ) \
            fill( matrixNN, matrixNNSeed ); \
            tensorOps::Rij_add_AikAjk< N, M >( matrixNN, matrixA ); \
            CHECK_EQUALITY_2D( N, N, matrixNN, result )

          #define _TEST_PERMS( matrixNN, matrixA0, matrixA1, matrixA2, matrixA3 ) \
            _TEST( matrixNN, matrixA0 ); \
            _TEST( matrixNN, matrixA1 ); \
            _TEST( matrixNN, matrixA2 ); \
            _TEST( matrixNN, matrixA3 )

          T matrixNN_local[ N ][ N ];

          _TEST_PERMS( matrixNN_IJK[ 0 ], matrixA_IJK[ 0 ], matrixA_IKJ[ 0 ], matrixA_KJI[ 0 ], matrixA_local );
          _TEST_PERMS( matrixNN_IKJ[ 0 ], matrixA_IJK[ 0 ], matrixA_IKJ[ 0 ], matrixA_KJI[ 0 ], matrixA_local );
          _TEST_PERMS( matrixNN_KJI[ 0 ], matrixA_IJK[ 0 ], matrixA_IKJ[ 0 ], matrixA_KJI[ 0 ], matrixA_local );
          _TEST_PERMS( matrixNN_local, matrixA_IJK[ 0 ], matrixA_IKJ[ 0 ], matrixA_KJI[ 0 ], matrixA_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testTranspose()
  {
    ArrayViewT< T, 3, 2 > const matrixA_IJK = m_matrixA_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixA_KJI = m_matrixA_KJI.toView();

    ArrayViewT< T const, 3, 2 > const matrixMN_IJK_view = m_matrixMN_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixMN_IKJ_view = m_matrixMN_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixMN_KJI_view = m_matrixMN_KJI.toViewConst();
    T const ( &matrixMN_local )[ M ][ N ] = m_matrixMN_local;

    std::ptrdiff_t const matrixSeed = m_matrixASeed;

    forall< POLICY >( 1, [=] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( dstMatrix, srcMatrix ) \
            fill( dstMatrix, matrixSeed ); \
            tensorOps::transpose< N, M >( dstMatrix, srcMatrix ); \
            for( int i = 0; i < N; ++i ) \
            { \
              for( int j = 0; j < M; ++j ) \
              { \
                PORTABLE_EXPECT_EQ( dstMatrix[ i ][ j ], srcMatrix[ j ][ i ] ); \
              } \
            }

          #define _TEST_PERMS( dstMatrix, srcMatrix0, srcMatrix1, srcMatrix2, srcMatrix3 ) \
            _TEST( dstMatrix, srcMatrix0 ); \
            _TEST( dstMatrix, srcMatrix1 ); \
            _TEST( dstMatrix, srcMatrix2 ); \
            _TEST( dstMatrix, srcMatrix3 )

          T matrix_local[ N ][ M ];

          _TEST_PERMS( matrixA_IJK[ 0 ], matrixMN_IJK_view[ 0 ], matrixMN_IKJ_view[ 0 ], matrixMN_KJI_view[ 0 ], matrixMN_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], matrixMN_IJK_view[ 0 ], matrixMN_IKJ_view[ 0 ], matrixMN_KJI_view[ 0 ], matrixMN_local );
          _TEST_PERMS( matrixA_IJK[ 0 ], matrixMN_IJK_view[ 0 ], matrixMN_IKJ_view[ 0 ], matrixMN_KJI_view[ 0 ], matrixMN_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], matrixMN_IJK_view[ 0 ], matrixMN_IKJ_view[ 0 ], matrixMN_KJI_view[ 0 ], matrixMN_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], matrixMN_IJK_view[ 0 ], matrixMN_IKJ_view[ 0 ], matrixMN_KJI_view[ 0 ], matrixMN_local );
          _TEST_PERMS( matrixA_IKJ[ 0 ], matrixMN_IJK_view[ 0 ], matrixMN_IKJ_view[ 0 ], matrixMN_KJI_view[ 0 ], matrixMN_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], matrixMN_IJK_view[ 0 ], matrixMN_IKJ_view[ 0 ], matrixMN_KJI_view[ 0 ], matrixMN_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], matrixMN_IJK_view[ 0 ], matrixMN_IKJ_view[ 0 ], matrixMN_KJI_view[ 0 ], matrixMN_local );
          _TEST_PERMS( matrixA_KJI[ 0 ], matrixMN_IJK_view[ 0 ], matrixMN_IKJ_view[ 0 ], matrixMN_KJI_view[ 0 ], matrixMN_local );
          _TEST_PERMS( matrix_local, matrixMN_IJK_view[ 0 ], matrixMN_IKJ_view[ 0 ], matrixMN_KJI_view[ 0 ], matrixMN_local );
          _TEST_PERMS( matrix_local, matrixMN_IJK_view[ 0 ], matrixMN_IKJ_view[ 0 ], matrixMN_KJI_view[ 0 ], matrixMN_local );
          _TEST_PERMS( matrix_local, matrixMN_IJK_view[ 0 ], matrixMN_IKJ_view[ 0 ], matrixMN_KJI_view[ 0 ], matrixMN_local );

      #undef _TEST_PERMS
      #undef _TEST
        } );
  }

private:
  std::ptrdiff_t const m_matrixASeed = 0;
  ArrayT< T, RAJA::PERM_IJK > m_matrixA_IJK { 1, N, M };
  ArrayT< T, RAJA::PERM_IKJ > m_matrixA_IKJ { 1, N, M };
  ArrayT< T, RAJA::PERM_KJI > m_matrixA_KJI { 1, N, M };
  T m_matrixA_local[ N ][ M ];

  std::ptrdiff_t const m_matrixBSeed = m_matrixASeed + N * M;
  ArrayT< T, RAJA::PERM_IJK > m_matrixB_IJK { 1, N, M };
  ArrayT< T, RAJA::PERM_IKJ > m_matrixB_IKJ { 1, N, M };
  ArrayT< T, RAJA::PERM_KJI > m_matrixB_KJI { 1, N, M };
  T m_matrixB_local[ N ][ M ];

  std::ptrdiff_t const m_matrixNNSeed = m_matrixBSeed + N * M;
  ArrayT< T, RAJA::PERM_IJK > m_matrixNN_IJK { 1, N, N };
  ArrayT< T, RAJA::PERM_IKJ > m_matrixNN_IKJ { 1, N, N };
  ArrayT< T, RAJA::PERM_KJI > m_matrixNN_KJI { 1, N, N };
  T m_matrixNN_local[ N ][ N ];

  std::ptrdiff_t const m_matrixMNSeed = m_matrixNNSeed + N * N;
  ArrayT< T, RAJA::PERM_IJK > m_matrixMN_IJK { 1, M, N };
  ArrayT< T, RAJA::PERM_IKJ > m_matrixMN_IKJ { 1, M, N };
  ArrayT< T, RAJA::PERM_KJI > m_matrixMN_KJI { 1, M, N };
  T m_matrixMN_local[ M ][ N ];

  std::ptrdiff_t const m_vectorNSeed = m_matrixMNSeed + N * M;
  ArrayT< T, RAJA::PERM_IJ > m_vectorN_IJ { 1, N };
  ArrayT< T, RAJA::PERM_JI > m_vectorN_JI { 1, N };
  T m_vectorN_local[ N ];

  std::ptrdiff_t const m_vectorMSeed = m_vectorNSeed + N;
  ArrayT< T, RAJA::PERM_IJ > m_vectorM_IJ { 1, M };
  ArrayT< T, RAJA::PERM_JI > m_vectorM_JI { 1, M };
  T m_vectorM_local[ M ];
};


using TwoSizesTestTypes = ::testing::Types<
  std::tuple< double, std::integral_constant< int, 2 >, std::integral_constant< int, 3 >, serialPolicy >
  , std::tuple< int, std::integral_constant< int, 5 >, std::integral_constant< int, 4 >, serialPolicy >
  , std::tuple< double, std::integral_constant< int, 3 >, std::integral_constant< int, 3 >, serialPolicy >

#if ( defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP) ) && defined(LVARRAY_USE_CHAI)
  , std::tuple< double, std::integral_constant< int, 2 >, std::integral_constant< int, 3 >, parallelDevicePolicy< 32 > >
  , std::tuple< int, std::integral_constant< int, 5 >, std::integral_constant< int, 4 >, parallelDevicePolicy< 32 > >
  , std::tuple< double, std::integral_constant< int, 3 >, std::integral_constant< int, 3 >, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( TwoSizesTest, TwoSizesTestTypes, );

} // namespace testing
} // namespace LvArray
