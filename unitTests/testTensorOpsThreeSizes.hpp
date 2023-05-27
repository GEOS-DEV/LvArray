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

template< typename T_N_M_L_POLICY_TUPLE >
class ThreeSizesTest : public ::testing::Test
{
public:
  using T = std::tuple_element_t< 0, T_N_M_L_POLICY_TUPLE >;
  static constexpr std::ptrdiff_t N = std::tuple_element_t< 1, T_N_M_L_POLICY_TUPLE > {};
  static constexpr std::ptrdiff_t M = std::tuple_element_t< 2, T_N_M_L_POLICY_TUPLE > {};
  static constexpr std::ptrdiff_t L = std::tuple_element_t< 3, T_N_M_L_POLICY_TUPLE > {};
  using POLICY = std::tuple_element_t< 4, T_N_M_L_POLICY_TUPLE >;

  void SetUp() override
  {
    fill( m_matrixNM_IJK.toSlice(), m_matrixNMSeed );
    fill( m_matrix_NM_IKJ.toSlice(), m_matrixNMSeed );
    fill( m_matrix_NM_KJI.toSlice(), m_matrixNMSeed );
    fill( m_matrixNM_local, m_matrixNMSeed );

    fill( m_matrixNL_IJK.toSlice(), m_matrixNLSeed );
    fill( m_matrixNL_IKJ.toSlice(), m_matrixNLSeed );
    fill( m_matrixNL_KJI.toSlice(), m_matrixNLSeed );
    fill( m_matrixNL_local, m_matrixNLSeed );

    fill( m_matrixLM_IJK.toSlice(), m_matrixLMSeed );
    fill( m_matrixLM_IKJ.toSlice(), m_matrixLMSeed );
    fill( m_matrixLM_KJI.toSlice(), m_matrixLMSeed );
    fill( m_matrixLM_local, m_matrixLMSeed );

    fill( m_matrixML_IJK.toSlice(), m_matrixMLSeed );
    fill( m_matrixML_IKJ.toSlice(), m_matrixMLSeed );
    fill( m_matrixML_KJI.toSlice(), m_matrixMLSeed );
    fill( m_matrixML_local, m_matrixMLSeed );
  }

  void testAikBkj()
  {
    T result[ N ][ M ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    {
      for( std::ptrdiff_t j = 0; j < M; ++j )
      {
        T dot = 0;
        for( std::ptrdiff_t k = 0; k < L; ++k )
        {
          dot += m_matrixNL_local[ i ][ k ] * m_matrixLM_local[ k ][ j ];
        }
        result[ i ][ j ] = dot;
      }
    }

    ArrayViewT< T, 3, 2 > const matrixNM_IJK = m_matrixNM_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixNM_IKJ = m_matrix_NM_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixNM_KJI = m_matrix_NM_KJI.toView();

    ArrayViewT< T const, 3, 2 > const matrixML_IJK = m_matrixNL_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixML_IKJ = m_matrixNL_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixML_KJI = m_matrixNL_KJI.toViewConst();
    T const ( &matrixML_local )[ N ][ L ] = m_matrixNL_local;

    ArrayViewT< T const, 3, 2 > const matrixLM_IJK = m_matrixLM_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixLM_IKJ = m_matrixLM_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixLM_KJI = m_matrixLM_KJI.toViewConst();
    T const ( &matrixLM_local )[ L ][ M ] = m_matrixLM_local;

    std::ptrdiff_t const matrixNMSeed = m_matrixNMSeed;

    forall< POLICY >( 1,
                      [result, matrixNM_IJK, matrixNM_IKJ, matrixNM_KJI, matrixML_IJK, matrixML_IKJ, matrixML_KJI, matrixML_local, matrixLM_IJK, matrixLM_IKJ,
                       matrixLM_KJI,
                       matrixLM_local, matrixNMSeed ] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( matrixNM, matrixML, matrixLM ) \
            fill( matrixNM, matrixNMSeed ); \
            tensorOps::Rij_eq_AikBkj< N, M, L >( matrixNM, matrixML, matrixLM ); \
            CHECK_EQUALITY_2D( N, M, matrixNM, result )

          #define _TEST_PERMS( matrixNM, matrixML, matrixLM0, matrixLM1, matrixLM2, matrixLM3 ) \
            _TEST( matrixNM, matrixML, matrixLM0 ); \
            _TEST( matrixNM, matrixML, matrixLM1 ); \
            _TEST( matrixNM, matrixML, matrixLM2 ); \
            _TEST( matrixNM, matrixML, matrixLM3 )

          T matrixNM_local[ N ][ M ];

          _TEST_PERMS( matrixNM_IJK[ 0 ], matrixML_IJK[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_IJK[ 0 ], matrixML_KJI[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_IJK[ 0 ], matrixML_local, matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_IKJ[ 0 ], matrixML_IJK[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_IKJ[ 0 ], matrixML_IKJ[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_IKJ[ 0 ], matrixML_local, matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_KJI[ 0 ], matrixML_IJK[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_KJI[ 0 ], matrixML_IKJ[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_KJI[ 0 ], matrixML_KJI[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_KJI[ 0 ], matrixML_local, matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_local, matrixML_IJK[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_local, matrixML_IKJ[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_local, matrixML_KJI[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_local, matrixML_local, matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testPlusAikBkj()
  {
    T result[ N ][ M ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    {
      for( std::ptrdiff_t j = 0; j < M; ++j )
      {
        T dot = 0;
        for( std::ptrdiff_t k = 0; k < L; ++k )
        {
          dot += m_matrixNL_local[ i ][ k ] * m_matrixLM_local[ k ][ j ];
        }
        result[ i ][ j ] = m_matrixNM_local[ i ][ j ] + dot;
      }
    }

    ArrayViewT< T, 3, 2 > const matrixNM_IJK = m_matrixNM_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixNM_IKJ = m_matrix_NM_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixNM_KJI = m_matrix_NM_KJI.toView();

    ArrayViewT< T const, 3, 2 > const matrixML_IJK = m_matrixNL_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixML_IKJ = m_matrixNL_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixML_KJI = m_matrixNL_KJI.toViewConst();
    T const ( &matrixML_local )[ N ][ L ] = m_matrixNL_local;

    ArrayViewT< T const, 3, 2 > const matrixLM_IJK = m_matrixLM_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixLM_IKJ = m_matrixLM_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixLM_KJI = m_matrixLM_KJI.toViewConst();
    T const ( &matrixLM_local )[ L ][ M ] = m_matrixLM_local;

    std::ptrdiff_t const matrixNMSeed = m_matrixNMSeed;

    forall< POLICY >( 1,
                      [result, matrixNM_IJK, matrixNM_IKJ, matrixNM_KJI, matrixML_IJK, matrixML_IKJ, matrixML_KJI, matrixML_local, matrixLM_IJK, matrixLM_IKJ,
                       matrixLM_KJI,
                       matrixLM_local, matrixNMSeed ] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( matrixNM, matrixML, matrixLM ) \
            fill( matrixNM, matrixNMSeed ); \
            tensorOps::Rij_add_AikBkj< N, M, L >( matrixNM, matrixML, matrixLM ); \
            CHECK_EQUALITY_2D( N, M, matrixNM, result )

          #define _TEST_PERMS( matrixNM, matrixML, matrixLM0, matrixLM1, matrixLM2, matrixLM3 ) \
            _TEST( matrixNM, matrixML, matrixLM0 ); \
            _TEST( matrixNM, matrixML, matrixLM1 ); \
            _TEST( matrixNM, matrixML, matrixLM2 ); \
            _TEST( matrixNM, matrixML, matrixLM3 )

          T matrixNM_local[ N ][ M ];

          _TEST_PERMS( matrixNM_IJK[ 0 ], matrixML_IJK[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_IJK[ 0 ], matrixML_KJI[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_IJK[ 0 ], matrixML_local, matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_IKJ[ 0 ], matrixML_IJK[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_IKJ[ 0 ], matrixML_IKJ[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_IKJ[ 0 ], matrixML_local, matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_KJI[ 0 ], matrixML_IJK[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_KJI[ 0 ], matrixML_IKJ[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_KJI[ 0 ], matrixML_KJI[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_KJI[ 0 ], matrixML_local, matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_local, matrixML_IJK[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_local, matrixML_IKJ[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_local, matrixML_KJI[ 0 ], matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );
          _TEST_PERMS( matrixNM_local, matrixML_local, matrixLM_IJK[ 0 ], matrixLM_IKJ[ 0 ], matrixLM_KJI[ 0 ], matrixLM_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testAikBjk()
  {
    T result[ N ][ M ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    {
      for( std::ptrdiff_t j = 0; j < M; ++j )
      {
        T dot = 0;
        for( std::ptrdiff_t k = 0; k < L; ++k )
        {
          dot += m_matrixNL_local[ i ][ k ] * m_matrixML_local[ j ][ k ];
        }
        result[ i ][ j ] = dot;
      }
    }

    ArrayViewT< T, 3, 2 > const matrixNM_IJK = m_matrixNM_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixNM_IKJ = m_matrix_NM_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixNM_KJI = m_matrix_NM_KJI.toView();

    ArrayViewT< T const, 3, 2 > const matrixNL_IJK = m_matrixNL_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixNL_IKJ = m_matrixNL_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixNL_KJI = m_matrixNL_KJI.toViewConst();
    T const ( &matrixNL_local )[ N ][ L ] = m_matrixNL_local;

    ArrayViewT< T const, 3, 2 > const matrixML_IJK = m_matrixML_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixML_IKJ = m_matrixML_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixML_KJI = m_matrixML_KJI.toViewConst();
    T const ( &matrixML_local )[ M ][ L ] = m_matrixML_local;

    std::ptrdiff_t const matrixNMSeed = m_matrixNMSeed;

    forall< POLICY >( 1,
                      [result, matrixNM_IJK, matrixNM_IKJ, matrixNM_KJI, matrixNL_IJK, matrixNL_IKJ, matrixNL_KJI, matrixNL_local, matrixML_IJK, matrixML_IKJ,
                       matrixML_KJI,
                       matrixML_local, matrixNMSeed ] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( matrixNM, matrixNL, matrixML ) \
            fill( matrixNM, matrixNMSeed ); \
            tensorOps::Rij_eq_AikBjk< N, M, L >( matrixNM, matrixNL, matrixML ); \
            CHECK_EQUALITY_2D( N, M, matrixNM, result )

          #define _TEST_PERMS( matrixNM, matrixNL, matrixML0, matrixML1, matrixML2, matrixML3 ) \
            _TEST( matrixNM, matrixNL, matrixML0 ); \
            _TEST( matrixNM, matrixNL, matrixML1 ); \
            _TEST( matrixNM, matrixNL, matrixML2 ); \
            _TEST( matrixNM, matrixNL, matrixML3 )

          T matrixNM_local[ N ][ M ];

          _TEST_PERMS( matrixNM_IJK[ 0 ], matrixNL_IJK[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_IJK[ 0 ], matrixNL_KJI[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_IJK[ 0 ], matrixNL_local, matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_IKJ[ 0 ], matrixNL_IJK[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_IKJ[ 0 ], matrixNL_IKJ[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_IKJ[ 0 ], matrixNL_local, matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_KJI[ 0 ], matrixNL_IJK[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_KJI[ 0 ], matrixNL_IKJ[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_KJI[ 0 ], matrixNL_KJI[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_KJI[ 0 ], matrixNL_local, matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_local, matrixNL_IJK[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_local, matrixNL_IKJ[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_local, matrixNL_KJI[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_local, matrixNL_local, matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testPlusAikBjk()
  {
    T result[ N ][ M ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    {
      for( std::ptrdiff_t j = 0; j < M; ++j )
      {
        T dot = 0;
        for( std::ptrdiff_t k = 0; k < L; ++k )
        {
          dot += m_matrixNL_local[ i ][ k ] * m_matrixML_local[ j ][ k ];
        }
        result[ i ][ j ] = m_matrixNM_local[ i ][ j ] + dot;
      }
    }

    ArrayViewT< T, 3, 2 > const matrixNM_IJK = m_matrixNM_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixNM_IKJ = m_matrix_NM_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixNM_KJI = m_matrix_NM_KJI.toView();

    ArrayViewT< T const, 3, 2 > const matrixNL_IJK = m_matrixNL_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixNL_IKJ = m_matrixNL_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixNL_KJI = m_matrixNL_KJI.toViewConst();
    T const ( &matrixNL_local )[ N ][ L ] = m_matrixNL_local;

    ArrayViewT< T const, 3, 2 > const matrixML_IJK = m_matrixML_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixML_IKJ = m_matrixML_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixML_KJI = m_matrixML_KJI.toViewConst();
    T const ( &matrixML_local )[ M ][ L ] = m_matrixML_local;

    std::ptrdiff_t const matrixNMSeed = m_matrixNMSeed;

    forall< POLICY >( 1,
                      [result, matrixNM_IJK, matrixNM_IKJ, matrixNM_KJI, matrixNL_IJK, matrixNL_IKJ, matrixNL_KJI, matrixNL_local, matrixML_IJK, matrixML_IKJ,
                       matrixML_KJI,
                       matrixML_local, matrixNMSeed ] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( matrixNM, matrixNL, matrixML ) \
            fill( matrixNM, matrixNMSeed ); \
            tensorOps::Rij_add_AikBjk< N, M, L >( matrixNM, matrixNL, matrixML ); \
            CHECK_EQUALITY_2D( N, M, matrixNM, result )

          #define _TEST_PERMS( matrixNM, matrixNL, matrixML0, matrixML1, matrixML2, matrixML3 ) \
            _TEST( matrixNM, matrixNL, matrixML0 ); \
            _TEST( matrixNM, matrixNL, matrixML1 ); \
            _TEST( matrixNM, matrixNL, matrixML2 ); \
            _TEST( matrixNM, matrixNL, matrixML3 )

          T matrixNM_local[ N ][ M ];

          _TEST_PERMS( matrixNM_IJK[ 0 ], matrixNL_IJK[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_IJK[ 0 ], matrixNL_KJI[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_IJK[ 0 ], matrixNL_local, matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_IKJ[ 0 ], matrixNL_IJK[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_IKJ[ 0 ], matrixNL_IKJ[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_IKJ[ 0 ], matrixNL_local, matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_KJI[ 0 ], matrixNL_IJK[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_KJI[ 0 ], matrixNL_IKJ[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_KJI[ 0 ], matrixNL_KJI[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_KJI[ 0 ], matrixNL_local, matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_local, matrixNL_IJK[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_local, matrixNL_IKJ[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_local, matrixNL_KJI[ 0 ], matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );
          _TEST_PERMS( matrixNM_local, matrixNL_local, matrixML_IJK[ 0 ], matrixML_IKJ[ 0 ], matrixML_KJI[ 0 ], matrixML_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testAkiBkj()
  {
    T result[ M ][ L ];
    for( std::ptrdiff_t i = 0; i < M; ++i )
    {
      for( std::ptrdiff_t j = 0; j < L; ++j )
      {
        T dot = 0;
        for( std::ptrdiff_t k = 0; k < N; ++k )
        {
          dot += m_matrixNM_local[ k ][ i ] * m_matrixNL_local[ k ][ j ];
        }
        result[ i ][ j ] = dot;
      }
    }

    ArrayViewT< T, 3, 2 > const matrixML_IJK = m_matrixML_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixML_IKJ = m_matrixML_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixML_KJI = m_matrixML_KJI.toView();

    ArrayViewT< T const, 3, 2 > const matrixNM_IJK = m_matrixNM_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixNM_IKJ = m_matrix_NM_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixNM_KJI = m_matrix_NM_KJI.toViewConst();
    T const ( &matrixNM_local )[ N ][ M ] = m_matrixNM_local;

    ArrayViewT< T const, 3, 2 > const matrixNL_IJK = m_matrixNL_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixNL_IKJ = m_matrixNL_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixNL_KJI = m_matrixNL_KJI.toViewConst();
    T const ( &matrixNL_local )[ N ][ L ] = m_matrixNL_local;

    std::ptrdiff_t const matrixMLSeed = m_matrixMLSeed;

    forall< POLICY >( 1,
                      [result, matrixML_IJK, matrixML_IKJ, matrixML_KJI, matrixNM_IJK, matrixNM_IKJ, matrixNM_KJI, matrixNM_local,
                       matrixNL_IJK, matrixNL_IKJ, matrixNL_KJI, matrixNL_local, matrixMLSeed] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( matrixML, matrixNM, matrixNL ) \
            fill( matrixML, matrixMLSeed ); \
            tensorOps::Rij_eq_AkiBkj< M, L, N >( matrixML, matrixNM, matrixNL ); \
            CHECK_EQUALITY_2D( M, L, matrixML, result )

          #define _TEST_PERMS( matrixML, matrixNM, matrixNL0, matrixNL1, matrixNL2, matrixNL3 ) \
            _TEST( matrixML, matrixNM, matrixNL0 ); \
            _TEST( matrixML, matrixNM, matrixNL1 ); \
            _TEST( matrixML, matrixNM, matrixNL2 ); \
            _TEST( matrixML, matrixNM, matrixNL3 )

          T matrixML_local[ M ][ L ];

          _TEST_PERMS( matrixML_IJK[ 0 ], matrixNM_IJK[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_IJK[ 0 ], matrixNM_IKJ[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_IJK[ 0 ], matrixNM_KJI[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_IJK[ 0 ], matrixNM_local, matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_IKJ[ 0 ], matrixNM_IJK[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_IKJ[ 0 ], matrixNM_IKJ[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_IKJ[ 0 ], matrixNM_KJI[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_IKJ[ 0 ], matrixNM_local, matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_KJI[ 0 ], matrixNM_IJK[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_KJI[ 0 ], matrixNM_IKJ[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_KJI[ 0 ], matrixNM_KJI[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_KJI[ 0 ], matrixNM_local, matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_local, matrixNM_IJK[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_local, matrixNM_IKJ[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_local, matrixNM_KJI[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_local, matrixNM_local, matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );

      #undef _TEST_PERMS
      #undef _TEST
        } );
  }

  void testPlusAkiBkj()
  {
    T result[ M ][ L ];
    for( std::ptrdiff_t i = 0; i < M; ++i )
    {
      for( std::ptrdiff_t j = 0; j < L; ++j )
      {
        T dot = 0;
        for( std::ptrdiff_t k = 0; k < N; ++k )
        {
          dot += m_matrixNM_local[ k ][ i ] * m_matrixNL_local[ k ][ j ];
        }
        result[ i ][ j ] = m_matrixML_local[ i ][ j ] + dot;
      }
    }

    ArrayViewT< T, 3, 2 > const matrixML_IJK = m_matrixML_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixML_IKJ = m_matrixML_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixML_KJI = m_matrixML_KJI.toView();

    ArrayViewT< T const, 3, 2 > const matrixNM_IJK = m_matrixNM_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixNM_IKJ = m_matrix_NM_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixNM_KJI = m_matrix_NM_KJI.toViewConst();
    T const ( &matrixNM_local )[ N ][ M ] = m_matrixNM_local;

    ArrayViewT< T const, 3, 2 > const matrixNL_IJK = m_matrixNL_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const matrixNL_IKJ = m_matrixNL_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const matrixNL_KJI = m_matrixNL_KJI.toViewConst();
    T const ( &matrixNL_local )[ N ][ L ] = m_matrixNL_local;

    std::ptrdiff_t const matrixMLSeed = m_matrixMLSeed;

    forall< POLICY >( 1,
                      [result, matrixML_IJK, matrixML_IKJ, matrixML_KJI, matrixNM_IJK, matrixNM_IKJ, matrixNM_KJI, matrixNM_local,
                       matrixNL_IJK, matrixNL_IKJ, matrixNL_KJI, matrixNL_local, matrixMLSeed] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( matrixML, matrixNM, matrixNL ) \
            fill( matrixML, matrixMLSeed ); \
            tensorOps::Rij_add_AkiBkj< M, L, N >( matrixML, matrixNM, matrixNL ); \
            CHECK_EQUALITY_2D( M, L, matrixML, result )

          #define _TEST_PERMS( matrixML, matrixNM, matrixNL0, matrixNL1, matrixNL2, matrixNL3 ) \
            _TEST( matrixML, matrixNM, matrixNL0 ); \
            _TEST( matrixML, matrixNM, matrixNL1 ); \
            _TEST( matrixML, matrixNM, matrixNL2 ); \
            _TEST( matrixML, matrixNM, matrixNL3 )

          T matrixML_local[ M ][ L ];

          _TEST_PERMS( matrixML_IJK[ 0 ], matrixNM_IJK[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_IJK[ 0 ], matrixNM_IKJ[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_IJK[ 0 ], matrixNM_KJI[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_IJK[ 0 ], matrixNM_local, matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_IKJ[ 0 ], matrixNM_IJK[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_IKJ[ 0 ], matrixNM_IKJ[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_IKJ[ 0 ], matrixNM_KJI[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_IKJ[ 0 ], matrixNM_local, matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_KJI[ 0 ], matrixNM_IJK[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_KJI[ 0 ], matrixNM_IKJ[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_KJI[ 0 ], matrixNM_KJI[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_KJI[ 0 ], matrixNM_local, matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_local, matrixNM_IJK[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_local, matrixNM_IKJ[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_local, matrixNM_KJI[ 0 ], matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );
          _TEST_PERMS( matrixML_local, matrixNM_local, matrixNL_IJK[ 0 ], matrixNL_IKJ[ 0 ], matrixNL_KJI[ 0 ], matrixNL_local );

      #undef _TEST_PERMS
      #undef _TEST
        } );
  }


private:
  std::ptrdiff_t const m_matrixNMSeed = 0;
  ArrayT< T, RAJA::PERM_IJK > m_matrixNM_IJK { 1, N, M };
  ArrayT< T, RAJA::PERM_IKJ > m_matrix_NM_IKJ { 1, N, M };
  ArrayT< T, RAJA::PERM_KJI > m_matrix_NM_KJI { 1, N, M };
  T m_matrixNM_local[ N ][ M ];

  std::ptrdiff_t const m_matrixNLSeed = m_matrixNMSeed + N * M;
  ArrayT< T, RAJA::PERM_IJK > m_matrixNL_IJK { 1, N, L };
  ArrayT< T, RAJA::PERM_IKJ > m_matrixNL_IKJ { 1, N, L };
  ArrayT< T, RAJA::PERM_KJI > m_matrixNL_KJI { 1, N, L };
  T m_matrixNL_local[ N ][ L ];

  std::ptrdiff_t const m_matrixLMSeed = m_matrixNMSeed + N * L;
  ArrayT< T, RAJA::PERM_IJK > m_matrixLM_IJK { 1, L, M };
  ArrayT< T, RAJA::PERM_IKJ > m_matrixLM_IKJ { 1, L, M };
  ArrayT< T, RAJA::PERM_KJI > m_matrixLM_KJI { 1, L, M };
  T m_matrixLM_local[ L ][ M ];

  std::ptrdiff_t const m_matrixMLSeed = m_matrixLMSeed + L * M;
  ArrayT< T, RAJA::PERM_IJK > m_matrixML_IJK { 1, M, L };
  ArrayT< T, RAJA::PERM_IKJ > m_matrixML_IKJ { 1, M, L };
  ArrayT< T, RAJA::PERM_KJI > m_matrixML_KJI { 1, M, L };
  T m_matrixML_local[ M ][ L ];
};


using ThreeSizesTestTypes = ::testing::Types<
  std::tuple< double,
              std::integral_constant< int, 2 >,
              std::integral_constant< int, 3 >,
              std::integral_constant< int, 4 >,
              serialPolicy >
  , std::tuple< int,
                std::integral_constant< int, 5 >,
                std::integral_constant< int, 4 >,
                std::integral_constant< int, 6 >,
                serialPolicy >
  , std::tuple< double,
                std::integral_constant< int, 3 >,
                std::integral_constant< int, 3 >,
                std::integral_constant< int, 3 >,
                serialPolicy >

#if ( defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP) ) && defined(LVARRAY_USE_CHAI)
  , std::tuple< double,
                std::integral_constant< int, 2 >,
                std::integral_constant< int, 3 >,
                std::integral_constant< int, 4 >,
                parallelDevicePolicy< 32 > >
  , std::tuple< int,
                std::integral_constant< int, 5 >,
                std::integral_constant< int, 4 >,
                std::integral_constant< int, 6 >,
                parallelDevicePolicy< 32 > >
  , std::tuple< double,
                std::integral_constant< int, 3 >,
                std::integral_constant< int, 3 >,
                std::integral_constant< int, 3 >,
                parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( ThreeSizesTest, ThreeSizesTestTypes, );

} // namespace testing
} // namespace LvArray
