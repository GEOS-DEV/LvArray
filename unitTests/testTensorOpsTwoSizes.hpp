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

#pragma once

// Source includes
#include "tensorOps.hpp"
#include "Array.hpp"
#include "testUtils.hpp"
#include "ArrayUtilities.hpp"
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

    ArrayView< T, 3, 2 > const & matrixA_IJK = m_matrixA_IJK.toView();
    ArrayView< T, 3, 1 > const & matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayView< T, 3, 0 > const & matrixA_KJI = m_matrixA_KJI.toView();

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
    ArrayView< T, 3, 2 > const & matrixA_IJK = m_matrixA_IJK.toView();
    ArrayView< T, 3, 1 > const & matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayView< T, 3, 0 > const & matrixA_KJI = m_matrixA_KJI.toView();

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

    ArrayView< T, 3, 2 > const & matrixA_IJK = m_matrixA_IJK.toView();
    ArrayView< T, 3, 1 > const & matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayView< T, 3, 0 > const & matrixA_KJI = m_matrixA_KJI.toView();

    ArrayView< T const, 2, 1 > const & vectorN_IJ = m_vectorN_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorN_JI = m_vectorN_JI.toViewConst();
    T const ( &vectorN_local )[ N ] = m_vectorN_local;

    ArrayView< T const, 2, 1 > const & vectorM_IJ = m_vectorM_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorM_JI = m_vectorM_JI.toViewConst();
    T const ( &vectorM_local )[ M ] = m_vectorM_local;

    std::ptrdiff_t const matrixSeed = m_matrixASeed;

    forall< POLICY >( 1, [result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, vectorN_IJ, vectorN_JI, vectorN_local,
                          vectorM_IJ, vectorM_JI, vectorM_local, matrixSeed] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( matrix, vectorN, vectorM ) \
            fill( matrix, matrixSeed ); \
            tensorOps::AiBj< N, M >( matrix, vectorN, vectorM ); \
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

    ArrayView< T, 3, 2 > const & matrixA_IJK = m_matrixA_IJK.toView();
    ArrayView< T, 3, 1 > const & matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayView< T, 3, 0 > const & matrixA_KJI = m_matrixA_KJI.toView();

    ArrayView< T const, 2, 1 > const & vectorN_IJ = m_vectorN_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorN_JI = m_vectorN_JI.toViewConst();
    T const ( &vectorN_local )[ N ] = m_vectorN_local;

    ArrayView< T const, 2, 1 > const & vectorM_IJ = m_vectorM_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorM_JI = m_vectorM_JI.toViewConst();
    T const ( &vectorM_local )[ M ] = m_vectorM_local;

    std::ptrdiff_t const matrixSeed = m_matrixASeed;

    forall< POLICY >( 1,
                      [result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, vectorN_IJ, vectorN_JI, vectorN_local, vectorM_IJ, vectorM_JI, vectorM_local, matrixSeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( matrix, vectorN, vectorM ) \
            fill( matrix, matrixSeed ); \
            tensorOps::plusAiBj< N, M >( matrix, vectorN, vectorM ); \
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

    ArrayView< T const, 3, 2 > const & matrixA_IJK = m_matrixA_IJK.toViewConst();
    ArrayView< T const, 3, 1 > const & matrixA_IKJ = m_matrixA_IKJ.toViewConst();
    ArrayView< T const, 3, 0 > const & matrixA_KJI = m_matrixA_KJI.toViewConst();
    T const ( &matrix_local )[ N ][ M ] = m_matrixA_local;

    ArrayView< T, 2, 1 > const & vectorN_IJ = m_vectorN_IJ.toView();
    ArrayView< T, 2, 0 > const & vectorN_JI = m_vectorN_JI.toView();

    ArrayView< T const, 2, 1 > const & vectorM_IJ = m_vectorM_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorM_JI = m_vectorM_JI.toViewConst();
    T const ( &vectorM_local )[ M ] = m_vectorM_local;

    std::ptrdiff_t const vectorNSeed = m_vectorNSeed;

    forall< POLICY >( 1,
                      [result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, matrix_local, vectorN_IJ, vectorN_JI, vectorM_IJ, vectorM_JI, vectorM_local, vectorNSeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( matrix, vectorN, vectorM ) \
            fill( vectorN, vectorNSeed ); \
            tensorOps::AijBj< N, M >( vectorN, matrix, vectorM ); \
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

    ArrayView< T const, 3, 2 > const & matrixA_IJK = m_matrixA_IJK.toViewConst();
    ArrayView< T const, 3, 1 > const & matrixA_IKJ = m_matrixA_IKJ.toViewConst();
    ArrayView< T const, 3, 0 > const & matrixA_KJI = m_matrixA_KJI.toViewConst();
    T const ( &matrix_local )[ N ][ M ] = m_matrixA_local;

    ArrayView< T, 2, 1 > const & vectorN_IJ = m_vectorN_IJ.toView();
    ArrayView< T, 2, 0 > const & vectorN_JI = m_vectorN_JI.toView();

    ArrayView< T const, 2, 1 > const & vectorM_IJ = m_vectorM_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorM_JI = m_vectorM_JI.toViewConst();
    T const ( &vectorM_local )[ M ] = m_vectorM_local;

    std::ptrdiff_t const vectorNSeed = m_vectorNSeed;

    forall< POLICY >( 1,
                      [result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, matrix_local, vectorN_IJ, vectorN_JI, vectorM_IJ, vectorM_JI, vectorM_local, vectorNSeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( matrix, vectorN, vectorM ) \
            fill( vectorN, vectorNSeed ); \
            tensorOps::plusAijBj< N, M >( vectorN, matrix, vectorM ); \
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

    ArrayView< T const, 3, 2 > const & matrixA_IJK = m_matrixA_IJK.toViewConst();
    ArrayView< T const, 3, 1 > const & matrixA_IKJ = m_matrixA_IKJ.toViewConst();
    ArrayView< T const, 3, 0 > const & matrixA_KJI = m_matrixA_KJI.toViewConst();
    T const ( &matrix_local )[ N ][ M ] = m_matrixA_local;

    ArrayView< T const, 2, 1 > const & vectorN_IJ = m_vectorN_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorN_JI = m_vectorN_JI.toViewConst();
    T const ( &vectorN_local )[ N ] = m_vectorN_local;

    ArrayView< T, 2, 1 > const & vectorM_IJ = m_vectorM_IJ.toView();
    ArrayView< T, 2, 0 > const & vectorM_JI = m_vectorM_JI.toView();

    std::ptrdiff_t const vectorMSeed = m_vectorMSeed;

    forall< POLICY >( 1,
                      [result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, matrix_local, vectorN_IJ, vectorN_JI, vectorN_local, vectorM_IJ, vectorM_JI, vectorMSeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( matrix, vectorN, vectorM ) \
            fill( vectorM, vectorMSeed ); \
            tensorOps::AjiBj< N, M >( vectorM, matrix, vectorN ); \
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

    ArrayView< T const, 3, 2 > const & matrixA_IJK = m_matrixA_IJK.toViewConst();
    ArrayView< T const, 3, 1 > const & matrixA_IKJ = m_matrixA_IKJ.toViewConst();
    ArrayView< T const, 3, 0 > const & matrixA_KJI = m_matrixA_KJI.toViewConst();
    T const ( &matrix_local )[ N ][ M ] = m_matrixA_local;

    ArrayView< T const, 2, 1 > const & vectorN_IJ = m_vectorN_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorN_JI = m_vectorN_JI.toViewConst();
    T const ( &vectorN_local )[ N ] = m_vectorN_local;

    ArrayView< T, 2, 1 > const & vectorM_IJ = m_vectorM_IJ.toView();
    ArrayView< T, 2, 0 > const & vectorM_JI = m_vectorM_JI.toView();

    std::ptrdiff_t const vectorMSeed = m_vectorMSeed;

    forall< POLICY >( 1,
                      [result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, matrix_local, vectorN_IJ, vectorN_JI, vectorN_local, vectorM_IJ, vectorM_JI, vectorMSeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( matrix, vectorN, vectorM ) \
            fill( vectorM, vectorMSeed ); \
            tensorOps::plusAjiBj< N, M >( vectorM, matrix, vectorN ); \
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
    ArrayView< T, 3, 2 > const & matrixA_IJK = m_matrixA_IJK.toView();
    ArrayView< T, 3, 1 > const & matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayView< T, 3, 0 > const & matrixA_KJI = m_matrixA_KJI.toView();

    ArrayView< T const, 3, 2 > const & matrixB_IJK_view = m_matrixB_IJK.toViewConst();
    ArrayView< T const, 3, 1 > const & matrixB_IKJ_view = m_matrixB_IKJ.toViewConst();
    ArrayView< T const, 3, 0 > const & matrixB_KJI_view = m_matrixB_KJI.toViewConst();
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

    ArrayView< T, 3, 2 > const & matrixA_IJK = m_matrixA_IJK.toView();
    ArrayView< T, 3, 1 > const & matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayView< T, 3, 0 > const & matrixA_KJI = m_matrixA_KJI.toView();

    ArrayView< T const, 3, 2 > const & matrixB_IJK_view = m_matrixB_IJK.toViewConst();
    ArrayView< T const, 3, 1 > const & matrixB_IKJ_view = m_matrixB_IKJ.toViewConst();
    ArrayView< T const, 3, 0 > const & matrixB_KJI_view = m_matrixB_KJI.toViewConst();
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

    ArrayView< T, 3, 2 > const & matrixA_IJK = m_matrixA_IJK.toView();
    ArrayView< T, 3, 1 > const & matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayView< T, 3, 0 > const & matrixA_KJI = m_matrixA_KJI.toView();

    ArrayView< T const, 3, 2 > const & matrixB_IJK_view = m_matrixB_IJK.toViewConst();
    ArrayView< T const, 3, 1 > const & matrixB_IKJ_view = m_matrixB_IKJ.toViewConst();
    ArrayView< T const, 3, 0 > const & matrixB_KJI_view = m_matrixB_KJI.toViewConst();
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

private:
  std::ptrdiff_t const m_matrixASeed = 0;
  Array< T, 3, RAJA::PERM_IJK > m_matrixA_IJK { 1, N, M };
  Array< T, 3, RAJA::PERM_IKJ > m_matrixA_IKJ { 1, N, M };
  Array< T, 3, RAJA::PERM_KJI > m_matrixA_KJI { 1, N, M };
  T m_matrixA_local[ N ][ M ];

  std::ptrdiff_t const m_matrixBSeed = m_matrixASeed + N * M;
  Array< T, 3, RAJA::PERM_IJK > m_matrixB_IJK { 1, N, M };
  Array< T, 3, RAJA::PERM_IKJ > m_matrixB_IKJ { 1, N, M };
  Array< T, 3, RAJA::PERM_KJI > m_matrixB_KJI { 1, N, M };
  T m_matrixB_local[ N ][ M ];

  std::ptrdiff_t const m_vectorNSeed = m_matrixBSeed + N * M;
  Array< T, 2, RAJA::PERM_IJ > m_vectorN_IJ { 1, N };
  Array< T, 2, RAJA::PERM_JI > m_vectorN_JI { 1, N };
  T m_vectorN_local[ N ];

  std::ptrdiff_t const m_vectorMSeed = m_vectorNSeed + N;
  Array< T, 2, RAJA::PERM_IJ > m_vectorM_IJ { 1, M };
  Array< T, 2, RAJA::PERM_JI > m_vectorM_JI { 1, M };
  T m_vectorM_local[ M ];
};


using TwoSizesTestTypes = ::testing::Types<
  std::tuple< double, std::integral_constant< int, 2 >, std::integral_constant< int, 3 >, serialPolicy >
  , std::tuple< int, std::integral_constant< int, 5 >, std::integral_constant< int, 4 >, serialPolicy >
  , std::tuple< double, std::integral_constant< int, 3 >, std::integral_constant< int, 3 >, serialPolicy >
#if defined(USE_CUDA)
  , std::tuple< double, std::integral_constant< int, 2 >, std::integral_constant< int, 3 >, parallelDevicePolicy< 32 > >
  , std::tuple< int, std::integral_constant< int, 5 >, std::integral_constant< int, 4 >, parallelDevicePolicy< 32 > >
  , std::tuple< double, std::integral_constant< int, 3 >, std::integral_constant< int, 3 >, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( TwoSizesTest, TwoSizesTestTypes, );

} // namespace testing
} // namespace LvArray
