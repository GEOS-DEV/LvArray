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

// Source includes
#include "tensorOps.hpp"
#include "Array.hpp"
#include "testUtils.hpp"
#include "streamIO.hpp"
#include "testTensorOpsCommon.hpp"

// TPL includes
#include <gtest/gtest.h>


namespace LvArray
{
namespace testing
{

// These could be put into tensorOps if it was of use.
template< typename DST_MATRIX, typename SRC_SYM_MATRIX >
void symmetricToDense( std::integral_constant< int, 2 >, DST_MATRIX && dstMatrix, SRC_SYM_MATRIX const & srcSymMatrix )
{
  tensorOps::internal::checkSizes< 2, 2 >( dstMatrix );
  tensorOps::internal::checkSizes< 3 >( srcSymMatrix );

  dstMatrix[ 0 ][ 0 ] = srcSymMatrix[ 0 ];
  dstMatrix[ 1 ][ 1 ] = srcSymMatrix[ 1 ];

  dstMatrix[ 0 ][ 1 ] = srcSymMatrix[ 2 ];

  dstMatrix[ 1 ][ 0 ] = srcSymMatrix[ 2 ];
}

template< typename DST_MATRIX, typename SRC_SYM_MATRIX >
void symmetricToDense( std::integral_constant< int, 3 >, DST_MATRIX && dstMatrix, SRC_SYM_MATRIX const & srcSymMatrix )
{
  tensorOps::internal::checkSizes< 3, 3 >( dstMatrix );
  tensorOps::internal::checkSizes< 6 >( srcSymMatrix );

  dstMatrix[ 0 ][ 0 ] = srcSymMatrix[ 0 ];
  dstMatrix[ 1 ][ 1 ] = srcSymMatrix[ 1 ];
  dstMatrix[ 2 ][ 2 ] = srcSymMatrix[ 2 ];

  dstMatrix[ 0 ][ 1 ] = srcSymMatrix[ 5 ];
  dstMatrix[ 0 ][ 2 ] = srcSymMatrix[ 4 ];
  dstMatrix[ 1 ][ 2 ] = srcSymMatrix[ 3 ];

  dstMatrix[ 1 ][ 0 ] = srcSymMatrix[ 5 ];
  dstMatrix[ 2 ][ 0 ] = srcSymMatrix[ 4 ];
  dstMatrix[ 2 ][ 1 ] = srcSymMatrix[ 3 ];
}

template< typename DST_SYM_MATRIX, typename SRC_MATRIX >
void denseToSymmetric( std::integral_constant< int, 2 >, DST_SYM_MATRIX && dstSymMatrix, SRC_MATRIX const & srcMatrix )
{
  tensorOps::internal::checkSizes< 3 >( dstSymMatrix );
  tensorOps::internal::checkSizes< 2, 2 >( srcMatrix );

  dstSymMatrix[ 0 ] = srcMatrix[ 0 ][ 0 ];
  dstSymMatrix[ 1 ] = srcMatrix[ 1 ][ 1 ];
  dstSymMatrix[ 2 ] = srcMatrix[ 0 ][ 1 ];
  EXPECT_EQ( srcMatrix[ 0 ][ 1 ], srcMatrix[ 1 ][ 0 ] );
}

template< typename DST_SYM_MATRIX, typename SRC_MATRIX >
void denseToSymmetric( std::integral_constant< int, 3 >, DST_SYM_MATRIX && dstSymMatrix, SRC_MATRIX const & srcMatrix )
{
  tensorOps::internal::checkSizes< 6 >( dstSymMatrix );
  tensorOps::internal::checkSizes< 3, 3 >( srcMatrix );

  dstSymMatrix[ 0 ] = srcMatrix[ 0 ][ 0 ];
  dstSymMatrix[ 1 ] = srcMatrix[ 1 ][ 1 ];
  dstSymMatrix[ 2 ] = srcMatrix[ 2 ][ 2 ];
  dstSymMatrix[ 5 ] = srcMatrix[ 0 ][ 1 ];
  dstSymMatrix[ 4 ] = srcMatrix[ 0 ][ 2 ];
  dstSymMatrix[ 3 ] = srcMatrix[ 1 ][ 2 ];

  EXPECT_EQ( srcMatrix[ 0 ][ 1 ], srcMatrix[ 1 ][ 0 ] );
  EXPECT_EQ( srcMatrix[ 0 ][ 2 ], srcMatrix[ 2 ][ 0 ] );
  EXPECT_EQ( srcMatrix[ 1 ][ 2 ], srcMatrix[ 2 ][ 1 ] );
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

  void testDeterminant()
  {
    ArrayViewT< T, 3, 2 > const & matrixA_IJK = m_matrixA_IJK.toView();
    ArrayViewT< T, 3, 1 > const & matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayViewT< T, 3, 0 > const & matrixA_KJI = m_matrixA_KJI.toView();

    forall< POLICY >( 1, [matrixA_IJK, matrixA_IKJ, matrixA_KJI] LVARRAY_HOST_DEVICE ( int )
        {
          T result[ N ][ N ];
          T a_local[ N ][ N ];
          T det;
          for( int i = 0; i < numInverseTests; ++i )
          {
            #define _TEST( matrix ) \
              det = initInverse( i, matrix, result ); \
              PORTABLE_EXPECT_EQ( det, tensorOps::determinant< N >( matrix ) ); \

            _TEST( matrixA_IJK[ 0 ] );
            _TEST( matrixA_IKJ[ 0 ] );
            _TEST( matrixA_KJI[ 0 ] );
            _TEST( a_local );

        #undef _TEST
          }
        } );
  }

  void testInverseTwoArgs()
  {
    ArrayViewT< T, 3, 2 > const & matrixA_IJK = m_matrixA_IJK.toView();
    ArrayViewT< T, 3, 1 > const & matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayViewT< T, 3, 0 > const & matrixA_KJI = m_matrixA_KJI.toView();

    ArrayViewT< T, 3, 2 > const & matrixB_IJK = m_matrixB_IJK.toView();
    ArrayViewT< T, 3, 1 > const & matrixB_IKJ = m_matrixB_IKJ.toView();
    ArrayViewT< T, 3, 0 > const & matrixB_KJI = m_matrixB_KJI.toView();

    std::ptrdiff_t const aSeed = m_seedMatrixA;
    forall< POLICY >( 1, [matrixA_IJK, matrixA_IKJ, matrixA_KJI, matrixB_IJK, matrixB_IKJ, matrixB_KJI, aSeed] LVARRAY_HOST_DEVICE ( int )
        {
          T result[ N ][ N ];
          T a_local[ N ][ N ];
          T b_local[ N ][ N ];
          T det;
          for( int i = 0; i < numInverseTests; ++i )
          {
            #define _TEST( input, output ) \
              det = initInverse( i, input, result ); \
              PORTABLE_EXPECT_EQ( det, tensorOps::invert< N >( output, input ) ); \
              CHECK_EQUALITY_2D( N, N, output, result ); \
              fill( output, aSeed )

            #define _TEST_PERMS( input, output0, output1, output2, output3 ) \
              _TEST( input, output0 ); \
              _TEST( input, output1 ); \
              _TEST( input, output2 ); \
              _TEST( input, output3 )

            _TEST_PERMS( matrixA_IJK[ 0 ], matrixB_IJK[ 0 ], matrixB_IKJ[ 0 ], matrixB_KJI[ 0 ], b_local );
            _TEST_PERMS( matrixA_IKJ[ 0 ], matrixB_IJK[ 0 ], matrixB_IKJ[ 0 ], matrixB_KJI[ 0 ], b_local );
            _TEST_PERMS( matrixA_KJI[ 0 ], matrixB_IJK[ 0 ], matrixB_IKJ[ 0 ], matrixB_KJI[ 0 ], b_local );
            _TEST_PERMS( a_local, matrixB_IJK[ 0 ], matrixB_IKJ[ 0 ], matrixB_KJI[ 0 ], b_local );

            #undef _TEST_PERMS
            #undef _TEST
          }
        } );
  }

  void testInverseOneArg()
  {
    ArrayViewT< T, 3, 2 > const & matrixA_IJK = m_matrixA_IJK.toView();
    ArrayViewT< T, 3, 1 > const & matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayViewT< T, 3, 0 > const & matrixA_KJI = m_matrixA_KJI.toView();

    forall< POLICY >( 1, [matrixA_IJK, matrixA_IKJ, matrixA_KJI] LVARRAY_HOST_DEVICE ( int )
        {
          T result[ N ][ N ];
          T a_local[ N ][ N ];
          T det;
          for( int i = 0; i < numInverseTests; ++i )
          {
            #define _TEST( matrix ) \
              det = initInverse( i, matrix, result ); \
              PORTABLE_EXPECT_EQ( det, tensorOps::invert< N >( matrix ) ); \
              CHECK_EQUALITY_2D( N, N, matrix, result ); \

            _TEST( matrixA_IJK[ 0 ] );
            _TEST( matrixA_IKJ[ 0 ] );
            _TEST( matrixA_KJI[ 0 ] );
            _TEST( a_local );

            #undef _TEST
          }
        } );
  }

  void testSymAijBj()
  {
    T result[ N ];
    {
      T denseSymA[ N ][ N ];
      symmetricToDense( std::integral_constant< int, N > {}, denseSymA, m_symMatrixA_local );
      tensorOps::AijBj< N, N >( result, denseSymA, m_vectorB_local );
    }

    ArrayViewT< T, 2, 1 > const & vectorA_IJ = m_vectorA_IJ.toView();
    ArrayViewT< T, 2, 0 > const & vectorA_JI = m_vectorA_JI.toView();

    ArrayViewT< T const, 2, 1 > symMatrixA_IJ = m_symMatrixA_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > symMatrixA_JI = m_symMatrixA_JI.toViewConst();
    T const ( &symMatrixA_local )[ SYM_SIZE ] = m_symMatrixA_local;

    ArrayViewT< T const, 2, 1 > const & vectorB_IJ = m_vectorB_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const & vectorB_JI = m_vectorB_JI.toViewConst();
    T const ( &vectorB_local )[ N ] = m_vectorB_local;

    std::ptrdiff_t const vectorASeed = m_seedVectorA;

    forall< POLICY >( 1, [result, vectorA_IJ, vectorA_JI, symMatrixA_IJ, symMatrixA_JI, symMatrixA_local, vectorB_IJ, vectorB_JI, vectorB_local, vectorASeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( vectorA, symMatrix, vectorB ) \
            fill( vectorA, vectorASeed ); \
            tensorOps::symAijBj< N >( vectorA, symMatrix, vectorB ); \
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

  void plusSymAijBj()
  {
    T result[ N ];
    tensorOps::copy< N >( result, m_vectorA_local );
    {
      T denseSymA[ N ][ N ];
      symmetricToDense( std::integral_constant< int, N > {}, denseSymA, m_symMatrixA_local );
      tensorOps::plusAijBj< N, N >( result, denseSymA, m_vectorB_local );
    }

    ArrayViewT< T, 2, 1 > const & vectorA_IJ = m_vectorA_IJ.toView();
    ArrayViewT< T, 2, 0 > const & vectorA_JI = m_vectorA_JI.toView();

    ArrayViewT< T const, 2, 1 > symMatrixA_IJ = m_symMatrixA_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > symMatrixA_JI = m_symMatrixA_JI.toViewConst();
    T const ( &symMatrixA_local )[ SYM_SIZE ] = m_symMatrixA_local;

    ArrayViewT< T const, 2, 1 > const & vectorB_IJ = m_vectorB_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const & vectorB_JI = m_vectorB_JI.toViewConst();
    T const ( &vectorB_local )[ N ] = m_vectorB_local;

    std::ptrdiff_t const vectorASeed = m_seedVectorA;

    forall< POLICY >( 1, [result, vectorA_IJ, vectorA_JI, symMatrixA_IJ, symMatrixA_JI, symMatrixA_local, vectorB_IJ, vectorB_JI, vectorB_local, vectorASeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( vectorA, symMatrix, vectorB ) \
            fill( vectorA, vectorASeed ); \
            tensorOps::plusSymAijBj< N >( vectorA, symMatrix, vectorB ); \
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

  void testSymAikBjk()
  {
    T result[ N ][ N ];
    {
      T denseSymA[ N ][ N ];
      symmetricToDense( std::integral_constant< int, N > {}, denseSymA, m_symMatrixA_local );
      tensorOps::AikBjk< N, N, N >( result, denseSymA, m_matrixB_local );
    }

    ArrayViewT< T, 3, 2 > const & matrixA_IJK = m_matrixA_IJK.toView();
    ArrayViewT< T, 3, 1 > const & matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayViewT< T, 3, 0 > const & matrixA_KJI = m_matrixA_KJI.toView();

    ArrayViewT< T const, 2, 1 > symMatrixA_IJ = m_symMatrixA_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > symMatrixA_JI = m_symMatrixA_JI.toViewConst();
    T const ( &symMatrixA_local )[ SYM_SIZE ] = m_symMatrixA_local;

    ArrayViewT< T const, 3, 2 > const & matrixB_IJK = m_matrixB_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const & matrixB_IKJ = m_matrixB_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const & matrixB_KJI = m_matrixB_KJI.toViewConst();
    T const ( &matrixB_local )[ N ][ N ] = m_matrixB_local;

    std::ptrdiff_t const matrixASeed = m_seedMatrixA;

    forall< POLICY >( 1,
                      [result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, symMatrixA_local, symMatrixA_IJ, symMatrixA_JI, matrixB_IJK, matrixB_IKJ, matrixB_KJI,
                       matrixB_local,
                       matrixASeed] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( matrixA, symMatrix, matrixB ) \
            fill( matrixA, matrixASeed ); \
            tensorOps::symAikBjk< N >( matrixA, symMatrix, matrixB ); \
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

  void testAikSymBklAjl()
  {
    T result[ SYM_SIZE ];
    {
      T denseSymB[ N ][ N ];
      symmetricToDense( std::integral_constant< int, N > {}, denseSymB, m_symMatrixB_local );

      T temp[ N ][ N ];
      tensorOps::AikBjk< N, N, N >( temp, denseSymB, m_matrixA_local );

      T denseResult[ N ][ N ];
      tensorOps::AikBkj< N, N, N >( denseResult, m_matrixA_local, temp );
      denseToSymmetric( std::integral_constant< int, N > {}, result, denseResult );
    }

    ArrayViewT< T, 2, 1 > symMatrixA_IJ = m_symMatrixA_IJ.toView();
    ArrayViewT< T, 2, 0 > symMatrixA_JI = m_symMatrixA_JI.toView();

    ArrayViewT< T const, 3, 2 > const & matrixA_IJK = m_matrixA_IJK.toViewConst();
    ArrayViewT< T const, 3, 1 > const & matrixA_IKJ = m_matrixA_IKJ.toViewConst();
    ArrayViewT< T const, 3, 0 > const & matrixA_KJI = m_matrixA_KJI.toViewConst();
    T const ( &matrixA_local )[ N ][ N ] = m_matrixA_local;

    ArrayViewT< T const, 2, 1 > symMatrixB_IJ = m_symMatrixB_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > symMatrixB_JI = m_symMatrixB_JI.toViewConst();
    T const ( &symMatrixB_local )[ SYM_SIZE ] = m_symMatrixB_local;

    std::ptrdiff_t const matrixASeed = m_seedMatrixA;

    forall< POLICY >( 1,
                      [result, symMatrixA_IJ, symMatrixA_JI, matrixA_IJK, matrixA_IKJ, matrixA_KJI, matrixA_local, symMatrixB_IJ, symMatrixB_JI,
                       symMatrixB_local, matrixASeed] LVARRAY_HOST_DEVICE (
                        int )
        {
          #define _TEST( symMatrixA, matrixA, symMatrixB ) \
            fill( symMatrixA, matrixASeed ); \
            tensorOps::AikSymBklAjl< N >( symMatrixA, matrixA, symMatrixB ); \
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

private:

  static constexpr int numInverseTests = 2;

  template< typename A >
  LVARRAY_HOST_DEVICE
  static T initInverse( int i, A && src, T ( & result )[ 2 ][ 2 ] )
  {
    if( i == 0 )
    {
      T const matrix[ 2 ][ 2 ] = { { 1, 0 },
        { 0, 1 } };

      tensorOps::copy< N, N >( src, matrix );
      tensorOps::copy< N, N >( result, matrix );
      return 1;
    }
    if( i == 1 )
    {
      double const matrix[ 2 ][ 2 ] = { { 1, 2 },
        { 1, -2 } };

      double const inverse[ 2 ][ 2 ] = { { 2, 2 },
        { 1, -1 } };

      tensorOps::copy< N, N >( src, matrix );
      tensorOps::copy< N, N >( result, inverse );
      tensorOps::scale< N, N >( result, 1.0 / 4.0 );
      return -4;
    }

    LVARRAY_ERROR( "No test for " << i );
    return 0;
  }

  template< typename A >
  LVARRAY_HOST_DEVICE
  static T initInverse( int i, A && src, T ( & result )[ 3 ][ 3 ] )
  {
    if( i == 0 )
    {
      T const matrix[ 3 ][ 3 ] = { { 1, 0, 0 },
        { 0, 1, 0 },
        { 0, 0, 1 } };

      tensorOps::copy< N, N >( src, matrix );
      tensorOps::copy< N, N >( result, matrix );
      return 1;
    }
    if( i == 1 )
    {
      double const matrix[ 3 ][ 3 ] = { { 1, 2, 3 },
        { 1, 2, -3 },
        { 7, 8, 9 } };

      double const inverse[ 3 ][ 3 ] = { { -7, -1, 2 },
        {  5, 2, -1 },
        {  1, -1, 0 } };

      tensorOps::copy< N, N >( src, matrix );
      tensorOps::copy< N, N >( result, inverse );
      tensorOps::scale< N, N >( result, 1.0 / 6.0 );
      return -36;
    }

    LVARRAY_ERROR( "No test for " << i );
    return 0;
  }

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

// TODO: These tests can be run without chai and only using the c-arrays.
#if defined(USE_CUDA) && defined(USE_CHAI)
  , std::tuple< double, std::integral_constant< int, 2 >, parallelDevicePolicy< 32 > >
  , std::tuple< double, std::integral_constant< int, 3 >, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( FixedSizeSquareMatrixTest, FixedSizeSquareMatrixTestTypes, );

TYPED_TEST( FixedSizeSquareMatrixTest, determinant )
{
  this->testDeterminant();
}

TYPED_TEST( FixedSizeSquareMatrixTest, inverseTwoArgs )
{
  this->testInverseTwoArgs();
}

TYPED_TEST( FixedSizeSquareMatrixTest, inverseOneArg )
{
  this->testInverseOneArg();
}

TYPED_TEST( FixedSizeSquareMatrixTest, symAijBj )
{
  this->testSymAijBj();
}

TYPED_TEST( FixedSizeSquareMatrixTest, plusSymAijBj )
{
  this->plusSymAijBj();
}

TYPED_TEST( FixedSizeSquareMatrixTest, symAikBjk )
{
  this->testSymAikBjk();
}

TYPED_TEST( FixedSizeSquareMatrixTest, AikSymBklAjl )
{
  this->testAikSymBklAjl();
}

} // namespace testing
} // namespace LvArray
