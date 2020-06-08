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

template< typename T_N_POLICY_TUPLE >
class OneSizeTest : public ::testing::Test
{
public:
  using T = std::tuple_element_t< 0, T_N_POLICY_TUPLE >;
  static constexpr std::ptrdiff_t N = std::tuple_element_t< 1, T_N_POLICY_TUPLE > {};
  using POLICY = std::tuple_element_t< 2, T_N_POLICY_TUPLE >;

  void SetUp() override
  {
    fill( m_vectorA_IJ.toSlice(), m_seedVectorA );
    fill( m_vectorA_JI.toSlice(), m_seedVectorA );
    fill( m_vectorA_local, m_seedVectorA );

    fill( m_vectorB_IJ.toSlice(), m_seedVectorB );
    fill( m_vectorB_JI.toSlice(), m_seedVectorB );
    fill( m_vectorB_local, m_seedVectorB );

    fill( m_vectorC_IJ.toSlice(), m_seedVectorC );
    fill( m_vectorC_JI.toSlice(), m_seedVectorC );
    fill( m_vectorC_local, m_seedVectorC );

    fill( m_matrixA_IJK.toSlice(), m_seedMatrixA );
    fill( m_matrixA_IKJ.toSlice(), m_seedMatrixA );
    fill( m_matrixA_KJI.toSlice(), m_seedMatrixA );
    fill( m_matrixA_local, m_seedMatrixA );

    fill( m_matrixB_IJK.toSlice(), m_seedMatrixB );
    fill( m_matrixB_IKJ.toSlice(), m_seedMatrixB );
    fill( m_matrixB_KJI.toSlice(), m_seedMatrixB );
    fill( m_matrixB_local, m_seedMatrixB );
  }

  void testScale()
  {
    T scale = T( 3.14 );
    T result[ N ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    { result[ i ] = m_vectorA_local[ i ] * scale; }

    ArrayView< T, 2, 1 > const & vectorA_IJ = m_vectorA_IJ.toView();
    ArrayView< T, 2, 0 > const & vectorA_JI = m_vectorA_JI.toView();

    std::ptrdiff_t const aSeed = m_seedVectorA;

    forall< POLICY >( 1, [scale, result, vectorA_IJ, vectorA_JI, aSeed] LVARRAY_HOST_DEVICE ( int )
        {
          tensorOps::scale< N >( vectorA_IJ[ 0 ], scale );
          CHECK_EQUALITY_1D( N, vectorA_IJ[ 0 ], result );

          tensorOps::scale< N >( vectorA_JI[ 0 ], scale );
          CHECK_EQUALITY_1D( N, vectorA_JI[ 0 ], result );

          T vectorA_local[ N ];
          fill( vectorA_local, aSeed );
          tensorOps::scale< N >( vectorA_local, scale );
          CHECK_EQUALITY_1D( N, vectorA_local, result );
        } );
  }

  void testL2NormSquared()
  {
    T result = 0;
    for( std::ptrdiff_t i = 0; i < N; ++i )
    { result += m_vectorA_local[ i ] * m_vectorA_local[ i ]; }

    ArrayView< T const, 2, 1 > const & vectorA_IJ = m_vectorA_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorA_JI = m_vectorA_JI.toViewConst();
    T const ( &vectorA_local )[ N ] = m_vectorA_local;

    forall< POLICY >( 1, [result, vectorA_IJ, vectorA_JI, vectorA_local] LVARRAY_HOST_DEVICE ( int )
        {
          PORTABLE_EXPECT_EQ( tensorOps::l2NormSquared< N >( vectorA_IJ[ 0 ] ), result );
          PORTABLE_EXPECT_EQ( tensorOps::l2NormSquared< N >( vectorA_JI[ 0 ] ), result );
          PORTABLE_EXPECT_EQ( tensorOps::l2NormSquared< N >( vectorA_local ), result );
        } );
  }

  void testL2Norm()
  {
    double norm = 0;
    for( std::ptrdiff_t i = 0; i < N; ++i )
    { norm += m_vectorA_local[ i ] * m_vectorA_local[ i ]; }
    norm = sqrt( norm );

    ArrayView< T const, 2, 1 > const & vectorA_IJ = m_vectorA_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorA_JI = m_vectorA_JI.toViewConst();
    T const ( &vectorA_local )[ N ] = m_vectorA_local;

    forall< POLICY >( 1, [norm, vectorA_IJ, vectorA_JI, vectorA_local] LVARRAY_HOST_DEVICE ( int )
        {
          PORTABLE_EXPECT_EQ( tensorOps::l2Norm< N >( vectorA_IJ[ 0 ] ), norm );
          PORTABLE_EXPECT_EQ( tensorOps::l2Norm< N >( vectorA_JI[ 0 ] ), norm );
          PORTABLE_EXPECT_EQ( tensorOps::l2Norm< N >( vectorA_local ), norm );
        } );
  }

  void testNormalize()
  {
    double norm = 0;
    for( std::ptrdiff_t i = 0; i < N; ++i )
    { norm += m_vectorA_local[ i ] * m_vectorA_local[ i ]; }
    norm = sqrt( norm );

    T result[ N ];
    double const invNorm = 1.0 / norm;
    for( std::ptrdiff_t i = 0; i < N; ++i )
    { result [ i ] = m_vectorA_local[ i ] * invNorm; }

    ArrayView< T, 2, 1 > const & vectorA_IJ = m_vectorA_IJ.toView();
    ArrayView< T, 2, 0 > const & vectorA_JI = m_vectorA_JI.toView();

    std::ptrdiff_t const aSeed = m_seedVectorA;

    forall< POLICY >( 1, [norm, result, vectorA_IJ, vectorA_JI, aSeed] LVARRAY_HOST_DEVICE ( int )
        {
          PORTABLE_EXPECT_EQ( tensorOps::normalize< N >( vectorA_IJ[ 0 ] ), norm );
          CHECK_EQUALITY_1D( N, vectorA_IJ[ 0 ], result );

          PORTABLE_EXPECT_EQ( tensorOps::normalize< N >( vectorA_JI[ 0 ] ), norm );
          CHECK_EQUALITY_1D( N, vectorA_JI[ 0 ], result );

          T vectorA_local[ N ];
          fill( vectorA_local, aSeed );
          PORTABLE_EXPECT_EQ( tensorOps::normalize< N >( vectorA_local ), norm );
          CHECK_EQUALITY_1D( N, vectorA_local, result );
        } );
  }

  void testFill()
  {
    ArrayView< T, 2, 1 > const & vectorA_IJ = m_vectorA_IJ.toView();
    ArrayView< T, 2, 0 > const & vectorA_JI = m_vectorA_JI.toView();

    forall< POLICY >( 1, [vectorA_IJ, vectorA_JI] LVARRAY_HOST_DEVICE ( int )
        {
          for( int i = 0; i < 3; ++i )
          {
            T const value = 3.14 * i;
            tensorOps::fill< N >( vectorA_IJ[ 0 ], value );
            for( std::ptrdiff_t j = 0; j < N; ++j )
            { PORTABLE_EXPECT_EQ( vectorA_IJ( 0, j ), value ); }

            tensorOps::fill< N >( vectorA_JI[ 0 ], value );
            for( std::ptrdiff_t j = 0; j < N; ++j )
            { PORTABLE_EXPECT_EQ( vectorA_JI( 0, j ), value ); }

            T vectorA_local[ N ];
            tensorOps::fill< N >( vectorA_local, value );
            for( std::ptrdiff_t j = 0; j < N; ++j )
            { PORTABLE_EXPECT_EQ( vectorA_local[ j ], value ); }
          }
        } );
  }

  void testMaxAbsoluteEntry()
  {
    T maxEntry = 0;
    for( std::ptrdiff_t i = 0; i < N; ++i )
    { maxEntry = std::max( std::abs( maxEntry ), m_vectorA_local[ i ] ); }

    ArrayView< T, 2, 1 > const & vectorA_IJ = m_vectorA_IJ.toView();
    ArrayView< T, 2, 0 > const & vectorA_JI = m_vectorA_JI.toView();

    std::ptrdiff_t const seedVectorA = m_seedVectorA;

    forall< POLICY >( 1, [maxEntry, vectorA_IJ, vectorA_JI, seedVectorA] LVARRAY_HOST_DEVICE ( int )
        {
          PORTABLE_EXPECT_EQ( maxEntry, tensorOps::maxAbsoluteEntry< N >( vectorA_IJ[ 0 ] ) );

          PORTABLE_EXPECT_EQ( maxEntry, tensorOps::maxAbsoluteEntry< N >( vectorA_JI[ 0 ] ) );

          T vectorA_local[ N ];
          fill( vectorA_local, seedVectorA );
          PORTABLE_EXPECT_EQ( maxEntry, tensorOps::maxAbsoluteEntry< N >( vectorA_local ) );
        } );
  }

  void testCopy()
  {
    ArrayView< T, 2, 1 > const & vectorA_IJ = m_vectorA_IJ.toView();
    ArrayView< T, 2, 0 > const & vectorA_JI = m_vectorA_JI.toView();

    ArrayView< T const, 2, 1 > const & vectorB_IJ = m_vectorB_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorB_JI = m_vectorB_JI.toViewConst();
    T const ( &vectorB_local )[ N ] = m_vectorB_local;

    std::ptrdiff_t const aSeed = m_seedVectorA;

    forall< POLICY >( 1, [vectorA_IJ, vectorA_JI, vectorB_IJ, vectorB_JI, vectorB_local, aSeed] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( a, b ) \
            tensorOps::copy< N >( a, b ); \
            CHECK_EQUALITY_1D( N, a, b ); \
            fill( a, aSeed )

          #define _TEST_PERMS( a, b0, b1, b2 ) \
            _TEST( a, b0 ); \
            _TEST( a, b1 ); \
            _TEST( a, b2 )

          T vectorA_local[ N ];
          fill( vectorA_local, aSeed );

          _TEST_PERMS( vectorA_IJ[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_JI[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_local, vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testScaledCopy()
  {
    T scale = T( 3.14 );
    ArrayView< T, 2, 1 > const & vectorA_IJ = m_vectorA_IJ.toView();
    ArrayView< T, 2, 0 > const & vectorA_JI = m_vectorA_JI.toView();

    ArrayView< T const, 2, 1 > const & vectorB_IJ = m_vectorB_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorB_JI = m_vectorB_JI.toViewConst();
    T const ( &vectorB_local )[ N ] = m_vectorB_local;

    std::ptrdiff_t const aSeed = m_seedVectorA;

    forall< POLICY >( 1, [scale, vectorA_IJ, vectorA_JI, vectorB_IJ, vectorB_JI, vectorB_local, aSeed] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( a, b ) \
            tensorOps::scaledCopy< N >( a, b, scale ); \
            for( std::ptrdiff_t i = 0; i < N; ++i ) \
            { PORTABLE_EXPECT_EQ( a[ i ], scale * b[ i ] ); } \
            fill( a, aSeed )

          #define _TEST_PERMS( a, b0, b1, b2 ) \
            _TEST( a, b0 ); \
            _TEST( a, b1 ); \
            _TEST( a, b2 )

          T vectorA_local[ N ];
          fill( vectorA_local, aSeed );

          _TEST_PERMS( vectorA_IJ[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_JI[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_local, vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testAdd()
  {
    T result[ N ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    { result[ i ] = m_vectorA_local[ i ] + m_vectorB_local[ i ]; }

    ArrayView< T, 2, 1 > const & vectorA_IJ = m_vectorA_IJ.toView();
    ArrayView< T, 2, 0 > const & vectorA_JI = m_vectorA_JI.toView();

    ArrayView< T const, 2, 1 > const & vectorB_IJ = m_vectorB_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorB_JI = m_vectorB_JI.toViewConst();
    T const ( &vectorB_local )[ N ] = m_vectorB_local;

    std::ptrdiff_t const aSeed = m_seedVectorA;

    forall< POLICY >( 1, [result, vectorA_IJ, vectorA_JI, vectorB_IJ, vectorB_JI, vectorB_local, aSeed] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( a, b ) \
            tensorOps::add< N >( a, b ); \
            CHECK_EQUALITY_1D( N, a, result ); \
            fill( a, aSeed )

          #define _TEST_PERMS( a, b0, b1, b2 ) \
            _TEST( a, b0 ); \
            _TEST( a, b1 ); \
            _TEST( a, b2 )

          T vectorA_local[ N ];
          fill( vectorA_local, aSeed );

          _TEST_PERMS( vectorA_IJ[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_JI[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_local, vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );

      #undef _TEST_PERMS
      #undef _TEST
        } );
  }

  void testSubtract()
  {
    T result[ N ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    { result[ i ] = m_vectorA_local[ i ] - m_vectorB_local[ i ]; }

    ArrayView< T, 2, 1 > const & vectorA_IJ = m_vectorA_IJ.toView();
    ArrayView< T, 2, 0 > const & vectorA_JI = m_vectorA_JI.toView();

    ArrayView< T const, 2, 1 > const & vectorB_IJ = m_vectorB_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorB_JI = m_vectorB_JI.toViewConst();
    T const ( &vectorB_local )[ N ] = m_vectorB_local;

    std::ptrdiff_t const aSeed = m_seedVectorA;

    forall< POLICY >( 1, [result, vectorA_IJ, vectorA_JI, vectorB_IJ, vectorB_JI, vectorB_local, aSeed] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( a, b ) \
            tensorOps::subtract< N >( a, b ); \
            CHECK_EQUALITY_1D( N, a, result ); \
            fill( a, aSeed )

          #define _TEST_PERMS( a, b0, b1, b2 ) \
            _TEST( a, b0 ); \
            _TEST( a, b1 ); \
            _TEST( a, b2 )

          T vectorA_local[ N ];
          fill( vectorA_local, aSeed );

          _TEST_PERMS( vectorA_IJ[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_JI[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_local, vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );

      #undef _TEST_PERMS
      #undef _TEST
        } );
  }

  void testScaledAdd()
  {
    ArrayView< T, 2, 1 > const & vectorA_IJ = m_vectorA_IJ.toView();
    ArrayView< T, 2, 0 > const & vectorA_JI = m_vectorA_JI.toView();

    ArrayView< T const, 2, 1 > const & vectorB_IJ = m_vectorB_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorB_JI = m_vectorB_JI.toViewConst();
    T const ( &vectorB_local )[ N ] = m_vectorB_local;

    std::ptrdiff_t const aSeed = m_seedVectorA;

    forall< POLICY >( 1, [vectorA_IJ, vectorA_JI, vectorB_IJ, vectorB_JI, vectorB_local, aSeed] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( a, b ) \
            tensorOps::scaledAdd< N >( a, b, scale ); \
            CHECK_EQUALITY_1D( N, a, result ); \
            fill( a, aSeed )

          #define _TEST_PERMS( a, b0, b1, b2 ) \
            _TEST( a, b0 ); \
            _TEST( a, b1 ); \
            _TEST( a, b2 )

          T vectorA_local[ N ];
          fill( vectorA_local, aSeed );

          T const scale = T( 3.14 );
          T result[ N ];
          for( std::ptrdiff_t i = 0; i < N; ++i )
          { result[ i ] = vectorA_local[ i ] + scale * vectorB_local[ i ]; }

          _TEST_PERMS( vectorA_IJ[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_JI[ 0 ], vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );
          _TEST_PERMS( vectorA_local, vectorB_IJ[ 0 ], vectorB_JI[ 0 ], vectorB_local );

          #undef _TEST_PERMS
          #undef _TEST
        } );
  }

  void testAiBi()
  {
    T expectedValue = 0;
    for( std::ptrdiff_t i = 0; i < N; ++i )
    { expectedValue += m_vectorA_local[ i ] * m_vectorB_local[ i ]; }

    ArrayView< T const, 2, 1 > const & vectorA_IJ = m_vectorA_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorA_JI = m_vectorA_JI.toViewConst();
    T const ( &vectorA_local )[ N ] = m_vectorA_local;

    ArrayView< T const, 2, 1 > const & vectorB_IJ = m_vectorB_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorB_JI = m_vectorB_JI.toViewConst();
    T const ( &vectorB_local )[ N ] = m_vectorB_local;

    forall< POLICY >( 1, [expectedValue, vectorA_IJ, vectorA_JI, vectorB_IJ, vectorB_JI, vectorA_local, vectorB_local] LVARRAY_HOST_DEVICE ( int )
        {
          PORTABLE_EXPECT_EQ( tensorOps::AiBi< N >( vectorA_IJ[ 0 ], vectorB_IJ[ 0 ] ), expectedValue );
          PORTABLE_EXPECT_EQ( tensorOps::AiBi< N >( vectorA_IJ[ 0 ], vectorB_JI[ 0 ] ), expectedValue );
          PORTABLE_EXPECT_EQ( tensorOps::AiBi< N >( vectorA_IJ[ 0 ], vectorB_local ), expectedValue );

          PORTABLE_EXPECT_EQ( tensorOps::AiBi< N >( vectorA_JI[ 0 ], vectorB_IJ[ 0 ] ), expectedValue );
          PORTABLE_EXPECT_EQ( tensorOps::AiBi< N >( vectorA_JI[ 0 ], vectorB_JI[ 0 ] ), expectedValue );
          PORTABLE_EXPECT_EQ( tensorOps::AiBi< N >( vectorA_JI[ 0 ], vectorB_local ), expectedValue );

          PORTABLE_EXPECT_EQ( tensorOps::AiBi< N >( vectorA_local, vectorB_IJ[ 0 ] ), expectedValue );
          PORTABLE_EXPECT_EQ( tensorOps::AiBi< N >( vectorA_local, vectorB_JI[ 0 ] ), expectedValue );
          PORTABLE_EXPECT_EQ( tensorOps::AiBi< N >( vectorA_local, vectorB_local ), expectedValue );
        } );
  }

  void testElementWiseMultiplication()
  {
    T result[ N ];
    for( std::ptrdiff_t i = 0; i < N; ++i )
    { result[ i ] = m_vectorB_local[ i ] * m_vectorC_local[ i ]; }

    ArrayView< T, 2, 1 > const & vectorA_IJ = m_vectorA_IJ.toView();
    ArrayView< T, 2, 0 > const & vectorA_JI = m_vectorA_JI.toView();

    ArrayView< T const, 2, 1 > const & vectorB_IJ = m_vectorB_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorB_JI = m_vectorB_JI.toViewConst();
    T const ( &vectorB_local )[ N ] = m_vectorB_local;

    ArrayView< T const, 2, 1 > const & vectorC_IJ = m_vectorC_IJ.toViewConst();
    ArrayView< T const, 2, 0 > const & vectorC_JI = m_vectorC_JI.toViewConst();
    T const ( &vectorC_local )[ N ] = m_vectorC_local;

    std::ptrdiff_t const aSeed = m_seedVectorA;

    forall< POLICY >( 1, [result, vectorA_IJ, vectorA_JI, vectorB_IJ, vectorB_JI, vectorB_local,
                          vectorC_IJ, vectorC_JI, vectorC_local, aSeed] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( a, b, c ) \
            fill( a, aSeed ); \
            tensorOps::elementWiseMultiplication< N >( a, b, c ); \
            CHECK_EQUALITY_1D( N, a, result )

          #define _TEST_PERMS( a, b, c0, c1, c2 ) \
            _TEST( a, b, c0 ); \
            _TEST( a, b, c1 ); \
            _TEST( a, b, c2 )

          T vectorA_local[ N ];

          _TEST_PERMS( vectorA_IJ[ 0 ], vectorB_IJ[ 0 ], vectorC_IJ[ 0 ], vectorC_JI[ 0 ], vectorC_local );
          _TEST_PERMS( vectorA_IJ[ 0 ], vectorB_JI[ 0 ], vectorC_IJ[ 0 ], vectorC_JI[ 0 ], vectorC_local );
          _TEST_PERMS( vectorA_IJ[ 0 ], vectorB_local, vectorC_IJ[ 0 ], vectorC_JI[ 0 ], vectorC_local );
          _TEST_PERMS( vectorA_JI[ 0 ], vectorB_IJ[ 0 ], vectorC_IJ[ 0 ], vectorC_JI[ 0 ], vectorC_local );
          _TEST_PERMS( vectorA_JI[ 0 ], vectorB_JI[ 0 ], vectorC_IJ[ 0 ], vectorC_JI[ 0 ], vectorC_local );
          _TEST_PERMS( vectorA_JI[ 0 ], vectorB_local, vectorC_IJ[ 0 ], vectorC_JI[ 0 ], vectorC_local );
          _TEST_PERMS( vectorA_local, vectorB_IJ[ 0 ], vectorC_JI[ 0 ], vectorC_JI[ 0 ], vectorC_local );
          _TEST_PERMS( vectorA_local, vectorB_JI[ 0 ], vectorC_JI[ 0 ], vectorC_JI[ 0 ], vectorC_local );
          _TEST_PERMS( vectorA_local, vectorB_local, vectorC_JI[ 0 ], vectorC_JI[ 0 ], vectorC_local );

      #undef _TEST_PERMS
      #undef _TEST
        } );
  }

  void testAddIdentity()
  {
    T value = T( 3.14 );
    T result[ N ][ N ];
    tensorOps::copy< N, N >( result, m_matrixA_local );

    for( std::ptrdiff_t i = 0; i < N; ++i )
    { result[ i ][ i ] += value; }

    ArrayView< T, 3, 2 > const & matrixA_IJK = m_matrixA_IJK.toView();
    ArrayView< T, 3, 1 > const & matrixA_IKJ = m_matrixA_IKJ.toView();
    ArrayView< T, 3, 0 > const & matrixA_KJI = m_matrixA_KJI.toView();

    std::ptrdiff_t const seedMatrixA = m_seedMatrixA;

    forall< POLICY >( 1, [value, result, matrixA_IJK, matrixA_IKJ, matrixA_KJI, seedMatrixA] LVARRAY_HOST_DEVICE ( int )
        {
          tensorOps::addIdentity< N >( matrixA_IJK[ 0 ], value );
          CHECK_EQUALITY_2D( N, N, matrixA_IJK[ 0 ], result );

          tensorOps::addIdentity< N >( matrixA_IKJ[ 0 ], value );
          CHECK_EQUALITY_2D( N, N, matrixA_IKJ[ 0 ], result );

          tensorOps::addIdentity< N >( matrixA_KJI[ 0 ], value );
          CHECK_EQUALITY_2D( N, N, matrixA_KJI[ 0 ], result );

          T matrixA_local[ N ][ N ];
          fill( matrixA_local, seedMatrixA );
          tensorOps::addIdentity< N >( matrixA_local, value );
          CHECK_EQUALITY_2D( N, N, matrixA_local, result );
        } );
  }

private:
  std::ptrdiff_t const m_seedVectorA = 0;
  Array< T, 2, RAJA::PERM_IJ > m_vectorA_IJ { 1, N };
  Array< T, 2, RAJA::PERM_JI > m_vectorA_JI { 1, N };
  T m_vectorA_local[ N ];

  std::ptrdiff_t const m_seedVectorB = m_seedVectorA + N;
  Array< T, 2, RAJA::PERM_IJ > m_vectorB_IJ { 1, N };
  Array< T, 2, RAJA::PERM_JI > m_vectorB_JI { 1, N };
  T m_vectorB_local[ N ];

  std::ptrdiff_t const m_seedVectorC = m_seedVectorB + N;
  Array< T, 2, RAJA::PERM_IJ > m_vectorC_IJ { 1, N };
  Array< T, 2, RAJA::PERM_JI > m_vectorC_JI { 1, N };
  T m_vectorC_local[ N ];

  std::ptrdiff_t const m_seedMatrixA = m_seedVectorC + N;
  Array< T, 3, RAJA::PERM_IJK > m_matrixA_IJK { 1, N, N };
  Array< T, 3, RAJA::PERM_IKJ > m_matrixA_IKJ { 1, N, N };
  Array< T, 3, RAJA::PERM_KJI > m_matrixA_KJI { 1, N, N };
  T m_matrixA_local[ N ][ N ];

  std::ptrdiff_t const m_seedMatrixB = m_seedMatrixA + N * N;
  Array< T, 3, RAJA::PERM_IJK > m_matrixB_IJK { 1, N, N };
  Array< T, 3, RAJA::PERM_IKJ > m_matrixB_IKJ { 1, N, N };
  Array< T, 3, RAJA::PERM_KJI > m_matrixB_KJI { 1, N, N };
  T m_matrixB_local[ N ][ N ];
};


using OneSizeTestTypes = ::testing::Types<
  std::tuple< double, std::integral_constant< int, 2 >, serialPolicy >
  , std::tuple< int, std::integral_constant< int, 3 >, serialPolicy >
  , std::tuple< double, std::integral_constant< int, 6 >, serialPolicy >
#if defined(USE_CUDA)
  , std::tuple< double, std::integral_constant< int, 2 >, parallelDevicePolicy< 32 > >
  , std::tuple< int, std::integral_constant< int, 3 >, parallelDevicePolicy< 32 > >
  , std::tuple< double, std::integral_constant< int, 6 >, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( OneSizeTest, OneSizeTestTypes, );

TYPED_TEST( OneSizeTest, scale )
{
  this->testScale();
}

TYPED_TEST( OneSizeTest, l2NormSquared )
{
  this->testL2NormSquared();
}

TYPED_TEST( OneSizeTest, l2Norm )
{
  this->testL2Norm();
}

TYPED_TEST( OneSizeTest, normalize )
{
  this->testNormalize();
}

TYPED_TEST( OneSizeTest, fill )
{
  this->testFill();
}

TYPED_TEST( OneSizeTest, maxAbsoluteEntry )
{
  this->testMaxAbsoluteEntry();
}

TYPED_TEST( OneSizeTest, copy )
{
  this->testCopy();
}

TYPED_TEST( OneSizeTest, scaledCopy )
{
  this->testScaledCopy();
}

TYPED_TEST( OneSizeTest, add )
{
  this->testAdd();
}

TYPED_TEST( OneSizeTest, subtract )
{
  this->testSubtract();
}

TYPED_TEST( OneSizeTest, scaledAdd )
{
  this->testScaledAdd();
}

TYPED_TEST( OneSizeTest, AiBi )
{
  this->testAiBi();
}

TYPED_TEST( OneSizeTest, elementWiseMultiplication )
{
  this->testElementWiseMultiplication();
}

TYPED_TEST( OneSizeTest, addIdentity )
{
  this->testAddIdentity();
}

} // namespace testing
} // namespace LvArray
