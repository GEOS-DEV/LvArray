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

template< typename T_POLICY_TUPLE >
class NoSizeTest : public ::testing::Test
{
public:
  using T = std::tuple_element_t< 0, T_POLICY_TUPLE >;
  using POLICY = std::tuple_element_t< 1, T_POLICY_TUPLE >;

  void SetUp() override
  {
    fill( m_vectorA2_IJ.toSlice(), m_seedVectorA2 );
    fill( m_vectorA2_JI.toSlice(), m_seedVectorA2 );
    fill( m_vectorA2_local, m_seedVectorA2 );

    fill( m_vectorA3_IJ.toSlice(), m_seedVectorA3 );
    fill( m_vectorA3_JI.toSlice(), m_seedVectorA3 );
    fill( m_vectorA3_local, m_seedVectorA3 );

    fill( m_vectorB3_IJ.toSlice(), m_seedVectorB3 );
    fill( m_vectorB3_JI.toSlice(), m_seedVectorB3 );
    fill( m_vectorB3_local, m_seedVectorB3 );

    fill( m_vectorC3_IJ.toSlice(), m_seedVectorC3 );
    fill( m_vectorC3_JI.toSlice(), m_seedVectorC3 );
    fill( m_vectorC3_local, m_seedVectorC3 );

    fill( m_vectorA6_IJ.toSlice(), m_seedVectorA6 );
    fill( m_vectorA6_JI.toSlice(), m_seedVectorA6 );
    fill( m_vectorA6_local, m_seedVectorA6 );

    fill( m_matrixA2_IJK.toSlice(), m_seedMatrixA2 );
    fill( m_matrixA2_IKJ.toSlice(), m_seedMatrixA2 );
    fill( m_matrixA2_KJI.toSlice(), m_seedMatrixA2 );
    fill( m_matrixA2_local, m_seedMatrixA2 );

    fill( m_matrixA3_IJK.toSlice(), m_seedMatrixA3 );
    fill( m_matrixA3_IKJ.toSlice(), m_seedMatrixA3 );
    fill( m_matrixA3_KJI.toSlice(), m_seedMatrixA3 );
    fill( m_matrixA3_local, m_seedMatrixA3 );
  }

  void testInit2()
  {
    ArrayViewT< T const, 2, 1 > const vectorA2_IJ = m_vectorA2_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const vectorA2_JI = m_vectorA2_JI.toViewConst();
    T const ( &vectorA2_local )[ 2 ] = m_vectorA2_local;

    forall< POLICY >( 1, [vectorA2_IJ, vectorA2_JI, vectorA2_local] LVARRAY_HOST_DEVICE ( int )
        {
          {
            T const result[ 3 ] = LVARRAY_TENSOROPS_INIT_LOCAL_2( vectorA2_IJ[ 0 ] );
            for( std::ptrdiff_t i = 0; i < 2; ++i )
            { PORTABLE_EXPECT_EQ( result[ i ], vectorA2_IJ( 0, i ) ); }
          }

          {
            T const result[ 3 ] = LVARRAY_TENSOROPS_INIT_LOCAL_2( 4 + vectorA2_JI[ 0 ] );
            for( std::ptrdiff_t i = 0; i < 2; ++i )
            { PORTABLE_EXPECT_EQ( result[ i ], 4 + vectorA2_JI( 0, i ) ); }
          }

          {
            T const result[ 3 ] = LVARRAY_TENSOROPS_INIT_LOCAL_2( 4 + 2 * vectorA2_local );
            for( std::ptrdiff_t i = 0; i < 2; ++i )
            { PORTABLE_EXPECT_EQ( result[ i ], 4 + 2 * vectorA2_local[ i ] ); }
          }
        } );
  }

  void testInit3()
  {
    ArrayViewT< T const, 2, 1 > const vectorA3_IJ = m_vectorA3_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const vectorA3_JI = m_vectorA3_JI.toViewConst();
    T const ( &vectorA3_local )[ 3 ] = m_vectorA3_local;

    forall< POLICY >( 1, [vectorA3_IJ, vectorA3_JI, vectorA3_local] LVARRAY_HOST_DEVICE ( int )
        {
          {
            T const result[ 3 ] = LVARRAY_TENSOROPS_INIT_LOCAL_3( vectorA3_IJ[ 0 ] );
            for( std::ptrdiff_t i = 0; i < 3; ++i )
            { PORTABLE_EXPECT_EQ( result[ i ], vectorA3_IJ( 0, i ) ); }
          }

          {
            T const result[ 3 ] = LVARRAY_TENSOROPS_INIT_LOCAL_3( 4 + vectorA3_JI[ 0 ] );
            for( std::ptrdiff_t i = 0; i < 3; ++i )
            { PORTABLE_EXPECT_EQ( result[ i ], 4 + vectorA3_JI( 0, i ) ); }
          }

          {
            T const result[ 3 ] = LVARRAY_TENSOROPS_INIT_LOCAL_3( 4 + 2 * vectorA3_local );
            for( std::ptrdiff_t i = 0; i < 3; ++i )
            { PORTABLE_EXPECT_EQ( result[ i ], 4 + 2 * vectorA3_local[ i ] ); }
          }
        } );
  }

  void testInit6()
  {
    ArrayViewT< T const, 2, 1 > const vectorA6_IJ = m_vectorA6_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const vectorA6_JI = m_vectorA6_JI.toViewConst();
    T const ( &vectorA6_local )[ 6 ] = m_vectorA6_local;

    forall< POLICY >( 1, [vectorA6_IJ, vectorA6_JI, vectorA6_local] LVARRAY_HOST_DEVICE ( int )
        {
          {
            T const result[ 6 ] = LVARRAY_TENSOROPS_INIT_LOCAL_6( vectorA6_IJ[ 0 ] );
            for( std::ptrdiff_t i = 0; i < 6; ++i )
            { PORTABLE_EXPECT_EQ( result[ i ], vectorA6_IJ( 0, i ) ); }
          }

          {
            T const result[ 6 ] = LVARRAY_TENSOROPS_INIT_LOCAL_6( 4 + vectorA6_JI[ 0 ] );
            for( std::ptrdiff_t i = 0; i < 6; ++i )
            { PORTABLE_EXPECT_EQ( result[ i ], 4 + vectorA6_JI( 0, i ) ); }
          }

          {
            T const result[ 6 ] = LVARRAY_TENSOROPS_INIT_LOCAL_6( 4 + 2 * vectorA6_local );
            for( std::ptrdiff_t i = 0; i < 6; ++i )
            { PORTABLE_EXPECT_EQ( result[ i ], 4 + 2 * vectorA6_local[ i ] ); }
          }
        } );
  }

  void testInit2x2()
  {
    ArrayViewT< T, 3, 2 > const matrixA2_IJK = m_matrixA2_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixA2_IKJ = m_matrixA2_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixA2_KJI = m_matrixA2_KJI.toView();
    T const ( &matrixA2_local )[ 2 ][ 2 ] = m_matrixA2_local;

    forall< POLICY >( 1, [matrixA2_IJK, matrixA2_IKJ, matrixA2_KJI, matrixA2_local] LVARRAY_HOST_DEVICE ( int )
        {
          {
            T const result[ 2 ][ 2 ] = LVARRAY_TENSOROPS_INIT_LOCAL_2x2( matrixA2_IJK[ 0 ] );
            for( std::ptrdiff_t i = 0; i < 2; ++i )
            {
              for( std::ptrdiff_t j = 0; j < 2; ++j )
              {
                PORTABLE_EXPECT_EQ( result[ i ][ j ], matrixA2_IJK( 0, i, j ) );
              }
            }
          }

          {
            T const result[ 2 ][ 2 ] = LVARRAY_TENSOROPS_INIT_LOCAL_2x2( 4 + matrixA2_IKJ[ 0 ] );
            for( std::ptrdiff_t i = 0; i < 2; ++i )
            {
              for( std::ptrdiff_t j = 0; j < 2; ++j )
              {
                PORTABLE_EXPECT_EQ( result[ i ][ j ], 4 + matrixA2_IKJ( 0, i, j ) );
              }
            }
          }

          {
            T const result[ 2 ][ 2 ] = LVARRAY_TENSOROPS_INIT_LOCAL_2x2( 1 + 10 / matrixA2_KJI[ 0 ] );
            for( std::ptrdiff_t i = 0; i < 2; ++i )
            {
              for( std::ptrdiff_t j = 0; j < 2; ++j )
              {
                PORTABLE_EXPECT_EQ( result[ i ][ j ], 1 + 10 / matrixA2_KJI( 0, i, j ) );
              }
            }
          }

          {
            T const result[ 2 ][ 2 ] = LVARRAY_TENSOROPS_INIT_LOCAL_2x2( 4 + 2 * matrixA2_local );
            for( std::ptrdiff_t i = 0; i < 2; ++i )
            {
              for( std::ptrdiff_t j = 0; j < 2; ++j )
              {
                PORTABLE_EXPECT_EQ( result[ i ][ j ], 4 + 2 * matrixA2_local[ i ][ j ] );
              }
            }
          }
        } );
  }

  void testInit3x3()
  {
    ArrayViewT< T, 3, 2 > const matrixA3_IJK = m_matrixA3_IJK.toView();
    ArrayViewT< T, 3, 1 > const matrixA3_IKJ = m_matrixA3_IKJ.toView();
    ArrayViewT< T, 3, 0 > const matrixA3_KJI = m_matrixA3_KJI.toView();
    T const ( &matrixA3_local )[ 3 ][ 3 ] = m_matrixA3_local;

    forall< POLICY >( 1, [matrixA3_IJK, matrixA3_IKJ, matrixA3_KJI, matrixA3_local] LVARRAY_HOST_DEVICE ( int )
        {
          {
            T const result[ 3 ][ 3 ] = LVARRAY_TENSOROPS_INIT_LOCAL_3x3( matrixA3_IJK[ 0 ] );
            for( std::ptrdiff_t i = 0; i < 3; ++i )
            {
              for( std::ptrdiff_t j = 0; j < 3; ++j )
              {
                PORTABLE_EXPECT_EQ( result[ i ][ j ], matrixA3_IJK( 0, i, j ) );
              }
            }
          }

          {
            T const result[ 3 ][ 3 ] = LVARRAY_TENSOROPS_INIT_LOCAL_3x3( 4 + matrixA3_IKJ[ 0 ] );
            for( std::ptrdiff_t i = 0; i < 3; ++i )
            {
              for( std::ptrdiff_t j = 0; j < 3; ++j )
              {
                PORTABLE_EXPECT_EQ( result[ i ][ j ], 4 + matrixA3_IKJ( 0, i, j ) );
              }
            }
          }

          {
            T const result[ 3 ][ 3 ] = LVARRAY_TENSOROPS_INIT_LOCAL_3x3( 1 + 10 / matrixA3_KJI[ 0 ] );
            for( std::ptrdiff_t i = 0; i < 3; ++i )
            {
              for( std::ptrdiff_t j = 0; j < 3; ++j )
              {
                PORTABLE_EXPECT_EQ( result[ i ][ j ], 1 + 10 / matrixA3_KJI( 0, i, j ) );
              }
            }
          }

          {
            T const result[ 3 ][ 3 ] = LVARRAY_TENSOROPS_INIT_LOCAL_3x3( 4 + 3 * matrixA3_local );
            for( std::ptrdiff_t i = 0; i < 3; ++i )
            {
              for( std::ptrdiff_t j = 0; j < 3; ++j )
              {
                PORTABLE_EXPECT_EQ( result[ i ][ j ], 4 + 3 * matrixA3_local[ i ][ j ] );
              }
            }
          }
        } );
  }

  void testCrossProduct()
  {
    T result[ 3 ];
    result[ 0 ] = m_vectorB3_local[ 1 ] * m_vectorC3_local[ 2 ] - m_vectorB3_local[ 2 ] * m_vectorC3_local[ 1 ];
    result[ 1 ] = m_vectorB3_local[ 2 ] * m_vectorC3_local[ 0 ] - m_vectorB3_local[ 0 ] * m_vectorC3_local[ 2 ];
    result[ 2 ] = m_vectorB3_local[ 0 ] * m_vectorC3_local[ 1 ] - m_vectorB3_local[ 1 ] * m_vectorC3_local[ 0 ];

    ArrayViewT< T, 2, 1 > const vectorA3_IJ = m_vectorA3_IJ.toView();
    ArrayViewT< T, 2, 0 > const vectorA3_JI = m_vectorA3_JI.toView();

    ArrayViewT< T const, 2, 1 > const vectorB3_IJ = m_vectorB3_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const vectorB3_JI = m_vectorB3_JI.toViewConst();
    T const ( &vectorB3_local )[ 3 ] = m_vectorB3_local;

    ArrayViewT< T const, 2, 1 > const vectorC3_IJ = m_vectorC3_IJ.toViewConst();
    ArrayViewT< T const, 2, 0 > const vectorC3_JI = m_vectorC3_JI.toViewConst();
    T const ( &vectorC3_local )[ 3 ] = m_vectorC3_local;

    std::ptrdiff_t const aSeed = m_seedVectorA3;

    forall< POLICY >( 1, [result, vectorA3_IJ, vectorA3_JI, vectorB3_IJ, vectorB3_JI, vectorB3_local,
                          vectorC3_IJ, vectorC3_JI, vectorC3_local, aSeed] LVARRAY_HOST_DEVICE ( int )
        {
          #define _TEST( dst, src0, src1 ) \
            fill( dst, aSeed ); \
            tensorOps::crossProduct( dst, src0, src1 ); \
            CHECK_EQUALITY_1D( 3, dst, result ); \
            PORTABLE_EXPECT_EQ( tensorOps::AiBi< 3 >( dst, src0 ), 0 ); \
            PORTABLE_EXPECT_EQ( tensorOps::AiBi< 3 >( dst, src1 ), 0 )

          #define _TEST_PERMS( dst, srcA, srcB0, srcB1, srcB2 ) \
            _TEST( dst, srcA, srcB0 ); \
            _TEST( dst, srcA, srcB1 ); \
            _TEST( dst, srcA, srcB2 )

          T vectorA3_local[ 3 ];
          _TEST_PERMS( vectorA3_IJ[ 0 ], vectorB3_IJ[ 0 ], vectorC3_IJ[ 0 ], vectorC3_JI[ 0 ], vectorC3_local );
          _TEST_PERMS( vectorA3_IJ[ 0 ], vectorB3_JI[ 0 ], vectorC3_IJ[ 0 ], vectorC3_JI[ 0 ], vectorC3_local );
          _TEST_PERMS( vectorA3_IJ[ 0 ], vectorB3_local, vectorC3_IJ[ 0 ], vectorC3_JI[ 0 ], vectorC3_local );
          _TEST_PERMS( vectorA3_JI[ 0 ], vectorB3_IJ[ 0 ], vectorC3_IJ[ 0 ], vectorC3_JI[ 0 ], vectorC3_local );
          _TEST_PERMS( vectorA3_JI[ 0 ], vectorB3_JI[ 0 ], vectorC3_IJ[ 0 ], vectorC3_JI[ 0 ], vectorC3_local );
          _TEST_PERMS( vectorA3_JI[ 0 ], vectorB3_local, vectorC3_IJ[ 0 ], vectorC3_JI[ 0 ], vectorC3_local );
          _TEST_PERMS( vectorA3_local, vectorB3_IJ[ 0 ], vectorC3_IJ[ 0 ], vectorC3_JI[ 0 ], vectorC3_local );
          _TEST_PERMS( vectorA3_local, vectorB3_JI[ 0 ], vectorC3_IJ[ 0 ], vectorC3_JI[ 0 ], vectorC3_local );
          _TEST_PERMS( vectorA3_local, vectorB3_local, vectorC3_IJ[ 0 ], vectorC3_JI[ 0 ], vectorC3_local );

      #undef _TEST_PERMS
      #undef _TEST
        } );
  }

private:
  std::ptrdiff_t const m_seedVectorA2 = 0;
  ArrayT< T, RAJA::PERM_IJ > m_vectorA2_IJ { 1, 2 };
  ArrayT< T, RAJA::PERM_JI > m_vectorA2_JI { 1, 2 };
  T m_vectorA2_local[ 2 ];

  std::ptrdiff_t const m_seedVectorA3 = m_seedVectorA2 + 2;
  ArrayT< T, RAJA::PERM_IJ > m_vectorA3_IJ { 1, 3 };
  ArrayT< T, RAJA::PERM_JI > m_vectorA3_JI { 1, 3 };
  T m_vectorA3_local[ 3 ];

  std::ptrdiff_t const m_seedVectorB3 = m_seedVectorA3 + 3;
  ArrayT< T, RAJA::PERM_IJ > m_vectorB3_IJ { 1, 3 };
  ArrayT< T, RAJA::PERM_JI > m_vectorB3_JI { 1, 3 };
  T m_vectorB3_local[ 3 ];

  std::ptrdiff_t const m_seedVectorC3 = m_seedVectorB3 + 3;
  ArrayT< T, RAJA::PERM_IJ > m_vectorC3_IJ { 1, 3 };
  ArrayT< T, RAJA::PERM_JI > m_vectorC3_JI { 1, 3 };
  T m_vectorC3_local[ 3 ];

  std::ptrdiff_t const m_seedVectorA6 = m_seedVectorC3 + 3;
  ArrayT< T, RAJA::PERM_IJ > m_vectorA6_IJ { 1, 6 };
  ArrayT< T, RAJA::PERM_JI > m_vectorA6_JI { 1, 6 };
  T m_vectorA6_local[ 6 ];

  std::ptrdiff_t const m_seedMatrixA2 = m_seedVectorA6 + 6;
  ArrayT< T, RAJA::PERM_IJK > m_matrixA2_IJK { 1, 2, 2 };
  ArrayT< T, RAJA::PERM_IKJ > m_matrixA2_IKJ { 1, 2, 2 };
  ArrayT< T, RAJA::PERM_KJI > m_matrixA2_KJI { 1, 2, 2 };
  T m_matrixA2_local[ 2 ][ 2 ];

  std::ptrdiff_t const m_seedMatrixA3 = m_seedMatrixA2 + 4;
  ArrayT< T, RAJA::PERM_IJK > m_matrixA3_IJK { 1, 3, 3 };
  ArrayT< T, RAJA::PERM_IKJ > m_matrixA3_IKJ { 1, 3, 3 };
  ArrayT< T, RAJA::PERM_KJI > m_matrixA3_KJI { 1, 3, 3 };
  T m_matrixA3_local[ 3 ][ 3 ];
};

using NoSizeTestTypes = ::testing::Types<
  std::tuple< double, serialPolicy >
  , std::tuple< int, serialPolicy >

#if ( defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP) ) && defined(LVARRAY_USE_CHAI)
  , std::tuple< double, parallelDevicePolicy< 32 > >
  , std::tuple< int, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( NoSizeTest, NoSizeTestTypes, );

TYPED_TEST( NoSizeTest, init2 )
{
  this->testInit2();
}

TYPED_TEST( NoSizeTest, init3 )
{
  this->testInit3();
}

TYPED_TEST( NoSizeTest, init6 )
{
  this->testInit6();
}

TYPED_TEST( NoSizeTest, init2x2 )
{
  this->testInit2x2();
}

TYPED_TEST( NoSizeTest, init3x3 )
{
  this->testInit3x3();
}

TYPED_TEST( NoSizeTest, crossProduct )
{
  this->testCrossProduct();
}

} // namespace testing
} // namespace LvArray
