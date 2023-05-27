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

template< typename T_FLOAT_M_POLICY_TUPLE >
class InverseTest : public ::testing::Test
{
public:
  using T = std::tuple_element_t< 0, T_FLOAT_M_POLICY_TUPLE >;
  using FLOAT = std::tuple_element_t< 1, T_FLOAT_M_POLICY_TUPLE >;
  static constexpr std::ptrdiff_t M = std::tuple_element_t< 2, T_FLOAT_M_POLICY_TUPLE > {};
  using POLICY = std::tuple_element_t< 3, T_FLOAT_M_POLICY_TUPLE >;

  static_assert( M == 2 || M == 3, "Other sizes not supported." );
  static constexpr std::ptrdiff_t SYM_SIZE = tensorOps::SYM_SIZE< M >;

  void determinant()
  {
    ArrayViewT< T, 3, 2 > const srcMatrix_IJK = m_srcMatrix_IJK.toView();
    ArrayViewT< T, 3, 1 > const srcMatrix_IKJ = m_srcMatrix_IKJ.toView();
    ArrayViewT< T, 3, 0 > const srcMatrix_KJI = m_srcMatrix_KJI.toView();

    ArrayT< T, RAJA::PERM_IJK > arrayOfMatrices( numMatrices, M, M );

    T scale = 100;
    for( T & value : arrayOfMatrices )
    { value = randomValue( scale, m_gen ); }

    ArrayViewT< T const, 3, 2 > const matrices = arrayOfMatrices.toViewConst();

    forall< POLICY >( matrices.size( 0 ), [=] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          T srcLocal[ M ][ M ];
          #define _TEST( matrix ) \
            tensorOps::copy< M, M >( matrix, matrices[ i ] ); \
            checkDeterminant( scale, matrix )

          _TEST( srcMatrix_IJK[ i ] );
          _TEST( srcMatrix_IKJ[ i ] );
          _TEST( srcMatrix_KJI[ i ] );
          _TEST( srcLocal );

      #undef _TEST
        } );
  }

  void symDeterminant()
  {
    ArrayViewT< T, 2, 1 > const srcSymMatrix_IJ = m_srcSymMatrix_IJ.toView();
    ArrayViewT< T, 2, 0 > const srcSymMatrix_JI = m_srcSymMatrix_JI.toView();

    ArrayT< T, RAJA::PERM_IJ > arrayOfMatrices( numMatrices, SYM_SIZE );

    T scale = 100;
    for( T & value : arrayOfMatrices )
    { value = randomValue( scale, m_gen ); }

    ArrayViewT< T const, 2, 1 > const matrices = arrayOfMatrices.toViewConst();

    forall< POLICY >( matrices.size( 0 ), [=] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          T srcLocal[ SYM_SIZE ];
          #define _TEST( matrix ) \
            tensorOps::copy< SYM_SIZE >( matrix, matrices[ i ] ); \
            checkSymDeterminant( scale, matrix )

          _TEST( srcSymMatrix_IJ[ i ] );
          _TEST( srcSymMatrix_JI[ i ] );
          _TEST( srcLocal );

      #undef _TEST
        } );
  }

  void inverseTwoArgs()
  {
    ArrayViewT< T, 3, 2 > const srcMatrix_IJK = m_srcMatrix_IJK.toView();
    ArrayViewT< T, 3, 1 > const srcMatrix_IKJ = m_srcMatrix_IKJ.toView();
    ArrayViewT< T, 3, 0 > const srcMatrix_KJI = m_srcMatrix_KJI.toView();

    ArrayViewT< FLOAT, 3, 2 > const dstMatrix_IJK = m_dstMatrix_IJK.toView();
    ArrayViewT< FLOAT, 3, 1 > const dstMatrix_IKJ = m_dstMatrix_IKJ.toView();
    ArrayViewT< FLOAT, 3, 0 > const dstMatrix_KJI = m_dstMatrix_KJI.toView();

    ArrayT< T, RAJA::PERM_IJK > arrayOfMatrices( numMatrices, M, M );

    T scale = 100;
    for( T & value : arrayOfMatrices )
    { value = randomValue( scale, m_gen ); }

    ArrayViewT< T const, 3, 2 > const matrices = arrayOfMatrices.toViewConst();

    forall< POLICY >( matrices.size( 0 ), [=] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          T srcLocal[ M ][ M ];
          FLOAT dstLocal[ M ][ M ];
          T det;

          #define _TEST( output, input ) \
            tensorOps::copy< M, M >( input, matrices[ i ] ); \
            det = tensorOps::invert< M >( output, input ); \
            checkInverse( scale, det, output, input )

          #define _TEST_PERMS( input, output0, output1, output2, output3 ) \
            _TEST( output0, input ); \
            _TEST( output1, input ); \
            _TEST( output2, input ); \
            _TEST( output3, input )

          _TEST_PERMS( srcMatrix_IJK[ i ], dstMatrix_IJK[ i ], dstMatrix_IKJ[ i ], dstMatrix_KJI[ i ], dstLocal );
          _TEST_PERMS( srcMatrix_IKJ[ i ], dstMatrix_IJK[ i ], dstMatrix_IKJ[ i ], dstMatrix_KJI[ i ], dstLocal );
          _TEST_PERMS( srcMatrix_KJI[ i ], dstMatrix_IJK[ i ], dstMatrix_IKJ[ i ], dstMatrix_KJI[ i ], dstLocal );
          _TEST_PERMS( srcLocal, dstMatrix_IJK[ i ], dstMatrix_IKJ[ i ], dstMatrix_KJI[ i ], dstLocal );

      #undef _TEST_PERMS
      #undef _TEST
        } );
  }

  void inverseOneArg()
  {
    ArrayViewT< T, 3, 2 > const srcMatrix_IJK = m_srcMatrix_IJK.toView();
    ArrayViewT< T, 3, 1 > const srcMatrix_IKJ = m_srcMatrix_IKJ.toView();
    ArrayViewT< T, 3, 0 > const srcMatrix_KJI = m_srcMatrix_KJI.toView();

    ArrayT< T, RAJA::PERM_IJK > arrayOfMatrices( numMatrices, M, M );

    T scale = 100;
    for( T & value : arrayOfMatrices )
    { value = randomValue( scale, m_gen ); }

    ArrayViewT< T const, 3, 2 > const matrices = arrayOfMatrices.toViewConst();

    forall< POLICY >( matrices.size( 0 ), [=] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          T initialMatrix[ M ][ M ];
          T srcLocal[ M ][ M ];
          T det;

          #define _TEST( matrix ) \
            tensorOps::copy< M, M >( initialMatrix, matrices[ i ] ); \
            tensorOps::copy< M, M >( matrix, initialMatrix ); \
            det = tensorOps::invert< M >( matrix ); \
            checkInverse( scale, det, matrix, initialMatrix )

          _TEST( srcMatrix_IJK[ i ] );
          _TEST( srcMatrix_IKJ[ i ] );
          _TEST( srcMatrix_KJI[ i ] );
          _TEST( srcLocal );

      #undef _TEST
        } );
  }

  void symInverseTwoArgs()
  {
    ArrayViewT< T, 2, 1 > const srcSymMatrix_IJ = m_srcSymMatrix_IJ.toView();
    ArrayViewT< T, 2, 0 > const srcSymMatrix_JI = m_srcSymMatrix_JI.toView();

    ArrayViewT< FLOAT, 2, 1 > const dstSymMatrix_IJ = m_dstSymMatrix_IJ.toView();
    ArrayViewT< FLOAT, 2, 0 > const dstSymMatrix_JI = m_dstSymMatrix_JI.toView();

    ArrayT< T, RAJA::PERM_IJ > arrayOfMatrices( numMatrices, SYM_SIZE );

    T scale = 100;
    for( T & value : arrayOfMatrices )
    { value = randomValue( scale, m_gen ); }

    ArrayViewT< T const, 2, 1 > const matrices = arrayOfMatrices.toViewConst();

    forall< POLICY >( matrices.size( 0 ), [=] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          T srcLocal[ SYM_SIZE ];
          FLOAT dstLocal[ SYM_SIZE ];
          T det;

          #define _TEST( output, input ) \
            tensorOps::copy< SYM_SIZE >( input, matrices[ i ] ); \
            det = tensorOps::symInvert< M >( output, input ); \
            checkSymInverse( scale, det, output, input )

          #define _TEST_PERMS( input, output0, output1, output2 ) \
            _TEST( output0, input ); \
            _TEST( output1, input ); \
            _TEST( output2, input )

          _TEST_PERMS( srcSymMatrix_IJ[ i ], dstSymMatrix_IJ[ i ], dstSymMatrix_JI[ i ], dstLocal );
          _TEST_PERMS( srcSymMatrix_JI[ i ], dstSymMatrix_IJ[ i ], dstSymMatrix_JI[ i ], dstLocal );
          _TEST_PERMS( srcLocal, dstSymMatrix_IJ[ i ], dstSymMatrix_JI[ i ], dstLocal );

      #undef _TEST_PERMS
      #undef _TEST
        } );
  }

  void symInverseOneArg()
  {
    ArrayViewT< T, 2, 1 > const srcSymMatrix_IJ = m_srcSymMatrix_IJ.toView();
    ArrayViewT< T, 2, 0 > const srcSymMatrix_JI = m_srcSymMatrix_JI.toView();

    ArrayT< T, RAJA::PERM_IJ > arrayOfMatrices( numMatrices, SYM_SIZE );

    T scale = 100;
    for( T & value : arrayOfMatrices )
    { value = randomValue( scale, m_gen ); }

    ArrayViewT< T const, 2, 1 > const matrices = arrayOfMatrices.toViewConst();

    forall< POLICY >( matrices.size( 0 ), [=] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          T initialMatrix[ SYM_SIZE ];
          T srcLocal[ SYM_SIZE ];
          T det;

          #define _TEST( matrix ) \
            tensorOps::copy< SYM_SIZE >( initialMatrix, matrices[ i ] ); \
            tensorOps::copy< SYM_SIZE >( matrix, initialMatrix ); \
            det = tensorOps::symInvert< M >( matrix ); \
            checkSymInverse( scale, det, matrix, initialMatrix )

          _TEST( srcSymMatrix_IJ[ i ] );
          _TEST( srcSymMatrix_JI[ i ] );
          _TEST( srcLocal );

      #undef _TEST
        } );
  }


private:

  template< typename MATRIX >
  static void LVARRAY_HOST_DEVICE checkDeterminant( T const scale, MATRIX && matrix )
  {
    T const originalDet = tensorOps::determinant< M >( matrix );

    // If T is integral then there should be no error.
    double const epsilon = ( std::is_floating_point< T >::value ) ? NumericLimitsNC< T >{}.epsilon : NumericLimitsNC< double >{}.min;
    double const epsilonScale3 = epsilon * scale * scale * scale;
    double const epsilonScale6 = epsilonScale3 * scale * scale * scale;

    // The determinant of the product of two matrices is the product of their determinants.
    T copy[ M ][ M ];
    T result[ M ][ M ];
    tensorOps::copy< M, M >( copy, matrix );
    tensorOps::Rij_eq_AikBkj< M, M, M >( result, copy, matrix );

    PORTABLE_EXPECT_NEAR( originalDet * originalDet, tensorOps::determinant< M >( result ), 4 * epsilonScale6 );

    // The transpose has the same determinant.
    tensorOps::transpose< M >( matrix );
    PORTABLE_EXPECT_NEAR( originalDet, tensorOps::determinant< M >( matrix ), 3 * epsilonScale3 );

    // Adding one row to another doesn't change the determinant.
    tensorOps::scaledAdd< M >( matrix[ 0 ], matrix[ 1 ], 2 );
    PORTABLE_EXPECT_NEAR( originalDet, tensorOps::determinant< M >( matrix ), 3 * epsilonScale3 );

    // Setting one row equal to another makes the determinant zero.
    tensorOps::copy< M >( matrix[ 1 ], matrix[ 0 ] );
    PORTABLE_EXPECT_NEAR( 0, tensorOps::determinant< M >( matrix ), 5 * epsilonScale3 );

    LVARRAY_UNUSED_VARIABLE( originalDet );
    LVARRAY_UNUSED_VARIABLE( epsilonScale3 );
  }

  template< typename MATRIX >
  static void LVARRAY_HOST_DEVICE checkSymDeterminant( T const scale, MATRIX && matrix )
  {
    FLOAT const epsilon = NumericLimitsNC< FLOAT >{}.epsilon;

    FLOAT dense[ M ][ M ];
    tensorOps::symmetricToDense< M >( dense, matrix );

    PORTABLE_EXPECT_NEAR( tensorOps::symDeterminant< M >( matrix ),
                          tensorOps::determinant< M >( dense ),
                          3 * scale * scale * scale * epsilon )
  }

  template< typename MATRIX_A, typename MATRIX_B >
  static void LVARRAY_HOST_DEVICE checkInverse( T const scale,
                                                T const det,
                                                MATRIX_A const & inverse,
                                                MATRIX_B const & source )
  {
    FLOAT const epsilon = NumericLimitsNC< FLOAT >{}.epsilon;

    // The bounds for this specific check need to be increased a lot for XL. About 100x for
    // 2x2 even more for 3x3. I'm not sure why, especially since the check below passes.
    #if !defined( __ibmxl__ ) || defined( __CUDA_ARCH__ )
    PORTABLE_EXPECT_NEAR( det, tensorOps::determinant< M >( source ), scale * epsilon );
    #endif

    PORTABLE_EXPECT_NEAR( 1.0 / det, tensorOps::determinant< M >( inverse ), scale * epsilon );

    FLOAT product[ M ][ M ];
    tensorOps::Rij_eq_AikBkj< M, M, M >( product, inverse, source );

    for( int i = 0; i < M; ++i )
    {
      for( int j = 0; j < M; ++j )
      { PORTABLE_EXPECT_NEAR( product[ i ][ j ], i == j, 5 * scale * epsilon ); }
    }
  }

  template< typename MATRIX_A, typename MATRIX_B >
  static void LVARRAY_HOST_DEVICE checkSymInverse( T const scale,
                                                   T const det,
                                                   MATRIX_A const & inverse,
                                                   MATRIX_B const & source )
  {
    FLOAT const epsilon = NumericLimitsNC< FLOAT >{}.epsilon;

    PORTABLE_EXPECT_NEAR( det, tensorOps::symDeterminant< M >( source ), 2.5 * scale * scale * scale * epsilon );
    PORTABLE_EXPECT_NEAR( 1.0 / det, tensorOps::symDeterminant< M >( inverse ), scale * epsilon );

    FLOAT denseInverse[ M ][ M ];
    tensorOps::symmetricToDense< M >( denseInverse, inverse );

    FLOAT denseSource[ M ][ M ];
    tensorOps::symmetricToDense< M >( denseSource, source );

    FLOAT product[ M ][ M ];
    tensorOps::Rij_eq_AikBkj< M, M, M >( product, denseInverse, denseSource );

    for( int i = 0; i < M; ++i )
    {
      for( int j = 0; j < M; ++j )
      { PORTABLE_EXPECT_NEAR( product[ i ][ j ], i == j, 4 * scale * epsilon ); }
    }
  }


  std::mt19937_64 m_gen;

  static constexpr INDEX_TYPE const numMatrices = 100;

  ArrayT< T, RAJA::PERM_IJK > m_srcMatrix_IJK { numMatrices, M, M };
  ArrayT< T, RAJA::PERM_IKJ > m_srcMatrix_IKJ { numMatrices, M, M };
  ArrayT< T, RAJA::PERM_KJI > m_srcMatrix_KJI { numMatrices, M, M };

  ArrayT< FLOAT, RAJA::PERM_IJK > m_dstMatrix_IJK { numMatrices, M, M };
  ArrayT< FLOAT, RAJA::PERM_IKJ > m_dstMatrix_IKJ { numMatrices, M, M };
  ArrayT< FLOAT, RAJA::PERM_KJI > m_dstMatrix_KJI { numMatrices, M, M };

  ArrayT< T, RAJA::PERM_IJ > m_srcSymMatrix_IJ { numMatrices, SYM_SIZE };
  ArrayT< T, RAJA::PERM_JI > m_srcSymMatrix_JI { numMatrices, SYM_SIZE };

  ArrayT< FLOAT, RAJA::PERM_IJ > m_dstSymMatrix_IJ { numMatrices, SYM_SIZE};
  ArrayT< FLOAT, RAJA::PERM_JI > m_dstSymMatrix_JI { numMatrices, SYM_SIZE };
};

using InverseTestTypes = ::testing::Types<
  std::tuple< double, double, std::integral_constant< int, 2 >, serialPolicy >
  , std::tuple< float, float, std::integral_constant< int, 2 >, serialPolicy >
  , std::tuple< int, double, std::integral_constant< int, 2 >, serialPolicy >
  , std::tuple< double, double, std::integral_constant< int, 3 >, serialPolicy >
  , std::tuple< float, float, std::integral_constant< int, 3 >, serialPolicy >
  , std::tuple< int, double, std::integral_constant< int, 3 >, serialPolicy >

#if ( defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP) ) && defined(LVARRAY_USE_CHAI)
  , std::tuple< double, double, std::integral_constant< int, 2 >, parallelDevicePolicy< 32 > >
  , std::tuple< float, float, std::integral_constant< int, 2 >, parallelDevicePolicy< 32 > >
  , std::tuple< int, double, std::integral_constant< int, 2 >, parallelDevicePolicy< 32 > >
  , std::tuple< double, double, std::integral_constant< int, 3 >, parallelDevicePolicy< 32 > >
  , std::tuple< float, float, std::integral_constant< int, 3 >, parallelDevicePolicy< 32 > >
  , std::tuple< int, double, std::integral_constant< int, 3 >, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( InverseTest, InverseTestTypes, );


template< typename ARGS >
class InverseFloatOnlyTest : public InverseTest< ARGS >
{
  static_assert( std::is_same< typename InverseTest< ARGS >::T, typename InverseTest< ARGS >::FLOAT >::value, "should be the same." );
};

using InverseFloatOnlyTestTypes = ::testing::Types<
  std::tuple< double, double, std::integral_constant< int, 2 >, serialPolicy >
  , std::tuple< float, float, std::integral_constant< int, 2 >, serialPolicy >
  , std::tuple< double, double, std::integral_constant< int, 3 >, serialPolicy >
  , std::tuple< float, float, std::integral_constant< int, 3 >, serialPolicy >

#if ( defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP) ) && defined(LVARRAY_USE_CHAI)
  , std::tuple< double, double, std::integral_constant< int, 2 >, parallelDevicePolicy< 32 > >
  , std::tuple< float, float, std::integral_constant< int, 2 >, parallelDevicePolicy< 32 > >
  , std::tuple< double, double, std::integral_constant< int, 3 >, parallelDevicePolicy< 32 > >
  , std::tuple< float, float, std::integral_constant< int, 3 >, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( InverseFloatOnlyTest, InverseFloatOnlyTestTypes, );

} // namespace testing
} // namespace LvArray
