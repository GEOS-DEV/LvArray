/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "Array.hpp"
#include "testUtils.hpp"

// TPL inclues
#include <gtest/gtest.h>

namespace LvArray
{
namespace testing
{

using INDEX_TYPE = std::ptrdiff_t;

template< typename T, typename PERMUTATION >
using ArrayT = Array< T, typeManipulation::getDimension< PERMUTATION >, PERMUTATION, INDEX_TYPE, DEFAULT_BUFFER >;

template< typename T >
void check( ArraySlice< T const, 1, 0, INDEX_TYPE > const slice )
{
  INDEX_TYPE offset = 0;
  for( INDEX_TYPE i = 0; i < slice.size( 0 ); ++i )
  {
    EXPECT_EQ( slice( i ), offset++ );
  }
}

template< typename T, int USD >
void check( ArraySlice< T const, 2, USD, INDEX_TYPE > const slice )
{
  INDEX_TYPE offset = 0;
  for( INDEX_TYPE i = 0; i < slice.size( 0 ); ++i )
  {
    for( INDEX_TYPE j = 0; j < slice.size( 1 ); ++j )
    {
      EXPECT_EQ( slice( i, j ), offset++ );
    }
  }
}

template< typename T, int USD >
void check( ArraySlice< T const, 3, USD, INDEX_TYPE > const slice )
{
  INDEX_TYPE offset = 0;
  for( INDEX_TYPE i = 0; i < slice.size( 0 ); ++i )
  {
    for( INDEX_TYPE j = 0; j < slice.size( 1 ); ++j )
    {
      for( INDEX_TYPE k = 0; k < slice.size( 2 ); ++k )
      {
        EXPECT_EQ( slice( i, j, k ), offset++ );
      }
    }
  }
}

template< typename T, int USD >
void check( ArraySlice< T const, 4, USD, INDEX_TYPE > const slice )
{
  INDEX_TYPE offset = 0;
  for( INDEX_TYPE i = 0; i < slice.size( 0 ); ++i )
  {
    for( INDEX_TYPE j = 0; j < slice.size( 1 ); ++j )
    {
      for( INDEX_TYPE k = 0; k < slice.size( 2 ); ++k )
      {
        for( INDEX_TYPE l = 0; l < slice.size( 3 ); ++l )
        {
          EXPECT_EQ( slice( i, j, k, l ), offset++ );
        }
      }
    }
  }
}

template< typename ARRAY >
class ForValuesInSlice : public ::testing::Test
{
public:
  static constexpr int NDIM = ARRAY::NDIM;

  void test()
  {
    initialize();

    check( m_array.toSliceConst() );

    INDEX_TYPE offset = 0;
    forValuesInSlice( m_array.toSliceConst(), [&offset] ( auto const & val )
    {
      EXPECT_EQ( val, offset++ );
    } );

    forValuesInSliceWithIndices( m_array.toSliceConst(), [slice=m_array.toSliceConst()] ( auto const & val, auto const ... indices )
    {
      EXPECT_EQ( &val, &slice( indices ... ) );
    } );
  }

protected:

  void initialize()
  {
    INDEX_TYPE dims[ NDIM ];
    for( int dim = 0; dim < NDIM; ++dim )
    {
      dims[ dim ] = 10 + dim;
    }

    m_array.resize( NDIM, dims );

    INDEX_TYPE offset = 0;
    forValuesInSlice( m_array.toSlice(), [&offset] ( auto & val )
    {
      val = offset++;
    } );
  }

  ARRAY m_array;
};

using ForValuesInSliceTypes = ::testing::Types<
  // All 1D permutations
  ArrayT< int, RAJA::PERM_I >
  // All 2D permutations
  , ArrayT< int, RAJA::PERM_IJ >
  , ArrayT< int, RAJA::PERM_JI >
  // All 3D permutations
  , ArrayT< int, RAJA::PERM_IJK >
  , ArrayT< int, RAJA::PERM_IKJ >
  , ArrayT< int, RAJA::PERM_JIK >
  , ArrayT< int, RAJA::PERM_JKI >
  , ArrayT< int, RAJA::PERM_KIJ >
  , ArrayT< int, RAJA::PERM_KJI >
  // Some 4D permutations
  , ArrayT< int, RAJA::PERM_IJKL >
  , ArrayT< int, RAJA::PERM_LKJI >
  , ArrayT< int, RAJA::PERM_IKLJ >
  , ArrayT< int, RAJA::PERM_KLIJ >
  >;

TYPED_TEST_SUITE( ForValuesInSlice, ForValuesInSliceTypes, );

TYPED_TEST( ForValuesInSlice, test )
{
  this->test();
}

TEST( ForValuesInSlice, scalar )
{
  int x;
  forValuesInSlice( x, []( int & val )
  {
    val = 5;
  } );

  EXPECT_EQ( x, 5 );
}


template< typename T, int USD_SRC >
void checkSums( ArraySlice< T const, 2, USD_SRC, INDEX_TYPE > const src,
                ArraySlice< T const, 1, 0, INDEX_TYPE > const sums )
{
  for( INDEX_TYPE j = 0; j < src.size( 1 ); ++j )
  {
    T sum {};
    for( INDEX_TYPE i = 0; i < src.size( 0 ); ++i )
    {
      sum += src( i, j );
    }

    EXPECT_EQ( sum, sums( j ) );
  }
}

template< typename T, int USD_SRC, int USD_SUMS >
void checkSums( ArraySlice< T const, 3, USD_SRC, INDEX_TYPE > const src,
                ArraySlice< T const, 2, USD_SUMS, INDEX_TYPE > const sums )
{
  for( INDEX_TYPE j = 0; j < src.size( 1 ); ++j )
  {
    for( INDEX_TYPE k = 0; k < src.size( 2 ); ++k )
    {
      T sum {};
      for( INDEX_TYPE i = 0; i < src.size( 0 ); ++i )
      {
        sum += src( i, j, k );
      }

      EXPECT_EQ( sum, sums( j, k ) );
    }
  }
}

template< typename T, int USD_SRC, int USD_SUMS >
void checkSums( ArraySlice< T const, 4, USD_SRC, INDEX_TYPE > const src,
                ArraySlice< T const, 3, USD_SUMS, INDEX_TYPE > const sums )
{
  for( INDEX_TYPE j = 0; j < src.size( 1 ); ++j )
  {
    for( INDEX_TYPE k = 0; k < src.size( 2 ); ++k )
    {
      for( INDEX_TYPE l = 0; l < src.size( 3 ); ++l )
      {
        T sum {};
        for( INDEX_TYPE i = 0; i < src.size( 0 ); ++i )
        {
          sum += src( i, j, k, l );
        }

        EXPECT_EQ( sum, sums( j, k, l ) );
      }
    }
  }
}

template< typename ARRAY_PERM_PAIR >
class SumOverFirstDimension : public ForValuesInSlice< typename ARRAY_PERM_PAIR::first_type >
{
public:
  using ARRAY = typename ARRAY_PERM_PAIR::first_type;
  using T = typename ARRAY::value_type;
  static constexpr int NDIM = ARRAY::NDIM;

  using PERM = typename ARRAY_PERM_PAIR::second_type;

  void test()
  {
    this->initialize();
    INDEX_TYPE dims[ NDIM - 1 ];
    for( INDEX_TYPE dim = 0; dim < NDIM - 1; ++dim )
    {
      dims[ dim ] = this->m_array.size( dim + 1 );
    }

    m_sums.resize( NDIM - 1, dims );

    sumOverFirstDimension( this->m_array.toSliceConst(), m_sums.toSlice() );

    checkSums( this->m_array.toSliceConst(), m_sums.toSliceConst() );
  }

private:
  ArrayT< T, PERM > m_sums;
};

using SumOverFirstDimensionTypes = ::testing::Types<
  // All 2D permutations by all 1D permutations
  std::pair< ArrayT< int, RAJA::PERM_IJ >, RAJA::PERM_I >
  , std::pair< ArrayT< int, RAJA::PERM_IJ >, RAJA::PERM_I >
  // All 3D permutations by RAJA::PERM_IJ
  , std::pair< ArrayT< int, RAJA::PERM_IJK >, RAJA::PERM_IJ >
  , std::pair< ArrayT< int, RAJA::PERM_IKJ >, RAJA::PERM_IJ >
  , std::pair< ArrayT< int, RAJA::PERM_JIK >, RAJA::PERM_IJ >
  , std::pair< ArrayT< int, RAJA::PERM_JKI >, RAJA::PERM_IJ >
  , std::pair< ArrayT< int, RAJA::PERM_KIJ >, RAJA::PERM_IJ >
  , std::pair< ArrayT< int, RAJA::PERM_KJI >, RAJA::PERM_IJ >
  // All 3D permutations by RAJA::PERM_JI
  , std::pair< ArrayT< int, RAJA::PERM_IJK >, RAJA::PERM_JI >
  , std::pair< ArrayT< int, RAJA::PERM_IKJ >, RAJA::PERM_JI >
  , std::pair< ArrayT< int, RAJA::PERM_JIK >, RAJA::PERM_JI >
  , std::pair< ArrayT< int, RAJA::PERM_JKI >, RAJA::PERM_JI >
  , std::pair< ArrayT< int, RAJA::PERM_KIJ >, RAJA::PERM_JI >
  , std::pair< ArrayT< int, RAJA::PERM_KJI >, RAJA::PERM_JI >
  // Some 4D permutations by some 3D permutations
  , std::pair< ArrayT< int, RAJA::PERM_IJKL >, RAJA::PERM_IJK >
  , std::pair< ArrayT< int, RAJA::PERM_LKJI >, RAJA::PERM_KJI >
  , std::pair< ArrayT< int, RAJA::PERM_JKIL >, RAJA::PERM_JIK >
  , std::pair< ArrayT< int, RAJA::PERM_KILJ >, RAJA::PERM_IKJ >
  , std::pair< ArrayT< int, RAJA::PERM_LJIK >, RAJA::PERM_KIJ >
  , std::pair< ArrayT< int, RAJA::PERM_ILJK >, RAJA::PERM_IJK >
  >;

TYPED_TEST_SUITE( SumOverFirstDimension, SumOverFirstDimensionTypes, );

TYPED_TEST( SumOverFirstDimension, test )
{
  this->test();
}

TEST( SumOverFirstDimension, OneDimensional )
{
  INDEX_TYPE const size = 10;
  ArrayT< int, RAJA::PERM_I > array( size );

  for( INDEX_TYPE i = 0; i < size; ++i )
  {
    array( i ) = i + 1;
  }

  int sum = 0;
  sumOverFirstDimension( array.toSliceConst(), sum );
  EXPECT_EQ( sum, ( size * size + size ) / 2 );
}


} /// namespace testing
} /// namespace LvArray
