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

/// Source includes
#include "Array.hpp"

/// TPL inclues
#include <gtest/gtest.h>

namespace LvArray
{
namespace testing
{

template< typename T >
void check( ArraySlice< T const, 1, 0 > const & view )
{
  std::ptrdiff_t offset = 0;
  for( std::ptrdiff_t i = 0; i < view.size( 0 ); ++i )
  {
    EXPECT_EQ( view( i ), offset++ );
  }
}

template< typename T, int USD >
void check( ArraySlice< T const, 2, USD > const & view )
{
  std::ptrdiff_t offset = 0;
  for( std::ptrdiff_t i = 0; i < view.size( 0 ); ++i )
  {
    for( std::ptrdiff_t j = 0; j < view.size( 1 ); ++j )
    {
      EXPECT_EQ( view( i, j ), offset++ );
    }
  }
}

template< typename T, int USD >
void check( ArraySlice< T const, 3, USD > const & view )
{
  std::ptrdiff_t offset = 0;
  for( std::ptrdiff_t i = 0; i < view.size( 0 ); ++i )
  {
    for( std::ptrdiff_t j = 0; j < view.size( 1 ); ++j )
    {
      for( std::ptrdiff_t k = 0; k < view.size( 2 ); ++k )
      {
        EXPECT_EQ( view( i, j, k ), offset++ );
      }
    }
  }
}

template< typename T, int USD >
void check( ArraySlice< T const, 4, USD > const & view )
{
  std::ptrdiff_t offset = 0;
  for( std::ptrdiff_t i = 0; i < view.size( 0 ); ++i )
  {
    for( std::ptrdiff_t j = 0; j < view.size( 1 ); ++j )
    {
      for( std::ptrdiff_t k = 0; k < view.size( 2 ); ++k )
      {
        for( std::ptrdiff_t l = 0; l < view.size( 3 ); ++l )
        {
          EXPECT_EQ( view( i, j, k, l ), offset++ );
        }
      }
    }
  }
}

template< typename ARRAY >
class ForValuesInSlice : public ::testing::Test
{
public:
  static constexpr int NDIM = ARRAY::ndim;

  void test()
  {
    initialize();

    check( m_array.toSliceConst() );

    std::ptrdiff_t offset = 0;
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
    std::ptrdiff_t dims[ NDIM ];
    for( int dim = 0; dim < NDIM; ++dim )
    {
      dims[ dim ] = 10 + dim;
    }

    m_array.resize( NDIM, dims );

    std::ptrdiff_t offset = 0;
    forValuesInSlice( m_array.toSlice(), [&offset] ( auto & val )
    {
      val = offset++;
    } );
  }

  ARRAY m_array;
};

using ForValuesInSliceTypes = ::testing::Types<
  // All 1D permutations
  Array< int, 1 >
  // All 2D permutations
  , Array< int, 2, RAJA::PERM_IJ >
  , Array< int, 2, RAJA::PERM_JI >
  // All 3D permutations
  , Array< int, 3, RAJA::PERM_IJK >
  , Array< int, 3, RAJA::PERM_IKJ >
  , Array< int, 3, RAJA::PERM_JIK >
  , Array< int, 3, RAJA::PERM_JKI >
  , Array< int, 3, RAJA::PERM_KIJ >
  , Array< int, 3, RAJA::PERM_KJI >
  // Some 4D permutations
  , Array< int, 4, RAJA::PERM_IJKL >
  , Array< int, 4, RAJA::PERM_LKJI >
  , Array< int, 4, RAJA::PERM_IKLJ >
  , Array< int, 4, RAJA::PERM_KLIJ >
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
void checkSums( ArraySlice< T const, 2, USD_SRC > const & src, ArraySlice< T const, 1, 0 > const & sums )
{
  for( std::ptrdiff_t j = 0; j < src.size( 1 ); ++j )
  {
    T sum {};
    for( std::ptrdiff_t i = 0; i < src.size( 0 ); ++i )
    {
      sum += src( i, j );
    }

    EXPECT_EQ( sum, sums( j ) );
  }
}

template< typename T, int USD_SRC, int USD_SUMS >
void checkSums( ArraySlice< T const, 3, USD_SRC > const & src, ArraySlice< T const, 2, USD_SUMS > const & sums )
{
  for( std::ptrdiff_t j = 0; j < src.size( 1 ); ++j )
  {
    for( std::ptrdiff_t k = 0; k < src.size( 2 ); ++k )
    {
      T sum {};
      for( std::ptrdiff_t i = 0; i < src.size( 0 ); ++i )
      {
        sum += src( i, j, k );
      }

      EXPECT_EQ( sum, sums( j, k ) );
    }
  }
}

template< typename T, int USD_SRC, int USD_SUMS >
void checkSums( ArraySlice< T const, 4, USD_SRC > const & src, ArraySlice< T const, 3, USD_SUMS > const & sums )
{
  for( std::ptrdiff_t j = 0; j < src.size( 1 ); ++j )
  {
    for( std::ptrdiff_t k = 0; k < src.size( 2 ); ++k )
    {
      for( std::ptrdiff_t l = 0; l < src.size( 3 ); ++l )
      {
        T sum {};
        for( std::ptrdiff_t i = 0; i < src.size( 0 ); ++i )
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
  static constexpr int NDIM = ARRAY::ndim;

  using PERM = typename ARRAY_PERM_PAIR::second_type;

  void test()
  {
    this->initialize();
    std::ptrdiff_t dims[ NDIM - 1 ];
    for( std::ptrdiff_t dim = 0; dim < NDIM - 1; ++dim )
    {
      dims[ dim ] = this->m_array.size( dim + 1 );
    }

    m_sums.resize( NDIM - 1, dims );

    sumOverFirstDimension( this->m_array.toSliceConst(), m_sums.toSlice() );

    checkSums( this->m_array.toSliceConst(), m_sums.toSliceConst() );
  }

private:
  Array< T, NDIM - 1, PERM > m_sums;
};

using SumOverFirstDimensionTypes = ::testing::Types<
  // All 2D permutations by all 1D permutations
  std::pair< Array< int, 2, RAJA::PERM_IJ >, RAJA::PERM_I >
  , std::pair< Array< int, 2, RAJA::PERM_IJ >, RAJA::PERM_I >
  // All 3D permutations by RAJA::PERM_IJ
  , std::pair< Array< int, 3, RAJA::PERM_IJK >, RAJA::PERM_IJ >
  , std::pair< Array< int, 3, RAJA::PERM_IKJ >, RAJA::PERM_IJ >
  , std::pair< Array< int, 3, RAJA::PERM_JIK >, RAJA::PERM_IJ >
  , std::pair< Array< int, 3, RAJA::PERM_JKI >, RAJA::PERM_IJ >
  , std::pair< Array< int, 3, RAJA::PERM_KIJ >, RAJA::PERM_IJ >
  , std::pair< Array< int, 3, RAJA::PERM_KJI >, RAJA::PERM_IJ >
  // All 3D permutations by RAJA::PERM_JI
  , std::pair< Array< int, 3, RAJA::PERM_IJK >, RAJA::PERM_JI >
  , std::pair< Array< int, 3, RAJA::PERM_IKJ >, RAJA::PERM_JI >
  , std::pair< Array< int, 3, RAJA::PERM_JIK >, RAJA::PERM_JI >
  , std::pair< Array< int, 3, RAJA::PERM_JKI >, RAJA::PERM_JI >
  , std::pair< Array< int, 3, RAJA::PERM_KIJ >, RAJA::PERM_JI >
  , std::pair< Array< int, 3, RAJA::PERM_KJI >, RAJA::PERM_JI >
  // Some 4D permutations by some 3D permutations
  , std::pair< Array< int, 4, RAJA::PERM_IJKL >, RAJA::PERM_IJK >
  , std::pair< Array< int, 4, RAJA::PERM_LKJI >, RAJA::PERM_KJI >
  , std::pair< Array< int, 4, RAJA::PERM_JKIL >, RAJA::PERM_JIK >
  , std::pair< Array< int, 4, RAJA::PERM_KILJ >, RAJA::PERM_IKJ >
  , std::pair< Array< int, 4, RAJA::PERM_LJIK >, RAJA::PERM_KIJ >
  , std::pair< Array< int, 4, RAJA::PERM_ILJK >, RAJA::PERM_IJK >
  >;

TYPED_TEST_SUITE( SumOverFirstDimension, SumOverFirstDimensionTypes, );

TYPED_TEST( SumOverFirstDimension, test )
{
  this->test();
}

TEST( SumOverFirstDimension, OneDimensional )
{
  std::ptrdiff_t const size = 10;
  Array< int, 1 > array( size );

  for( std::ptrdiff_t i = 0; i < size; ++i )
  {
    array( i ) = i + 1;
  }

  int sum = 0;
  sumOverFirstDimension( array.toSliceConst(), sum );
  EXPECT_EQ( sum, ( size * size + size ) / 2 );
}


} /// namespace testing
} /// namespace LvArray
