/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "Array.hpp"
#include "output.hpp"
#include "testUtils.hpp"
#include "MallocBuffer.hpp"

// TPL includes
#include <gtest/gtest.h>

// System includes
#include <random>

namespace LvArray
{
namespace testing
{

using INDEX_TYPE = std::ptrdiff_t;

inline INDEX_TYPE randomInteger( INDEX_TYPE const min, INDEX_TYPE const max )
{
  static std::mt19937_64 gen;
  return std::uniform_int_distribution< INDEX_TYPE >( min, max )( gen );
}
template< typename ARRAY1D >
class Array1DTest : public ::testing::Test
{
public:
  using T = typename ARRAY1D::value_type;

  void compareToReference() const
  {
    EXPECT_EQ( m_array.size(), m_ref.size() );
    EXPECT_EQ( m_array.empty(), m_ref.empty() );
    if( m_array.empty() )
    {
      EXPECT_EQ( m_array.size(), 0 );
      return;
    }

    for( INDEX_TYPE i = 0; i < m_array.size(); ++i )
    {
      EXPECT_EQ( m_array[ i ], m_ref[ i ] );
      EXPECT_EQ( m_array( i ), m_ref[ i ] );
    }

    const T * ptr = m_array.data();
    const T * refPtr = m_ref.data();
    for( INDEX_TYPE i = 0; i < m_array.size(); ++i )
    {
      EXPECT_EQ( ptr[ i ], refPtr[ i ] );
    }
  }

  #define COMPARE_TO_REFERENCE { SCOPED_TRACE( "" ); compareToReference(); \
  }

  void emplace_back( INDEX_TYPE const nVals )
  {
    COMPARE_TO_REFERENCE

      LVARRAY_ERROR_IF_NE( nVals % 2, 0 );

    for( INDEX_TYPE i = 0; i < nVals / 2; ++i )
    {
      T const val( randomInteger( 0, 1000 ) );
      m_array.emplace_back( val );
      m_ref.emplace_back( val );

      INDEX_TYPE const seed = randomInteger( 0, 1000 );
      m_array.emplace_back( seed );
      m_ref.emplace_back( seed );
    }

    COMPARE_TO_REFERENCE
  }

  void emplace( INDEX_TYPE const nToInsert )
  {
    COMPARE_TO_REFERENCE

      LVARRAY_ERROR_IF_NE( nToInsert % 2, 0 );

    for( INDEX_TYPE i = 0; i < nToInsert / 2; ++i )
    {
      {
        T const val( randomInteger( 0, 1000 ) );
        INDEX_TYPE const position = randomInteger( 0, m_array.size() );
        m_array.emplace( position, val );
        m_ref.emplace( m_ref.begin() + position, val );
      }

      {
        INDEX_TYPE const seed = randomInteger( 0, 1000 );
        INDEX_TYPE const position = randomInteger( 0, m_array.size() );
        m_array.emplace( position, seed );
        m_ref.emplace( m_ref.begin() + position, seed );
      }
    }

    COMPARE_TO_REFERENCE
  }

  void insert( INDEX_TYPE const nToInsert )
  {
    COMPARE_TO_REFERENCE

    constexpr INDEX_TYPE MAX_VALS_PER_INSERT = 20;

    INDEX_TYPE nInserted = 0;
    while( nInserted < nToInsert )
    {
      // Insert from a std::vector
      {
        INDEX_TYPE const nToInsertThisIter = randomInteger( 0, math::min( MAX_VALS_PER_INSERT, nToInsert - nInserted ) );
        std::vector< T > valsToInsert( nToInsertThisIter );

        for( T & val : valsToInsert )
        { val = T( randomInteger( 0, 1000 ) ); }

        INDEX_TYPE const position = randomInteger( 0, m_array.size() );
        m_array.insert( position, valsToInsert.begin(), valsToInsert.end() );
        m_ref.insert( m_ref.begin() + position, valsToInsert.begin(), valsToInsert.end() );

        nInserted += nToInsertThisIter;
      }

      // Insert from a std::list
      {
        INDEX_TYPE const nToInsertThisIter = randomInteger( 0, math::min( MAX_VALS_PER_INSERT, nToInsert - nInserted ) );
        std::list< T > valsToInsert( nToInsertThisIter );
        for( T & val : valsToInsert )
        { val = T( randomInteger( 0, 1000 ) ); }

        INDEX_TYPE const position = randomInteger( 0, m_array.size() );
        m_array.insert( position, valsToInsert.begin(), valsToInsert.end() );
        m_ref.insert( m_ref.begin() + position, valsToInsert.begin(), valsToInsert.end() );

        nInserted += nToInsertThisIter;
      }
    }

    LVARRAY_ERROR_IF_NE( nInserted, nToInsert );
    COMPARE_TO_REFERENCE
  }

  void pop_back( INDEX_TYPE const nToPop )
  {
    COMPARE_TO_REFERENCE

      LVARRAY_ERROR_IF_GT( nToPop, m_array.size() );

    for( INDEX_TYPE i = 0; i < nToPop; ++i )
    {
      m_array.pop_back();
      m_ref.pop_back();

      if( i % 10 == 0 )
      { COMPARE_TO_REFERENCE }
    }

    COMPARE_TO_REFERENCE
  }

  void erase( std::ptrdiff_t const nToErase )
  {
    COMPARE_TO_REFERENCE

      LVARRAY_ERROR_IF_GT( nToErase, m_array.size() );

    for( std::ptrdiff_t i = 0; i < nToErase; ++i )
    {
      std::ptrdiff_t const position = randomInteger( 0, m_array.size() - 1 );
      m_array.erase( position );
      m_ref.erase( m_ref.begin() + position );
    }

    COMPARE_TO_REFERENCE
  }

protected:

  ARRAY1D m_array;
  std::vector< T > m_ref;

  #undef COMPARE_TO_REFERENCE
};

using Array1DTestTypes = ::testing::Types<
  Array< int, 1, RAJA::PERM_I, INDEX_TYPE, MallocBuffer >
  , Array< Tensor, 1, RAJA::PERM_I, INDEX_TYPE, MallocBuffer >
  , Array< TestString, 1, RAJA::PERM_I, INDEX_TYPE, MallocBuffer >
#if defined(LVARRAY_USE_CHAI)
  , Array< int, 1, RAJA::PERM_I, INDEX_TYPE, ChaiBuffer >
  , Array< Tensor, 1, RAJA::PERM_I, INDEX_TYPE, ChaiBuffer >
  , Array< TestString, 1, RAJA::PERM_I, INDEX_TYPE, ChaiBuffer >
#endif
  >;

TYPED_TEST_SUITE( Array1DTest, Array1DTestTypes, );

TYPED_TEST( Array1DTest, emplace_back )
{
  this->emplace_back( 1000 );
}

TYPED_TEST( Array1DTest, emplace )
{
  this->emplace( 100 );
}

TYPED_TEST( Array1DTest, pop_back )
{
  this->emplace_back( 1000 );
  this->pop_back( 400 );
}

TYPED_TEST( Array1DTest, erase )
{
  this->emplace_back( 1000 );
  this->erase( 400 );
}

TYPED_TEST( Array1DTest, combinedOperations )
{
  for( int i = 0; i < 3; ++i )
  {
    this->emplace_back( 100 );
    this->erase( 25 );
    this->emplace( 50 );
    this->pop_back( 50 );
    this->erase( 75 );
  }
}

} // namespace testing
} // namespace LvArray

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
