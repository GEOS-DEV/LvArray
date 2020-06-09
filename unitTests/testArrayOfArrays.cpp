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
#include "ArrayOfArrays.hpp"
#include "testUtils.hpp"
#include "Array.hpp"
#include "MallocBuffer.hpp"

// TPL includes
#include <gtest/gtest.h>

// System includes
#include <vector>
#include <random>

namespace LvArray
{
namespace testing
{

using INDEX_TYPE = std::ptrdiff_t;

// This is for debugging.
template< typename ARRAY_OF_ARRAYS >
void print( ARRAY_OF_ARRAYS const & array )
{
  std::cout << "{" << std::endl;

  for( INDEX_TYPE i = 0; i < array.size(); ++i )
  {
    std::cout << "\t{";
    for( INDEX_TYPE j = 0; j < array.sizeOfArray( i ); ++j )
    {
      std::cout << array[ i ][ j ] << ", ";
    }

    for( INDEX_TYPE j = array.sizeOfArray( i ); j < array.capacityOfArray( i ); ++j )
    {
      std::cout << "X" << ", ";
    }

    std::cout << "}" << std::endl;
  }

  std::cout << "}" << std::endl;
}

template< class ARRAY_OF_ARRAYS >
class ArrayOfArraysTest : public ::testing::Test
{
public:
  using T = typename ARRAY_OF_ARRAYS::value_type;

  using ViewType = std::remove_reference_t< decltype( std::declval< ARRAY_OF_ARRAYS >().toView() ) >;
  using ViewTypeConstSizes = std::remove_reference_t< decltype( std::declval< ARRAY_OF_ARRAYS >().toViewConstSizes() ) >;
  using ViewTypeConst = std::remove_reference_t< decltype( std::declval< ARRAY_OF_ARRAYS >().toViewConst() ) >;

  void compareToReference( ViewTypeConst const & view ) const
  {
    ASSERT_EQ( view.size(), m_ref.size());

    for( INDEX_TYPE i = 0; i < view.size(); ++i )
    {
      ASSERT_EQ( view.sizeOfArray( i ), m_ref[ i ].size());

      for( INDEX_TYPE j = 0; j < view.sizeOfArray( i ); ++j )
      {
        EXPECT_EQ( view[ i ][ j ], m_ref[ i ][ j ] );
        EXPECT_EQ( view( i, j ), view[ i ][ j ] );
      }

      INDEX_TYPE j = 0;
      for( T const & val : view[ i ] )
      {
        EXPECT_EQ( val, m_ref[ i ][ j ] );
        ++j;
      }

      EXPECT_EQ( j, view.sizeOfArray( i ));
    }
  }

  #define COMPARE_TO_REFERENCE { SCOPED_TRACE( "" ); this->compareToReference( m_array.toViewConst() ); \
  }

  void appendArray( INDEX_TYPE const nArrays, INDEX_TYPE const maxAppendSize )
  {
    COMPARE_TO_REFERENCE;

    LVARRAY_ERROR_IF_NE( nArrays % 2, 0 );

    for( INDEX_TYPE i = 0; i < nArrays / 2; ++i )
    {
      // Append using a std::vector.
      {
        INDEX_TYPE const nValues = rand( 0, maxAppendSize );

        std::vector< T > toAppend;
        for( INDEX_TYPE j = 0; j < nValues; ++j )
        { toAppend.emplace_back( 2 * i * LARGE_NUMBER + j ); }

        m_array.appendArray( toAppend.begin(), toAppend.end() );
        m_ref.emplace_back( toAppend.begin(), toAppend.end() );
      }

      // Append using a std::list.
      {
        INDEX_TYPE const nValues = rand( 0, maxAppendSize );

        std::list< T > toAppend;
        for( INDEX_TYPE j = 0; j < nValues; ++j )
        { toAppend.emplace_back( ( 2 * i + 1 ) * LARGE_NUMBER + j ); }

        m_array.appendArray( toAppend.begin(), toAppend.end() );
        m_ref.emplace_back( toAppend.begin(), toAppend.end() );
      }
    }

    COMPARE_TO_REFERENCE;
  }

  void appendArray2( INDEX_TYPE const nArrays, INDEX_TYPE const maxAppendSize )
  {
    COMPARE_TO_REFERENCE;

    std::vector< T > arrayToAppend( maxAppendSize );

    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      INDEX_TYPE const nValues = rand( 0, maxAppendSize );
      arrayToAppend.resize( nValues );
      m_array.appendArray( nValues );

      for( INDEX_TYPE j = 0; j < nValues; ++j )
      {
        T const value = T( i * LARGE_NUMBER + j );
        arrayToAppend[ j ] = value;
        m_array( m_array.size() - 1, j ) = value;
      }

      m_ref.push_back( arrayToAppend );
    }

    COMPARE_TO_REFERENCE;
  }

  void insertArray( INDEX_TYPE const nArrays, INDEX_TYPE const maxAppendSize )
  {
    COMPARE_TO_REFERENCE;

    LVARRAY_ERROR_IF_NE( nArrays % 2, 0 );

    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      // Insert using a std::vector
      {
        INDEX_TYPE const nValues = rand( 0, maxAppendSize );

        std::vector< T > toInsert;
        for( INDEX_TYPE j = 0; j < nValues; ++j )
        { toInsert.emplace_back( 2 * i * LARGE_NUMBER + j ); }

        INDEX_TYPE const insertPos = rand( 0, m_array.size() );
        m_array.insertArray( insertPos, toInsert.begin(), toInsert.end() );
        m_ref.emplace( m_ref.begin() + insertPos, toInsert.begin(), toInsert.end() );
      }

      // Insert using a std::list
      {
        INDEX_TYPE const nValues = rand( 0, maxAppendSize );

        std::list< T > toInsert;
        for( INDEX_TYPE j = 0; j < nValues; ++j )
        { toInsert.emplace_back( ( 2 * i + 1 ) * LARGE_NUMBER + j ); }

        INDEX_TYPE const insertPos = rand( 0, m_array.size() );
        m_array.insertArray( insertPos, toInsert.begin(), toInsert.end() );
        m_ref.emplace( m_ref.begin() + insertPos, toInsert.begin(), toInsert.end() );
      }
    }

    COMPARE_TO_REFERENCE;
  }

  void eraseArray( INDEX_TYPE const nArrays )
  {
    COMPARE_TO_REFERENCE;

    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      INDEX_TYPE const removePos = rand( 0, m_array.size() -1 );
      m_array.eraseArray( removePos );
      m_ref.erase( m_ref.begin() + removePos );
    }

    COMPARE_TO_REFERENCE;
  }

  void resize( INDEX_TYPE const newSize, INDEX_TYPE const capacityPerArray=0 )
  {
    COMPARE_TO_REFERENCE;

    std::vector< INDEX_TYPE > oldSizes;
    std::vector< INDEX_TYPE > oldCapacities;
    for( INDEX_TYPE i = 0; i < m_array.size(); ++i )
    {
      oldSizes.push_back( m_array.sizeOfArray( i ) );
      oldCapacities.push_back( m_array.capacityOfArray( i ) );
    }

    m_array.resize( newSize, capacityPerArray );

    for( INDEX_TYPE i = 0; i < std::min( newSize, INDEX_TYPE( oldSizes.size() ) ); ++i )
    {
      EXPECT_EQ( m_array.sizeOfArray( i ), oldSizes[ i ] );
      EXPECT_EQ( m_array.capacityOfArray( i ), oldCapacities[ i ] );
    }

    for( INDEX_TYPE i = INDEX_TYPE( oldSizes.size() ); i < newSize; ++i )
    {
      EXPECT_EQ( m_array.sizeOfArray( i ), 0 );
      EXPECT_EQ( m_array.capacityOfArray( i ), capacityPerArray );
    }

    m_ref.resize( newSize );

    COMPARE_TO_REFERENCE;
  }

  template< typename POLICY >
  void resizeFromCapacities( INDEX_TYPE const newSize, INDEX_TYPE const maxCapacity )
  {
    COMPARE_TO_REFERENCE;

    std::vector< INDEX_TYPE > newCapacities( newSize );

    for( INDEX_TYPE & capacity : newCapacities )
    {
      capacity = rand( 0, maxCapacity );
    }

    m_array.template resizeFromCapacities< POLICY >( newSize, newCapacities.data() );

    EXPECT_EQ( m_array.size(), newSize );
    for( INDEX_TYPE i = 0; i < m_array.size(); ++i )
    {
      EXPECT_EQ( m_array.sizeOfArray( i ), 0 );
      EXPECT_EQ( m_array.capacityOfArray( i ), newCapacities[ i ] );
    }

    m_ref.clear();
    m_ref.resize( newSize );
    COMPARE_TO_REFERENCE;
  }

  void resize()
  {
    COMPARE_TO_REFERENCE;

    INDEX_TYPE const newSize = rand( 0, 2 * m_array.size());
    m_array.resize( newSize );
    m_ref.resize( newSize );

    COMPARE_TO_REFERENCE;
  }

  void resizeArray( INDEX_TYPE const maxSize )
  {
    COMPARE_TO_REFERENCE;

    INDEX_TYPE const nArrays = m_array.size();
    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      INDEX_TYPE const newSize = rand( 0, maxSize );

      T const defaultValue = T( i );
      m_array.resizeArray( i, newSize, defaultValue );
      m_ref[ i ].resize( newSize, defaultValue );
    }

    COMPARE_TO_REFERENCE;
  }

  void emplaceBack( INDEX_TYPE const maxAppendSize )
  {
    COMPARE_TO_REFERENCE;

    INDEX_TYPE const nArrays = m_array.size();
    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      INDEX_TYPE const nValues = 2 * rand( 0, maxAppendSize / 2 );
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );

      for( INDEX_TYPE j = 0; j < nValues; ++j )
      {
        T const value = T( i * LARGE_NUMBER + arraySize + 2 * j );
        m_array.emplaceBack( i, value );
        m_ref[ i ].emplace_back( value );

        INDEX_TYPE const seed = i * LARGE_NUMBER + arraySize + 2 * j + 1;
        m_array.emplaceBack( i, seed );
        m_ref[ i ].emplace_back( seed );
      }
    }

    COMPARE_TO_REFERENCE;
  }

  void appendToArray( INDEX_TYPE const maxAppendSize )
  {
    COMPARE_TO_REFERENCE;

    INDEX_TYPE const nArrays = m_array.size();

    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      // Append using a std::vector.
      {
        INDEX_TYPE const nValues = rand( 0, maxAppendSize / 2 );
        INDEX_TYPE const arraySize = m_array.sizeOfArray( i );

        std::vector< T > toAppend;
        for( INDEX_TYPE j = 0; j < nValues; ++j )
        { toAppend.emplace_back( i * LARGE_NUMBER + arraySize + j ); }

        m_array.appendToArray( i, toAppend.begin(), toAppend.end() );
        m_ref[ i ].insert( m_ref[ i ].end(), toAppend.begin(), toAppend.end() );
      }

      // Append using a std::list.
      {
        INDEX_TYPE const nValues = rand( 0, maxAppendSize / 2 );
        INDEX_TYPE const arraySize = m_array.sizeOfArray( i );

        std::list< T > toAppend;
        for( INDEX_TYPE j = 0; j < nValues; ++j )
        { toAppend.emplace_back( i * LARGE_NUMBER + arraySize + j ); }

        m_array.appendToArray( i, toAppend.begin(), toAppend.end() );
        m_ref[ i ].insert( m_ref[ i ].end(), toAppend.begin(), toAppend.end() );
      }
    }

    COMPARE_TO_REFERENCE;
  }

  void emplace( INDEX_TYPE const maxInsertSize )
  {
    COMPARE_TO_REFERENCE;

    INDEX_TYPE const nArrays = m_array.size();
    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      INDEX_TYPE const nValues = 2 * rand( 0, maxInsertSize / 2 );
      for( INDEX_TYPE j = 0; j < nValues; ++j )
      {
        {
          T const value = T( i * LARGE_NUMBER + arraySize + 2 * j );
          INDEX_TYPE const insertPos = rand( 0, arraySize + 2 * j );
          m_array.emplace( i, insertPos, value );
          m_ref[ i ].insert( m_ref[ i ].begin() + insertPos, value );
        }

        {
          INDEX_TYPE const seed = i * LARGE_NUMBER + arraySize + 2 * j + 1;
          INDEX_TYPE const insertPos = rand( 0, arraySize + 2 * j + 1 );
          m_array.emplace( i, insertPos, seed );
          m_ref[ i ].emplace( m_ref[ i ].begin() + insertPos, seed );
        }
      }
    }

    COMPARE_TO_REFERENCE;
  }

  void insertIntoArray( INDEX_TYPE const maxInsertSize )
  {
    COMPARE_TO_REFERENCE;


    INDEX_TYPE const nArrays = m_array.size();
    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      // Insert using a std::vector
      {
        INDEX_TYPE const nValues = rand( 0, maxInsertSize / 2 );
        INDEX_TYPE const arraySize = m_array.sizeOfArray( i );

        std::vector< T > toInsert;
        for( INDEX_TYPE j = 0; j < nValues; ++j )
        { toInsert.emplace_back( i * LARGE_NUMBER + arraySize + j ); }

        INDEX_TYPE const insertPos = rand( 0, arraySize );
        m_array.insertIntoArray( i, insertPos, toInsert.begin(), toInsert.end() );
        m_ref[ i ].insert( m_ref[ i ].begin() + insertPos, toInsert.begin(), toInsert.end() );
      }

      // Insert using a std::list
      {
        INDEX_TYPE const nValues = rand( 0, maxInsertSize / 2 );
        INDEX_TYPE const arraySize = m_array.sizeOfArray( i );

        std::list< T > toInsert;
        for( INDEX_TYPE j = 0; j < nValues; ++j )
        { toInsert.emplace_back( i * LARGE_NUMBER + arraySize + j ); }

        INDEX_TYPE const insertPos = rand( 0, arraySize );
        m_array.insertIntoArray( i, insertPos, toInsert.begin(), toInsert.end() );
        m_ref[ i ].insert( m_ref[ i ].begin() + insertPos, toInsert.begin(), toInsert.end() );
      }
    }

    COMPARE_TO_REFERENCE;
  }

  void eraseFromArray( INDEX_TYPE const maxRemoves )
  {
    COMPARE_TO_REFERENCE;

    INDEX_TYPE const nArrays = m_array.size();
    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      INDEX_TYPE const nToRemove = rand( 0, min( arraySize, maxRemoves ));
      for( INDEX_TYPE j = 0; j < nToRemove; ++j )
      {
        INDEX_TYPE const removePosition = rand( 0, arraySize - j - 1 );

        m_array.eraseFromArray( i, removePosition );
        m_ref[ i ].erase( m_ref[ i ].begin() + removePosition );
      }
    }

    COMPARE_TO_REFERENCE;
  }

  void eraseMultipleFromArray( INDEX_TYPE const maxRemoves )
  {
    COMPARE_TO_REFERENCE;

    INDEX_TYPE const nArrays = m_array.size();
    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      if( arraySize == 0 ) continue;

      INDEX_TYPE const removePosition = rand( 0, arraySize - 1 );
      INDEX_TYPE const nToRemove = rand( 0, min( maxRemoves, arraySize - removePosition ));

      m_array.eraseFromArray( i, removePosition, nToRemove );
      m_ref[ i ].erase( m_ref[ i ].begin() + removePosition, m_ref[ i ].begin() + removePosition + nToRemove );
    }

    COMPARE_TO_REFERENCE;
  }

  void compress()
  {
    COMPARE_TO_REFERENCE;

    m_array.compress();
    T const * const values = m_array[0];

    INDEX_TYPE curOffset = 0;
    for( INDEX_TYPE i = 0; i < m_array.size(); ++i )
    {
      ASSERT_EQ( m_array.sizeOfArray( i ), m_array.capacityOfArray( i ));

      T const * const curValues = m_array[ i ];
      ASSERT_EQ( values + curOffset, curValues );

      curOffset += m_array.sizeOfArray( i );
    }

    COMPARE_TO_REFERENCE;
  }

  void fill()
  {
    COMPARE_TO_REFERENCE;

    T const * const values = m_array[0];
    T const * const endValues = m_array[m_array.size() - 1];

    for( INDEX_TYPE i = 0; i < m_array.size(); ++i )
    {
      fillArray( i );
    }

    T const * const newValues = m_array[0];
    T const * const newEndValues = m_array[m_array.size() - 1];
    ASSERT_EQ( values, newValues );
    ASSERT_EQ( endValues, newEndValues );

    COMPARE_TO_REFERENCE;
  }

  void capacityOfArray( INDEX_TYPE const maxCapacity )
  {
    COMPARE_TO_REFERENCE;

    for( INDEX_TYPE i = 0; i < m_array.size(); ++i )
    {
      INDEX_TYPE const capacity = m_array.capacityOfArray( i );
      INDEX_TYPE const newCapacity = rand( 0, maxCapacity );

      if( newCapacity < capacity )
      {
        T const * const values = m_array[ i ];
        m_array.setCapacityOfArray( i, newCapacity );
        T const * const newValues = m_array[ i ];

        ASSERT_EQ( values, newValues );
        m_ref[ i ].resize( newCapacity );
      }
      else
      {
        m_array.setCapacityOfArray( i, newCapacity );
        fillArray( i );
      }
    }

    COMPARE_TO_REFERENCE;
  }

  void deepCopy()
  {
    COMPARE_TO_REFERENCE;

    ARRAY_OF_ARRAYS copy( m_array );

    ASSERT_EQ( m_array.size(), copy.size());

    for( INDEX_TYPE i = 0; i < m_array.size(); ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      ASSERT_EQ( arraySize, copy.sizeOfArray( i ));

      T const * const values = m_array[ i ];
      T const * const valuesCopy = copy[ i ];
      ASSERT_NE( values, valuesCopy );

      for( INDEX_TYPE j = 0; j < arraySize; ++j )
      {
        ASSERT_EQ( m_array[ i ][ j ], copy[ i ][ j ] );
        copy[ i ][ j ] = T( -i * LARGE_NUMBER - j );
      }
    }

    COMPARE_TO_REFERENCE;
  }

  void shallowCopy()
  {
    ViewType const copy( m_array.toView() );

    ASSERT_EQ( m_array.size(), copy.size());

    for( INDEX_TYPE i = 0; i < m_array.size(); ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      ASSERT_EQ( arraySize, copy.sizeOfArray( i ));
      T const * const values = m_array[ i ];
      T const * const valuesCopy = copy[ i ];
      ASSERT_EQ( values, valuesCopy );

      for( INDEX_TYPE j = 0; j < arraySize; ++j )
      {
        ASSERT_EQ( m_array[ i ][ j ], copy[ i ][ j ] );

        copy[ i ][ j ] = T( -i * LARGE_NUMBER - j );

        ASSERT_EQ( m_array[ i ][ j ], T( -i * LARGE_NUMBER - j ));
        ASSERT_EQ( m_array[ i ][ j ], copy[ i ][ j ] );
      }
    }
  }

protected:

  void fillArray( INDEX_TYPE const i )
  {
    INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
    INDEX_TYPE const nToAppend = m_array.capacityOfArray( i ) - arraySize;
    T const * const values = m_array[ i ];
    T const * const endValues = m_array[m_array.size() - 1];
    INDEX_TYPE const capacity = m_array.capacityOfArray( i );

    for( INDEX_TYPE j = 0; j < nToAppend; ++j )
    {
      T const valueToAppend( LARGE_NUMBER * i + j );
      m_array.emplaceBack( i, valueToAppend );
      m_ref[ i ].emplace_back( valueToAppend );
    }

    T const * const newValues = m_array[ i ];
    T const * const newEndValues = m_array[m_array.size() - 1];
    INDEX_TYPE const endCapacity = m_array.capacityOfArray( i );
    ASSERT_EQ( values, newValues );
    ASSERT_EQ( endValues, newEndValues );
    ASSERT_EQ( capacity, endCapacity );
  }

  INDEX_TYPE rand( INDEX_TYPE const min, INDEX_TYPE const max )
  { return std::uniform_int_distribution< INDEX_TYPE >( min, max )( m_gen ); }

  static constexpr INDEX_TYPE LARGE_NUMBER = 1E6;

  ARRAY_OF_ARRAYS m_array;

  std::vector< std::vector< T > > m_ref;

  std::mt19937_64 m_gen;


  /// Check that the move, toView, and toViewConst methods of ARRAY_OF_ARRAYS are detected.
  static_assert( bufferManipulation::HasMemberFunction_move< ARRAY_OF_ARRAYS >,
                 "ARRAY_OF_ARRAYS has a move method." );
  static_assert( HasMemberFunction_toView< ARRAY_OF_ARRAYS >,
                 "ARRAY_OF_ARRAYS has a toView method." );
  static_assert( HasMemberFunction_toViewConst< ARRAY_OF_ARRAYS >,
                 "ARRAY_OF_ARRAYS has a toViewConst method." );

  /// Check that the move and toViewConst methods of ViewType are detected.
  static_assert( bufferManipulation::HasMemberFunction_move< ViewType >,
                 "ViewType has a move method." );
  static_assert( HasMemberFunction_toView< ViewType >,
                 "ViewType has a toView method." );
  static_assert( HasMemberFunction_toViewConst< ViewType >,
                 "ViewType has a toViewConst method." );

  /// Check that the move and toViewConst methods of ViewTypeConstSizes are detected.
  static_assert( bufferManipulation::HasMemberFunction_move< ViewTypeConstSizes >,
                 "ViewTypeConstSizes has a move method." );
  static_assert( HasMemberFunction_toView< ViewTypeConstSizes >,
                 "ViewTypeConstSizes has a toView method." );
  static_assert( HasMemberFunction_toViewConst< ViewTypeConstSizes >,
                 "ViewTypeConstSizes has a toViewConst method." );

  /// Check that the move and toViewConst methods of ViewTypeConst are detected.
  static_assert( bufferManipulation::HasMemberFunction_move< ViewTypeConst >,
                 "ViewTypeConst has a move method." );
  static_assert( HasMemberFunction_toView< ViewTypeConst >,
                 "ViewTypeConst has a toView method." );
  static_assert( HasMemberFunction_toViewConst< ViewTypeConst >,
                 "ViewTypeConst has a toViewConst method." );

  /// Check that GetViewType and GetViewTypeConst are correct for ARRAY_OF_ARRAYS
  static_assert( std::is_same< typename GetViewType< ARRAY_OF_ARRAYS >::type, ViewType const >::value,
                 "The view type of ARRAY_OF_ARRAYS is ViewType const." );
  static_assert( std::is_same< typename GetViewTypeConst< ARRAY_OF_ARRAYS >::type, ViewTypeConst const >::value,
                 "The const view type of ARRAY_OF_ARRAYS is ViewTypeConst const." );

  /// Check that GetViewType and GetViewTypeConst are correct for ViewType
  static_assert( std::is_same< typename GetViewType< ViewType >::type, ViewType const >::value,
                 "The view type of ViewType is ViewType const." );
  static_assert( std::is_same< typename GetViewTypeConst< ViewType >::type, ViewTypeConst const >::value,
                 "The const view type of ViewType is ViewTypeConst const." );

  /// Check that GetViewType and GetViewTypeConst are correct for ViewTypeConstSizes
  static_assert( std::is_same< typename GetViewType< ViewTypeConstSizes >::type, ViewTypeConstSizes const >::value,
                 "The view type of ViewTypeConstSizes is ViewTypeConstSizes const." );
  static_assert( std::is_same< typename GetViewTypeConst< ViewTypeConstSizes >::type, ViewTypeConst const >::value,
                 "The const view type of ViewTypeConstSizes is ViewTypeConst const." );

  /// Check that GetViewType and GetViewTypeConst are correct for ViewTypeConst
  static_assert( std::is_same< typename GetViewType< ViewTypeConst >::type, ViewTypeConst const >::value,
                 "The view type of ViewTypeConst is ViewTypeConst const." );
  static_assert( std::is_same< typename GetViewTypeConst< ViewTypeConst >::type, ViewTypeConst const >::value,
                 "The const view type of ViewTypeConst is ViewTypeConst const." );
};

using ArrayOfArraysTestTypes = ::testing::Types<
  ArrayOfArrays< int, INDEX_TYPE, MallocBuffer >
  , ArrayOfArrays< Tensor, INDEX_TYPE, MallocBuffer >
  , ArrayOfArrays< TestString, INDEX_TYPE, MallocBuffer >
#if defined(USE_CHAI)
  , ArrayOfArrays< int, INDEX_TYPE, NewChaiBuffer >
  , ArrayOfArrays< Tensor, INDEX_TYPE, NewChaiBuffer >
  , ArrayOfArrays< TestString, INDEX_TYPE, NewChaiBuffer >
#endif
  >;
TYPED_TEST_SUITE( ArrayOfArraysTest, ArrayOfArraysTestTypes, );


TYPED_TEST( ArrayOfArraysTest, emptyConstruction )
{
  TypeParam a;
  ASSERT_EQ( a.size(), 0 );
  ASSERT_EQ( a.capacity(), 0 );
}


TYPED_TEST( ArrayOfArraysTest, sizedConstruction )
{
  TypeParam a( 10 );
  ASSERT_EQ( a.size(), 10 );
  ASSERT_EQ( a.capacity(), 10 );

  for( INDEX_TYPE i = 0; i < a.size(); ++i )
  {
    ASSERT_EQ( a.sizeOfArray( i ), 0 );
    ASSERT_EQ( a.capacityOfArray( i ), 0 );
  }
}

TYPED_TEST( ArrayOfArraysTest, sizedHintConstruction )
{
  TypeParam a( 10, 20 );
  ASSERT_EQ( a.size(), 10 );
  ASSERT_EQ( a.capacity(), 10 );

  for( INDEX_TYPE i = 0; i < a.size(); ++i )
  {
    ASSERT_EQ( a.sizeOfArray( i ), 0 );
    ASSERT_EQ( a.capacityOfArray( i ), 20 );
  }
}

TYPED_TEST( ArrayOfArraysTest, appendArray )
{
  this->appendArray( 100, 10 );
}


TYPED_TEST( ArrayOfArraysTest, appendArray2 )
{
  this->appendArray2( 100, 10 );
}

TYPED_TEST( ArrayOfArraysTest, insertArray )
{
  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->insertArray( 100, 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, copyConstruction )
{
  this->appendArray( 100, 10 );
  TypeParam a( this->m_array );

  ASSERT_NE( &a( 0, 0 ), &this->m_array( 0, 0 ) );

  this->compareToReference( this->m_array.toViewConst() );
  this->compareToReference( a.toViewConst() );
}

TYPED_TEST( ArrayOfArraysTest, copyAssignment )
{
  this->appendArray( 100, 10 );
  TypeParam a;
  ASSERT_EQ( a.size(), 0 );
  a = this->m_array;

  ASSERT_NE( &a( 0, 0 ), &this->m_array( 0, 0 ) );

  this->compareToReference( this->m_array.toViewConst() );
  this->compareToReference( a.toViewConst() );
}

TYPED_TEST( ArrayOfArraysTest, moveConstruction )
{
  this->appendArray( 100, 10 );
  TypeParam a( std::move( this->m_array ) );
  ASSERT_EQ( this->m_array.size(), 0 );

  this->compareToReference( a.toViewConst() );
}

TYPED_TEST( ArrayOfArraysTest, moveAssignment )
{
  this->appendArray( 100, 10 );
  TypeParam a;
  ASSERT_EQ( a.size(), 0 );
  a = std::move( this->m_array );
  ASSERT_EQ( this->m_array.size(), 0 );

  this->compareToReference( a.toViewConst() );
}

TYPED_TEST( ArrayOfArraysTest, eraseArray )
{
  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->appendArray( 100, 10 );
    this->eraseArray( this->m_array.size() / 2 );
  }
}

TYPED_TEST( ArrayOfArraysTest, resize )
{
  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->insertArray( 100, 10 );
    this->resize();
  }
}

TYPED_TEST( ArrayOfArraysTest, resizeWithCapacity )
{
  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->resize( 100, 10 );
    this->emplace( 5 );
    this->resize( 150, 10 );
    this->emplace( 5 );
  }
}

TYPED_TEST( ArrayOfArraysTest, resizeFromCapacities )
{
  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->template resizeFromCapacities< serialPolicy >( 100, 10 );
    this->emplace( 10 );

#if defined( USE_OPENMP )
    this->template resizeFromCapacities< parallelHostPolicy >( 150, 10 );
    this->emplace( 10 );
#endif
  }
}

TYPED_TEST( ArrayOfArraysTest, resizeArray )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->resizeArray( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, emplaceBack )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->emplaceBack( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, appendToArray )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, emplace )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->emplace( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, insertIntoArray )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->insertIntoArray( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, eraseFromArray )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
    this->eraseFromArray( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, eraseMultipleFromArray )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
    this->eraseMultipleFromArray( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, compress )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
    this->compress();
  }
}

TYPED_TEST( ArrayOfArraysTest, capacity )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->resizeArray( 10 );
    this->fill();
  }
}

TYPED_TEST( ArrayOfArraysTest, capacityOfArray )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->capacityOfArray( 30 );
  }
}

TYPED_TEST( ArrayOfArraysTest, deepCopy )
{
  this->resize( 100 );

  this->appendToArray( 30 );
  this->deepCopy();
}

TYPED_TEST( ArrayOfArraysTest, shallowCopy )
{
  this->resize( 100 );

  this->appendToArray( 30 );
  this->shallowCopy();
}

template< typename ARRAY_OF_ARRAYS_POLICY_PAIR >
class ArrayOfArraysViewTest : public ArrayOfArraysTest< typename ARRAY_OF_ARRAYS_POLICY_PAIR::first_type >
{
public:
  using ARRAY_OF_ARRAYS = typename ARRAY_OF_ARRAYS_POLICY_PAIR::first_type;
  using POLICY = typename ARRAY_OF_ARRAYS_POLICY_PAIR::second_type;

  using T = typename ARRAY_OF_ARRAYS::value_type;

  using ParentClass = ArrayOfArraysTest< ARRAY_OF_ARRAYS >;

  using ViewType = std::remove_const_t< typename ARRAY_OF_ARRAYS::ViewType >;

  template< typename T >
  using Array1D = typename ArrayConverter< ARRAY_OF_ARRAYS >::template Array< T, 1, RAJA::PERM_I >;

  template< typename T >
  using ArrayView1D = typename ArrayConverter< ARRAY_OF_ARRAYS >::template ArrayView< T, 1, 0 >;

  template< typename T >
  using Array2D = typename ArrayConverter< ARRAY_OF_ARRAYS >::template Array< T, 2, RAJA::PERM_IJ >;

  template< typename T >
  using ArrayView2D = typename ArrayConverter< ARRAY_OF_ARRAYS >::template ArrayView< T, 2, 1 >;


  void modifyInKernel()
  {
    COMPARE_TO_REFERENCE;

    INDEX_TYPE const nArrays = m_array.size();

    // Update the view on the device.
    ViewType const & view = m_array.toView();
    forall< POLICY >( nArrays, [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          INDEX_TYPE const sizeOfArray = view.sizeOfArray( i );
          for( INDEX_TYPE j = 0; j < sizeOfArray; ++j )
          {
            PORTABLE_EXPECT_EQ( &view[ i ][ j ], &view( i, j ) );
            view[ i ][ j ] += T( view[ i ][ j ] );
            view( i, j ) += T( view( i, j ) );
          }
        } );

    // Update the reference.
    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      for( INDEX_TYPE j = 0; j < INDEX_TYPE( m_ref[ i ].size() ); ++j )
      {
        m_ref[ i ][ j ] += T( m_ref[ i ][ j ] );
        m_ref[ i ][ j ] += T( m_ref[ i ][ j ] );
      }
    }

    // Move the view back to the host and compare with the reference.
    forall< serialPolicy >( 1, [view, this] ( INDEX_TYPE )
    {
      SCOPED_TRACE( "" );
      this->compareToReference( view.toViewConst() );
    } );
  }

  void memoryMotionMove()
  {
    COMPARE_TO_REFERENCE;

    INDEX_TYPE const nArrays = m_array.size();

    // Update the view on the device.
    ViewType const & view = m_array.toView();
    forall< POLICY >( nArrays, [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          INDEX_TYPE const sizeOfArray = view.sizeOfArray( i );
          for( INDEX_TYPE j = 0; j < sizeOfArray; ++j )
          {
            PORTABLE_EXPECT_EQ( &view[ i ][ j ], &view( i, j ) );
            view[ i ][ j ] += T( view[ i ][ j ] );
            view( i, j ) += T( view( i, j ) );
          }
        } );

    // Update the reference.
    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      for( INDEX_TYPE j = 0; j < INDEX_TYPE( m_ref[ i ].size() ); ++j )
      {
        m_ref[ i ][ j ] += T( m_ref[ i ][ j ] );
        m_ref[ i ][ j ] += T( m_ref[ i ][ j ] );
      }
    }

    // Move the view back to the host and compare with the reference.
    m_array.move( MemorySpace::CPU );
    COMPARE_TO_REFERENCE;
  }

  void emplaceBack()
  {
    COMPARE_TO_REFERENCE;

    INDEX_TYPE const nArrays = m_array.size();
    Array1D< Array1D< INDEX_TYPE > > seedsToAppend( nArrays );

    // Populate the seedsToAppend array and append the values to the reference.
    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      INDEX_TYPE maxAppends = m_array.capacityOfArray( i ) - arraySize;
      INDEX_TYPE const nSeeds = rand( 0, maxAppends / 2 );

      seedsToAppend[ i ].resize( nSeeds );

      for( INDEX_TYPE j = 0; j < nSeeds; ++j )
      {
        INDEX_TYPE const seed = i * LARGE_NUMBER + arraySize + 2 * j;
        m_ref[ i ].emplace_back( T( seed ) );
        m_ref[ i ].emplace_back( seed + 1 );
        seedsToAppend[ i ][ j ] = seed;
      }
    }

    // Append to the view on the device.
    ViewType const & view = m_array.toView();
    ArrayView1D< ArrayView1D< INDEX_TYPE const > const > const & seeds = seedsToAppend.toViewConst();
    forall< POLICY >( nArrays, [view, seeds] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < seeds[ i ].size(); ++j )
          {
            view.emplaceBack( i, T( seeds[ i ][ j ] ) );
            view.emplaceBack( i, seeds[ i ][ j ] + 1 );
          }
        } );

    // Move the view back to the host and compare with the reference.
    m_array.move( MemorySpace::CPU );
    COMPARE_TO_REFERENCE;
  }

  void view_appendToArray()
  {
    COMPARE_TO_REFERENCE;

    INDEX_TYPE const nArrays = m_array.size();
    Array1D< Array1D< T > > valuesToAppend( nArrays );

    // Populate the valuesToAppend array and update the reference.
    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      INDEX_TYPE const maxAppends = m_array.capacityOfArray( i ) - arraySize;
      INDEX_TYPE const nValues = rand( 0, maxAppends );
      valuesToAppend[ i ].resize( nValues );

      for( INDEX_TYPE j = 0; j < nValues; ++j )
      {
        T const value = T( i * LARGE_NUMBER + arraySize + j );
        m_ref[ i ].push_back( value );
        valuesToAppend[ i ][ j ] = value;
      }
    }

    // Append to the view on the device.
    ViewType const & view = m_array.toView();
    ArrayView1D< ArrayView1D< T const > const > const & toAppend = valuesToAppend.toViewConst();
    forall< POLICY >( nArrays, [view, toAppend] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          view.appendToArray( i, toAppend[ i ].begin(), toAppend[ i ].end() );
        } );

    // Move the view back to the host and compare with the reference.
    m_array.move( MemorySpace::CPU );
    COMPARE_TO_REFERENCE;
  }

  void view_emplace()
  {
    COMPARE_TO_REFERENCE;

    INDEX_TYPE const nArrays = m_array.size();
    Array1D< Array1D< INDEX_TYPE > > seedsToInsert( nArrays );
    Array1D< Array1D< INDEX_TYPE > > positionsToInsert( nArrays );

    // Populate the valuesToInsert and positionsToInsert arrays and update the reference.
    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      INDEX_TYPE const maxInserts = m_array.capacityOfArray( i ) - arraySize;
      INDEX_TYPE const nSeeds = rand( 0, maxInserts / 2 );
      seedsToInsert[ i ].resize( nSeeds );
      positionsToInsert[ i ].resize( nSeeds );

      for( INDEX_TYPE j = 0; j < nSeeds; ++j )
      {
        INDEX_TYPE const seed = i * LARGE_NUMBER + arraySize + 2 * j;
        INDEX_TYPE const insertPos = rand( 0, arraySize + j );

        m_ref[ i ].emplace( m_ref[ i ].begin() + insertPos, T( seed ) );
        m_ref[ i ].emplace( m_ref[ i ].begin() + insertPos, seed + 1 );

        seedsToInsert[ i ][ j ] = seed;
        positionsToInsert[ i ][ j ] = insertPos;
      }
    }

    // Insert into the view on the device.
    ViewType const & view = m_array.toView();
    ArrayView1D< ArrayView1D< INDEX_TYPE const > const > const & positions = positionsToInsert.toViewConst();
    ArrayView1D< ArrayView1D< INDEX_TYPE const > const > const & seeds = seedsToInsert.toViewConst();
    forall< POLICY >( nArrays, [view, positions, seeds] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < positions[ i ].size(); ++j )
          {
            view.emplace( i, positions[ i ][ j ], T( seeds[ i ][ j ] ) );
            view.emplace( i, positions[ i ][ j ], seeds[ i ][ j ] + 1 );
          }
        } );

    // Move the view back to the host and compare with the reference.
    m_array.move( MemorySpace::CPU );
    COMPARE_TO_REFERENCE;
  }

  void view_insertIntoArray()
  {
    COMPARE_TO_REFERENCE;

    INDEX_TYPE const nArrays = m_array.size();
    Array1D< Array1D< T > > valuesToInsert( nArrays );
    Array1D< INDEX_TYPE > positionsToInsert( nArrays );

    // Populate the valuesToInsert and positionsToInsert arrays and update the reference.
    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      INDEX_TYPE const maxInserts = m_array.capacityOfArray( i ) - arraySize;
      INDEX_TYPE const nValues = rand( 0, maxInserts );
      valuesToInsert[ i ].resize( nValues );

      INDEX_TYPE const insertPos = rand( 0, arraySize );
      positionsToInsert[ i ] = insertPos;

      for( INDEX_TYPE j = 0; j < nValues; ++j )
      {
        T const value = T( i * LARGE_NUMBER + arraySize + j );
        valuesToInsert[ i ][ j ] = value;
      }

      m_ref[ i ].insert( m_ref[ i ].begin() + insertPos, valuesToInsert[ i ].begin(), valuesToInsert[ i ].end() );
    }

    // Insert into the view on the device.
    ViewType const & view = m_array.toView();
    ArrayView1D< INDEX_TYPE const > const & positions = positionsToInsert.toViewConst();
    ArrayView1D< ArrayView1D< T const > const > const & values = valuesToInsert.toViewConst();
    forall< POLICY >( nArrays, [view, positions, values] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          view.insertIntoArray( i, positions[ i ], values[ i ].begin(), values[ i ].end() );
        } );

    // Move the view back to the host and compare with the reference.
    m_array.move( MemorySpace::CPU );
    COMPARE_TO_REFERENCE;
  }

  void view_erase( INDEX_TYPE const maxRemoves )
  {
    COMPARE_TO_REFERENCE;

    INDEX_TYPE const nArrays = m_array.size();
    Array1D< Array1D< INDEX_TYPE > > positionsToRemove( nArrays );

    // Populate the positionsToRemove array and update the reference.
    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      INDEX_TYPE const nToRemove = rand( 0, min( arraySize, maxRemoves ));
      positionsToRemove[ i ].resize( nToRemove );

      for( INDEX_TYPE j = 0; j < nToRemove; ++j )
      {
        INDEX_TYPE const removePosition = rand( 0, arraySize - j - 1 );
        m_ref[ i ].erase( m_ref[ i ].begin() + removePosition );
        positionsToRemove[ i ][ j ] = removePosition;
      }
    }

    // Remove the view on the device.
    ViewType const & view = m_array.toView();
    ArrayView1D< ArrayView1D< INDEX_TYPE const > const > const & positions = positionsToRemove.toViewConst();
    forall< POLICY >( nArrays, [view, positions] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < positions[ i ].size(); ++j )
          {
            view.eraseFromArray( i, positions[ i ][ j ] );
          }
        } );

    // Move the view back to the host and compare with the reference.
    m_array.move( MemorySpace::CPU );
    COMPARE_TO_REFERENCE;
  }

  void view_eraseMultiple( INDEX_TYPE const maxRemoves )
  {
    COMPARE_TO_REFERENCE;

    INDEX_TYPE const nArrays = m_array.size();
    Array2D< INDEX_TYPE > toRemove( nArrays, 2 );

    // Populate the toRemove array and update the reference.
    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      INDEX_TYPE const arraySize = m_array.sizeOfArray( i );
      if( arraySize == 0 ) continue;

      INDEX_TYPE const removePosition = rand( 0, arraySize - 1 );
      INDEX_TYPE const nToRemove = rand( 0, min( maxRemoves, arraySize - removePosition ));

      toRemove[ i ][ 0 ] = removePosition;
      toRemove[ i ][ 1 ] = nToRemove;
      m_ref[ i ].erase( m_ref[ i ].begin() + removePosition, m_ref[ i ].begin() + removePosition + nToRemove );
    }

    // Remove from the view on the device.
    ViewType const & view = m_array.toView();
    ArrayView2D< INDEX_TYPE const > removals = toRemove.toViewConst();
    forall< POLICY >( nArrays, [view, removals] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          if( view.sizeOfArray( i ) == 0 ) return;
          view.eraseFromArray( i, removals[ i ][ 0 ], removals[ i ][ 1 ] );
        } );

    // Move the view back to the host and compare with the reference.
    m_array.move( MemorySpace::CPU );
    COMPARE_TO_REFERENCE;
  }

protected:
  using ParentClass::rand;
  using ParentClass::LARGE_NUMBER;
  using ParentClass::m_gen;
  using ParentClass::m_array;
  using ParentClass::m_ref;
};

using ArrayOfArraysViewTestTypes = ::testing::Types<
  std::pair< ArrayOfArrays< int, INDEX_TYPE, MallocBuffer >, serialPolicy >
  , std::pair< ArrayOfArrays< Tensor, INDEX_TYPE, MallocBuffer >, serialPolicy >
  , std::pair< ArrayOfArrays< TestString, INDEX_TYPE, MallocBuffer >, serialPolicy >
#if defined(USE_CHAI)
  , std::pair< ArrayOfArrays< int, INDEX_TYPE, NewChaiBuffer >, serialPolicy >
  , std::pair< ArrayOfArrays< Tensor, INDEX_TYPE, NewChaiBuffer >, serialPolicy >
  , std::pair< ArrayOfArrays< TestString, INDEX_TYPE, NewChaiBuffer >, serialPolicy >
#endif

#if defined(USE_CUDA) && defined(USE_CHAI)
  , std::pair< ArrayOfArrays< int, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< ArrayOfArrays< Tensor, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
#endif
  >;
TYPED_TEST_SUITE( ArrayOfArraysViewTest, ArrayOfArraysViewTestTypes, );

TYPED_TEST( ArrayOfArraysViewTest, memoryMotion )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->appendToArray( 30 );
    this->modifyInKernel();
  }
}

TYPED_TEST( ArrayOfArraysViewTest, memoryMotionMove )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->appendToArray( 30 );
    this->memoryMotionMove();
  }
}

TYPED_TEST( ArrayOfArraysViewTest, emplaceBack )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
    this->emplaceBack();
  }
}

TYPED_TEST( ArrayOfArraysViewTest, appendToArray )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
    this->view_appendToArray();
  }
}

TYPED_TEST( ArrayOfArraysViewTest, emplace )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
    this->view_emplace();
  }
}

TYPED_TEST( ArrayOfArraysViewTest, insertIntoArray )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
    this->view_insertIntoArray();
  }
}

TYPED_TEST( ArrayOfArraysViewTest, erase )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
    this->view_erase( 10 );
  }
}

TYPED_TEST( ArrayOfArraysViewTest, eraseMultiple )
{
  this->resize( 100 );

  for( INDEX_TYPE i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
    this->view_eraseMultiple( 10 );
  }
}


template< typename ARRAY_OF_ARRAYS_POLICY_PAIR >
class ArrayOfArraysViewAtomicTest : public ArrayOfArraysViewTest< ARRAY_OF_ARRAYS_POLICY_PAIR >
{
public:
  using ARRAY_OF_ARRAYS = typename ARRAY_OF_ARRAYS_POLICY_PAIR::first_type;
  using POLICY = typename ARRAY_OF_ARRAYS_POLICY_PAIR::second_type;
  using AtomicPolicy = typename RAJAHelper< POLICY >::AtomicPolicy;


  using T = typename ARRAY_OF_ARRAYS::value_type;

  using ParentClass = ArrayOfArraysViewTest< ARRAY_OF_ARRAYS_POLICY_PAIR >;

  using ViewType = std::remove_const_t< typename ARRAY_OF_ARRAYS::ViewType >;


  void emplaceBackAtomic( INDEX_TYPE const numThreads, INDEX_TYPE const appendsPerArrayPerThread )
  {
    INDEX_TYPE const nArrays = m_array.size();

    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      ASSERT_EQ( m_array.sizeOfArray( i ), 0 );
    }

    ViewType const & view = m_array.toView();
    forall< POLICY >( numThreads,
                      [view, nArrays, appendsPerArrayPerThread] LVARRAY_HOST_DEVICE ( INDEX_TYPE const threadNum )
        {
          for( INDEX_TYPE i = 0; i < nArrays; ++i )
          {
            for( INDEX_TYPE j = 0; j < appendsPerArrayPerThread; ++j )
            {
              T const value = T( i * LARGE_NUMBER + threadNum * appendsPerArrayPerThread + j );
              view.template emplaceBackAtomic< AtomicPolicy >( i, value );
            }
          }
        } );

    // Now sort each array and check that the values are as expected.
    m_array.move( MemorySpace::CPU );
    INDEX_TYPE const appendsPerArray = numThreads * appendsPerArrayPerThread;
    for( INDEX_TYPE i = 0; i < nArrays; ++i )
    {
      ASSERT_EQ( m_array.sizeOfArray( i ), appendsPerArray );
      ASSERT_LE( m_array.sizeOfArray( i ), m_array.capacityOfArray( i ));

      T * const subArrayPtr = m_array[ i ];
      std::sort( subArrayPtr, subArrayPtr + appendsPerArray );

      for( INDEX_TYPE j = 0; j < appendsPerArray; ++j )
      {
        T const value = T( i * LARGE_NUMBER + j );
        EXPECT_EQ( value, m_array( i, j ) ) << i << ", " << j;
      }
    }
  }

protected:
  using ParentClass::rand;
  using ParentClass::LARGE_NUMBER;
  using ParentClass::m_gen;
  using ParentClass::m_array;
  using ParentClass::m_ref;
};

using ArrayOfArraysViewAtomicTestTypes = ::testing::Types<
  std::pair< ArrayOfArrays< int, INDEX_TYPE, MallocBuffer >, serialPolicy >
  , std::pair< ArrayOfArrays< Tensor, INDEX_TYPE, MallocBuffer >, serialPolicy >
  , std::pair< ArrayOfArrays< TestString, INDEX_TYPE, MallocBuffer >, serialPolicy >
#if defined(USE_CHAI)
  , std::pair< ArrayOfArrays< int, INDEX_TYPE, NewChaiBuffer >, serialPolicy >
  , std::pair< ArrayOfArrays< Tensor, INDEX_TYPE, NewChaiBuffer >, serialPolicy >
  , std::pair< ArrayOfArrays< TestString, INDEX_TYPE, NewChaiBuffer >, serialPolicy >
#endif

#if defined(USE_OPENMP)
  , std::pair< ArrayOfArrays< int, INDEX_TYPE, MallocBuffer >, parallelHostPolicy >
  , std::pair< ArrayOfArrays< Tensor, INDEX_TYPE, MallocBuffer >, parallelHostPolicy >
  , std::pair< ArrayOfArrays< TestString, INDEX_TYPE, MallocBuffer >, parallelHostPolicy >
#endif
#if defined(USE_OPENMP) && defined(USE_CHAI)
  , std::pair< ArrayOfArrays< int, INDEX_TYPE, NewChaiBuffer >, parallelHostPolicy >
  , std::pair< ArrayOfArrays< Tensor, INDEX_TYPE, NewChaiBuffer >, parallelHostPolicy >
  , std::pair< ArrayOfArrays< TestString, INDEX_TYPE, NewChaiBuffer >, parallelHostPolicy >
#endif

#if defined(USE_CUDA) && defined(USE_CHAI)
  , std::pair< ArrayOfArrays< int, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< ArrayOfArrays< Tensor, INDEX_TYPE, NewChaiBuffer >, parallelDevicePolicy< 32 > >
#endif
  >;
TYPED_TEST_SUITE( ArrayOfArraysViewAtomicTest, ArrayOfArraysViewAtomicTestTypes, );

TYPED_TEST( ArrayOfArraysViewAtomicTest, atomicAppend )
{
  INDEX_TYPE const NUM_THREADS = 10;
  INDEX_TYPE const APPENDS_PER_THREAD = 10;

  this->resize( 100, NUM_THREADS * APPENDS_PER_THREAD );
  this->emplaceBackAtomic( NUM_THREADS, APPENDS_PER_THREAD );
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
