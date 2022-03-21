/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
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

template< class ARRAY_OF_ARRAYS >
class ArrayOfArraysTest : public ::testing::Test
{
public:
  using T = typename ARRAY_OF_ARRAYS::ValueType;
  using IndexType = typename ARRAY_OF_ARRAYS::IndexType;

  using ViewType = typeManipulation::ViewType< ARRAY_OF_ARRAYS >;
  using ViewTypeConstSizes = typeManipulation::ViewTypeConstSizes< ARRAY_OF_ARRAYS >;
  using ViewTypeConst = typeManipulation::ViewTypeConst< ARRAY_OF_ARRAYS >;

  void compareToReference( ViewTypeConst const & view ) const
  {
    ASSERT_EQ( view.size(), m_ref.size());

    for( IndexType i = 0; i < view.size(); ++i )
    {
      ASSERT_EQ( view.sizeOfArray( i ), m_ref[ i ].size());

      for( IndexType j = 0; j < view.sizeOfArray( i ); ++j )
      {
        EXPECT_EQ( view[ i ][ j ], m_ref[ i ][ j ] );
        EXPECT_EQ( view( i, j ), view[ i ][ j ] );
      }

      IndexType j = 0;
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

  void appendArray( IndexType const nArrays, IndexType const maxAppendSize )
  {
    COMPARE_TO_REFERENCE;

    LVARRAY_ERROR_IF_NE( nArrays % 2, 0 );

    for( IndexType i = 0; i < nArrays / 2; ++i )
    {
      // Append using a std::vector.
      {
        IndexType const nValues = rand( 0, maxAppendSize );

        std::vector< T > toAppend;
        for( IndexType j = 0; j < nValues; ++j )
        { toAppend.emplace_back( 2 * i * LARGE_NUMBER + j ); }

        m_array.appendArray( toAppend.begin(), toAppend.end() );
        m_ref.emplace_back( toAppend.begin(), toAppend.end() );
      }

      // Append using a std::list.
      {
        IndexType const nValues = rand( 0, maxAppendSize );

        std::list< T > toAppend;
        for( IndexType j = 0; j < nValues; ++j )
        { toAppend.emplace_back( ( 2 * i + 1 ) * LARGE_NUMBER + j ); }

        m_array.appendArray( toAppend.begin(), toAppend.end() );
        m_ref.emplace_back( toAppend.begin(), toAppend.end() );
      }
    }

    COMPARE_TO_REFERENCE;
  }

  void appendArray2( IndexType const nArrays, IndexType const maxAppendSize )
  {
    COMPARE_TO_REFERENCE;

    std::vector< T > arrayToAppend( maxAppendSize );

    for( IndexType i = 0; i < nArrays; ++i )
    {
      IndexType const nValues = rand( 0, maxAppendSize );
      arrayToAppend.resize( nValues );
      m_array.appendArray( nValues );

      for( IndexType j = 0; j < nValues; ++j )
      {
        T const value = T( i * LARGE_NUMBER + j );
        arrayToAppend[ j ] = value;
        m_array( m_array.size() - 1, j ) = value;
      }

      m_ref.push_back( arrayToAppend );
    }

    COMPARE_TO_REFERENCE;
  }

  void insertArray( IndexType const nArrays, IndexType const maxAppendSize )
  {
    COMPARE_TO_REFERENCE;

    LVARRAY_ERROR_IF_NE( nArrays % 2, 0 );

    for( IndexType i = 0; i < nArrays; ++i )
    {
      // Insert using a std::vector
      {
        IndexType const nValues = rand( 0, maxAppendSize );

        std::vector< T > toInsert;
        for( IndexType j = 0; j < nValues; ++j )
        { toInsert.emplace_back( 2 * i * LARGE_NUMBER + j ); }

        IndexType const insertPos = rand( 0, m_array.size() );
        m_array.insertArray( insertPos, toInsert.begin(), toInsert.end() );
        m_ref.emplace( m_ref.begin() + insertPos, toInsert.begin(), toInsert.end() );
      }

      // Insert using a std::list
      {
        IndexType const nValues = rand( 0, maxAppendSize );

        std::list< T > toInsert;
        for( IndexType j = 0; j < nValues; ++j )
        { toInsert.emplace_back( ( 2 * i + 1 ) * LARGE_NUMBER + j ); }

        IndexType const insertPos = rand( 0, m_array.size() );
        m_array.insertArray( insertPos, toInsert.begin(), toInsert.end() );
        m_ref.emplace( m_ref.begin() + insertPos, toInsert.begin(), toInsert.end() );
      }
    }

    COMPARE_TO_REFERENCE;
  }

  void eraseArray( IndexType const nArrays )
  {
    COMPARE_TO_REFERENCE;

    for( IndexType i = 0; i < nArrays; ++i )
    {
      IndexType const removePos = rand( 0, m_array.size() -1 );
      m_array.eraseArray( removePos );
      m_ref.erase( m_ref.begin() + removePos );
    }

    COMPARE_TO_REFERENCE;
  }

  void resize( IndexType const newSize, IndexType const capacityPerArray=0 )
  {
    COMPARE_TO_REFERENCE;

    std::vector< IndexType > oldSizes;
    std::vector< IndexType > oldCapacities;
    for( IndexType i = 0; i < m_array.size(); ++i )
    {
      oldSizes.push_back( m_array.sizeOfArray( i ) );
      oldCapacities.push_back( m_array.capacityOfArray( i ) );
    }

    m_array.resize( newSize, capacityPerArray );

    for( IndexType i = 0; i < math::min( newSize, IndexType( oldSizes.size() ) ); ++i )
    {
      EXPECT_EQ( m_array.sizeOfArray( i ), oldSizes[ i ] );
      EXPECT_EQ( m_array.capacityOfArray( i ), oldCapacities[ i ] );
    }

    for( IndexType i = IndexType( oldSizes.size() ); i < newSize; ++i )
    {
      EXPECT_EQ( m_array.sizeOfArray( i ), 0 );
      EXPECT_EQ( m_array.capacityOfArray( i ), capacityPerArray );
    }

    m_ref.resize( newSize );

    COMPARE_TO_REFERENCE;
  }

  template< typename POLICY >
  void resizeFromCapacities( IndexType const newSize, IndexType const maxCapacity )
  {
    COMPARE_TO_REFERENCE;

    std::vector< IndexType > newCapacities( newSize );

    for( IndexType & capacity : newCapacities )
    {
      capacity = rand( 0, maxCapacity );
    }

    m_array.template resizeFromCapacities< POLICY >( newSize, newCapacities.data() );

    EXPECT_EQ( m_array.size(), newSize );
    for( IndexType i = 0; i < m_array.size(); ++i )
    {
      EXPECT_EQ( m_array.sizeOfArray( i ), 0 );
      EXPECT_EQ( m_array.capacityOfArray( i ), newCapacities[ i ] );
    }

    m_ref.clear();
    m_ref.resize( newSize );
    COMPARE_TO_REFERENCE;
  }

  void resizeFromOffsets( IndexType const newSize, IndexType const maxCapacity )
  {
    COMPARE_TO_REFERENCE;

    std::vector< IndexType > newCapacities( newSize );

    for( IndexType & capacity : newCapacities )
    {
      capacity = rand( 0, maxCapacity );
    }

    std::vector< IndexType > newOffsets( newSize + 1 );

    IndexType totalOffset = 0;
    for( IndexType i = 0; i < newSize; ++i )
    {
      newOffsets[i] = totalOffset;
      totalOffset += newCapacities[i];
    }
    newOffsets.back() = totalOffset;

    m_array.resizeFromOffsets( newSize, newOffsets.data() );

    EXPECT_EQ( m_array.size(), newSize );
    for( IndexType i = 0; i < m_array.size(); ++i )
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

    IndexType const newSize = rand( 0, 2 * m_array.size());
    m_array.resize( newSize );
    m_ref.resize( newSize );

    COMPARE_TO_REFERENCE;
  }

  void resizeArray( IndexType const maxSize )
  {
    COMPARE_TO_REFERENCE;

    IndexType const nArrays = m_array.size();
    for( IndexType i = 0; i < nArrays; ++i )
    {
      IndexType const newSize = rand( 0, maxSize );

      T const defaultValue = T( i );
      m_array.resizeArray( i, newSize, defaultValue );
      m_ref[ i ].resize( newSize, defaultValue );
    }

    COMPARE_TO_REFERENCE;
  }

  void emplaceBack( IndexType const maxAppendSize )
  {
    COMPARE_TO_REFERENCE;

    IndexType const nArrays = m_array.size();
    for( IndexType i = 0; i < nArrays; ++i )
    {
      IndexType const nValues = 2 * rand( 0, maxAppendSize / 2 );
      IndexType const arraySize = m_array.sizeOfArray( i );

      for( IndexType j = 0; j < nValues; ++j )
      {
        T const value = T( i * LARGE_NUMBER + arraySize + 2 * j );
        m_array.emplaceBack( i, value );
        m_ref[ i ].emplace_back( value );

        IndexType const seed = i * LARGE_NUMBER + arraySize + 2 * j + 1;
        m_array.emplaceBack( i, seed );
        m_ref[ i ].emplace_back( seed );
      }
    }

    COMPARE_TO_REFERENCE;
  }

  void appendToArray( IndexType const maxAppendSize )
  {
    COMPARE_TO_REFERENCE;

    IndexType const nArrays = m_array.size();

    for( IndexType i = 0; i < nArrays; ++i )
    {
      // Append using a std::vector.
      {
        IndexType const nValues = rand( 0, maxAppendSize / 2 );
        IndexType const arraySize = m_array.sizeOfArray( i );

        std::vector< T > toAppend;
        for( IndexType j = 0; j < nValues; ++j )
        { toAppend.emplace_back( i * LARGE_NUMBER + arraySize + j ); }

        m_array.appendToArray( i, toAppend.begin(), toAppend.end() );
        m_ref[ i ].insert( m_ref[ i ].end(), toAppend.begin(), toAppend.end() );
      }

      // Append using a std::list.
      {
        IndexType const nValues = rand( 0, maxAppendSize / 2 );
        IndexType const arraySize = m_array.sizeOfArray( i );

        std::list< T > toAppend;
        for( IndexType j = 0; j < nValues; ++j )
        { toAppend.emplace_back( i * LARGE_NUMBER + arraySize + j ); }

        m_array.appendToArray( i, toAppend.begin(), toAppend.end() );
        m_ref[ i ].insert( m_ref[ i ].end(), toAppend.begin(), toAppend.end() );
      }
    }

    COMPARE_TO_REFERENCE;
  }

  void emplace( IndexType const maxInsertSize )
  {
    COMPARE_TO_REFERENCE;

    IndexType const nArrays = m_array.size();
    for( IndexType i = 0; i < nArrays; ++i )
    {
      IndexType const arraySize = m_array.sizeOfArray( i );
      IndexType const nValues = 2 * rand( 0, maxInsertSize / 2 );
      for( IndexType j = 0; j < nValues; ++j )
      {
        {
          T const value = T( i * LARGE_NUMBER + arraySize + 2 * j );
          IndexType const insertPos = rand( 0, arraySize + 2 * j );
          m_array.emplace( i, insertPos, value );
          m_ref[ i ].insert( m_ref[ i ].begin() + insertPos, value );
        }

        {
          IndexType const seed = i * LARGE_NUMBER + arraySize + 2 * j + 1;
          IndexType const insertPos = rand( 0, arraySize + 2 * j + 1 );
          m_array.emplace( i, insertPos, seed );
          m_ref[ i ].emplace( m_ref[ i ].begin() + insertPos, seed );
        }
      }
    }

    COMPARE_TO_REFERENCE;
  }

  void insertIntoArray( IndexType const maxInsertSize )
  {
    COMPARE_TO_REFERENCE;


    IndexType const nArrays = m_array.size();
    for( IndexType i = 0; i < nArrays; ++i )
    {
      // Insert using a std::vector
      {
        IndexType const nValues = rand( 0, maxInsertSize / 2 );
        IndexType const arraySize = m_array.sizeOfArray( i );

        std::vector< T > toInsert;
        for( IndexType j = 0; j < nValues; ++j )
        { toInsert.emplace_back( i * LARGE_NUMBER + arraySize + j ); }

        IndexType const insertPos = rand( 0, arraySize );
        m_array.insertIntoArray( i, insertPos, toInsert.begin(), toInsert.end() );
        m_ref[ i ].insert( m_ref[ i ].begin() + insertPos, toInsert.begin(), toInsert.end() );
      }

      // Insert using a std::list
      {
        IndexType const nValues = rand( 0, maxInsertSize / 2 );
        IndexType const arraySize = m_array.sizeOfArray( i );

        std::list< T > toInsert;
        for( IndexType j = 0; j < nValues; ++j )
        { toInsert.emplace_back( i * LARGE_NUMBER + arraySize + j ); }

        IndexType const insertPos = rand( 0, arraySize );
        m_array.insertIntoArray( i, insertPos, toInsert.begin(), toInsert.end() );
        m_ref[ i ].insert( m_ref[ i ].begin() + insertPos, toInsert.begin(), toInsert.end() );
      }
    }

    COMPARE_TO_REFERENCE;
  }

  void eraseFromArray( IndexType const maxRemoves )
  {
    COMPARE_TO_REFERENCE;

    IndexType const nArrays = m_array.size();
    for( IndexType i = 0; i < nArrays; ++i )
    {
      IndexType const arraySize = m_array.sizeOfArray( i );
      IndexType const nToRemove = rand( 0, math::min( arraySize, maxRemoves ));
      for( IndexType j = 0; j < nToRemove; ++j )
      {
        IndexType const removePosition = rand( 0, arraySize - j - 1 );

        m_array.eraseFromArray( i, removePosition );
        m_ref[ i ].erase( m_ref[ i ].begin() + removePosition );
      }
    }

    COMPARE_TO_REFERENCE;
  }

  void eraseMultipleFromArray( IndexType const maxRemoves )
  {
    COMPARE_TO_REFERENCE;

    IndexType const nArrays = m_array.size();
    for( IndexType i = 0; i < nArrays; ++i )
    {
      IndexType const arraySize = m_array.sizeOfArray( i );
      if( arraySize == 0 ) continue;

      IndexType const removePosition = rand( 0, arraySize - 1 );
      IndexType const nToRemove = rand( 0, math::min( maxRemoves, arraySize - removePosition ));

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

    IndexType curOffset = 0;
    for( IndexType i = 0; i < m_array.size(); ++i )
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

    for( IndexType i = 0; i < m_array.size(); ++i )
    {
      fillArray( i );
    }

    T const * const newValues = m_array[0];
    T const * const newEndValues = m_array[m_array.size() - 1];
    ASSERT_EQ( values, newValues );
    ASSERT_EQ( endValues, newEndValues );

    COMPARE_TO_REFERENCE;
  }

  void capacityOfArray( IndexType const maxCapacity )
  {
    COMPARE_TO_REFERENCE;

    for( IndexType i = 0; i < m_array.size(); ++i )
    {
      IndexType const capacity = m_array.capacityOfArray( i );
      IndexType const newCapacity = rand( 0, maxCapacity );

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

    for( IndexType i = 0; i < m_array.size(); ++i )
    {
      IndexType const arraySize = m_array.sizeOfArray( i );
      ASSERT_EQ( arraySize, copy.sizeOfArray( i ));

      T const * const values = m_array[ i ];
      T const * const valuesCopy = copy[ i ];
      ASSERT_NE( values, valuesCopy );

      for( IndexType j = 0; j < arraySize; ++j )
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

    for( IndexType i = 0; i < m_array.size(); ++i )
    {
      IndexType const arraySize = m_array.sizeOfArray( i );
      ASSERT_EQ( arraySize, copy.sizeOfArray( i ));
      T const * const values = m_array[ i ];
      T const * const valuesCopy = copy[ i ];
      ASSERT_EQ( values, valuesCopy );

      for( IndexType j = 0; j < arraySize; ++j )
      {
        ASSERT_EQ( m_array[ i ][ j ], copy[ i ][ j ] );

        copy[ i ][ j ] = T( -i * LARGE_NUMBER - j );

        ASSERT_EQ( m_array[ i ][ j ], T( -i * LARGE_NUMBER - j ));
        ASSERT_EQ( m_array[ i ][ j ], copy[ i ][ j ] );
      }
    }
  }

protected:

  void fillArray( IndexType const i )
  {
    IndexType const arraySize = m_array.sizeOfArray( i );
    IndexType const nToAppend = m_array.capacityOfArray( i ) - arraySize;
    T const * const values = m_array[ i ];
    T const * const endValues = m_array[m_array.size() - 1];
    IndexType const capacity = m_array.capacityOfArray( i );

    for( IndexType j = 0; j < nToAppend; ++j )
    {
      T const valueToAppend( LARGE_NUMBER * i + j );
      m_array.emplaceBack( i, valueToAppend );
      m_ref[ i ].emplace_back( valueToAppend );
    }

    T const * const newValues = m_array[ i ];
    T const * const newEndValues = m_array[m_array.size() - 1];
    IndexType const endCapacity = m_array.capacityOfArray( i );
    ASSERT_EQ( values, newValues );
    ASSERT_EQ( endValues, newEndValues );
    ASSERT_EQ( capacity, endCapacity );
  }

  IndexType rand( IndexType const min, IndexType const max )
  { return std::uniform_int_distribution< IndexType >( min, max )( m_gen ); }

  static constexpr IndexType LARGE_NUMBER = 1E6;

  ARRAY_OF_ARRAYS m_array;

  std::vector< std::vector< T > > m_ref;

  std::mt19937_64 m_gen;
};

using ArrayOfArraysTestTypes = ::testing::Types<
  ArrayOfArrays< int, std::ptrdiff_t, MallocBuffer >
  , ArrayOfArrays< Tensor, std::ptrdiff_t, MallocBuffer >
  , ArrayOfArrays< TestString, std::ptrdiff_t, MallocBuffer >
#if defined(LVARRAY_USE_CHAI)
  , ArrayOfArrays< int, std::ptrdiff_t, ChaiBuffer >
  , ArrayOfArrays< Tensor, std::ptrdiff_t, ChaiBuffer >
  , ArrayOfArrays< TestString, std::ptrdiff_t, ChaiBuffer >
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

  using IndexT = typename TypeParam::IndexType;
  for( IndexT i = 0; i < a.size(); ++i )
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

  using IndexT = typename TypeParam::IndexType;
  for( IndexT i = 0; i < a.size(); ++i )
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
  for( int i = 0; i < 3; ++i )
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
  for( int i = 0; i < 3; ++i )
  {
    this->appendArray( 100, 10 );
    this->eraseArray( this->m_array.size() / 2 );
  }
}

TYPED_TEST( ArrayOfArraysTest, resize )
{
  for( int i = 0; i < 3; ++i )
  {
    this->insertArray( 100, 10 );
    this->resize();
  }
}

TYPED_TEST( ArrayOfArraysTest, resizeWithCapacity )
{
  for( int i = 0; i < 3; ++i )
  {
    this->resize( 100, 10 );
    this->emplace( 5 );
    this->resize( 150, 10 );
    this->emplace( 5 );
  }
}

TYPED_TEST( ArrayOfArraysTest, resizeFromCapacities )
{
  for( int i = 0; i < 3; ++i )
  {
    this->template resizeFromCapacities< serialPolicy >( 100, 10 );
    this->emplace( 10 );

#if defined( RAJA_ENABLE_OPENMP )
    this->template resizeFromCapacities< parallelHostPolicy >( 150, 10 );
    this->emplace( 10 );
#endif
  }
}

TYPED_TEST( ArrayOfArraysTest, resizeFromOffsets )
{
  for( int i = 0; i < 3; ++i )
  {
    this->resizeFromOffsets( 100, 10 );
    this->emplace( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, resizeArray )
{
  this->resize( 100 );

  for( int i = 0; i < 3; ++i )
  {
    this->resizeArray( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, emplaceBack )
{
  this->resize( 100 );

  for( int i = 0; i < 3; ++i )
  {
    this->emplaceBack( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, appendToArray )
{
  this->resize( 100 );

  for( int i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, emplace )
{
  this->resize( 100 );

  for( int i = 0; i < 3; ++i )
  {
    this->emplace( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, insertIntoArray )
{
  this->resize( 100 );

  for( int i = 0; i < 3; ++i )
  {
    this->insertIntoArray( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, eraseFromArray )
{
  this->resize( 100 );

  for( int i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
    this->eraseFromArray( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, eraseMultipleFromArray )
{
  this->resize( 100 );

  for( int i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
    this->eraseMultipleFromArray( 10 );
  }
}

TYPED_TEST( ArrayOfArraysTest, compress )
{
  this->resize( 100 );

  for( int i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
    this->compress();
  }
}

TYPED_TEST( ArrayOfArraysTest, capacity )
{
  this->resize( 100 );

  for( int i = 0; i < 3; ++i )
  {
    this->resizeArray( 10 );
    this->fill();
  }
}

TYPED_TEST( ArrayOfArraysTest, capacityOfArray )
{
  this->resize( 100 );

  for( int i = 0; i < 3; ++i )
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

template< typename U, typename T >
struct ToArray
{};

template< typename U, typename T, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
struct ToArray< U, ArrayOfArrays< T, INDEX_TYPE, BUFFER_TYPE > >
{
  using OneD = Array< U, 1, RAJA::PERM_I, INDEX_TYPE, BUFFER_TYPE >;
  using OneDView = ArrayView< U, 1, 0, INDEX_TYPE, BUFFER_TYPE >;

  using TwoD = Array< U, 2, RAJA::PERM_IJ, INDEX_TYPE, BUFFER_TYPE >;
  using TwoDView = ArrayView< U, 2, 1, INDEX_TYPE, BUFFER_TYPE >;
};


template< typename ARRAY_OF_ARRAYS_POLICY_PAIR >
class ArrayOfArraysViewTest : public ArrayOfArraysTest< typename ARRAY_OF_ARRAYS_POLICY_PAIR::first_type >
{
public:
  using ARRAY_OF_ARRAYS = typename ARRAY_OF_ARRAYS_POLICY_PAIR::first_type;
  using POLICY = typename ARRAY_OF_ARRAYS_POLICY_PAIR::second_type;

  using ParentClass = ArrayOfArraysTest< ARRAY_OF_ARRAYS >;

  using T = typename ParentClass::T;
  using IndexType = typename ParentClass::IndexType;

  using ViewType = typeManipulation::ViewType< ARRAY_OF_ARRAYS >;

  template< typename U >
  using Array1D = typename ToArray< U, ARRAY_OF_ARRAYS >::OneD;

  template< typename U >
  using ArrayView1D = typename ToArray< U, ARRAY_OF_ARRAYS >::OneDView;

  template< typename U >
  using Array2D = typename ToArray< U, ARRAY_OF_ARRAYS >::TwoD;

  template< typename U >
  using ArrayView2D = typename ToArray< U, ARRAY_OF_ARRAYS >::TwoDView;

  void modifyInKernel()
  {
    COMPARE_TO_REFERENCE;

    IndexType const nArrays = m_array.size();

    // Update the view on the device.
    ViewType const view = m_array.toView();
    forall< POLICY >( nArrays, [view] LVARRAY_HOST_DEVICE ( IndexType const i )
        {
          IndexType const sizeOfArray = view.sizeOfArray( i );
          for( IndexType j = 0; j < sizeOfArray; ++j )
          {
            PORTABLE_EXPECT_EQ( &view[ i ][ j ], &view( i, j ) );
            view[ i ][ j ] += T( view[ i ][ j ] );
            view( i, j ) += T( view( i, j ) );
          }
        } );

    // Update the reference.
    for( IndexType i = 0; i < nArrays; ++i )
    {
      for( IndexType j = 0; j < IndexType( m_ref[ i ].size() ); ++j )
      {
        m_ref[ i ][ j ] += T( m_ref[ i ][ j ] );
        m_ref[ i ][ j ] += T( m_ref[ i ][ j ] );
      }
    }

    // Move the view back to the host and compare with the reference.
    forall< serialPolicy >( 1, [view, this] ( IndexType )
    {
      SCOPED_TRACE( "" );
      this->compareToReference( view.toViewConst() );
    } );
  }

  void memoryMotionMove()
  {
    COMPARE_TO_REFERENCE;

    IndexType const nArrays = m_array.size();

    // Update the view on the device.
    ViewType const view = m_array.toView();
    forall< POLICY >( nArrays, [view] LVARRAY_HOST_DEVICE ( IndexType const i )
        {
          IndexType const sizeOfArray = view.sizeOfArray( i );
          for( IndexType j = 0; j < sizeOfArray; ++j )
          {
            PORTABLE_EXPECT_EQ( &view[ i ][ j ], &view( i, j ) );
            view[ i ][ j ] += T( view[ i ][ j ] );
            view( i, j ) += T( view( i, j ) );
          }
        } );

    // Update the reference.
    for( IndexType i = 0; i < nArrays; ++i )
    {
      for( IndexType j = 0; j < IndexType( m_ref[ i ].size() ); ++j )
      {
        m_ref[ i ][ j ] += T( m_ref[ i ][ j ] );
        m_ref[ i ][ j ] += T( m_ref[ i ][ j ] );
      }
    }

    // Move the view back to the host and compare with the reference.
    m_array.move( MemorySpace::host );
    COMPARE_TO_REFERENCE;
  }

  void emplaceBack()
  {
    COMPARE_TO_REFERENCE;

    IndexType const nArrays = m_array.size();
    Array1D< Array1D< IndexType > > seedsToAppend( nArrays );

    // Populate the seedsToAppend array and append the values to the reference.
    for( IndexType i = 0; i < nArrays; ++i )
    {
      IndexType const arraySize = m_array.sizeOfArray( i );
      IndexType maxAppends = m_array.capacityOfArray( i ) - arraySize;
      IndexType const nSeeds = rand( 0, maxAppends / 2 );

      seedsToAppend[ i ].resize( nSeeds );

      for( IndexType j = 0; j < nSeeds; ++j )
      {
        IndexType const seed = i * LARGE_NUMBER + arraySize + 2 * j;
        m_ref[ i ].emplace_back( T( seed ) );
        m_ref[ i ].emplace_back( seed + 1 );
        seedsToAppend[ i ][ j ] = seed;
      }
    }

    // Append to the view on the device.
    ViewType const view = m_array.toView();
    ArrayView1D< ArrayView1D< IndexType const > const > const seeds = seedsToAppend.toNestedViewConst();
    forall< POLICY >( nArrays, [view, seeds] LVARRAY_HOST_DEVICE ( IndexType const i )
        {
          for( IndexType j = 0; j < seeds[ i ].size(); ++j )
          {
            view.emplaceBack( i, T( seeds[ i ][ j ] ) );
            view.emplaceBack( i, seeds[ i ][ j ] + 1 );
          }
        } );

    // Move the view back to the host and compare with the reference.
    m_array.move( MemorySpace::host );
    COMPARE_TO_REFERENCE;
  }

  void view_appendToArray()
  {
    COMPARE_TO_REFERENCE;

    IndexType const nArrays = m_array.size();
    Array1D< Array1D< T > > valuesToAppend( nArrays );

    // Populate the valuesToAppend array and update the reference.
    for( IndexType i = 0; i < nArrays; ++i )
    {
      IndexType const arraySize = m_array.sizeOfArray( i );
      IndexType const maxAppends = m_array.capacityOfArray( i ) - arraySize;
      IndexType const nValues = rand( 0, maxAppends );
      valuesToAppend[ i ].resize( nValues );

      for( IndexType j = 0; j < nValues; ++j )
      {
        T const value = T( i * LARGE_NUMBER + arraySize + j );
        m_ref[ i ].push_back( value );
        valuesToAppend[ i ][ j ] = value;
      }
    }

    // Append to the view on the device.
    ViewType const view = m_array.toView();
    ArrayView1D< ArrayView1D< T const > const > const toAppend = valuesToAppend.toNestedViewConst();
    forall< POLICY >( nArrays, [view, toAppend] LVARRAY_HOST_DEVICE ( IndexType const i )
        {
          view.appendToArray( i, toAppend[ i ].begin(), toAppend[ i ].end() );
        } );

    // Move the view back to the host and compare with the reference.
    m_array.move( MemorySpace::host );
    COMPARE_TO_REFERENCE;
  }

  void view_emplace()
  {
    COMPARE_TO_REFERENCE;

    IndexType const nArrays = m_array.size();
    Array1D< Array1D< IndexType > > seedsToInsert( nArrays );
    Array1D< Array1D< IndexType > > positionsToInsert( nArrays );

    // Populate the valuesToInsert and positionsToInsert arrays and update the reference.
    for( IndexType i = 0; i < nArrays; ++i )
    {
      IndexType const arraySize = m_array.sizeOfArray( i );
      IndexType const maxInserts = m_array.capacityOfArray( i ) - arraySize;
      IndexType const nSeeds = rand( 0, maxInserts / 2 );
      seedsToInsert[ i ].resize( nSeeds );
      positionsToInsert[ i ].resize( nSeeds );

      for( IndexType j = 0; j < nSeeds; ++j )
      {
        IndexType const seed = i * LARGE_NUMBER + arraySize + 2 * j;
        IndexType const insertPos = rand( 0, arraySize + j );

        m_ref[ i ].emplace( m_ref[ i ].begin() + insertPos, T( seed ) );
        m_ref[ i ].emplace( m_ref[ i ].begin() + insertPos, seed + 1 );

        seedsToInsert[ i ][ j ] = seed;
        positionsToInsert[ i ][ j ] = insertPos;
      }
    }

    // Insert into the view on the device.
    ViewType const view = m_array.toView();
    ArrayView1D< ArrayView1D< IndexType const > const > const positions = positionsToInsert.toNestedViewConst();
    ArrayView1D< ArrayView1D< IndexType const > const > const seeds = seedsToInsert.toNestedViewConst();
    forall< POLICY >( nArrays, [view, positions, seeds] LVARRAY_HOST_DEVICE ( IndexType const i )
        {
          for( IndexType j = 0; j < positions[ i ].size(); ++j )
          {
            view.emplace( i, positions[ i ][ j ], T( seeds[ i ][ j ] ) );
            view.emplace( i, positions[ i ][ j ], seeds[ i ][ j ] + 1 );
          }
        } );

    // Move the view back to the host and compare with the reference.
    m_array.move( MemorySpace::host );
    COMPARE_TO_REFERENCE;
  }

  void view_insertIntoArray()
  {
    COMPARE_TO_REFERENCE;

    IndexType const nArrays = m_array.size();
    Array1D< Array1D< T > > valuesToInsert( nArrays );
    Array1D< IndexType > positionsToInsert( nArrays );

    // Populate the valuesToInsert and positionsToInsert arrays and update the reference.
    for( IndexType i = 0; i < nArrays; ++i )
    {
      IndexType const arraySize = m_array.sizeOfArray( i );
      IndexType const maxInserts = m_array.capacityOfArray( i ) - arraySize;
      IndexType const nValues = rand( 0, maxInserts );
      valuesToInsert[ i ].resize( nValues );

      IndexType const insertPos = rand( 0, arraySize );
      positionsToInsert[ i ] = insertPos;

      for( IndexType j = 0; j < nValues; ++j )
      {
        T const value = T( i * LARGE_NUMBER + arraySize + j );
        valuesToInsert[ i ][ j ] = value;
      }

      m_ref[ i ].insert( m_ref[ i ].begin() + insertPos, valuesToInsert[ i ].begin(), valuesToInsert[ i ].end() );
    }

    // Insert into the view on the device.
    ViewType const view = m_array.toView();
    ArrayView1D< IndexType const > const positions = positionsToInsert.toViewConst();
    ArrayView1D< ArrayView1D< T const > const > const values = valuesToInsert.toNestedViewConst();
    forall< POLICY >( nArrays, [view, positions, values] LVARRAY_HOST_DEVICE ( IndexType const i )
        {
          view.insertIntoArray( i, positions[ i ], values[ i ].begin(), values[ i ].end() );
        } );

    // Move the view back to the host and compare with the reference.
    m_array.move( MemorySpace::host );
    COMPARE_TO_REFERENCE;
  }

  void view_erase( IndexType const maxRemoves )
  {
    COMPARE_TO_REFERENCE;

    IndexType const nArrays = m_array.size();
    Array1D< Array1D< IndexType > > positionsToRemove( nArrays );

    // Populate the positionsToRemove array and update the reference.
    for( IndexType i = 0; i < nArrays; ++i )
    {
      IndexType const arraySize = m_array.sizeOfArray( i );
      IndexType const nToRemove = rand( 0, math::min( arraySize, maxRemoves ));
      positionsToRemove[ i ].resize( nToRemove );

      for( IndexType j = 0; j < nToRemove; ++j )
      {
        IndexType const removePosition = rand( 0, arraySize - j - 1 );
        m_ref[ i ].erase( m_ref[ i ].begin() + removePosition );
        positionsToRemove[ i ][ j ] = removePosition;
      }
    }

    // Remove the view on the device.
    ViewType const view = m_array.toView();
    ArrayView1D< ArrayView1D< IndexType const > const > const positions = positionsToRemove.toNestedViewConst();
    forall< POLICY >( nArrays, [view, positions] LVARRAY_HOST_DEVICE ( IndexType const i )
        {
          for( IndexType j = 0; j < positions[ i ].size(); ++j )
          {
            view.eraseFromArray( i, positions[ i ][ j ] );
          }
        } );

    // Move the view back to the host and compare with the reference.
    m_array.move( MemorySpace::host );
    COMPARE_TO_REFERENCE;
  }

  void view_eraseMultiple( IndexType const maxRemoves )
  {
    COMPARE_TO_REFERENCE;

    IndexType const nArrays = m_array.size();
    Array2D< IndexType > toRemove( nArrays, 2 );

    // Populate the toRemove array and update the reference.
    for( IndexType i = 0; i < nArrays; ++i )
    {
      IndexType const arraySize = m_array.sizeOfArray( i );
      if( arraySize == 0 ) continue;

      IndexType const removePosition = rand( 0, arraySize - 1 );
      IndexType const nToRemove = rand( 0, math::min( maxRemoves, arraySize - removePosition ));

      toRemove[ i ][ 0 ] = removePosition;
      toRemove[ i ][ 1 ] = nToRemove;
      m_ref[ i ].erase( m_ref[ i ].begin() + removePosition, m_ref[ i ].begin() + removePosition + nToRemove );
    }

    // Remove from the view on the device.
    ViewType const view = m_array.toView();
    ArrayView2D< IndexType const > const removals = toRemove.toViewConst();
    forall< POLICY >( nArrays, [view, removals] LVARRAY_HOST_DEVICE ( IndexType const i )
        {
          if( view.sizeOfArray( i ) == 0 ) return;
          view.eraseFromArray( i, removals[ i ][ 0 ], removals[ i ][ 1 ] );
        } );

    // Move the view back to the host and compare with the reference.
    m_array.move( MemorySpace::host );
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
  std::pair< ArrayOfArrays< int, std::ptrdiff_t, MallocBuffer >, serialPolicy >
  , std::pair< ArrayOfArrays< Tensor, std::ptrdiff_t, MallocBuffer >, serialPolicy >
  , std::pair< ArrayOfArrays< TestString, std::ptrdiff_t, MallocBuffer >, serialPolicy >
#if defined(LVARRAY_USE_CHAI)
  , std::pair< ArrayOfArrays< int, std::ptrdiff_t, ChaiBuffer >, serialPolicy >
  , std::pair< ArrayOfArrays< Tensor, std::ptrdiff_t, ChaiBuffer >, serialPolicy >
  , std::pair< ArrayOfArrays< TestString, std::ptrdiff_t, ChaiBuffer >, serialPolicy >
#endif

#if ( defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP) ) && defined(LVARRAY_USE_CHAI)
  , std::pair< ArrayOfArrays< int, std::ptrdiff_t, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< ArrayOfArrays< Tensor, std::ptrdiff_t, ChaiBuffer >, parallelDevicePolicy< 32 > >
#endif
  >;
TYPED_TEST_SUITE( ArrayOfArraysViewTest, ArrayOfArraysViewTestTypes, );

TYPED_TEST( ArrayOfArraysViewTest, memoryMotion )
{
  this->resize( 100 );

  for( int i = 0; i < 3; ++i )
  {
    this->appendToArray( 30 );
    this->modifyInKernel();
  }
}

TYPED_TEST( ArrayOfArraysViewTest, memoryMotionMove )
{
  this->resize( 100 );

  for( int i = 0; i < 3; ++i )
  {
    this->appendToArray( 30 );
    this->memoryMotionMove();
  }
}

TYPED_TEST( ArrayOfArraysViewTest, emplaceBack )
{
  this->resize( 100 );

  for( int i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
    this->emplaceBack();
  }
}

TYPED_TEST( ArrayOfArraysViewTest, appendToArray )
{
  this->resize( 100 );

  for( int i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
    this->view_appendToArray();
  }
}

TYPED_TEST( ArrayOfArraysViewTest, emplace )
{
  this->resize( 100 );

  for( int i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
    this->view_emplace();
  }
}

TYPED_TEST( ArrayOfArraysViewTest, insertIntoArray )
{
  this->resize( 100 );

  for( int i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
    this->view_insertIntoArray();
  }
}

TYPED_TEST( ArrayOfArraysViewTest, erase )
{
  this->resize( 100 );

  for( int i = 0; i < 3; ++i )
  {
    this->appendToArray( 10 );
    this->view_erase( 10 );
  }
}

TYPED_TEST( ArrayOfArraysViewTest, eraseMultiple )
{
  this->resize( 100 );

  for( int i = 0; i < 3; ++i )
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


  using ParentClass = ArrayOfArraysViewTest< ARRAY_OF_ARRAYS_POLICY_PAIR >;

  using T = typename ParentClass::T;
  using IndexType = typename ParentClass::IndexType;
  using ViewType = typename ParentClass::ViewType;


  void emplaceBackAtomic( IndexType const numThreads, IndexType const appendsPerArrayPerThread )
  {
    IndexType const nArrays = m_array.size();

    for( IndexType i = 0; i < nArrays; ++i )
    {
      ASSERT_EQ( m_array.sizeOfArray( i ), 0 );
    }

    ViewType const view = m_array.toView();
    forall< POLICY >( numThreads,
                      [view, nArrays, appendsPerArrayPerThread] LVARRAY_HOST_DEVICE ( IndexType const threadNum )
        {
          for( IndexType i = 0; i < nArrays; ++i )
          {
            for( IndexType j = 0; j < appendsPerArrayPerThread; ++j )
            {
              T const value = T( i * LARGE_NUMBER + threadNum * appendsPerArrayPerThread + j );
              view.template emplaceBackAtomic< AtomicPolicy >( i, value );
            }
          }
        } );

    // Now sort each array and check that the values are as expected.
    m_array.move( MemorySpace::host );
    IndexType const appendsPerArray = numThreads * appendsPerArrayPerThread;
    for( IndexType i = 0; i < nArrays; ++i )
    {
      ASSERT_EQ( m_array.sizeOfArray( i ), appendsPerArray );
      ASSERT_LE( m_array.sizeOfArray( i ), m_array.capacityOfArray( i ));

      T * const subArrayPtr = m_array[ i ];
      std::sort( subArrayPtr, subArrayPtr + appendsPerArray );

      for( IndexType j = 0; j < appendsPerArray; ++j )
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
  std::pair< ArrayOfArrays< int, std::ptrdiff_t, MallocBuffer >, serialPolicy >
  , std::pair< ArrayOfArrays< Tensor, std::ptrdiff_t, MallocBuffer >, serialPolicy >
  , std::pair< ArrayOfArrays< TestString, std::ptrdiff_t, MallocBuffer >, serialPolicy >
#if defined(LVARRAY_USE_CHAI)
  , std::pair< ArrayOfArrays< int, std::ptrdiff_t, ChaiBuffer >, serialPolicy >
  , std::pair< ArrayOfArrays< Tensor, std::ptrdiff_t, ChaiBuffer >, serialPolicy >
  , std::pair< ArrayOfArrays< TestString, std::ptrdiff_t, ChaiBuffer >, serialPolicy >
#endif

#if defined(RAJA_ENABLE_OPENMP)
  , std::pair< ArrayOfArrays< int, std::ptrdiff_t, MallocBuffer >, parallelHostPolicy >
  , std::pair< ArrayOfArrays< Tensor, std::ptrdiff_t, MallocBuffer >, parallelHostPolicy >
  , std::pair< ArrayOfArrays< TestString, std::ptrdiff_t, MallocBuffer >, parallelHostPolicy >
#endif
#if defined(RAJA_ENABLE_OPENMP) && defined(LVARRAY_USE_CHAI)
  , std::pair< ArrayOfArrays< int, std::ptrdiff_t, ChaiBuffer >, parallelHostPolicy >
  , std::pair< ArrayOfArrays< Tensor, std::ptrdiff_t, ChaiBuffer >, parallelHostPolicy >
  , std::pair< ArrayOfArrays< TestString, std::ptrdiff_t, ChaiBuffer >, parallelHostPolicy >
#endif

#if ( defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP) ) && defined(LVARRAY_USE_CHAI)
  , std::pair< ArrayOfArrays< int, std::ptrdiff_t, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< ArrayOfArrays< Tensor, std::ptrdiff_t, ChaiBuffer >, parallelDevicePolicy< 32 > >
#endif
  >;
TYPED_TEST_SUITE( ArrayOfArraysViewAtomicTest, ArrayOfArraysViewAtomicTestTypes, );

TYPED_TEST( ArrayOfArraysViewAtomicTest, atomicAppend )
{
  int const NUM_THREADS = 10;
  int const APPENDS_PER_THREAD = 10;

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
