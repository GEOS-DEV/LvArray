/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */


#include "ArrayOfSets.hpp"
#include "ArrayOfArrays.hpp"
#include "Array.hpp"
#include "testUtils.hpp"
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

template< typename U, typename T >
struct ToArray
{};

template< typename U, typename T, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
struct ToArray< U, ArrayOfSets< T, INDEX_TYPE, BUFFER_TYPE > >
{
  using OneD = Array< U, 1, RAJA::PERM_I, INDEX_TYPE, BUFFER_TYPE >;
  using OneDView = ArrayView< U, 1, 0, INDEX_TYPE, BUFFER_TYPE >;
  using AoA = ArrayOfArrays< U, INDEX_TYPE, BUFFER_TYPE >;
};

template< class ARRAY_OF_SETS >
class ArrayOfSetsTest : public ::testing::Test
{
public:
  using T = typename ARRAY_OF_SETS::ValueType;
  using IndexType = typename ARRAY_OF_SETS::IndexType;
  using ViewType = typeManipulation::ViewType< ARRAY_OF_SETS >;
  using ViewTypeConst = typeManipulation::ViewTypeConst< ARRAY_OF_SETS >;

  using ArrayOfArraysT = typename ToArray< T, ARRAY_OF_SETS >::AoA;

  void compareToReference( ViewTypeConst const & view )
  {
    view.consistencyCheck();

    ASSERT_EQ( view.size(), m_ref.size() );

    for( IndexType i = 0; i < view.size(); ++i )
    {
      ASSERT_EQ( view.sizeOfSet( i ), m_ref[ i ].size());

      typename std::set< T >::iterator it = m_ref[ i ].begin();
      for( IndexType j = 0; j < view.sizeOfSet( i ); ++j )
      {
        EXPECT_EQ( view[ i ][ j ], *it++ );
        EXPECT_EQ( view( i, j ), view[ i ][ j ] );
      }
      EXPECT_EQ( it, m_ref[ i ].end());

      it = m_ref[ i ].begin();
      for( T const & val : view[ i ] )
      {
        EXPECT_EQ( val, *it++ );
        EXPECT_TRUE( view.contains( i, val ) );
      }
      EXPECT_EQ( it, m_ref[ i ].end() );
    }
  }

  // Use this macro to call compareToReference with better error messages.
  #define COMPARE_TO_REFERENCE { SCOPED_TRACE( "" ); this->compareToReference( m_array.toViewConst() ); \
  }

  void appendSet( IndexType const nSets, IndexType const maxInserts, IndexType const maxValue )
  {
    COMPARE_TO_REFERENCE

    std::vector< T > arrayToAppend( maxInserts );

    for( IndexType i = 0; i < nSets; ++i )
    {
      IndexType const nValues = rand( 0, maxInserts );
      arrayToAppend.resize( nValues );

      for( IndexType j = 0; j < nValues; ++j )
      {
        arrayToAppend[ j ] = T( rand( 0, maxValue ));
      }

      m_array.appendSet( nValues );
      EXPECT_EQ( nValues, m_array.capacityOfSet( m_array.size() - 1 ) );

      for( T const & val : arrayToAppend )
      {
        m_array.insertIntoSet( m_array.size() - 1, val );
      }

      m_ref.push_back( std::set< T >( arrayToAppend.begin(), arrayToAppend.end() ) );
    }

    COMPARE_TO_REFERENCE
  }

  void insertSet( IndexType const nSets, IndexType const maxInserts, IndexType const maxValue )
  {
    COMPARE_TO_REFERENCE

    std::vector< T > arrayToAppend( maxInserts );

    for( IndexType i = 0; i < nSets; ++i )
    {
      IndexType const nValues = rand( 0, maxInserts );
      arrayToAppend.resize( nValues );

      for( IndexType j = 0; j < nValues; ++j )
      {
        arrayToAppend[ j ] = T( rand( 0, maxValue ));
      }

      IndexType insertPos = rand( 0, m_array.size());

      m_array.insertSet( insertPos, nValues );
      EXPECT_EQ( nValues, m_array.capacityOfSet( insertPos ) );

      for( T const & val : arrayToAppend )
      {
        m_array.insertIntoSet( insertPos, val );
      }

      m_ref.insert( m_ref.begin() + insertPos, std::set< T >( arrayToAppend.begin(), arrayToAppend.end()));
    }

    COMPARE_TO_REFERENCE
  }

  void eraseSet( IndexType const nSets )
  {
    COMPARE_TO_REFERENCE

    for( IndexType i = 0; i < nSets; ++i )
    {
      IndexType const removePos = rand( 0, m_array.size() -1 );
      m_array.eraseSet( removePos );
      m_ref.erase( m_ref.begin() + removePos );
    }

    COMPARE_TO_REFERENCE
  }

  void resize( IndexType const newSize, IndexType const capacityPerSet=0 )
  {
    COMPARE_TO_REFERENCE

    m_array.resize( newSize, capacityPerSet );
    m_ref.resize( newSize );

    COMPARE_TO_REFERENCE
  }

  void resize()
  {
    COMPARE_TO_REFERENCE

    IndexType const newSize = rand( 0, 2 * m_array.size());
    m_array.resize( newSize );
    m_ref.resize( newSize );

    COMPARE_TO_REFERENCE
  }

  template< typename POLICY >
  void resizeFromCapacities( IndexType const newSize, IndexType const maxCapacity )
  {
    COMPARE_TO_REFERENCE;

    std::vector< IndexType > newCapacities( newSize );

    for( IndexType & capacity : newCapacities )
    { capacity = rand( 0, maxCapacity ); }

    m_array.template resizeFromCapacities< POLICY >( newSize, newCapacities.data() );

    EXPECT_EQ( m_array.size(), newSize );
    for( IndexType i = 0; i < m_array.size(); ++i )
    {
      EXPECT_EQ( m_array.sizeOfSet( i ), 0 );
      EXPECT_EQ( m_array.capacityOfSet( i ), newCapacities[ i ] );
    }

    m_ref.clear();
    m_ref.resize( newSize );
    COMPARE_TO_REFERENCE;
  }

  void insertIntoSet( IndexType const maxInserts, IndexType const maxValue )
  {
    COMPARE_TO_REFERENCE

    IndexType const nSets = m_array.size();
    for( IndexType i = 0; i < nSets; ++i )
    {
      IndexType const nValues = rand( 0, maxInserts );

      for( IndexType j = 0; j < nValues; ++j )
      {
        T const value = T( rand( 0, maxValue ));
        EXPECT_EQ( m_array.insertIntoSet( i, value ), m_ref[ i ].insert( value ).second );
      }
    }

    COMPARE_TO_REFERENCE
  }

  void insertMultipleIntoSet( IndexType const maxInserts, IndexType const maxValue )
  {
    COMPARE_TO_REFERENCE

    IndexType const nSets = m_array.size();

    for( IndexType i = 0; i < nSets; ++i )
    {
      IndexType const nValues = rand( 0, maxInserts / 2 );

      // Insert using a std::vector
      {
        std::vector< T > valuesToInsert( nValues );

        for( IndexType j = 0; j < nValues; ++j )
        { valuesToInsert[ j ] = T( rand( 0, maxValue ) ); }

        IndexType const numUnique = sortedArrayManipulation::makeSortedUnique( valuesToInsert.begin(), valuesToInsert.end() );
        valuesToInsert.resize( numUnique );

        IndexType const numInserted = m_array.insertIntoSet( i, valuesToInsert.begin(), valuesToInsert.end() );
        EXPECT_EQ( numInserted, insertIntoRef( i, valuesToInsert ) );
      }

      /// Insert using a std::set
      {
        std::set< T > valuesToInsert;

        for( IndexType j = 0; j < nValues; ++j )
        { valuesToInsert.insert( T( rand( 0, maxValue ) ) ); }

        IndexType const numInserted = m_array.insertIntoSet( i, valuesToInsert.begin(), valuesToInsert.end() );
        EXPECT_EQ( numInserted, insertIntoRef( i, valuesToInsert ) );
      }
    }

    COMPARE_TO_REFERENCE
  }

  void removeFromSet( IndexType const maxInserts, IndexType const maxValue )
  {
    COMPARE_TO_REFERENCE

    IndexType const nSets = m_array.size();
    for( IndexType i = 0; i < nSets; ++i )
    {
      IndexType const nValues = rand( 0, maxInserts );

      for( IndexType j = 0; j < nValues; ++j )
      {
        T const value = T( rand( 0, maxValue ));
        EXPECT_EQ( m_array.removeFromSet( i, value ), m_ref[ i ].erase( value ) );
      }
    }

    COMPARE_TO_REFERENCE
  }

  void removeMultipleFromSet( IndexType const maxInserts, IndexType const maxValue )
  {
    COMPARE_TO_REFERENCE

    IndexType const nSets = m_array.size();

    for( IndexType i = 0; i < nSets; ++i )
    {
      IndexType const nValues = rand( 0, maxInserts );

      // Insert using a std::vector
      {
        std::vector< T > valuesToRemove( nValues );

        for( IndexType j = 0; j < nValues; ++j )
        { valuesToRemove[ j ] = T( rand( 0, maxValue ) ); }

        IndexType const numUnique = sortedArrayManipulation::makeSortedUnique( valuesToRemove.begin(), valuesToRemove.end() );
        valuesToRemove.resize( numUnique );

        IndexType const numRemoved = m_array.removeFromSet( i, valuesToRemove.begin(), valuesToRemove.end() );
        EXPECT_EQ( numRemoved, removeFromRef( i, valuesToRemove ) );
      }

      /// Insert using a std::set
      {
        std::set< T > valuesToRemove;

        for( IndexType j = 0; j < nValues; ++j )
        { valuesToRemove.insert( T( rand( 0, maxValue ) ) ); }

        IndexType const numRemoved = m_array.removeFromSet( i, valuesToRemove.begin(), valuesToRemove.end() );
        EXPECT_EQ( numRemoved, removeFromRef( i, valuesToRemove ) );
      }
    }

    COMPARE_TO_REFERENCE
  }

  void compress()
  {
    COMPARE_TO_REFERENCE

    m_array.compress();
    T const * const values = m_array[0];

    IndexType curOffset = 0;
    for( IndexType i = 0; i < m_array.size(); ++i )
    {
      EXPECT_EQ( m_array.sizeOfSet( i ), m_array.capacityOfSet( i ));

      T const * const curValues = m_array[ i ];
      EXPECT_EQ( values + curOffset, curValues );

      curOffset += m_array.sizeOfSet( i );
    }

    COMPARE_TO_REFERENCE
  }

  void reserveSet( IndexType const maxIncrease )
  {
    IndexType const nSets = m_array.size();
    for( IndexType i = 0; i < nSets; ++i )
    {
      IndexType const increment = rand( -maxIncrease, maxIncrease );
      IndexType const newCapacity = std::max( IndexType( 0 ), m_array.sizeOfSet( i ) + increment );

      m_array.reserveCapacityOfSet( i, newCapacity );
      ASSERT_GE( m_array.capacityOfSet( i ), newCapacity );
    }
  }

  void fill()
  {
    COMPARE_TO_REFERENCE

    T const * const values = m_array[ 0 ];
    T const * const endValues = m_array[m_array.size() - 1];

    for( IndexType i = 0; i < m_array.size(); ++i )
    {
      fillSet( i );
    }

    EXPECT_EQ( m_array[ 0 ].dataIfContiguous(), values );
    EXPECT_EQ( m_array[ m_array.size() - 1 ].dataIfContiguous(), endValues );

    COMPARE_TO_REFERENCE
  }

  void deepCopy()
  {
    COMPARE_TO_REFERENCE

    ARRAY_OF_SETS copy( m_array );

    EXPECT_EQ( m_array.size(), copy.size());

    for( IndexType i = 0; i < m_array.size(); ++i )
    {
      IndexType const setSize = m_array.sizeOfSet( i );
      EXPECT_EQ( setSize, copy.sizeOfSet( i ));

      T const * const values = m_array[ i ];
      T const * const valuesCopy = copy[ i ];
      ASSERT_NE( values, valuesCopy );

      for( IndexType j = setSize -1; j >= 0; --j )
      {
        T const val = m_array( i, j );
        EXPECT_EQ( val, copy( i, j ));
        copy.removeFromSet( i, val );
      }

      EXPECT_EQ( copy.sizeOfSet( i ), 0 );
      EXPECT_EQ( m_array.sizeOfSet( i ), setSize );
    }

    COMPARE_TO_REFERENCE
  }

  void shallowCopy()
  {
    ViewType const copy( m_array.toView());

    EXPECT_EQ( m_array.size(), copy.size());

    for( IndexType i = 0; i < m_array.size(); ++i )
    {
      IndexType const setSize = m_array.sizeOfSet( i );
      EXPECT_EQ( setSize, copy.sizeOfSet( i ));

      T const * const values = m_array[ i ];
      T const * const valuesCopy = copy[ i ];
      EXPECT_EQ( values, valuesCopy );

      for( IndexType j = setSize -1; j >= 0; --j )
      {
        EXPECT_EQ( j, m_array.sizeOfSet( i ) - 1 );
        EXPECT_EQ( j, copy.sizeOfSet( i ) - 1 );

        T const val = m_array( i, j );
        EXPECT_EQ( val, copy( i, j ));
        copy.removeFromSet( i, val );
      }

      EXPECT_EQ( copy.sizeOfSet( i ), 0 );
      EXPECT_EQ( m_array.sizeOfSet( i ), 0 );
    }
  }

protected:

  template< typename CONTAINER >
  IndexType insertIntoRef( IndexType const i, CONTAINER const & values )
  {
    IndexType numInserted = 0;
    for( T const & val : values )
    { numInserted += m_ref[ i ].insert( val ).second; }

    return numInserted;
  }

  template< typename CONTAINER >
  IndexType removeFromRef( IndexType const i, CONTAINER const & values )
  {
    IndexType numRemoved = 0;
    for( T const & val : values )
    { numRemoved += m_ref[ i ].erase( val ); }

    return numRemoved;
  }

  void fillSet( IndexType const i )
  {
    IndexType const setSize = m_array.sizeOfSet( i );
    IndexType const setCapacity = m_array.capacityOfSet( i );
    IndexType nToInsert = setCapacity - setSize;

    T const * const values = m_array[ i ];
    T const * const endValues = m_array[m_array.size() - 1];

    while( nToInsert > 0 )
    {
      T const testValue = T( rand( 0, 2 * setCapacity ));
      bool const inserted = m_array.insertIntoSet( i, testValue );
      EXPECT_EQ( inserted, m_ref[ i ].insert( testValue ).second );
      nToInsert -= inserted;
    }

    ASSERT_EQ( m_array[ i ].dataIfContiguous(), values );
    ASSERT_EQ( m_array[ m_array.size() - 1 ].dataIfContiguous(), endValues );
    ASSERT_EQ( setCapacity, m_array.capacityOfSet( i ));
  }

  IndexType rand( IndexType const min, IndexType const max )
  { return std::uniform_int_distribution< IndexType >( min, max )( m_gen ); }

  static constexpr IndexType LARGE_NUMBER = 1E6;

  ARRAY_OF_SETS m_array;

  std::vector< std::set< T > > m_ref;

  std::mt19937_64 m_gen;
};

using ArrayOfSetsTestTypes = ::testing::Types<
  ArrayOfSets< int, std::ptrdiff_t, MallocBuffer >
  , ArrayOfSets< Tensor, std::ptrdiff_t, MallocBuffer >
  , ArrayOfSets< TestString, std::ptrdiff_t, MallocBuffer >
#if defined(LVARRAY_USE_CHAI)
  , ArrayOfSets< int, std::ptrdiff_t, ChaiBuffer >
  , ArrayOfSets< Tensor, std::ptrdiff_t, ChaiBuffer >
  , ArrayOfSets< TestString, std::ptrdiff_t, ChaiBuffer >
#endif
  >;
TYPED_TEST_SUITE( ArrayOfSetsTest, ArrayOfSetsTestTypes, );

int const DEFAULT_MAX_INSERTS = 25;
int const DEFAULT_MAX_VALUE = 2 * DEFAULT_MAX_INSERTS;

TYPED_TEST( ArrayOfSetsTest, emptyConstruction )
{
  TypeParam a;
  ASSERT_EQ( a.size(), 0 );
  ASSERT_EQ( a.capacity(), 0 );
}

TYPED_TEST( ArrayOfSetsTest, sizedConstruction )
{
  TypeParam a( 10 );
  ASSERT_EQ( a.size(), 10 );
  ASSERT_EQ( a.capacity(), 10 );

  using IndexT = typename TypeParam::IndexType;
  for( IndexT i = 0; i < a.size(); ++i )
  {
    ASSERT_EQ( a.sizeOfSet( i ), 0 );
    ASSERT_EQ( a.capacityOfSet( i ), 0 );
  }
}

TYPED_TEST( ArrayOfSetsTest, sizedHintConstruction )
{
  TypeParam a( 10, 20 );
  ASSERT_EQ( a.size(), 10 );
  ASSERT_EQ( a.capacity(), 10 );

  using IndexT = typename TypeParam::IndexType;
  for( IndexT i = 0; i < a.size(); ++i )
  {
    ASSERT_EQ( a.sizeOfSet( i ), 0 );
    ASSERT_EQ( a.capacityOfSet( i ), 20 );
  }
}

TYPED_TEST( ArrayOfSetsTest, appendSet )
{
  this->appendSet( 50, DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
}

TYPED_TEST( ArrayOfSetsTest, insertSet )
{
  for( int i = 0; i < 2; ++i )
  {
    this->insertSet( 50, DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
  }
}

TYPED_TEST( ArrayOfSetsTest, eraseArray )
{
  for( int i = 0; i < 2; ++i )
  {
    this->appendSet( 50, DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
    this->eraseSet( this->m_array.size() / 2 );
  }
}

TYPED_TEST( ArrayOfSetsTest, resize )
{
  for( int i = 0; i < 2; ++i )
  {
    this->insertSet( 50, DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
    this->resize();
  }
}

TYPED_TEST( ArrayOfSetsTest, resizeFromCapacities )
{
  for( int i = 0; i < 2; ++i )
  {
    this->template resizeFromCapacities< serialPolicy >( 100, DEFAULT_MAX_INSERTS );
    this->insertIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );

#if defined( RAJA_ENABLE_OPENMP )
    this->template resizeFromCapacities< parallelHostPolicy >( 150, DEFAULT_MAX_INSERTS );
    this->insertIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
#endif
  }
}

TYPED_TEST( ArrayOfSetsTest, insertIntoSet )
{
  this->resize( 50 );
  for( int i = 0; i < 2; ++i )
  {
    this->insertIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
  }
}

TYPED_TEST( ArrayOfSetsTest, insertMultipleIntoSet )
{
  this->resize( 1 );
  for( int i = 0; i < 2; ++i )
  {
    this->insertMultipleIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
  }
}

TYPED_TEST( ArrayOfSetsTest, removeFromSet )
{
  this->resize( 50 );
  for( int i = 0; i < 2; ++i )
  {
    this->insertIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
    this->removeFromSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
  }
}

TYPED_TEST( ArrayOfSetsTest, removeMultipleFromSet )
{
  this->resize( 50 );
  for( int i = 0; i < 2; ++i )
  {
    this->insertIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
    this->removeMultipleFromSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
  }
}

TYPED_TEST( ArrayOfSetsTest, compress )
{
  this->resize( 50 );
  for( int i = 0; i < 2; ++i )
  {
    this->insertIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
    this->compress();
  }
}

TYPED_TEST( ArrayOfSetsTest, capacity )
{
  this->resize( 50 );
  for( int i = 0; i < 2; ++i )
  {
    this->reserveSet( 50 );
    this->fill();
  }
}

TYPED_TEST( ArrayOfSetsTest, deepCopy )
{
  this->resize( 50 );
  this->insertIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
  this->deepCopy();
}

TYPED_TEST( ArrayOfSetsTest, shallowCopy )
{
  this->resize( 50 );
  this->insertIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
  this->shallowCopy();
}

// This is testing capabilities of the ArrayOfArrays class, however it needs to first populate
// the ArrayOfSets so it involves less code duplication to put it here.
TYPED_TEST( ArrayOfSetsTest, ArrayOfArraysStealFrom )
{
  typename ArrayOfSetsTest< TypeParam >::ArrayOfArraysT array;

  this->resize( 50 );
  this->insertIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
  array.assimilate( std::move( this->m_array ) );

  ASSERT_EQ( array.size(), this->m_ref.size());

  using IndexT = typename TypeParam::IndexType;
  for( IndexT i = 0; i < array.size(); ++i )
  {
    ASSERT_EQ( array.sizeOfArray( i ), this->m_ref[ i ].size());

    auto it = this->m_ref[ i ].begin();
    for( IndexT j = 0; j < array.sizeOfArray( i ); ++j )
    {
      EXPECT_EQ( array[ i ][ j ], *it++ );
    }
    EXPECT_EQ( it, this->m_ref[ i ].end());
  }
}

template< class ARRAY_OF_SETS_POLICY_PAIR >
class ArrayOfSetsViewTest : public ArrayOfSetsTest< typename ARRAY_OF_SETS_POLICY_PAIR::first_type >
{
public:

  using ARRAY_OF_SETS = typename ARRAY_OF_SETS_POLICY_PAIR::first_type;
  using POLICY = typename ARRAY_OF_SETS_POLICY_PAIR::second_type;
  using ParentClass = ArrayOfSetsTest< ARRAY_OF_SETS >;

  using typename ParentClass::T;
  using typename ParentClass::IndexType;
  using typename ParentClass::ViewType;
  using typename ParentClass::ViewTypeConst;
  using typename ParentClass::ArrayOfArraysT;

  template< typename U >
  using Array1D = typename ToArray< U, ARRAY_OF_SETS >::OneD;

  template< typename U >
  using ArrayView1D = typename ToArray< U, ARRAY_OF_SETS >::OneDView;

  void memoryMotion()
  {
    COMPARE_TO_REFERENCE

    IndexType const nSets = m_array.size();

    // Update the view on the device.
    ViewType const view = m_array.toView();
    forall< POLICY >( nSets, [view] LVARRAY_HOST_DEVICE ( IndexType const i )
        {
          IndexType const sizeOfSet = view.sizeOfSet( i );
          for( IndexType j = 0; j < sizeOfSet; ++j )
          {
            PORTABLE_EXPECT_EQ( &view[ i ][ j ], &view( i, j ) );
          }

          for( T const & val : view[ i ] )
          {
            PORTABLE_EXPECT_EQ( view.contains( i, val ), true );
          }
        } );

    // Move the view back to the host and compare with the reference.
    forall< serialPolicy >( 1, [view, this] ( IndexType )
    {
      this->compareToReference( view.toViewConst() );
    } );
  }

  void memoryMotionMove()
  {
    COMPARE_TO_REFERENCE

    IndexType const nSets = m_array.size();

    ViewType const view = m_array.toView();
    forall< POLICY >( nSets, [view] LVARRAY_HOST_DEVICE ( IndexType const i )
        {
          IndexType const sizeOfSet = view.sizeOfSet( i );
          for( IndexType j = 0; j < sizeOfSet; ++j )
          {
            PORTABLE_EXPECT_EQ( &view[ i ][ j ], &view( i, j ) );
          }

          for( T const & val : view[ i ] )
          {
            PORTABLE_EXPECT_EQ( view.contains( i, val ), true );
          }
        } );

    // Move the view back to the host and compare with the reference.
    m_array.move( MemorySpace::host );
    COMPARE_TO_REFERENCE
  }

  void insertView()
  {
    COMPARE_TO_REFERENCE

    Array1D< Array1D< T > > toInsert = createValues( true, false );
    ArrayView1D< ArrayView1D< T const > const > const toInsertView = toInsert.toNestedViewConst();

    ViewType const view = m_array.toView();
    forall< POLICY >( m_array.size(), [view, toInsertView] LVARRAY_HOST_DEVICE ( IndexType const i )
        {
          for( IndexType j = 0; j < toInsertView[ i ].size(); ++j )
          {
            view.insertIntoSet( i, toInsertView[ i ][ j ] );
          }
        } );

    m_array.move( MemorySpace::host );
    COMPARE_TO_REFERENCE
  }

  void insertMultipleView()
  {
    COMPARE_TO_REFERENCE

    Array1D< Array1D< T > > const toInsert = createValues( true, true );
    ArrayView1D< ArrayView1D< T const > const > const toInsertView = toInsert.toNestedViewConst();

    ViewType const view = m_array.toView();
    forall< POLICY >( m_array.size(), [view, toInsertView] LVARRAY_HOST_DEVICE ( IndexType const i )
        {
          view.insertIntoSet( i, toInsertView[ i ].begin(), toInsertView[ i ].end() );
        } );

    m_array.move( MemorySpace::host );
    COMPARE_TO_REFERENCE
  }

  void removeView()
  {
    COMPARE_TO_REFERENCE

    Array1D< Array1D< T > > const toRemove = createValues( false, false );
    ArrayView1D< ArrayView1D< T const > const > const toRemoveView = toRemove.toNestedViewConst();

    ViewType const view = m_array.toView();
    forall< POLICY >( m_array.size(), [view, toRemoveView] LVARRAY_HOST_DEVICE ( IndexType const i )
        {
          for( IndexType j = 0; j < toRemoveView[ i ].size(); ++j )
          {
            view.removeFromSet( i, toRemoveView[ i ][ j ] );
          }
        } );

    m_array.move( MemorySpace::host );
    COMPARE_TO_REFERENCE
  }

  void removeMultipleView()
  {
    COMPARE_TO_REFERENCE

    Array1D< Array1D< T > > const toRemove = createValues( false, true );
    ArrayView1D< ArrayView1D< T const > const > const toRemoveView = toRemove.toNestedViewConst();

    ViewType const view = m_array.toView();
    forall< POLICY >( m_array.size(), [view, toRemoveView] LVARRAY_HOST_DEVICE ( IndexType const i )
        {
          view.removeFromSet( i, toRemoveView[ i ].begin(), toRemoveView[ i ].end() );
        } );

    m_array.move( MemorySpace::host );
    COMPARE_TO_REFERENCE
  }

  // This is not a View test, but it needs a POLICY type argument, so it's placed here for convenience
  void assimilate( IndexType const maxValue, IndexType const maxInserts, sortedArrayManipulation::Description const desc )
  {
    IndexType const nSets = m_array.size();
    ArrayOfArraysT arrayToSteal( nSets );

    for( IndexType i = 0; i < nSets; ++i )
    {
      IndexType const nValues = rand( 0, maxInserts );

      for( IndexType j = 0; j < nValues; ++j )
      {
        T const value = T( rand( 0, maxValue ));
        bool const insertSuccess = m_ref[ i ].insert( value ).second;

        if( sortedArrayManipulation::isUnique( desc ))
        {
          if( insertSuccess )
          {
            arrayToSteal.emplaceBack( i, value );
          }
        }
        else
        {
          arrayToSteal.emplaceBack( i, value );
        }
      }

      if( sortedArrayManipulation::isSorted( desc ))
      {
        T * const values = arrayToSteal[ i ];
        std::sort( values, values + arrayToSteal.sizeOfArray( i ));
      }
    }

    m_array.template assimilate< POLICY >( std::move( arrayToSteal ), desc );

    m_array.move( MemorySpace::host );
    COMPARE_TO_REFERENCE
  }

protected:

  Array1D< Array1D< T > > createValues( bool const insert, bool const sortedUnique )
  {
    IndexType const nSets = m_array.size();

    Array1D< Array1D< T > > values( nSets );

    for( IndexType i = 0; i < nSets; ++i )
    {
      IndexType const sizeOfSet = m_array.sizeOfSet( i );
      IndexType const capacityOfSet = m_array.capacityOfSet( i );

      IndexType const upperBound = insert ? capacityOfSet - sizeOfSet : capacityOfSet;
      IndexType const nValues = rand( 0, upperBound );

      values[ i ].resize( nValues );

      for( IndexType j = 0; j < nValues; ++j )
      {
        T const value = T( rand( 0, 2 * capacityOfSet ) );
        values[ i ][ j ] = value;

        if( insert )
        { m_ref[ i ].insert( value ); }
        else
        { m_ref[ i ].erase( value ); }
      }

      if( sortedUnique )
      {
        IndexType const numUnique = sortedArrayManipulation::makeSortedUnique( values[ i ].begin(), values[ i ].end() );
        values[ i ].resize( numUnique );
      }
    }

    return values;
  }

  using ParentClass::rand;
  using ParentClass::LARGE_NUMBER;
  using ParentClass::m_array;
  using ParentClass::m_ref;
  using ParentClass::m_gen;
};

using ArrayOfSetsViewTestTypes = ::testing::Types<
  std::pair< ArrayOfSets< int, std::ptrdiff_t, MallocBuffer >, serialPolicy >
  , std::pair< ArrayOfSets< Tensor, std::ptrdiff_t, MallocBuffer >, serialPolicy >
  , std::pair< ArrayOfSets< TestString, std::ptrdiff_t, MallocBuffer >, serialPolicy >
#if defined(LVARRAY_USE_CHAI)
  , std::pair< ArrayOfSets< int, std::ptrdiff_t, ChaiBuffer >, serialPolicy >
  , std::pair< ArrayOfSets< Tensor, std::ptrdiff_t, ChaiBuffer >, serialPolicy >
  , std::pair< ArrayOfSets< TestString, std::ptrdiff_t, ChaiBuffer >, serialPolicy >
#endif

#if ( defined(LVARRAY_USE_CUDA)  || defined(LVARRAY_USE_HIP) ) && defined(LVARRAY_USE_CHAI)
  , std::pair< ArrayOfSets< int, std::ptrdiff_t, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< ArrayOfSets< Tensor, std::ptrdiff_t, ChaiBuffer >, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( ArrayOfSetsViewTest, ArrayOfSetsViewTestTypes, );

TYPED_TEST( ArrayOfSetsViewTest, memoryMotion )
{
  this->resize( 50 );
  for( int i = 0; i < 2; ++i )
  {
    this->insertIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
    this->memoryMotion();
  }
}

TYPED_TEST( ArrayOfSetsViewTest, memoryMotionMove )
{
  this->resize( 50 );
  for( int i = 0; i < 2; ++i )
  {
    this->insertIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
    this->memoryMotionMove();
  }
}

TYPED_TEST( ArrayOfSetsViewTest, insert )
{
  this->resize( 50, 10 );
  for( int i = 0; i < 2; ++i )
  {
    this->insertView();
  }
}

TYPED_TEST( ArrayOfSetsViewTest, insertMultiple )
{
  this->resize( 50, 10 );
  for( int i = 0; i < 2; ++i )
  {
    this->insertMultipleView();
  }
}

TYPED_TEST( ArrayOfSetsViewTest, remove )
{
  this->resize( 50, 10 );
  for( int i = 0; i < 2; ++i )
  {
    this->insertMultipleIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
    this->removeView();
  }
}

TYPED_TEST( ArrayOfSetsViewTest, removeMultiple )
{
  this->resize( 50, 10 );
  for( int i = 0; i < 2; ++i )
  {
    this->insertMultipleIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
    this->removeMultipleView();
  }
}

TYPED_TEST( ArrayOfSetsViewTest, assimilateSortedUnique )
{
  this->resize( 50 );
  this->assimilate( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE, sortedArrayManipulation::SORTED_UNIQUE );
}

TYPED_TEST( ArrayOfSetsViewTest, assimilateUnsortedNoDuplicates )
{
  this->resize( 50 );
  this->assimilate( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE, sortedArrayManipulation::UNSORTED_NO_DUPLICATES );
}

TYPED_TEST( ArrayOfSetsViewTest, assimilateSortedWithDuplicates )
{
  this->resize( 50 );
  this->assimilate( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE, sortedArrayManipulation::SORTED_WITH_DUPLICATES );
}

TYPED_TEST( ArrayOfSetsViewTest, assimilateUnsortedWithDuplicates )
{
  this->resize( 50 );
  this->assimilate( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE, sortedArrayManipulation::UNSORTED_WITH_DUPLICATES );
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
