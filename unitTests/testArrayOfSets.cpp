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


#include "ArrayOfSets.hpp"
#include "ArrayOfArrays.hpp"
#include "Array.hpp"
#include "testUtils.hpp"

/// TPL includes
#include <gtest/gtest.h>

/// System includes
#include <vector>
#include <random>

namespace LvArray
{
namespace testing
{

using INDEX_TYPE = std::ptrdiff_t;

template< class T >
using ViewType = ArrayOfSetsView< T, INDEX_TYPE const >;

template< class T >
using REF_TYPE = std::vector< std::set< T > >;

template< class T >
void compareToReference( ViewType< T const > const & view, REF_TYPE< T > const & ref )
{
  view.consistencyCheck();

  ASSERT_EQ( view.size(), ref.size());

  for( INDEX_TYPE i = 0; i < view.size(); ++i )
  {
    ASSERT_EQ( view.sizeOfSet( i ), ref[i].size());

    typename std::set< T >::iterator it = ref[i].begin();
    for( INDEX_TYPE j = 0; j < view.sizeOfSet( i ); ++j )
    {
      EXPECT_EQ( view[i][j], *it++ );
      EXPECT_EQ( view( i, j ), view[i][j] );
    }
    EXPECT_EQ( it, ref[i].end());

    it = ref[i].begin();
    for( T const & val : view.getIterableSet( i ))
    {
      EXPECT_EQ( val, *it++ );
      EXPECT_TRUE( view.contains( i, val ));
    }
    EXPECT_EQ( it, ref[i].end());
  }
}

// Use this macro to call compareToReference with better error messages.
#define COMPARE_TO_REFERENCE( view, ref ) { SCOPED_TRACE( "" ); compareToReference( view, ref ); \
}

template< class T >
void printArray( ViewType< T const > const & view )
{
  std::cout << "{" << std::endl;

  for( INDEX_TYPE i = 0; i < view.size(); ++i )
  {
    std::cout << "\t{";
    for( INDEX_TYPE j = 0; j < view.sizeOfSet( i ); ++j )
    {
      std::cout << view[i][j] << ", ";
    }

    for( INDEX_TYPE j = view.sizeOfSet( i ); j < view.capacityOfSet( i ); ++j )
    {
      std::cout << "X" << ", ";
    }

    std::cout << "}" << std::endl;
  }

  std::cout << "}" << std::endl;
}

template< class T >
class ArrayOfSetsTest : public ::testing::Test
{
public:

  void appendSet( INDEX_TYPE const nSets, INDEX_TYPE const maxInserts, INDEX_TYPE const maxValue )
  {
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );

    std::vector< T > arrayToAppend( maxInserts );

    for( INDEX_TYPE i = 0; i < nSets; ++i )
    {
      INDEX_TYPE const nValues = rand( 0, maxInserts );
      arrayToAppend.resize( nValues );

      for( INDEX_TYPE j = 0; j < nValues; ++j )
      {
        arrayToAppend[j] = T( rand( 0, maxValue ));
      }

      m_array.appendSet( nValues );
      EXPECT_EQ( nValues, m_array.capacityOfSet( m_array.size() - 1 ) );

      for( T const & val : arrayToAppend )
      {
        m_array.insertIntoSet( m_array.size() - 1, val );
      }

      m_ref.push_back( std::set< T >( arrayToAppend.begin(), arrayToAppend.end() ) );
    }

    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );
  }

  void insertSet( INDEX_TYPE const nSets, INDEX_TYPE const maxInserts, INDEX_TYPE const maxValue )
  {
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );

    std::vector< T > arrayToAppend( maxInserts );

    for( INDEX_TYPE i = 0; i < nSets; ++i )
    {
      INDEX_TYPE const nValues = rand( 0, maxInserts );
      arrayToAppend.resize( nValues );

      for( INDEX_TYPE j = 0; j < nValues; ++j )
      {
        arrayToAppend[j] = T( rand( 0, maxValue ));
      }

      INDEX_TYPE insertPos = rand( 0, m_array.size());

      m_array.insertSet( insertPos, nValues );
      EXPECT_EQ( nValues, m_array.capacityOfSet( insertPos ) );

      for( T const & val : arrayToAppend )
      {
        m_array.insertIntoSet( insertPos, val );
      }

      m_ref.insert( m_ref.begin() + insertPos, std::set< T >( arrayToAppend.begin(), arrayToAppend.end()));
    }

    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );
  }

  void eraseSet( INDEX_TYPE const nSets )
  {
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );

    for( INDEX_TYPE i = 0; i < nSets; ++i )
    {
      INDEX_TYPE const removePos = rand( 0, m_array.size() -1 );
      m_array.eraseSet( removePos );
      m_ref.erase( m_ref.begin() + removePos );
    }

    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );
  }

  void resize( INDEX_TYPE const newSize, INDEX_TYPE const capacityPerSet=0 )
  {
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );

    m_array.resize( newSize, capacityPerSet );
    m_ref.resize( newSize );

    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );
  }

  void resize()
  {
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );

    INDEX_TYPE const newSize = rand( 0, 2 * m_array.size());
    m_array.resize( newSize );
    m_ref.resize( newSize );

    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );
  }

  void insertIntoSet( INDEX_TYPE const maxInserts, INDEX_TYPE const maxValue )
  {
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );

    INDEX_TYPE const nSets = m_array.size();
    for( INDEX_TYPE i = 0; i < nSets; ++i )
    {
      INDEX_TYPE const nValues = rand( 0, maxInserts );

      for( INDEX_TYPE j = 0; j < nValues; ++j )
      {
        T const value = T( rand( 0, maxValue ));
        EXPECT_EQ( m_array.insertIntoSet( i, value ), m_ref[i].insert( value ).second );
      }
    }

    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );
  }

  void insertMultipleIntoSet( INDEX_TYPE const maxInserts, INDEX_TYPE const maxValue )
  {
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );

    INDEX_TYPE const nSets = m_array.size();

    for( INDEX_TYPE i = 0; i < nSets; ++i )
    {
      INDEX_TYPE const nValues = rand( 0, maxInserts / 2 );

      // Insert using a std::vector
      {
        std::vector< T > valuesToInsert( nValues );

        for( INDEX_TYPE j = 0; j < nValues; ++j )
        { valuesToInsert[ j ] = T( rand( 0, maxValue ) ); }

        INDEX_TYPE const numUnique = sortedArrayManipulation::makeSortedUnique( valuesToInsert.begin(), valuesToInsert.end() );
        valuesToInsert.resize( numUnique );

        INDEX_TYPE const numInserted = m_array.insertIntoSet( i, valuesToInsert.begin(), valuesToInsert.end() );
        EXPECT_EQ( numInserted, insertIntoRef( i, valuesToInsert ) );
      }

      /// Insert using a std::set
      {
        std::set< T > valuesToInsert;

        for( INDEX_TYPE j = 0; j < nValues; ++j )
        { valuesToInsert.insert( T( rand( 0, maxValue ) ) ); }

        INDEX_TYPE const numInserted = m_array.insertIntoSet( i, valuesToInsert.begin(), valuesToInsert.end() );
        EXPECT_EQ( numInserted, insertIntoRef( i, valuesToInsert ) );
      }
    }

    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );
  }

  void removeFromSet( INDEX_TYPE const maxInserts, INDEX_TYPE const maxValue )
  {
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );

    INDEX_TYPE const nSets = m_array.size();
    for( INDEX_TYPE i = 0; i < nSets; ++i )
    {
      INDEX_TYPE const nValues = rand( 0, maxInserts );

      for( INDEX_TYPE j = 0; j < nValues; ++j )
      {
        T const value = T( rand( 0, maxValue ));
        EXPECT_EQ( m_array.removeFromSet( i, value ), m_ref[i].erase( value ) );
      }
    }

    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );
  }

  void removeMultipleFromSet( INDEX_TYPE const maxInserts, INDEX_TYPE const maxValue )
  {
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );

    INDEX_TYPE const nSets = m_array.size();

    for( INDEX_TYPE i = 0; i < nSets; ++i )
    {
      INDEX_TYPE const nValues = rand( 0, maxInserts );

      // Insert using a std::vector
      {
        std::vector< T > valuesToRemove( nValues );

        for( INDEX_TYPE j = 0; j < nValues; ++j )
        { valuesToRemove[ j ] = T( rand( 0, maxValue ) ); }

        INDEX_TYPE const numUnique = sortedArrayManipulation::makeSortedUnique( valuesToRemove.begin(), valuesToRemove.end() );
        valuesToRemove.resize( numUnique );

        INDEX_TYPE const numRemoved = m_array.removeFromSet( i, valuesToRemove.begin(), valuesToRemove.end() );
        EXPECT_EQ( numRemoved, removeFromRef( i, valuesToRemove ) );
      }

      /// Insert using a std::set
      {
        std::set< T > valuesToRemove;

        for( INDEX_TYPE j = 0; j < nValues; ++j )
        { valuesToRemove.insert( T( rand( 0, maxValue ) ) ); }

        INDEX_TYPE const numRemoved = m_array.removeFromSet( i, valuesToRemove.begin(), valuesToRemove.end() );
        EXPECT_EQ( numRemoved, removeFromRef( i, valuesToRemove ) );
      }
    }

    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );
  }

  void compress()
  {
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );

    m_array.compress();
    T const * const values = m_array[0];

    INDEX_TYPE curOffset = 0;
    for( INDEX_TYPE i = 0; i < m_array.size(); ++i )
    {
      EXPECT_EQ( m_array.sizeOfSet( i ), m_array.capacityOfSet( i ));

      T const * const curValues = m_array[i];
      EXPECT_EQ( values + curOffset, curValues );

      curOffset += m_array.sizeOfSet( i );
    }

    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );
  }

  void reserveSet( INDEX_TYPE const maxIncrease )
  {
    INDEX_TYPE const nSets = m_array.size();
    for( INDEX_TYPE i = 0; i < nSets; ++i )
    {
      INDEX_TYPE const increment = rand( -maxIncrease, maxIncrease );
      INDEX_TYPE const newCapacity = std::max( INDEX_TYPE( 0 ), m_array.sizeOfSet( i ) + increment );

      m_array.reserveCapacityOfSet( i, newCapacity );
      ASSERT_GE( m_array.capacityOfSet( i ), newCapacity );
    }
  }

  void fill()
  {
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );

    T const * const values = m_array[0];
    T const * const endValues = m_array[m_array.size() - 1];

    for( INDEX_TYPE i = 0; i < m_array.size(); ++i )
    {
      fillSet( i );
    }

    EXPECT_EQ( m_array[0], values );
    EXPECT_EQ( m_array[m_array.size() - 1], endValues );

    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );
  }

  void deepCopy()
  {
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );

    ArrayOfSets< T > copy( m_array );

    EXPECT_EQ( m_array.size(), copy.size());

    for( INDEX_TYPE i = 0; i < m_array.size(); ++i )
    {
      INDEX_TYPE const setSize = m_array.sizeOfSet( i );
      EXPECT_EQ( setSize, copy.sizeOfSet( i ));

      T const * const values = m_array[i];
      T const * const valuesCopy = copy[i];
      ASSERT_NE( values, valuesCopy );

      for( INDEX_TYPE j = setSize -1; j >= 0; --j )
      {
        T const val = m_array( i, j );
        EXPECT_EQ( val, copy( i, j ));
        copy.removeFromSet( i, val );
      }

      EXPECT_EQ( copy.sizeOfSet( i ), 0 );
      EXPECT_EQ( m_array.sizeOfSet( i ), setSize );
    }

    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );
  }

  void shallowCopy()
  {
    ViewType< T > const copy( m_array.toView());

    EXPECT_EQ( m_array.size(), copy.size());

    for( INDEX_TYPE i = 0; i < m_array.size(); ++i )
    {
      INDEX_TYPE const setSize = m_array.sizeOfSet( i );
      EXPECT_EQ( setSize, copy.sizeOfSet( i ));

      T const * const values = m_array[i];
      T const * const valuesCopy = copy[i];
      EXPECT_EQ( values, valuesCopy );

      for( INDEX_TYPE j = setSize -1; j >= 0; --j )
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

  void stealFrom( INDEX_TYPE const maxValue, INDEX_TYPE const maxInserts, sortedArrayManipulation::Description const desc )
  {
    INDEX_TYPE const nSets = m_array.size();
    ArrayOfArrays< T > arrayToSteal( nSets );

    for( INDEX_TYPE i = 0; i < nSets; ++i )
    {
      INDEX_TYPE const nValues = rand( 0, maxInserts );

      for( INDEX_TYPE j = 0; j < nValues; ++j )
      {
        T const value = T( rand( 0, maxValue ));
        bool const insertSuccess = m_ref[i].insert( value ).second;

        if( sortedArrayManipulation::isUnique( desc ))
        {
          if( insertSuccess )
          {
            arrayToSteal.appendToArray( i, value );
          }
        }
        else
        {
          arrayToSteal.appendToArray( i, value );
        }
      }

      if( sortedArrayManipulation::isSorted( desc ))
      {
        T * const values = arrayToSteal[i];
        std::sort( values, values + arrayToSteal.sizeOfArray( i ));
      }
    }

    m_array.stealFrom( std::move( arrayToSteal ), desc );

    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );
  }

protected:

  template< typename CONTAINER >
  INDEX_TYPE insertIntoRef( INDEX_TYPE const i, CONTAINER const & values )
  {
    INDEX_TYPE numInserted = 0;
    for( T const & val : values )
    { numInserted += m_ref[ i ].insert( val ).second; }

    return numInserted;
  }

  template< typename CONTAINER >
  INDEX_TYPE removeFromRef( INDEX_TYPE const i, CONTAINER const & values )
  {
    INDEX_TYPE numRemoved = 0;
    for( T const & val : values )
    { numRemoved += m_ref[ i ].erase( val ); }

    return numRemoved;
  }

  void fillSet( INDEX_TYPE const i )
  {
    INDEX_TYPE const setSize = m_array.sizeOfSet( i );
    INDEX_TYPE const setCapacity = m_array.capacityOfSet( i );
    INDEX_TYPE nToInsert = setCapacity - setSize;

    T const * const values = m_array[i];
    T const * const endValues = m_array[m_array.size() - 1];

    while( nToInsert > 0 )
    {
      T const testValue = T( rand( 0, 2 * setCapacity ));
      bool const inserted = m_array.insertIntoSet( i, testValue );
      EXPECT_EQ( inserted, m_ref[i].insert( testValue ).second );
      nToInsert -= inserted;
    }

    ASSERT_EQ( m_array[i], values );
    ASSERT_EQ( m_array[m_array.size() - 1], endValues );
    ASSERT_EQ( setCapacity, m_array.capacityOfSet( i ));
  }

  INDEX_TYPE rand( INDEX_TYPE const min, INDEX_TYPE const max )
  { return std::uniform_int_distribution< INDEX_TYPE >( min, max )( m_gen ); }

  static constexpr INDEX_TYPE LARGE_NUMBER = 1E6;

  ArrayOfSets< T > m_array;

  std::vector< std::set< T > > m_ref;

  std::mt19937_64 m_gen;
};

using TestTypes = ::testing::Types<
  INDEX_TYPE
  , Tensor
  , TestString
  >;
TYPED_TEST_SUITE( ArrayOfSetsTest, TestTypes, );

INDEX_TYPE const DEFAULT_MAX_INSERTS = 25;
INDEX_TYPE const DEFAULT_MAX_VALUE = 2 * DEFAULT_MAX_INSERTS;

TYPED_TEST( ArrayOfSetsTest, emptyConstruction )
{
  ArrayOfSets< TypeParam > a;
  ASSERT_EQ( a.size(), 0 );
  ASSERT_EQ( a.capacity(), 0 );
}

TYPED_TEST( ArrayOfSetsTest, sizedConstruction )
{
  ArrayOfSets< TypeParam > a( 10 );
  ASSERT_EQ( a.size(), 10 );
  ASSERT_EQ( a.capacity(), 10 );

  for( INDEX_TYPE i = 0; i < a.size(); ++i )
  {
    ASSERT_EQ( a.sizeOfSet( i ), 0 );
    ASSERT_EQ( a.capacityOfSet( i ), 0 );
  }
}

TYPED_TEST( ArrayOfSetsTest, sizedHintConstruction )
{
  ArrayOfSets< TypeParam > a( 10, 20 );
  ASSERT_EQ( a.size(), 10 );
  ASSERT_EQ( a.capacity(), 10 );

  for( INDEX_TYPE i = 0; i < a.size(); ++i )
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
  for( INDEX_TYPE i = 0; i < 2; ++i )
  {
    this->insertSet( 50, DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
  }
}

TYPED_TEST( ArrayOfSetsTest, eraseArray )
{
  for( INDEX_TYPE i = 0; i < 2; ++i )
  {
    this->appendSet( 50, DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
    this->eraseSet( this->m_array.size() / 2 );
  }
}

TYPED_TEST( ArrayOfSetsTest, resize )
{
  for( INDEX_TYPE i = 0; i < 2; ++i )
  {
    this->insertSet( 50, DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
    this->resize();
  }
}

TYPED_TEST( ArrayOfSetsTest, insertIntoSet )
{
  this->resize( 50 );
  for( INDEX_TYPE i = 0; i < 2; ++i )
  {
    this->insertIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
  }
}

TYPED_TEST( ArrayOfSetsTest, insertMultipleIntoSet )
{
  this->resize( 1 );
  for( INDEX_TYPE i = 0; i < 2; ++i )
  {
    this->insertMultipleIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
  }
}

TYPED_TEST( ArrayOfSetsTest, removeFromSet )
{
  this->resize( 50 );
  for( INDEX_TYPE i = 0; i < 2; ++i )
  {
    this->insertIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
    this->removeFromSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
  }
}

TYPED_TEST( ArrayOfSetsTest, removeMultipleFromSet )
{
  this->resize( 50 );
  for( INDEX_TYPE i = 0; i < 2; ++i )
  {
    this->insertIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
    this->removeMultipleFromSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
  }
}

TYPED_TEST( ArrayOfSetsTest, compress )
{
  this->resize( 50 );
  for( INDEX_TYPE i = 0; i < 2; ++i )
  {
    this->insertIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
    this->compress();
  }
}

TYPED_TEST( ArrayOfSetsTest, capacity )
{
  this->resize( 50 );
  for( INDEX_TYPE i = 0; i < 2; ++i )
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

TYPED_TEST( ArrayOfSetsTest, stealFromSortedUnique )
{
  this->resize( 50 );
  this->stealFrom( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE, sortedArrayManipulation::SORTED_UNIQUE );
}

TYPED_TEST( ArrayOfSetsTest, stealFromUnsortedNoDuplicates )
{
  this->resize( 50 );
  this->stealFrom( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE, sortedArrayManipulation::UNSORTED_NO_DUPLICATES );
}

TYPED_TEST( ArrayOfSetsTest, stealFromSortedWithDuplicates )
{
  this->resize( 50 );
  this->stealFrom( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE, sortedArrayManipulation::SORTED_WITH_DUPLICATES );
}

TYPED_TEST( ArrayOfSetsTest, stealFromUnsortedWithDuplicates )
{
  this->resize( 50 );
  this->stealFrom( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE, sortedArrayManipulation::UNSORTED_WITH_DUPLICATES );
}

// Note this is testing capabilities of the ArrayOfArrays class, however it needs to first populate
// the ArrayOfSets so it involves less code duplication to put it here.
TYPED_TEST( ArrayOfSetsTest, ArrayOfArraysStealFrom )
{
  ArrayOfArrays< TypeParam > array;

  this->resize( 50 );
  this->insertIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
  array.stealFrom( std::move( this->m_array ));

  ASSERT_EQ( array.size(), this->m_ref.size());

  for( INDEX_TYPE i = 0; i < array.size(); ++i )
  {
    ASSERT_EQ( array.sizeOfArray( i ), this->m_ref[i].size());

    typename std::set< TypeParam >::iterator it = this->m_ref[i].begin();
    for( INDEX_TYPE j = 0; j < array.sizeOfArray( i ); ++j )
    {
      EXPECT_EQ( array[i][j], *it++ );
    }
    EXPECT_EQ( it, this->m_ref[i].end());
  }
}

template< class T_POLICY_PAIR >
class ArrayOfSetsViewTest : public ArrayOfSetsTest< typename T_POLICY_PAIR::first_type >
{
public:

  using T = typename T_POLICY_PAIR::first_type;
  using POLICY = typename T_POLICY_PAIR::second_type;

  void memoryMotion()
  {
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );

    INDEX_TYPE const nSets = m_array.size();

    // Update the view on the device.
    ArrayOfSetsView< T, INDEX_TYPE const > const & view = m_array.toView();
    forall< POLICY >( nSets, [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          INDEX_TYPE const sizeOfSet = view.sizeOfSet( i );
          for( INDEX_TYPE j = 0; j < sizeOfSet; ++j )
          {
            PORTABLE_EXPECT_EQ( &view[ i ][ j ], &view( i, j ) );
          }

          for( T const & val : view.getIterableSet( i ))
          {
            PORTABLE_EXPECT_EQ( view.contains( i, val ), true );
          }
        } );

    // Move the view back to the host and compare with the reference.
    forall< serialPolicy >( 1, [view, this] ( INDEX_TYPE )
    {
      COMPARE_TO_REFERENCE( view.toViewConst(), m_ref );
    } );
  }

  void memoryMotionMove()
  {
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );

    INDEX_TYPE const nSets = m_array.size();

    ArrayOfSetsView< T, INDEX_TYPE const > const & view = m_array.toView();
    forall< POLICY >( nSets, [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          INDEX_TYPE const sizeOfSet = view.sizeOfSet( i );
          for( INDEX_TYPE j = 0; j < sizeOfSet; ++j )
          {
            PORTABLE_EXPECT_EQ( &view[ i ][ j ], &view( i, j ) );
          }

          for( T const & val : view.getIterableSet( i ))
          {
            PORTABLE_EXPECT_EQ( view.contains( i, val ), true );
          }
        } );

    // Move the view back to the host and compare with the reference.
    m_array.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );
  }

  void insertView()
  {
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );

    Array< Array< T, 1 >, 1 > toInsert = createValues( true, false );
    ArrayView< ArrayView< T const, 1 > const, 1 > const & toInsertView = toInsert.toViewConst();

    ArrayOfSetsView< T, INDEX_TYPE const > const & view = m_array.toView();
    forall< POLICY >( m_array.size(), [view, toInsertView] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < toInsertView[ i ].size(); ++j )
          {
            view.insertIntoSet( i, toInsertView[ i ][ j ] );
          }
        } );

    m_array.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );
  }

  void insertMultipleView()
  {
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );

    Array< Array< T, 1 >, 1 > const toInsert = createValues( true, true );
    ArrayView< ArrayView< T const, 1 > const, 1 > const & toInsertView = toInsert.toViewConst();

    ArrayOfSetsView< T, INDEX_TYPE const > const & view = m_array.toView();
    forall< POLICY >( m_array.size(), [view, toInsertView] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          view.insertIntoSet( i, toInsertView[ i ].begin(), toInsertView[ i ].end() );
        } );

    m_array.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );
  }

  void removeView()
  {
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );

    Array< Array< T, 1 >, 1 > const toRemove = createValues( false, false );
    ArrayView< ArrayView< T const, 1 > const, 1 > const & toRemoveView = toRemove.toViewConst();

    ArrayOfSetsView< T, INDEX_TYPE const > const & view = m_array.toView();
    forall< POLICY >( m_array.size(), [view, toRemoveView] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          for( INDEX_TYPE j = 0; j < toRemoveView[ i ].size(); ++j )
          {
            view.removeFromSet( i, toRemoveView[ i ][ j ] );
          }
        } );

    m_array.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );
  }

  void removeMultipleView()
  {
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );

    Array< Array< T, 1 >, 1 > const toRemove = createValues( false, true );
    ArrayView< ArrayView< T const, 1 > const, 1 > const & toRemoveView = toRemove.toViewConst();

    ArrayOfSetsView< T, INDEX_TYPE const > const & view = m_array.toView();
    forall< POLICY >( m_array.size(), [view, toRemoveView] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
        {
          view.removeFromSet( i, toRemoveView[ i ].begin(), toRemoveView[ i ].end() );
        } );

    m_array.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_array.toViewConst(), m_ref );
  }

protected:

  Array< Array< T, 1 >, 1 > createValues( bool const insert, bool const sortedUnique )
  {
    INDEX_TYPE const nSets = m_array.size();

    Array< Array< T, 1 >, 1 > values( nSets );

    for( INDEX_TYPE i = 0; i < nSets; ++i )
    {
      INDEX_TYPE const sizeOfSet = m_array.sizeOfSet( i );
      INDEX_TYPE const capacityOfSet = m_array.capacityOfSet( i );

      INDEX_TYPE const upperBound = insert ? capacityOfSet - sizeOfSet : capacityOfSet;
      INDEX_TYPE const nValues = rand( 0, upperBound );

      values[ i ].resize( nValues );

      for( INDEX_TYPE j = 0; j < nValues; ++j )
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
        INDEX_TYPE const numUnique = sortedArrayManipulation::makeSortedUnique( values[ i ].begin(), values[ i ].end() );
        values[ i ].resize( numUnique );
      }
    }

    return values;
  }

  using ArrayOfSetsTest< T >::rand;
  using ArrayOfSetsTest< T >::LARGE_NUMBER;
  using ArrayOfSetsTest< T >::m_array;
  using ArrayOfSetsTest< T >::m_ref;
  using ArrayOfSetsTest< T >::m_gen;
};

using ArrayOfSetsViewTestTypes = ::testing::Types<
  std::pair< INDEX_TYPE, serialPolicy >
  , std::pair< Tensor, serialPolicy >
  , std::pair< TestString, serialPolicy >

#ifdef USE_OPENMP
  , std::pair< INDEX_TYPE, parallelHostPolicy >
  , std::pair< Tensor, parallelHostPolicy >
  , std::pair< TestString, parallelHostPolicy >
#endif

#ifdef USE_CUDA
  , std::pair< INDEX_TYPE, parallelDevicePolicy >
  , std::pair< Tensor, parallelDevicePolicy >
#endif
  >;

TYPED_TEST_SUITE( ArrayOfSetsViewTest, ArrayOfSetsViewTestTypes, );

TYPED_TEST( ArrayOfSetsViewTest, memoryMotion )
{
  this->resize( 50 );
  for( INDEX_TYPE i = 0; i < 2; ++i )
  {
    this->insertIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
    this->memoryMotion();
  }
}

TYPED_TEST( ArrayOfSetsViewTest, memoryMotionMove )
{
  this->resize( 50 );
  for( INDEX_TYPE i = 0; i < 2; ++i )
  {
    this->insertIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
    this->memoryMotionMove();
  }
}

TYPED_TEST( ArrayOfSetsViewTest, insert )
{
  this->resize( 50, 10 );
  for( INDEX_TYPE i = 0; i < 2; ++i )
  {
    this->insertView();
  }
}

TYPED_TEST( ArrayOfSetsViewTest, insertMultiple )
{
  this->resize( 50, 10 );
  for( INDEX_TYPE i = 0; i < 2; ++i )
  {
    this->insertMultipleView();
  }
}

TYPED_TEST( ArrayOfSetsViewTest, remove )
{
  this->resize( 50, 10 );
  for( INDEX_TYPE i = 0; i < 2; ++i )
  {
    this->insertMultipleIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
    this->removeView();
  }
}

TYPED_TEST( ArrayOfSetsViewTest, removeMultiple )
{
  this->resize( 50, 10 );
  for( INDEX_TYPE i = 0; i < 2; ++i )
  {
    this->insertMultipleIntoSet( DEFAULT_MAX_INSERTS, DEFAULT_MAX_VALUE );
    this->removeMultipleView();
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
