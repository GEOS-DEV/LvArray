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
#include "SparsityPattern.hpp"
#include "testUtils.hpp"
#include "Array.hpp"


// TPL includes
#include <gtest/gtest.h>

#if defined(USE_CHAI) && defined(USE_CUDA)
  #include <chai/util/forall.hpp>
#endif

// System includes
#include <vector>
#include <set>
#include <iterator>
#include <random>
#include <climits>

namespace LvArray
{
namespace testing
{

// Alias for SparsityPatternView
template< class COL_TYPE, class INDEX_TYPE=std::ptrdiff_t >
using ViewType = SparsityPatternView< COL_TYPE, INDEX_TYPE const >;

using INDEX_TYPE = std::ptrdiff_t;

/**
 * @brief Check that the SparsityPatternView is equivalent to the reference type.
 * @param [in] v the SparsityPatternView to check.
 * @param [in] m_ref the reference to check against.
 */
template< class COL_TYPE >
void compareToReference( ViewType< COL_TYPE const > const & v, std::vector< std::set< COL_TYPE > > const & m_ref )
{
  INDEX_TYPE const numRows = v.numRows();
  ASSERT_EQ( numRows, INDEX_TYPE( m_ref.size()));

  INDEX_TYPE ref_nnz = 0;
  for( INDEX_TYPE row = 0; row < numRows; ++row )
  {
    INDEX_TYPE const rowNNZ = v.numNonZeros( row );
    INDEX_TYPE const refRowNNZ = m_ref[row].size();
    ref_nnz += refRowNNZ;

    ASSERT_EQ( rowNNZ, refRowNNZ );

    if( rowNNZ == 0 )
    {
      ASSERT_TRUE( v.empty( row ));
    }
    else
    {
      ASSERT_FALSE( v.empty( row ));
    }

    auto it = m_ref[row].begin();
    COL_TYPE const * const columns = v.getColumns( row );
    for( INDEX_TYPE i = 0; i < v.numNonZeros( row ); ++i )
    {
      EXPECT_FALSE( v.empty( row, columns[i] ));
      EXPECT_EQ( columns[i], *it );
      it++;
    }
  }

  ASSERT_EQ( v.numNonZeros(), ref_nnz );

  if( v.numNonZeros() == 0 )
  {
    ASSERT_TRUE( v.empty());
  }
}

#define COMPARE_TO_REFERENCE( view, ref ) { SCOPED_TRACE( "" ); compareToReference( view, ref ); \
}

template< typename COL_TYPE >
class SparsityPatternTest : public ::testing::Test
{
public:

  void resize( INDEX_TYPE const nRows,
               INDEX_TYPE const nCols,
               INDEX_TYPE initialRowCapacity=0 )
  {
    m_sp.resize( nRows, nCols, initialRowCapacity );
    m_ref.resize( nRows );

    COMPARE_TO_REFERENCE( m_sp.toViewConst(), m_ref );
  }

  /**
   * @brief Test the insert method of the SparsityPattern.
   * @param [in] maxInserts the number of times to call insert.
   */
  void insertTest( INDEX_TYPE const maxInserts )
  {
    INDEX_TYPE const numRows = m_sp.numRows();
    ASSERT_EQ( numRows, INDEX_TYPE( m_ref.size()));

    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      std::set< COL_TYPE > & refRow = m_ref[row];
      ASSERT_EQ( m_sp.numNonZeros( row ), INDEX_TYPE( refRow.size()));

      for( INDEX_TYPE i = 0; i < maxInserts; ++i )
      {
        COL_TYPE const col = randCol();
        ASSERT_EQ( refRow.insert( col ).second, m_sp.insertNonZero( row, col ));
      }
    }

    COMPARE_TO_REFERENCE( m_sp.toViewConst(), m_ref );
  }

  /**
   * @brief Test the insertSorted method of the SparsityPattern.
   * @param [in] maxInserts the number of values to insert at a time.
   */
  void insertMultipleTest( INDEX_TYPE const maxInserts )
  {
    INDEX_TYPE const numRows = m_sp.numRows();
    ASSERT_EQ( numRows, INDEX_TYPE( m_ref.size() ) );

    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      INDEX_TYPE const nCols = rand( maxInserts / 2 );

      // Insert using a std::vector
      {
        std::vector< COL_TYPE > columnsToInsert( nCols );

        for( INDEX_TYPE j = 0; j < nCols; ++j )
        { columnsToInsert[ j ] = randCol(); }

        INDEX_TYPE const numUnique = sortedArrayManipulation::makeSortedUnique( columnsToInsert.begin(), columnsToInsert.end() );
        columnsToInsert.resize( numUnique );

        INDEX_TYPE const numInserted = m_sp.insertNonZeros( row, columnsToInsert.begin(), columnsToInsert.end() );
        EXPECT_EQ( numInserted, insertIntoRef( row, columnsToInsert ) );
      }

      /// Insert using a std::set
      {
        std::set< COL_TYPE > columnsToInsert;

        for( INDEX_TYPE j = 0; j < nCols; ++j )
        { columnsToInsert.insert( randCol() ); }

        INDEX_TYPE const numInserted = m_sp.insertNonZeros( row, columnsToInsert.begin(), columnsToInsert.end() );
        EXPECT_EQ( numInserted, insertIntoRef( row, columnsToInsert ) );
      }
    }

    COMPARE_TO_REFERENCE( m_sp.toViewConst(), m_ref );
  }

  /**
   * @brief Test the remove method of the SparsityPatternView.
   * @param [in] maxRemoves the number of times to call remove.
   */
  void removeTest( INDEX_TYPE const maxRemoves )
  {
    INDEX_TYPE const numRows = m_sp.numRows();
    ASSERT_EQ( numRows, INDEX_TYPE( m_ref.size()));

    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      std::set< COL_TYPE > & refRow = m_ref[row];
      ASSERT_EQ( m_sp.numNonZeros( row ), COL_TYPE( refRow.size()));

      for( INDEX_TYPE i = 0; i < maxRemoves; ++i )
      {
        COL_TYPE const col = randCol();
        ASSERT_EQ( m_sp.removeNonZero( row, col ), refRow.erase( col ));
      }
    }

    COMPARE_TO_REFERENCE( m_sp.toViewConst(), m_ref );
  }

  /**
   * @brief Test the removeSorted method of the SparsityPatternView.
   * @param [in] maxRemoves the number of values to remove at a time.
   */
  void removeMultipleTest( INDEX_TYPE const maxRemoves )
  {
    INDEX_TYPE const numRows = m_sp.numRows();
    ASSERT_EQ( numRows, INDEX_TYPE( m_ref.size()));

    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      INDEX_TYPE const nCols = rand( maxRemoves / 2 );

      // Insert using a std::vector
      {
        std::vector< COL_TYPE > columnsToRemove( nCols );

        for( INDEX_TYPE j = 0; j < nCols; ++j )
        { columnsToRemove[ j ] = randCol(); }

        INDEX_TYPE const numUnique = sortedArrayManipulation::makeSortedUnique( columnsToRemove.begin(), columnsToRemove.end() );
        columnsToRemove.resize( numUnique );

        INDEX_TYPE const numInserted = m_sp.removeNonZeros( row, columnsToRemove.begin(), columnsToRemove.end() );
        EXPECT_EQ( numInserted, removeFromRef( row, columnsToRemove ) );
      }

      /// Insert using a std::set
      {
        std::set< COL_TYPE > columnsToRemove;

        for( INDEX_TYPE j = 0; j < nCols; ++j )
        { columnsToRemove.insert( randCol() ); }

        INDEX_TYPE const numInserted = m_sp.removeNonZeros( row, columnsToRemove.begin(), columnsToRemove.end() );
        EXPECT_EQ( numInserted, removeFromRef( row, columnsToRemove ) );
      }
    }

    COMPARE_TO_REFERENCE( m_sp.toViewConst(), m_ref );
  }

  /**
   * @brief Test the empty method of the SparsityPatternView.
   */
  void emptyTest() const
  {
    INDEX_TYPE const numRows = m_sp.numRows();
    ASSERT_EQ( numRows, m_ref.size());

    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      std::set< COL_TYPE > const & refRow = m_ref[row];
      INDEX_TYPE const nnz = m_sp.numNonZeros( row );
      ASSERT_EQ( nnz, refRow.size());

      auto it = refRow.begin();
      for( INDEX_TYPE i = 0; i < nnz; ++i )
      {
        COL_TYPE const col = *it;
        EXPECT_FALSE( m_sp.empty( row, col ));
        EXPECT_NE( m_sp.empty( row, COL_TYPE( i )), refRow.count( COL_TYPE( i )));
        ++it;
      }
    }
  }

  /**
   * @brief Fill a row of the SparsityPatternView up to its capacity.
   * @param [in] row the row of the SparsityPatternView to fill.
   */
  void fillRow( INDEX_TYPE const row )
  {
    INDEX_TYPE nToInsert = m_sp.nonZeroCapacity( row ) - m_sp.numNonZeros( row );
    COL_TYPE testColum = COL_TYPE( m_sp.numColumns() - 1 );
    COL_TYPE const * const columns = m_sp.getColumns( row );

    ASSERT_GE( m_sp.numColumns(), m_sp.nonZeroCapacity( row ));

    while( nToInsert > 0 )
    {
      bool const success = m_ref[ row ].insert( testColum ).second;
      ASSERT_EQ( success, m_sp.insertNonZero( row, testColum ));
      nToInsert -= success;
      testColum -= 1;
    }

    COL_TYPE const * const newColumns = m_sp.getColumns( row );
    ASSERT_EQ( columns, newColumns );
  }

  /**
   * @brief Test the setRowCapacity method of SparsityPattern.
   */
  void rowCapacityTest()
  {
    INDEX_TYPE const numRows = m_sp.numRows();
    ASSERT_EQ( numRows, INDEX_TYPE( m_ref.size()));

    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      std::set< COL_TYPE > & refRow = m_ref[row];
      INDEX_TYPE const rowNNZ = m_sp.numNonZeros( row );
      ASSERT_EQ( rowNNZ, COL_TYPE( refRow.size()));

      // The new capacity will be the index of the row.
      INDEX_TYPE const new_capacity = row;
      if( new_capacity < rowNNZ )
      {
        COL_TYPE const * const columns = m_sp.getColumns( row );
        m_sp.setRowCapacity( row, new_capacity );

        // Erase the last values from the reference.
        INDEX_TYPE const difference = rowNNZ - new_capacity;
        auto erase_pos = refRow.end();
        std::advance( erase_pos, -difference );
        refRow.erase( erase_pos, refRow.end());

        // Check that the pointer to the row hasn't changed.
        COL_TYPE const * const newColumns = m_sp.getColumns( row );
        EXPECT_EQ( columns, newColumns );
      }
      else
      {
        m_sp.setRowCapacity( row, new_capacity );

        if( new_capacity > m_sp.numColumns())
        {
          EXPECT_EQ( m_sp.nonZeroCapacity( row ), m_sp.numColumns());
        }
        else
        {
          EXPECT_EQ( m_sp.nonZeroCapacity( row ), new_capacity );
        }

        fillRow( row );
      }
    }

    COMPARE_TO_REFERENCE( m_sp.toViewConst(), m_ref );
  }

  /**
   * @brief Test the compress method of SparsityPattern.
   */
  void compressTest()
  {
    m_sp.compress();

    COL_TYPE const * const columns = m_sp.getColumns( 0 );
    INDEX_TYPE const * const offsets = m_sp.getOffsets();

    INDEX_TYPE curOffset = 0;
    for( INDEX_TYPE row = 0; row < m_sp.numRows(); ++row )
    {
      // The last row will have all the extra capacity.
      if( row != m_sp.numRows() - 1 )
      {
        ASSERT_EQ( m_sp.numNonZeros( row ), m_sp.nonZeroCapacity( row ));
      }

      COL_TYPE const * const rowColumns = m_sp.getColumns( row );
      ASSERT_EQ( rowColumns, columns + curOffset );
      ASSERT_EQ( offsets[row], curOffset );

      curOffset += m_sp.numNonZeros( row );
    }

    COMPARE_TO_REFERENCE( m_sp.toViewConst(), m_ref );
  }

  /**
   * @brief Test the copy constructor of the SparsityPattern.
   */
  void deepCopyTest()
  {
    SparsityPattern< COL_TYPE > copy( m_sp );

    ASSERT_EQ( m_sp.numRows(), copy.numRows());
    ASSERT_EQ( m_sp.numColumns(), copy.numColumns());
    ASSERT_EQ( m_sp.numNonZeros(), copy.numNonZeros());

    INDEX_TYPE const totalNNZ = m_sp.numNonZeros();

    for( INDEX_TYPE row = 0; row < m_sp.numRows(); ++row )
    {
      INDEX_TYPE const nnz = m_sp.numNonZeros( row );
      ASSERT_EQ( nnz, copy.numNonZeros( row ));

      COL_TYPE const * const cols = m_sp.getColumns( row );
      COL_TYPE const * const cols_cpy = copy.getColumns( row );
      ASSERT_NE( cols, cols_cpy );

      // Iterate backwards and remove entries from copy.
      for( INDEX_TYPE i = nnz - 1; i >= 0; --i )
      {
        EXPECT_EQ( cols[i], cols_cpy[i] );
        copy.removeNonZero( row, cols_cpy[i] );
      }

      EXPECT_EQ( copy.numNonZeros( row ), 0 );
      EXPECT_EQ( m_sp.numNonZeros( row ), nnz );
    }

    EXPECT_EQ( copy.numNonZeros(), 0 );
    EXPECT_EQ( m_sp.numNonZeros(), totalNNZ );

    COMPARE_TO_REFERENCE( m_sp.toViewConst(), m_ref );
  }

  /**
   * @brief Test the copy constructor of the SparsityPatternView.
   */
  void shallowCopyTest() const
  {
    ViewType< COL_TYPE > const & view = m_sp;
    ViewType< COL_TYPE > copy( view );

    ASSERT_EQ( m_sp.numRows(), copy.numRows());
    ASSERT_EQ( m_sp.numColumns(), copy.numColumns());
    ASSERT_EQ( m_sp.numNonZeros(), copy.numNonZeros());

    for( INDEX_TYPE row = 0; row < m_sp.numRows(); ++row )
    {
      INDEX_TYPE const nnz = m_sp.numNonZeros( row );
      ASSERT_EQ( nnz, copy.numNonZeros( row ));

      COL_TYPE const * const cols = m_sp.getColumns( row );
      COL_TYPE const * const cols_cpy = copy.getColumns( row );
      ASSERT_EQ( cols, cols_cpy );

      // Iterate backwards and remove entries from copy.
      for( INDEX_TYPE i = nnz - 1; i >= 0; --i )
      {
        EXPECT_EQ( cols[i], cols_cpy[i] );
        copy.removeNonZero( row, cols_cpy[i] );
      }

      EXPECT_EQ( copy.numNonZeros( row ), 0 );
      EXPECT_EQ( m_sp.numNonZeros( row ), 0 );
    }

    EXPECT_EQ( copy.numNonZeros(), 0 );
    EXPECT_EQ( m_sp.numNonZeros(), 0 );
  }

protected:

  template< typename CONTAINER >
  INDEX_TYPE insertIntoRef( INDEX_TYPE const row, CONTAINER const & columns )
  {
    INDEX_TYPE numInserted = 0;
    for( COL_TYPE const & col : columns )
    { numInserted += m_ref[ row ].insert( col ).second; }

    return numInserted;
  }

  template< typename CONTAINER >
  INDEX_TYPE removeFromRef( INDEX_TYPE const row, CONTAINER const & columns )
  {
    INDEX_TYPE numRemoved = 0;
    for( COL_TYPE const & col : columns )
    { numRemoved += m_ref[ row ].erase( col ); }

    return numRemoved;
  }

  INDEX_TYPE rand( INDEX_TYPE const max )
  { return std::uniform_int_distribution< INDEX_TYPE >( 0, max )( m_gen ); }

  COL_TYPE randCol()
  { return rand( m_sp.numColumns() - 1 ); }

  std::mt19937_64 m_gen;

  SparsityPattern< COL_TYPE > m_sp;
  std::vector< std::set< COL_TYPE > > m_ref;
};

using TestTypes = ::testing::Types< int, unsigned int >;
TYPED_TEST_SUITE( SparsityPatternTest, TestTypes, );

INDEX_TYPE const NROWS = 100;
INDEX_TYPE const NCOLS = 150;
INDEX_TYPE const MAX_INSERTS = 75;

TYPED_TEST( SparsityPatternTest, constructionNoHint )
{
  SparsityPattern< TypeParam > sp( NROWS, NCOLS );
  EXPECT_EQ( sp.numRows(), NROWS );
  EXPECT_EQ( sp.numColumns(), NCOLS );
  EXPECT_EQ( sp.numNonZeros(), 0 );
  EXPECT_TRUE( sp.empty());

  for( INDEX_TYPE row = 0; row < NROWS; ++row )
  {
    EXPECT_EQ( sp.numNonZeros( row ), 0 );
    EXPECT_EQ( sp.nonZeroCapacity( row ), 0 );
    EXPECT_EQ( sp.empty( row ), true );
    TypeParam const * const columns = sp.getColumns( row );
    EXPECT_EQ( columns, nullptr );
  }
}

TYPED_TEST( SparsityPatternTest, constructionWithHint )
{
  constexpr int SIZE_HINT = 5;

  SparsityPattern< TypeParam > sp( NROWS, NCOLS, SIZE_HINT );
  EXPECT_EQ( sp.numRows(), NROWS );
  EXPECT_EQ( sp.numColumns(), NCOLS );
  EXPECT_EQ( sp.numNonZeros(), 0 );
  EXPECT_EQ( sp.nonZeroCapacity(), NROWS * SIZE_HINT );
  EXPECT_TRUE( sp.empty());

  for( INDEX_TYPE row = 0; row < NROWS; ++row )
  {
    EXPECT_EQ( sp.numNonZeros( row ), 0 );
    EXPECT_EQ( sp.nonZeroCapacity( row ), SIZE_HINT );
    EXPECT_EQ( sp.empty( row ), true );

    TypeParam const * const columns = sp.getColumns( row );
    ASSERT_NE( columns, nullptr );
  }
}

TYPED_TEST( SparsityPatternTest, insert )
{
  this->resize( NROWS, NCOLS );
  for( int i = 0; i < 2; ++i )
  {
    this->insertTest( MAX_INSERTS );
  }
}

TYPED_TEST( SparsityPatternTest, insertMultiple )
{
  this->resize( NROWS, NCOLS );
  for( int i = 0; i < 2; ++i )
  {
    this->insertMultipleTest( MAX_INSERTS );
  }
}

TYPED_TEST( SparsityPatternTest, remove )
{
  this->resize( NROWS, NCOLS );
  for( int i = 0; i < 2; ++i )
  {
    this->insertTest( MAX_INSERTS );
    this->removeTest( MAX_INSERTS );
  }
}

TYPED_TEST( SparsityPatternTest, removeMultiple )
{
  this->resize( NROWS, NCOLS );
  for( int i = 0; i < 2; ++i )
  {
    this->insertTest( MAX_INSERTS );
    this->removeMultipleTest( MAX_INSERTS );
  }
}

TYPED_TEST( SparsityPatternTest, empty )
{
  this->resize( NROWS, NCOLS );
  this->insertTest( MAX_INSERTS );
  this->emptyTest();
}

TYPED_TEST( SparsityPatternTest, capacity )
{
  this->resize( NROWS, NCOLS );
  ASSERT_EQ( this->m_sp.nonZeroCapacity(), 0 );

  // We reserve 2 * MAX_INSERTS per row because each row may grow to have 2 * MAX_INSERTS capacity.
  this->m_sp.reserveNonZeros( 2 * NROWS * MAX_INSERTS );
  ASSERT_EQ( this->m_sp.nonZeroCapacity(), 2 * NROWS * MAX_INSERTS );
  ASSERT_EQ( this->m_sp.numNonZeros(), 0 );

  for( INDEX_TYPE row = 0; row < NROWS; ++row )
  {
    ASSERT_EQ( this->m_sp.nonZeroCapacity( row ), 0 );
    ASSERT_EQ( this->m_sp.numNonZeros( row ), 0 );
  }

  TypeParam const * const columns = this->m_sp.getColumns( 0 );
  this->insertTest( MAX_INSERTS );
  ASSERT_EQ( this->m_sp.getColumns( 0 ), columns );
}

TYPED_TEST( SparsityPatternTest, rowCapacity )
{
  this->resize( NROWS, NCOLS, MAX_INSERTS );

  TypeParam const * const columns = this->m_sp.getColumns( 0 );
  this->insertTest( MAX_INSERTS );
  ASSERT_EQ( this->m_sp.getColumns( 0 ), columns );

  this->rowCapacityTest();
}

TYPED_TEST( SparsityPatternTest, compress )
{
  this->resize( NROWS, NCOLS );

  this->insertTest( MAX_INSERTS );
  this->compressTest();
}

TYPED_TEST( SparsityPatternTest, deepCopy )
{
  this->resize( NROWS, NCOLS );

  this->insertTest( MAX_INSERTS );
  this->deepCopyTest();
}

TYPED_TEST( SparsityPatternTest, shallowCopy )
{
  this->resize( NROWS, NCOLS );

  this->insertTest( MAX_INSERTS );
  this->shallowCopyTest();
}

#ifdef USE_CUDA

template< typename COL_POLICY_PAIR >
class SparsityPatternViewTest : public SparsityPatternTest< typename COL_POLICY_PAIR::first_type >
{
public:
  using COL_TYPE = typename COL_POLICY_PAIR::first_type;
  using POLICY = typename COL_POLICY_PAIR::second_type;

  /**
   * @brief Test the SparsityPatternView copy constructor in regards to memory motion.
   */
  void memoryMotionTest() const
  {
    INDEX_TYPE const numRows = m_sp.numRows();

    // Set the columns. The const cast here isn't necessary, we could remove and insert
    // into the SparsityPattern instead, but this is quicker.
    INDEX_TYPE curIndex = 0;
    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      COL_TYPE const * const columns = m_sp.getColumns( row );
      COL_TYPE * const columnsNC = const_cast< COL_TYPE * >(columns);
      for( INDEX_TYPE i = 0; i < m_sp.numNonZeros( row ); ++i )
      {
        columnsNC[i] = COL_TYPE( curIndex++ );
      }
    }

    // Capture the view on device and set the values. Here the const cast is necessary
    // because we don't want to test the insert/remove methods on device yet.
    SparsityPatternView< COL_TYPE, INDEX_TYPE const > const & view = m_sp.toView();
    forall< POLICY >( numRows,
                      [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE row )
        {
          COL_TYPE const * const columns = view.getColumns( row );
          COL_TYPE * const columnsNC = const_cast< COL_TYPE * >(columns);
          for( INDEX_TYPE i = 0; i < view.numNonZeros( row ); ++i )
          {
            columnsNC[i] *= columns[i];
          }
        } );

    // Check that the columns have been updated.
    curIndex = 0;
    forall< serialPolicy >( numRows,
                            [view, &curIndex]( INDEX_TYPE row )
    {
      for( INDEX_TYPE i = 0; i < view.numNonZeros( row ); ++i )
      {
        COL_TYPE val = COL_TYPE( curIndex ) * COL_TYPE( curIndex );
        EXPECT_EQ( val, view.getColumns( row )[i] );
        ++curIndex;
      }
    } );
  }

  /**
   * @brief Test the SparsityPattern move method.
   */
  void memoryMotionMoveTest()
  {
    INDEX_TYPE const numRows = m_sp.numRows();

    SparsityPatternView< COL_TYPE, INDEX_TYPE const > const & view = m_sp.toView();

    INDEX_TYPE curIndex = 0;
    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      COL_TYPE const * const columns = view.getColumns( row );
      COL_TYPE * const columnsNC = const_cast< COL_TYPE * >(columns);
      for( INDEX_TYPE i = 0; i < view.numNonZeros( row ); ++i )
      {
        columnsNC[i] = COL_TYPE( curIndex++ );
      }
    }

    forall< POLICY >( numRows,
                      [=] LVARRAY_HOST_DEVICE ( INDEX_TYPE row )
        {
          COL_TYPE const * const columns = view.getColumns( row );
          COL_TYPE * const columnsNC = const_cast< COL_TYPE * >(columns);
          for( INDEX_TYPE i = 0; i < view.numNonZeros( row ); ++i )
          {
            columnsNC[i] *= columns[i];
          }
        } );

    m_sp.move( chai::CPU );
    curIndex = 0;
    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      for( INDEX_TYPE i = 0; i < m_sp.numNonZeros( row ); ++i )
      {
        COL_TYPE val = COL_TYPE( curIndex ) * COL_TYPE( curIndex );
        EXPECT_EQ( val, m_sp.getColumns( row )[i] );
        curIndex++;
      }
    }
  }

  /**
   * @brief Test the const capture semantics.
   */
  void memoryMotionConstTest()
  {
    INDEX_TYPE const numRows = m_sp.numRows();
    INDEX_TYPE const numCols = m_sp.numColumns();

    // Create a view const and capture it on device.
    SparsityPatternView< COL_TYPE const, INDEX_TYPE const > const & constView = m_sp.toViewConst();
    forall< POLICY >( numRows,
                      [constView, numCols] LVARRAY_HOST_DEVICE ( INDEX_TYPE row )
        {
          COL_TYPE const * const columns = constView.getColumns( row );
          for( INDEX_TYPE i = 0; i < constView.numNonZeros( row ); ++i )
          {
            PORTABLE_EXPECT_EQ( arrayManipulation::isPositive( columns[ i ] ) && columns[ i ] < numCols, true );
          }
        } );

    // Fill in the rows on host.
    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      this->fillRow( row );
    }

    // Move the SparsityPattern back from device, this should be a no-op.
    m_sp.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_sp.toViewConst(), m_ref );
  }

  void insertViewTest()
  {
    COMPARE_TO_REFERENCE( m_sp.toViewConst(), m_ref );

    Array< Array< COL_TYPE, 1 >, 1 > toInsert = createColumns( true, false );
    ArrayView< ArrayView< COL_TYPE const, 1 > const, 1 > const & toInsertView = toInsert.toViewConst();

    SparsityPatternView< COL_TYPE, INDEX_TYPE const > const & view = m_sp.toView();
    forall< POLICY >( m_sp.numRows(), [view, toInsertView] LVARRAY_HOST_DEVICE ( INDEX_TYPE const row )
        {
          for( INDEX_TYPE j = 0; j < toInsertView[ row ].size(); ++j )
          {
            view.insertNonZero( row, toInsertView[ row ][ j ] );
          }
        } );

    m_sp.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_sp.toViewConst(), m_ref );
  }

  void insertMultipleViewTest()
  {
    COMPARE_TO_REFERENCE( m_sp.toViewConst(), m_ref );

    Array< Array< COL_TYPE, 1 >, 1 > const toInsert = createColumns( true, true );
    ArrayView< ArrayView< COL_TYPE const, 1 > const, 1 > const & toInsertView = toInsert.toViewConst();

    SparsityPatternView< COL_TYPE, INDEX_TYPE const > const & view = m_sp.toView();
    forall< POLICY >( m_sp.numRows(), [view, toInsertView] LVARRAY_HOST_DEVICE ( INDEX_TYPE const row )
        {
          view.insertNonZeros( row, toInsertView[ row ].begin(), toInsertView[ row ].end() );
        } );

    m_sp.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_sp.toViewConst(), m_ref );
  }

  void removeViewTest()
  {
    COMPARE_TO_REFERENCE( m_sp.toViewConst(), m_ref );

    Array< Array< COL_TYPE, 1 >, 1 > const toRemove = createColumns( false, false );
    ArrayView< ArrayView< COL_TYPE const, 1 > const, 1 > const & toRemoveView = toRemove.toViewConst();

    SparsityPatternView< COL_TYPE, INDEX_TYPE const > const & view = m_sp.toView();
    forall< POLICY >( m_sp.numRows(), [view, toRemoveView] LVARRAY_HOST_DEVICE ( INDEX_TYPE const row )
        {
          for( INDEX_TYPE j = 0; j < toRemoveView[ row ].size(); ++j )
          {
            view.removeNonZero( row, toRemoveView[ row ][ j ] );
          }
        } );

    m_sp.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_sp.toViewConst(), m_ref );
  }

  void removeMultipleViewTest()
  {
    COMPARE_TO_REFERENCE( m_sp.toViewConst(), m_ref );

    Array< Array< COL_TYPE, 1 >, 1 > const toRemove = createColumns( false, true );
    ArrayView< ArrayView< COL_TYPE const, 1 > const, 1 > const & toRemoveView = toRemove.toViewConst();

    SparsityPatternView< COL_TYPE, INDEX_TYPE const > const & view = m_sp.toView();
    forall< POLICY >( m_sp.numRows(), [view, toRemoveView] LVARRAY_HOST_DEVICE ( INDEX_TYPE const row )
        {
          view.removeNonZeros( row, toRemoveView[ row ].begin(), toRemoveView[ row ].end() );
        } );

    m_sp.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_sp.toViewConst(), m_ref );
  }

  /**
   * @brief Test the empty method of the SparsityPatternView on device.
   * @param [in] v the SparsityPatternView to test.
   */
  void emptyViewTest() const
  {
    INDEX_TYPE const numRows = m_sp.numRows();

    // Initialize each row to only contain even numbered columns.
    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      COL_TYPE const * const columns = m_sp.getColumns( row );
      COL_TYPE * const columnsNC = const_cast< COL_TYPE * >(columns);
      for( INDEX_TYPE i = 0; i < m_sp.numNonZeros( row ); ++i )
      {
        columnsNC[i] = COL_TYPE( 2 * i );
      }
    }

    // Check that each row contains the even columns and no odd columns on device.
    SparsityPatternView< COL_TYPE const, INDEX_TYPE const > const & view = m_sp.toView();
    forall< POLICY >( numRows,
                      [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE row )
        {
          for( INDEX_TYPE i = 0; i < view.numNonZeros( row ); ++i )
          {
            PORTABLE_EXPECT_EQ( view.empty( row, COL_TYPE( 2 * i ) ), false );
            PORTABLE_EXPECT_EQ( view.empty( row, COL_TYPE( 2 * i + 1 ) ), true );
          }
        } );
  }

protected:

  Array< Array< COL_TYPE, 1 >, 1 > createColumns( bool const insert, bool const sortedUnique )
  {
    INDEX_TYPE const numRows = m_sp.numRows();

    Array< Array< COL_TYPE, 1 >, 1 > columns( numRows );

    for( INDEX_TYPE i = 0; i < numRows; ++i )
    {
      INDEX_TYPE const nnz = m_sp.numNonZeros( i );
      INDEX_TYPE const capacity = m_sp.nonZeroCapacity( i );

      INDEX_TYPE const upperBound = insert ? capacity - nnz : capacity;
      INDEX_TYPE const nCols = rand( upperBound );

      columns[ i ].resize( nCols );

      for( INDEX_TYPE j = 0; j < nCols; ++j )
      {
        COL_TYPE const column = randCol();
        columns[ i ][ j ] = column;

        if( insert )
        { m_ref[ i ].insert( column ); }
        else
        { m_ref[ i ].erase( column ); }
      }

      if( sortedUnique )
      {
        INDEX_TYPE const numUnique = sortedArrayManipulation::makeSortedUnique( columns[ i ].begin(), columns[ i ].end() );
        columns[ i ].resize( numUnique );
      }
    }

    return columns;
  }

  using SparsityPatternTest< COL_TYPE >::m_sp;
  using SparsityPatternTest< COL_TYPE >::m_ref;
  using SparsityPatternTest< COL_TYPE >::rand;
  using SparsityPatternTest< COL_TYPE >::randCol;
};

using SparsityPatternViewTestTypes = ::testing::Types<
  std::pair< int, serialPolicy >
  , std::pair< uint, serialPolicy >

#ifdef USE_OPENMP
  , std::pair< int, parallelHostPolicy >
  , std::pair< uint, parallelHostPolicy >
#endif

#ifdef USE_CUDA
  , std::pair< int, parallelDevicePolicy >
  , std::pair< uint, parallelDevicePolicy >
#endif
  >;

TYPED_TEST_SUITE( SparsityPatternViewTest, SparsityPatternViewTestTypes );

TYPED_TEST( SparsityPatternViewTest, memoryMotion )
{
  this->resize( NROWS, NCOLS );
  this->insertTest( MAX_INSERTS );
  this->memoryMotionTest();
}

TYPED_TEST( SparsityPatternViewTest, memoryMotionMove )
{
  this->resize( NROWS, NCOLS );
  this->insertTest ( MAX_INSERTS );
  this->memoryMotionMoveTest();
}

TYPED_TEST( SparsityPatternViewTest, memoryMotionConst )
{
  this->resize( NROWS, NCOLS );
  this->insertTest( MAX_INSERTS );
  this->memoryMotionConstTest();
}

TYPED_TEST( SparsityPatternViewTest, insert )
{
  this->resize( NROWS, NCOLS, 20 );

  for( int i = 0; i < 2; ++i )
  {
    this->insertViewTest();
  }
}

TYPED_TEST( SparsityPatternViewTest, insertMultiple )
{
  this->resize( NROWS, NCOLS, 20 );

  for( int i = 0; i < 2; ++i )
  {
    this->insertMultipleViewTest();
  }
}

TYPED_TEST( SparsityPatternViewTest, remove )
{
  this->resize( NROWS, NCOLS, 20 );

  for( int i = 0; i < 2; ++i )
  {
    this->insertMultipleTest( 20 );
    this->removeViewTest();
  }
}

TYPED_TEST( SparsityPatternViewTest, removeMultiple )
{
  this->resize( NROWS, NCOLS, 20 );

  for( int i = 0; i < 2; ++i )
  {
    this->insertMultipleTest( 20 );
    this->removeMultipleViewTest();
  }
}

TYPED_TEST( SparsityPatternViewTest, empty )
{
  this->resize( NROWS, NCOLS, 20 );

  this->insertTest( 20 );
  this->emptyViewTest();
}

#endif

} // namespace testing
} // namespace LvArray

// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
