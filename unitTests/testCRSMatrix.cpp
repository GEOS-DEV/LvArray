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

#include "CRSMatrix.hpp"
#include "Array.hpp"
#include "ArrayOfArrays.hpp"
#include "ArrayOfSets.hpp"
#include "ArrayUtilities.hpp"
#include "testUtils.hpp"

// TPL includes
#include <gtest/gtest.h>

// System includes
#include <vector>
#include <set>
#include <utility>
#include <random>

namespace LvArray
{
namespace testing
{

using INDEX_TYPE = std::ptrdiff_t;
using COL_TYPE = int;

// A matrix is an array of rows. Where each row is a sorted collection of unique columns each paired with a value.
template< typename T >
using REF_TYPE = std::vector< std::map< COL_TYPE, T > >;

/**
 * @brief Check that the CRSMatrixView is equivalent to the reference type.
 * @param [in] view the CRSMatrixView to check.
 * @param [in] ref the reference to check against.
 */
template< typename T >
void compareToReference( CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const & view,
                         REF_TYPE< T > const & ref )
{
  INDEX_TYPE const numRows = view.numRows();
  ASSERT_EQ( numRows, ref.size());

  INDEX_TYPE ref_nnz = 0;
  for( INDEX_TYPE row = 0; row < numRows; ++row )
  {
    INDEX_TYPE const rowNNZ = view.numNonZeros( row );
    INDEX_TYPE const refRowNNZ = ref[row].size();
    ref_nnz += refRowNNZ;

    ASSERT_EQ( rowNNZ, refRowNNZ );

    if( rowNNZ == 0 )
    {
      ASSERT_TRUE( view.empty( row ));
    }
    else
    {
      ASSERT_FALSE( view.empty( row ));
    }

    auto it = ref[row].begin();
    COL_TYPE const * const columns = view.getColumns( row );
    T const * const entries = view.getEntries( row );
    for( INDEX_TYPE i = 0; i < rowNNZ; ++i )
    {
      EXPECT_FALSE( view.empty( row, columns[i] ));
      EXPECT_EQ( columns[i], it->first );
      EXPECT_EQ( entries[i], it->second );
      ++it;
    }
  }

  ASSERT_EQ( view.numNonZeros(), ref_nnz );

  if( view.numNonZeros() == 0 )
  {
    ASSERT_TRUE( view.empty());
  }
}

#define COMPARE_TO_REFERENCE( view, ref ) { SCOPED_TRACE( "" ); compareToReference( view, ref ); \
}

enum class AddType
{
  BINARY_SEARCH,
  LINEAR_SEARCH,
  HYBRID,
  UNSORTED_BINARY
};

template< typename T >
class CRSMatrixTest : public ::testing::Test
{
public:

  void resize( INDEX_TYPE const nRows,
               INDEX_TYPE const nCols,
               INDEX_TYPE initialRowCapacity=0 )
  {
    m_matrix.resize( nRows, nCols, initialRowCapacity );
    m_ref.resize( nRows );

    COMPARE_TO_REFERENCE( m_matrix.toViewConst(), m_ref );
  }

  /**
   * @brief Test the insert multiple method of the CRSMatrix.
   * @param [in] maxInserts the number of entries to insert at a time.
   * @param [in] oneAtATime If true the entries are inserted one at at time.
   */
  void insert( INDEX_TYPE const maxInserts, bool const oneAtATime=false )
  {
    INDEX_TYPE const numRows = m_matrix.numRows();

    // Create an Array of the columns and values to insert into each row.
    ArrayOfArrays< COL_TYPE > const columnsToInsert = createArrayOfColumns( maxInserts, oneAtATime );
    ArrayOfArrays< T > const valuesToInsert = createArrayOfValues( columnsToInsert.toViewConst() );

    // Insert the entries into the reference.
    insertIntoRef( columnsToInsert.toViewConst() );

    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      INDEX_TYPE const initialNNZ = m_matrix.numNonZeros( row );
      INDEX_TYPE nInserted = 0;

      // LVARRAY_LOG_VAR( m_matrix.getColumns( row ).toSliceConst() );
      // LVARRAY_LOG_VAR( m_matrix.getEntries( row ).toSliceConst() );
      // LVARRAY_LOG_VAR( columnsToInsert[ row ].toSliceConst() );
      // LVARRAY_LOG_VAR( valuesToInsert[ row ].toSliceConst() );

      if( oneAtATime )
      {
        for( INDEX_TYPE i = 0; i < columnsToInsert.sizeOfArray( row ); ++i )
        {
          nInserted += m_matrix.insertNonZero( row, columnsToInsert( row, i ), valuesToInsert( row, i ) );
        }
      }
      else
      { nInserted = m_matrix.insertNonZeros( row, columnsToInsert[ row ], valuesToInsert[ row ], columnsToInsert.sizeOfArray( row ) ); }

      // LVARRAY_LOG_VAR( m_matrix.getColumns( row ).toSliceConst() );
      // LVARRAY_LOG_VAR( m_matrix.getEntries( row ).toSliceConst() );
      // LVARRAY_LOG( "" );

      ASSERT_EQ( m_matrix.numNonZeros( row ) - initialNNZ, nInserted );
      ASSERT_EQ( m_matrix.numNonZeros( row ), m_ref[ row ].size());
    }

    COMPARE_TO_REFERENCE( m_matrix.toViewConst(), m_ref );
  }

  /**
   * @brief Fill a row of the CRSMatrixView up to its capacity.
   * @param [in] row the row of the CRSMatrixView to fill.
   */
  void fillRow( INDEX_TYPE const row )
  {
    INDEX_TYPE nToInsert = m_matrix.nonZeroCapacity( row ) - m_matrix.numNonZeros( row );
    COL_TYPE testColumn = COL_TYPE( m_matrix.numColumns() - 1 );
    COL_TYPE const * const columns = m_matrix.getColumns( row );
    T const * const entries = m_matrix.getEntries( row );

    ASSERT_GE( m_matrix.numColumns(), m_matrix.nonZeroCapacity( row ));

    while( nToInsert > 0 )
    {
      T const value( testColumn );
      bool const success = m_matrix.insertNonZero( row, testColumn, value );
      EXPECT_NE( success, m_ref[ row ].count( testColumn ) );
      m_ref[ row ][ testColumn ] = value;
      nToInsert -= success;
      testColumn -= 1;
    }

    EXPECT_EQ( m_matrix.numNonZeros( row ), m_ref[row].size());
    EXPECT_EQ( m_matrix.nonZeroCapacity( row ), m_matrix.numNonZeros( row ));

    EXPECT_EQ( m_matrix.getColumns( row ), columns );
    EXPECT_EQ( m_matrix.getEntries( row ), entries );
  }

  /**
   * @brief Test the setRowCapacity method of CRSMatrix.
   */
  void rowCapacityTest( INDEX_TYPE const maxInserts )
  {
    COL_TYPE const * const columns0 = m_matrix.getColumns( 0 );
    T const * const entries0 = m_matrix.getEntries( 0 );

    this->insert( maxInserts );

    ASSERT_EQ( m_matrix.getColumns( 0 ), columns0 );
    ASSERT_EQ( m_matrix.getEntries( 0 ), entries0 );

    INDEX_TYPE const numRows = m_matrix.numRows();
    ASSERT_EQ( numRows, m_ref.size());

    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      INDEX_TYPE const rowNNZ = m_matrix.numNonZeros( row );
      ASSERT_EQ( rowNNZ, m_ref[row].size());

      // The new capacity will be the index of the row.
      INDEX_TYPE const new_capacity = row;
      if( new_capacity < rowNNZ )
      {
        COL_TYPE const * const columns = m_matrix.getColumns( row );
        T const * const entries = m_matrix.getEntries( row );
        m_matrix.setRowCapacity( row, new_capacity );

        ASSERT_EQ( m_matrix.nonZeroCapacity( row ), new_capacity );
        ASSERT_EQ( m_matrix.nonZeroCapacity( row ), m_matrix.numNonZeros( row ));

        // Check that the pointers to the columns and entries haven't changed.
        ASSERT_EQ( m_matrix.getColumns( row ), columns );
        ASSERT_EQ( m_matrix.getEntries( row ), entries );

        // Erase the last entries from the reference.
        INDEX_TYPE const difference = rowNNZ - new_capacity;
        auto erase_pos = m_ref[row].end();
        std::advance( erase_pos, -difference );
        m_ref[row].erase( erase_pos, m_ref[row].end());

        ASSERT_EQ( m_matrix.numNonZeros( row ), m_ref[row].size());
      }
      else
      {
        m_matrix.setRowCapacity( row, new_capacity );
        ASSERT_EQ( m_matrix.nonZeroCapacity( row ), new_capacity );
        ASSERT_EQ( m_matrix.numNonZeros( row ), rowNNZ );

        fillRow( row );
      }
    }

    COMPARE_TO_REFERENCE( m_matrix.toViewConst(), m_ref );
  }

  /**
   * @brief Test the copy constructor of the CRSMatrix.
   */
  void deepCopyTest() const
  {
    CRSMatrix< T > copy( m_matrix );

    ASSERT_EQ( m_matrix.numRows(), copy.numRows());
    ASSERT_EQ( m_matrix.numColumns(), copy.numColumns());
    ASSERT_EQ( m_matrix.numNonZeros(), copy.numNonZeros());

    INDEX_TYPE const totalNNZ = m_matrix.numNonZeros();

    for( INDEX_TYPE row = 0; row < m_matrix.numRows(); ++row )
    {
      INDEX_TYPE const nnz = m_matrix.numNonZeros( row );
      ASSERT_EQ( nnz, copy.numNonZeros( row ));

      COL_TYPE const * const cols = m_matrix.getColumns( row );
      COL_TYPE const * const cols_cpy = copy.getColumns( row );
      ASSERT_NE( cols, cols_cpy );

      T const * const entries = m_matrix.getEntries( row );
      T const * const entries_cpy = copy.getEntries( row );
      ASSERT_NE( entries, entries_cpy );

      // Iterate backwards and remove entries from copy.
      for( INDEX_TYPE i = nnz - 1; i >= 0; --i )
      {
        EXPECT_EQ( cols[i], cols_cpy[i] );
        EXPECT_EQ( entries[i], entries_cpy[i] );

        copy.removeNonZero( row, cols_cpy[i] );
      }

      EXPECT_EQ( copy.numNonZeros( row ), 0 );
      EXPECT_EQ( m_matrix.numNonZeros( row ), nnz );
    }

    EXPECT_EQ( copy.numNonZeros(), 0 );
    EXPECT_EQ( m_matrix.numNonZeros(), totalNNZ );

    COMPARE_TO_REFERENCE( m_matrix.toViewConst(), m_ref );
  }

  /**
   * @brief Test the copy constructor of the CRSMatrixView.
   */
  void shallowCopyTest() const
  {
    CRSMatrixView< T > copy( m_matrix );

    ASSERT_EQ( m_matrix.numRows(), copy.numRows());
    ASSERT_EQ( m_matrix.numColumns(), copy.numColumns());
    ASSERT_EQ( m_matrix.numNonZeros(), copy.numNonZeros());

    for( INDEX_TYPE row = 0; row < m_matrix.numRows(); ++row )
    {
      INDEX_TYPE const nnz = m_matrix.numNonZeros( row );
      ASSERT_EQ( nnz, copy.numNonZeros( row ));

      COL_TYPE const * const cols = m_matrix.getColumns( row );
      COL_TYPE const * const cols_cpy = copy.getColumns( row );
      ASSERT_EQ( cols, cols_cpy );

      T const * const entries = m_matrix.getEntries( row );
      T const * const entries_cpy = copy.getEntries( row );
      ASSERT_EQ( entries, entries_cpy );

      // Iterate backwards and remove entries from copy.
      for( INDEX_TYPE i = nnz - 1; i >= 0; --i )
      {
        EXPECT_EQ( cols[i], cols_cpy[i] );
        EXPECT_EQ( entries[i], entries_cpy[i] );

        copy.removeNonZero( row, cols_cpy[i] );
      }

      EXPECT_EQ( copy.numNonZeros( row ), 0 );
      EXPECT_EQ( m_matrix.numNonZeros( row ), 0 );
    }

    EXPECT_EQ( copy.numNonZeros(), 0 );
    EXPECT_EQ( m_matrix.numNonZeros(), 0 );
    EXPECT_TRUE( copy.empty());
    EXPECT_TRUE( m_matrix.empty());
  }

  /**
   * @brief Test the reserveNonZeros method of the CRSMatrix.
   * @param [in] capacityPerRow the capacity to reserve for each row.
   */
  void reserveTest( INDEX_TYPE const capacityPerRow )
  {
    INDEX_TYPE const nRows = m_matrix.numRows();

    ASSERT_EQ( m_matrix.nonZeroCapacity(), 0 );

    m_matrix.reserveNonZeros( nRows * capacityPerRow );
    ASSERT_EQ( m_matrix.nonZeroCapacity(), nRows * capacityPerRow );
    ASSERT_EQ( m_matrix.numNonZeros(), 0 );

    for( INDEX_TYPE row = 0; row < nRows; ++row )
    {
      ASSERT_EQ( m_matrix.nonZeroCapacity( row ), 0 );
      ASSERT_EQ( m_matrix.numNonZeros( row ), 0 );
    }

    COL_TYPE const * const columns = m_matrix.getColumns( 0 );
    T const * const entries = m_matrix.getEntries( 0 );

    insert( capacityPerRow / 2 );

    COL_TYPE const * const newColumns = m_matrix.getColumns( 0 );
    T const * const newEntries = m_matrix.getEntries( 0 );

    EXPECT_EQ( newColumns, columns );
    EXPECT_EQ( newEntries, entries );
  }


  void setValues()
  {
    T const val = T( rand( 100 ) );
    m_matrix.setValues( val );

    for( auto & row : m_ref )
    {
      for( auto & kvPair : row )
      {
        kvPair.second = val;
      }
    }

    COMPARE_TO_REFERENCE( m_matrix.toViewConst(), m_ref );
  }

  void compress()
  {
    m_matrix.compress();

    T const * const entries = m_matrix.getEntries( 0 );
    COL_TYPE const * const columns = m_matrix.getColumns( 0 );
    INDEX_TYPE const * const offsets = m_matrix.getOffsets();

    INDEX_TYPE curOffset = 0;
    for( INDEX_TYPE row = 0; row < m_matrix.numRows(); ++row )
    {
      // The last row will have all the extra capacity.
      if( row != m_matrix.numRows() - 1 )
      {
        EXPECT_EQ( m_matrix.numNonZeros( row ), m_matrix.nonZeroCapacity( row ));
      }

      COL_TYPE const * const rowColumns = m_matrix.getColumns( row );
      EXPECT_EQ( rowColumns, columns + curOffset );

      T const * const rowEntries = m_matrix.getEntries( row );
      EXPECT_EQ( rowEntries, entries + curOffset );

      EXPECT_EQ( offsets[row], curOffset );
      curOffset += m_matrix.numNonZeros( row );
    }

    COMPARE_TO_REFERENCE( m_matrix.toViewConst(), m_ref );
  }

protected:

  ArrayOfArrays< COL_TYPE > createArrayOfColumns( INDEX_TYPE const maxCols, bool const oneAtATime )
  {
    INDEX_TYPE const numRows = m_matrix.numRows();

    // Create an array of the columns.
    ArrayOfArrays< COL_TYPE > columns;

    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      INDEX_TYPE const nCols = rand( maxCols );
      columns.appendArray( nCols );
      for( INDEX_TYPE i = 0; i < nCols; ++i )
      {
        columns( row, i ) = rand( m_matrix.numColumns() - 1 );
      }

      if( !oneAtATime && nCols != 0 )
      {
        INDEX_TYPE const numUnique = sortedArrayManipulation::makeSortedUnique( &columns( row, 0 ), &columns( row, 0 ) + nCols );
        columns.resizeArray( row, numUnique );
      }
    }

    return columns;
  }

  void insertIntoRef( ArrayOfArraysView< COL_TYPE const, INDEX_TYPE const, true > const & columnsToInsert )
  {
    for( INDEX_TYPE i = 0; i < columnsToInsert.size(); ++i )
    {
      for( INDEX_TYPE j = 0; j < columnsToInsert.sizeOfArray( i ); ++j )
      {
        COL_TYPE const col = columnsToInsert( i, j );
        m_ref[ i ][ col ] = T( col );
      }
    }
  }

  ArrayOfArrays< T > createArrayOfValues( ArrayOfArraysView< COL_TYPE const, INDEX_TYPE const, true > const & columns ) const
  {
    ArrayOfArrays< T > values;

    INDEX_TYPE const numRows = m_matrix.numRows();
    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      INDEX_TYPE const nCols = columns.sizeOfArray( row );
      values.appendArray( nCols );
      for( INDEX_TYPE i = 0; i < nCols; ++i )
      {
        values( row, i ) = T( columns( row, i ) );
      }
    }

    return values;
  }

  INDEX_TYPE rand( INDEX_TYPE const max )
  {
    return std::uniform_int_distribution< INDEX_TYPE >( 0, max )( m_gen );
  }

  std::mt19937_64 m_gen;

  CRSMatrix< T > m_matrix;

  REF_TYPE< T > m_ref;

  /// Check that the move, toView, and toViewConst methods of CRSMatrix< T > are detected.
  static_assert( bufferManipulation::HasMemberFunction_move< CRSMatrix< T > >,
                 "CRSMatrix< T > has a move method." );
  static_assert( HasMemberFunction_toView< CRSMatrix< T > >,
                 "CRSMatrix< T > has a toView method." );
  static_assert( HasMemberFunction_toViewConst< CRSMatrix< T > >,
                 "CRSMatrix< T > has a toViewConst method." );

  /// Check that the move and toViewConst methods of CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > are detected.
  static_assert( bufferManipulation::HasMemberFunction_move< CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > >,
                 "CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > has a move method." );
  static_assert( HasMemberFunction_toView< CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > >,
                 "CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > has a toView method." );
  static_assert( HasMemberFunction_toViewConst< CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > >,
                 "CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > has a toViewConst method." );

  /// Check that the move and toViewConst methods of CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > are detected.
  static_assert( bufferManipulation::HasMemberFunction_move< CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > >,
                 "CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > has a move method." );
  static_assert( HasMemberFunction_toView< CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > >,
                 "CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > has a toView method." );
  static_assert( HasMemberFunction_toViewConst< CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > >,
                 "CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > has a toViewConst method." );

  /// Check that the move and toViewConst methods of CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > are
  /// detected.
  static_assert( bufferManipulation::HasMemberFunction_move< CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > >,
                 "CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > has a move method." );
  static_assert( HasMemberFunction_toView< CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > >,
                 "CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > has a toView method." );
  static_assert( HasMemberFunction_toViewConst< CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > >,
                 "CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > has a toViewConst method." );

  /// Check that GetViewType and GetViewTypeConst are correct for CRSMatrix< T >
  static_assert( std::is_same_v< typename GetViewType< CRSMatrix< T > >::type,
                                 CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > const >,
                 "The view type of CRSMatrix< T > is CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > const." );
  static_assert( std::is_same_v< typename GetViewTypeConst< CRSMatrix< T > >::type,
                                 CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const >,
                 "The const view type of CRSMatrix< T > is CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const." );

  /// Check that GetViewType and GetViewTypeConst are correct for CRSMatrixView< T, COL_TYPE, INDEX_TYPE const >
  static_assert( std::is_same_v< typename GetViewType< CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > >::type,
                                 CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > const >,
                 "The view type of CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > is CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > const." );
  static_assert( std::is_same_v< typename GetViewTypeConst< CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > >::type,
                                 CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const >,
                 "The const view type of CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > is CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const." );

  /// Check that GetViewType and GetViewTypeConst are correct for CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const >
  static_assert( std::is_same_v< typename GetViewType< CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > >::type,
                                 CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > const >,
                 "The view type of CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > is CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > const." );
  static_assert( std::is_same_v< typename GetViewTypeConst< CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > >::type,
                                 CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const >,
                 "The const view type of CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > is CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const." );

  /// Check that GetViewType and GetViewTypeConst are correct for CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE
  /// const >
  static_assert( std::is_same_v< typename GetViewType< CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > >::type,
                                 CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const >,
                 "The view type of CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > is CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const." );
  static_assert( std::is_same_v< typename GetViewTypeConst< CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > >::type,
                                 CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const >,
                 "The const view type of CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > is CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const." );
};

using TestTypes = ::testing::Types<
  int
  , Tensor
  , TestString
  >;
TYPED_TEST_SUITE( CRSMatrixTest, TestTypes, );

const int DEFAULT_NROWS = 25;
const int DEFAULT_NCOLS = 40;
const int DEFAULT_MAX_INSERTS = DEFAULT_NCOLS / 2;

TYPED_TEST( CRSMatrixTest, construction )
{
  CRSMatrix< TypeParam > m( DEFAULT_NROWS, DEFAULT_NCOLS );
  EXPECT_EQ( m.numRows(), DEFAULT_NROWS );
  EXPECT_EQ( m.numColumns(), DEFAULT_NCOLS );
  EXPECT_EQ( m.numNonZeros(), 0 );
  EXPECT_TRUE( m.empty());

  for( INDEX_TYPE row = 0; row < DEFAULT_NROWS; ++row )
  {
    EXPECT_EQ( m.numNonZeros( row ), 0 );
    EXPECT_EQ( m.nonZeroCapacity( row ), 0 );
    EXPECT_EQ( m.empty( row ), true );

    COL_TYPE const * const columns = m.getColumns( row );
    EXPECT_EQ( columns, nullptr );

    TypeParam const * const entries = m.getEntries( row );
    EXPECT_EQ( entries, nullptr );
  }
}

TYPED_TEST( CRSMatrixTest, hintConstruction )
{
  CRSMatrix< TypeParam > m( DEFAULT_NROWS, DEFAULT_NCOLS, 5 );
  EXPECT_EQ( m.numRows(), DEFAULT_NROWS );
  EXPECT_EQ( m.numColumns(), DEFAULT_NCOLS );
  EXPECT_EQ( m.numNonZeros(), 0 );
  EXPECT_EQ( m.nonZeroCapacity(), DEFAULT_NROWS * 5 );
  EXPECT_TRUE( m.empty());

  for( INDEX_TYPE row = 0; row < DEFAULT_NROWS; ++row )
  {
    EXPECT_EQ( m.numNonZeros( row ), 0 );
    EXPECT_EQ( m.nonZeroCapacity( row ), 5 );
    EXPECT_EQ( m.empty( row ), true );

    COL_TYPE const * const columns = m.getColumns( row );
    ASSERT_NE( columns, nullptr );

    TypeParam const * const entries = m.getEntries( row );
    ASSERT_NE( entries, nullptr );
  }
}

TYPED_TEST( CRSMatrixTest, insert )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );

  for( int i = 0; i < 4; ++i )
  {
    this->insert( DEFAULT_MAX_INSERTS, true );
  }
}

TYPED_TEST( CRSMatrixTest, insertMultiple )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );

  for( int i = 0; i < 4; ++i )
  {
    this->insert( DEFAULT_MAX_INSERTS );
  }
}

TYPED_TEST( CRSMatrixTest, reserveNonZeros )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );
  this->reserveTest( DEFAULT_MAX_INSERTS );
}

TYPED_TEST( CRSMatrixTest, rowCapacity )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS, DEFAULT_MAX_INSERTS );
  this->rowCapacityTest( DEFAULT_MAX_INSERTS );
}

TYPED_TEST( CRSMatrixTest, deepCopy )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );
  this->insert( DEFAULT_MAX_INSERTS );
  this->deepCopyTest();
}

TYPED_TEST( CRSMatrixTest, shallowCopy )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );
  this->insert( DEFAULT_MAX_INSERTS );
  this->deepCopyTest();
}

TYPED_TEST( CRSMatrixTest, setValues )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );
  this->insert( DEFAULT_MAX_INSERTS );
  this->setValues();
}

TYPED_TEST( CRSMatrixTest, compress )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );
  this->insert( DEFAULT_MAX_INSERTS );
  this->compress();
}

template< typename POLICY_VALUE_PAIR >
class CRSMatrixViewTest : public CRSMatrixTest< typename POLICY_VALUE_PAIR::second_type >
{
public:

  using Policy = typename POLICY_VALUE_PAIR::first_type;
  using T = typename POLICY_VALUE_PAIR::second_type;

  CRSMatrixViewTest():
    CRSMatrixTest< T >(),
    m_view( this->m_matrix.toView() )
  {};

  /**
   * @brief Test the CRSMatrixView copy constructor in regards to memory motion.
   */
  void memoryMotionTest()
  {
    memoryMotionInit();

    // Check that the columns and entries have been updated.
    INDEX_TYPE curIndex = 0;
    CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > const & view = m_view;
    forall< serialPolicy >( m_view.numRows(),
                            [view, &curIndex]( INDEX_TYPE row )
    {
      memoryMotionCheckRow( view.toViewConst(), row, curIndex );
    } );
  }

  /**
   * @brief Test the CRSMatrix move method.
   */
  void memoryMotionMoveTest()
  {
    memoryMotionInit();

    this->m_matrix.move( chai::CPU );
    INDEX_TYPE curIndex = 0;
    for( INDEX_TYPE row = 0; row < m_view.numRows(); ++row )
    {
      memoryMotionCheckRow( m_view.toViewConst(), row, curIndex );
    }
  }

  /**
   * @brief Test the const capture semantics where COL_TYPE is const.
   */
  void memoryMotionConstTest()
  {
    INDEX_TYPE const numRows = m_view.numRows();
    INDEX_TYPE const numCols = m_view.numColumns();

    // Capture a view with const columns on device and update the entries.
    CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > const & view = m_view.toViewConstSizes();
    forall< Policy >( numRows,
                      [view, numCols] LVARRAY_HOST_DEVICE ( INDEX_TYPE row )
        {
          COL_TYPE const * const columns = view.getColumns( row );
          T * const entries = view.getEntries( row );
          for( INDEX_TYPE i = 0; i < view.numNonZeros( row ); ++i )
          {
            LVARRAY_ERROR_IF( !arrayManipulation::isPositive( columns[i] ) || columns[i] >= numCols, "Invalid column." );
            entries[i] = T( 2 * i + 7 );
          }
        } );

    // This should copy back the entries but not the columns.
    this->m_matrix.move( chai::CPU );
    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      ASSERT_EQ( m_view.numNonZeros( row ), this->m_ref[row].size());

      auto it = this->m_ref[row].begin();
      for( INDEX_TYPE i = 0; i < m_view.numNonZeros( row ); ++i )
      {
        // So the columns should be the same as the reference.
        COL_TYPE const col = m_view.getColumns( row )[i];
        EXPECT_EQ( col, it->first );
        EXPECT_EQ( m_view.getEntries( row )[i], T( 2 * i + 7 ));
        ++it;
      }
    }
  }

  /**
   * @brief Test the const capture semantics where both T and COL_TYPE are const.
   * @param [in] maxInserts the maximum number of inserts.
   */
  void memoryMotionConstConstTest( INDEX_TYPE const maxInserts )
  {
    INDEX_TYPE const numRows = m_view.numRows();
    INDEX_TYPE const numCols = m_view.numColumns();

    // Capture a view with const columns and const values on device.
    CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const & view = m_view.toViewConst();
    forall< Policy >( numRows,
                      [view, numCols] LVARRAY_HOST_DEVICE ( INDEX_TYPE row )
        {
          COL_TYPE const * const columns = view.getColumns( row );
          T const * const entries = view.getEntries( row );
          for( INDEX_TYPE i = 0; i < view.numNonZeros( row ); ++i )
          {
            LVARRAY_ERROR_IF( !arrayManipulation::isPositive( columns[i] ) || columns[i] >= numCols, "Invalid column." );
            LVARRAY_ERROR_IF( entries[i] != T( columns[i] ), "Incorrect value." );
          }
        }
                      );

    // Insert entries, this will modify the offsets, sizes, columns and entries.
    this->insert( maxInserts );

    // Move the matrix back to the host, this should copy nothing.
    this->m_matrix.move( chai::CPU );

    // And therefore the matrix should still equal the reference.
    COMPARE_TO_REFERENCE( m_view.toViewConst(), this->m_ref );
  }

  /**
   * @brief Test the CRSMatrixView insert methods.
   * @param [in] maxInserts the maximum number of inserts.
   * @param [in] oneAtATime If true the entries are inserted one at a time.
   */
  void insertIntoView( INDEX_TYPE const maxInserts, bool const oneAtATime=false )
  {
    INDEX_TYPE const numRows = this->m_matrix.numRows();

    // Create an Array of the columns and values to insert into each row.
    ArrayOfArrays< COL_TYPE > const columnsToInsert = this->createArrayOfColumns( maxInserts, oneAtATime );
    ArrayOfArrays< T > const valuesToInsert = this->createArrayOfValues( columnsToInsert.toViewConst() );

    // Reserve space for the inserts
    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      this->m_matrix.reserveNonZeros( row, this->m_matrix.numNonZeros( row ) + columnsToInsert.sizeOfArray( row ) );
    }

    // Insert the entries into the reference.
    this->insertIntoRef( columnsToInsert.toViewConst() );

    CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > const & view = m_view;
    ArrayOfArraysView< COL_TYPE const, INDEX_TYPE const, true > const & columnsView = columnsToInsert.toViewConst();
    ArrayOfArraysView< T const, INDEX_TYPE const, true > const & valuesView = valuesToInsert.toViewConst();
    if( oneAtATime )
    {
      forall< Policy >( numRows,
                        [view, columnsView] LVARRAY_HOST_DEVICE ( INDEX_TYPE const row )
          {
            INDEX_TYPE const initialNNZ = view.numNonZeros( row );
            INDEX_TYPE nInserted = 0;
            for( INDEX_TYPE i = 0; i < columnsView.sizeOfArray( row ); ++i )
            {
              COL_TYPE const col = columnsView( row, i );
              nInserted += view.insertNonZero( row, col, T( col ) );
            }

            PORTABLE_EXPECT_EQ( view.numNonZeros( row ) - initialNNZ, nInserted );
          }
                        );
    }
    else
    {
      forall< Policy >( numRows,
                        [view, columnsView, valuesView] LVARRAY_HOST_DEVICE ( INDEX_TYPE const row )
          {
            INDEX_TYPE const initialNNZ = view.numNonZeros( row );
            INDEX_TYPE const nInserted = view.insertNonZeros( row,
                                                              columnsView[row],
                                                              valuesView[row],
                                                              columnsView.sizeOfArray( row ) );
            PORTABLE_EXPECT_EQ( view.numNonZeros( row ) - initialNNZ, nInserted );
          }
                        );
    }

    this->m_matrix.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_view.toViewConst(), this->m_ref );
  }

  /**
   * @brief Test the CRSMatrixView remove method.
   * @param [in] maxRemoves the maximum number of removes.
   * @param [in] oneAtATime If true the columns are removed one at a time.
   */
  void removeFromView( INDEX_TYPE const maxRemoves, bool const oneAtATime=false )
  {
    INDEX_TYPE const numRows = m_view.numRows();

    // Create an Array of the columns and values to insert into each row.
    ArrayOfArrays< COL_TYPE > const columnsToRemove = this->createArrayOfColumns( maxRemoves, oneAtATime );

    // Insert the entries into the reference.
    for( INDEX_TYPE i = 0; i < columnsToRemove.size(); ++i )
    {
      for( INDEX_TYPE j = 0; j < columnsToRemove.sizeOfArray( i ); ++j )
      {
        COL_TYPE const col = columnsToRemove( i, j );
        this->m_ref[ i ].erase( col );
      }
    }

    CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > const & view = m_view;
    ArrayOfArraysView< COL_TYPE const, INDEX_TYPE const, true > const & columnsView = columnsToRemove.toViewConst();
    if( oneAtATime )
    {
      forall< Policy >( numRows,
                        [view, columnsView] LVARRAY_HOST_DEVICE ( INDEX_TYPE row )
          {
            INDEX_TYPE const initialNNZ = view.numNonZeros( row );
            INDEX_TYPE nRemoved = 0;
            for( INDEX_TYPE i = 0; i < columnsView.sizeOfArray( row ); ++i )
            {
              nRemoved += view.removeNonZero( row, columnsView( row, i ) );
            }

            PORTABLE_EXPECT_EQ( initialNNZ - view.numNonZeros( row ), nRemoved );
          } );
    }
    else
    {
      forall< Policy >( numRows,
                        [view, columnsView] LVARRAY_HOST_DEVICE ( INDEX_TYPE const row )
          {
            INDEX_TYPE const initialNNZ = view.numNonZeros( row );
            INDEX_TYPE const nRemoved = view.removeNonZeros( row, columnsView[row], columnsView.sizeOfArray( row ) );
            PORTABLE_EXPECT_EQ( initialNNZ - view.numNonZeros( row ), nRemoved );
          } );
    }

    this->m_matrix.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_view.toViewConst(), this->m_ref );
  }

  /**
   * @brief Test the empty method of the CRSMatrixView on device.
   */
  void emptyTest() const
  {
    INDEX_TYPE const numRows = m_view.numRows();

    // Initialize each row to only contain even numbered columns.
    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      COL_TYPE const * const columns = m_view.getColumns( row );
      COL_TYPE * const columnsNC = const_cast< COL_TYPE * >(columns);
      for( INDEX_TYPE i = 0; i < m_view.numNonZeros( row ); ++i )
      {
        columnsNC[i] = COL_TYPE( 2 * i );
      }
    }

    // Check that each row contains the even columns and no odd columns.
    CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const & view = m_view.toViewConst();
    forall< Policy >( numRows,
                      [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const row )
        {
          for( INDEX_TYPE i = 0; i < view.numNonZeros( row ); ++i )
          {
            PORTABLE_EXPECT_EQ( view.empty( row, COL_TYPE( 2 * i ) ), false );
            PORTABLE_EXPECT_EQ( view.empty( row, COL_TYPE( 2 * i + 1 ) ), true );
          }
        } );
  }

  /**
   * @brief Test the toSparsityPatternView method of the CRSMatrixView.
   */
  void sparsityPatternViewTest() const
  {
    INDEX_TYPE const numRows = m_view.numRows();

    SparsityPatternView< COL_TYPE const, INDEX_TYPE const > const & spView = m_view.toSparsityPatternView();

    // Check that each row contains the even columns and no odd columns on device.
    CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const & view = m_view.toViewConst();
    forall< Policy >( numRows,
                      [spView, view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const row )
        {
          INDEX_TYPE const nnz = view.numNonZeros( row );
          PORTABLE_EXPECT_EQ( nnz, spView.numNonZeros( row ) );

          COL_TYPE const * const columns = view.getColumns( row );
          COL_TYPE const * const spColumns = spView.getColumns( row );
          PORTABLE_EXPECT_EQ( columns, spColumns );

          for( INDEX_TYPE i = 0; i < nnz; ++i )
          {
            PORTABLE_EXPECT_EQ( columns[ i ], spColumns[ i ] );
            PORTABLE_EXPECT_EQ( view.empty( row, columns[ i ] ), spView.empty( row, columns[ i ] ) );
            PORTABLE_EXPECT_EQ( view.empty( row, COL_TYPE( i ) ), spView.empty( row, COL_TYPE( i ) ) );
          }
        } );
  }

  // This should be private but you can't have device lambdas need to be in public methods :(
  void memoryMotionInit() const
  {
    INDEX_TYPE const numRows = m_view.numRows();

    // Set the columns and entries. The const casts here isn't necessary, we could remove and insert
    // into the view instead, but this is quicker.
    INDEX_TYPE curIndex = 0;
    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      COL_TYPE const * const columns = m_view.getColumns( row );
      COL_TYPE * const columnsNC = const_cast< COL_TYPE * >(columns);
      T * const entries = m_view.getEntries( row );
      for( INDEX_TYPE i = 0; i < m_view.numNonZeros( row ); ++i )
      {
        columnsNC[i] = curIndex;
        entries[i] = T( curIndex++ );
      }
    }

    // Capture the view on device and set the entries. Here the const cast is necessary
    // because we don't want to test the insert/remove methods on device yet.
    CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > const & view = m_view;
    forall< Policy >( numRows,
                      [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const row )
        {
          COL_TYPE const * const columns = view.getColumns( row );
          COL_TYPE * const columnsNC = const_cast< COL_TYPE * >(columns);
          T * const entries = view.getEntries( row );
          for( INDEX_TYPE i = 0; i < view.numNonZeros( row ); ++i )
          {
            columnsNC[i] += columns[i];
            entries[i] += entries[i];
          }
        }
                      );
  }

protected:

  static void memoryMotionCheckRow( CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const & view,
                                    INDEX_TYPE const row, INDEX_TYPE & curIndex )
  {
    for( INDEX_TYPE i = 0; i < view.numNonZeros( row ); ++i )
    {
      EXPECT_EQ( COL_TYPE( curIndex ) * 2, view.getColumns( row )[i] );
      T expectedEntry( curIndex );
      expectedEntry += T( curIndex );
      EXPECT_EQ( expectedEntry, view.getEntries( row )[i] );
      ++curIndex;
    }
  }

  CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > const & m_view;
};

using ViewTestTypes = ::testing::Types<
  std::pair< serialPolicy, int >
  , std::pair< serialPolicy, Tensor >
  , std::pair< serialPolicy, TestString >
#if defined(USE_OPENMP)
  , std::pair< parallelHostPolicy, int >
  , std::pair< parallelHostPolicy, Tensor >
  , std::pair< parallelHostPolicy, TestString >
#endif
#if defined(USE_CUDA)
  , std::pair< parallelDevicePolicy< 256 >, int >
  , std::pair< parallelDevicePolicy< 256 >, Tensor >
#endif
  >;

TYPED_TEST_SUITE( CRSMatrixViewTest, ViewTestTypes, );

TYPED_TEST( CRSMatrixViewTest, memoryMotion )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );
  this->insert( DEFAULT_MAX_INSERTS );
  this->memoryMotionTest();
}

TYPED_TEST( CRSMatrixViewTest, memoryMotionMove )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );
  this->insert( DEFAULT_MAX_INSERTS );
  this->memoryMotionMoveTest();
}

TYPED_TEST( CRSMatrixViewTest, memoryMotionConst )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );
  this->insert( DEFAULT_MAX_INSERTS );
  this->memoryMotionConstTest();
}

TYPED_TEST( CRSMatrixViewTest, memoryMotionConstConst )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );
  this->insert( DEFAULT_MAX_INSERTS );
  this->memoryMotionConstConstTest( DEFAULT_MAX_INSERTS );
}

TYPED_TEST( CRSMatrixViewTest, insert )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );

  for( int i = 0; i < 4; ++i )
  {
    this->insertIntoView( DEFAULT_MAX_INSERTS, true );
  }
}

TYPED_TEST( CRSMatrixViewTest, insertMultiple )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );

  for( int i = 0; i < 4; ++i )
  {
    this->insertIntoView( DEFAULT_MAX_INSERTS );
  }
}

TYPED_TEST( CRSMatrixViewTest, remove )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );

  for( int i = 0; i < 4; ++i )
  {
    this->insert( DEFAULT_MAX_INSERTS );
    this->removeFromView( DEFAULT_MAX_INSERTS, true );
  }
}

TYPED_TEST( CRSMatrixViewTest, removeMultiple )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );

  for( int i = 0; i < 4; ++i )
  {
    this->insert( DEFAULT_MAX_INSERTS );
    this->removeFromView( DEFAULT_MAX_INSERTS );
  }
}

TYPED_TEST( CRSMatrixViewTest, empty )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );
  this->insert( DEFAULT_MAX_INSERTS );
  this->emptyTest();
}

TYPED_TEST( CRSMatrixViewTest, sparsityPatternView )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );
  this->insert( DEFAULT_MAX_INSERTS );
  this->sparsityPatternViewTest();
}

template< typename POLICY_VALUE_PAIR >
class CRSMatrixViewAtomicTest : public CRSMatrixViewTest< POLICY_VALUE_PAIR >
{
public:
  using Policy = typename POLICY_VALUE_PAIR::first_type;
  using AtomicPolicy = typename RAJAHelper< Policy >::AtomicPolicy;
  using T = typename POLICY_VALUE_PAIR::second_type;

  void addToRow( AddType const type, INDEX_TYPE const nThreads )
  {
    INDEX_TYPE const numRows = m_view.numRows();

    for( INDEX_TYPE row = 0; row < numRows; ++row )
    {
      INDEX_TYPE const nColsInRow = m_view.numNonZeros( row );

      // Array that holds the columns each thread will add to in the current row.
      ArrayOfArrays< COL_TYPE > columns;
      for( INDEX_TYPE threadID = 0; threadID < nThreads; ++threadID )
      {
        INDEX_TYPE const nColsToAddTo = this->rand( nColsInRow );
        columns.appendArray( nColsToAddTo );

        for( INDEX_TYPE i = 0; i < nColsToAddTo; ++i )
        {
          INDEX_TYPE const pos = this->rand( nColsInRow - 1 );
          columns( threadID, i ) = m_view.getColumns( row )[ pos ];
        }
      }

      if( type != AddType::UNSORTED_BINARY )
      {
        for( INDEX_TYPE threadID = 0; threadID < nThreads; ++threadID )
        {
          INDEX_TYPE const numUniqueCols =
            sortedArrayManipulation::makeSortedUnique( columns.getIterableArray( threadID ).begin(), columns.getIterableArray( threadID ).end() );
          columns.resizeArray( threadID, numUniqueCols );
        }
      }

      // Array that holds the values each thread will add to in the current row.
      ArrayOfArrays< T > values;
      for( INDEX_TYPE threadID = 0; threadID < nThreads; ++threadID )
      {
        INDEX_TYPE const nColsToAddTo = columns.sizeOfArray( threadID );
        values.appendArray( nColsToAddTo );

        for( INDEX_TYPE i = 0; i < nColsToAddTo; ++i )
        {
          COL_TYPE const column = columns( threadID, i );
          values( threadID, i ) = T( columns( threadID, i ) );
          this->m_ref[ row ][ column ] += values( threadID, i );
        }
      }

      CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > const & view = m_view.toViewConstSizes();
      ArrayOfArraysView< COL_TYPE const, INDEX_TYPE const, true > colView = columns.toViewConst();
      ArrayOfArraysView< T const, INDEX_TYPE const, true > valView = values.toViewConst();

      if( type == AddType::LINEAR_SEARCH )
      {
        forall< Policy >( nThreads, [view, colView, valView, row] LVARRAY_HOST_DEVICE ( INDEX_TYPE const threadID )
            {
              view.template addToRowLinearSearch< AtomicPolicy >( row, colView[ threadID ], valView[ threadID ], colView.sizeOfArray( threadID ) );
            } );
      }
      if( type == AddType::BINARY_SEARCH )
      {
        forall< Policy >( nThreads, [view, colView, valView, row] LVARRAY_HOST_DEVICE ( INDEX_TYPE const threadID )
            {
              view.template addToRowBinarySearch< AtomicPolicy >( row, colView[ threadID ], valView[ threadID ], colView.sizeOfArray( threadID ) );
            } );
      }
      if( type == AddType::HYBRID )
      {
        forall< Policy >( nThreads, [view, colView, valView, row] LVARRAY_HOST_DEVICE ( INDEX_TYPE const threadID )
            {
              view.template addToRow< AtomicPolicy >( row, colView[ threadID ], valView[ threadID ], colView.sizeOfArray( threadID ) );
            } );
      }
      if( type == AddType::UNSORTED_BINARY )
      {
        forall< Policy >( nThreads, [view, colView, valView, row] LVARRAY_HOST_DEVICE ( INDEX_TYPE const threadID )
            {
              view.template addToRowBinarySearchUnsorted< AtomicPolicy >( row, colView[ threadID ], valView[ threadID ], colView.sizeOfArray( threadID ) );
            } );
      }
    }

    this->m_matrix.move( chai::CPU );
    COMPARE_TO_REFERENCE( m_view.toViewConst(), this->m_ref );
  }

protected:
  using CRSMatrixViewTest< POLICY_VALUE_PAIR >::m_view;
};

using ViewAtomicTestTypes = ::testing::Types<
  std::pair< serialPolicy, int >
  , std::pair< serialPolicy, Tensor >
  , std::pair< serialPolicy, TestString >
#if defined(USE_OPENMP)
  , std::pair< parallelHostPolicy, int >
#endif
#if defined(USE_CUDA)
  , std::pair< parallelDevicePolicy< 256 >, int >
#endif
  >;

TYPED_TEST_SUITE( CRSMatrixViewAtomicTest, ViewAtomicTestTypes, );

TYPED_TEST( CRSMatrixViewAtomicTest, addToRowLinearSearch )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );
  this->insert( DEFAULT_MAX_INSERTS );
  this->addToRow( AddType::LINEAR_SEARCH, 10 );
}

TYPED_TEST( CRSMatrixViewAtomicTest, addToRowBinarySearch )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );
  this->insert( DEFAULT_MAX_INSERTS );
  this->addToRow( AddType::BINARY_SEARCH, 10 );
}

TYPED_TEST( CRSMatrixViewAtomicTest, addToRow )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );
  this->insert( DEFAULT_MAX_INSERTS );
  this->addToRow( AddType::HYBRID, 10 );
}

TYPED_TEST( CRSMatrixViewAtomicTest, addToRowBinarySearchUnsorted )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );
  this->insert( DEFAULT_MAX_INSERTS );
  this->addToRow( AddType::UNSORTED_BINARY, 10 );
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
