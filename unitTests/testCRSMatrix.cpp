/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

#include "CRSMatrix.hpp"
#include "Array.hpp"
#include "ArrayOfArrays.hpp"
#include "ArrayOfSets.hpp"
#include "output.hpp"
#include "testUtils.hpp"
#include "MallocBuffer.hpp"

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

enum class AddType
{
  BINARY_SEARCH,
  LINEAR_SEARCH,
  HYBRID,
  UNSORTED_BINARY
};

template< typename U, typename T >
struct ToArray
{};

template< typename U, typename T, typename COL_TYPE, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
struct ToArray< U, CRSMatrix< T, COL_TYPE, INDEX_TYPE, BUFFER_TYPE > >
{
  using AoA = ArrayOfArrays< U, INDEX_TYPE, BUFFER_TYPE >;
};

template< typename CRS_MATRIX >
class CRSMatrixTest : public ::testing::Test
{
public:

  using T = typename CRS_MATRIX::EntryType;
  using ColType = typename CRS_MATRIX::ColType;
  using IndexType = typename CRS_MATRIX::IndexType;

  using ViewType = typeManipulation::ViewType< CRS_MATRIX >;
  using ViewTypeConstSizes = typeManipulation::ViewTypeConstSizes< CRS_MATRIX >;
  using ViewTypeConst = typeManipulation::ViewTypeConst< CRS_MATRIX >;

  template< typename U >
  using ArrayOfArraysT = typename ToArray< U, CRS_MATRIX >::AoA;

  template< typename U >
  using ArrayOfArraysViewT = typeManipulation::ViewTypeConst< ArrayOfArraysT< std::remove_const_t< U > > >;

  void compareToReference( ViewTypeConst const & view ) const
  {
    IndexType const numRows = view.numRows();
    ASSERT_EQ( numRows, m_ref.size());

    IndexType ref_nnz = 0;
    for( IndexType row = 0; row < numRows; ++row )
    {
      IndexType const rowNNZ = view.numNonZeros( row );
      IndexType const refRowNNZ = m_ref[ row ].size();
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

      auto it = m_ref[ row ].begin();
      ColType const * const columns = view.getColumns( row );
      T const * const entries = view.getEntries( row );
      for( IndexType i = 0; i < rowNNZ; ++i )
      {
        EXPECT_FALSE( view.empty( row, columns[ i ] ));
        EXPECT_EQ( columns[ i ], it->first );
        EXPECT_EQ( entries[ i ], it->second );
        ++it;
      }
    }

    ASSERT_EQ( view.numNonZeros(), ref_nnz );

    if( view.numNonZeros() == 0 )
    {
      ASSERT_TRUE( view.empty());
    }
  }

  #define COMPARE_TO_REFERENCE { SCOPED_TRACE( "" ); this->compareToReference( this->m_matrix.toViewConst() ); \
  }


  void resize( IndexType const nRows,
               IndexType const nCols,
               IndexType initialRowCapacity=0 )
  {
    m_matrix.resize( nRows, nCols, initialRowCapacity );
    m_ref.resize( nRows );

    COMPARE_TO_REFERENCE
  }

  template< typename POLICY >
  void resizeFromRowCapacities( IndexType const numRows, IndexType const numCols )
  {
    COMPARE_TO_REFERENCE;

    std::vector< IndexType > newCapacities( numRows );

    for( IndexType & capacity : newCapacities )
    { capacity = rand( numCols ); }

    m_matrix.template resizeFromRowCapacities< POLICY >( numRows, numCols, newCapacities.data() );

    EXPECT_EQ( m_matrix.numRows(), numRows );
    EXPECT_EQ( m_matrix.numColumns(), numCols );
    for( IndexType row = 0; row < m_matrix.numRows(); ++row )
    {
      EXPECT_EQ( m_matrix.numNonZeros( row ), 0 );
      EXPECT_EQ( m_matrix.nonZeroCapacity( row ), newCapacities[ row ] );
    }

    m_ref.clear();
    m_ref.resize( numRows );
    COMPARE_TO_REFERENCE;
  }

  /**
   * @brief Test the insert multiple method of the CRSMatrix.
   * @param [in] maxInserts the number of entries to insert at a time.
   * @param [in] oneAtATime If true the entries are inserted one at at time.
   */
  void insert( IndexType const maxInserts, bool const oneAtATime=false )
  {
    IndexType const numRows = m_matrix.numRows();

    // Create an Array of the columns and values to insert into each row.
    ArrayOfArraysT< ColType > const columnsToInsert = createArrayOfColumns( maxInserts, oneAtATime );
    ArrayOfArraysT< T > const valuesToInsert = createArrayOfValues( columnsToInsert.toViewConst() );

    // Insert the entries into the reference.
    insertIntoRef( columnsToInsert.toViewConst() );

    for( IndexType row = 0; row < numRows; ++row )
    {
      IndexType const initialNNZ = m_matrix.numNonZeros( row );
      IndexType nInserted = 0;

      if( oneAtATime )
      {
        for( IndexType i = 0; i < columnsToInsert.sizeOfArray( row ); ++i )
        {
          nInserted += m_matrix.insertNonZero( row, columnsToInsert( row, i ), valuesToInsert( row, i ) );
        }
      }
      else
      { nInserted = m_matrix.insertNonZeros( row, columnsToInsert[ row ], valuesToInsert[ row ], columnsToInsert.sizeOfArray( row ) ); }

      ASSERT_EQ( m_matrix.numNonZeros( row ) - initialNNZ, nInserted );
      ASSERT_EQ( m_matrix.numNonZeros( row ), m_ref[ row ].size());
    }

    COMPARE_TO_REFERENCE
  }

  /**
   * @brief Fill a row of the CRSMatrixView up to its capacity.
   * @param [in] row the row of the CRSMatrixView to fill.
   */
  void fillRow( IndexType const row )
  {
    IndexType nToInsert = m_matrix.nonZeroCapacity( row ) - m_matrix.numNonZeros( row );
    ColType testColumn = ColType( m_matrix.numColumns() - 1 );
    ColType const * const columns = m_matrix.getColumns( row );
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

    EXPECT_EQ( m_matrix.getColumns( row ).dataIfContiguous(), columns );
    EXPECT_EQ( m_matrix.getEntries( row ).dataIfContiguous(), entries );
  }

  /**
   * @brief Test the setRowCapacity method of CRSMatrix.
   */
  void rowCapacityTest( IndexType const maxInserts )
  {
    ColType const * const columns0 = m_matrix.getColumns( 0 );
    T const * const entries0 = m_matrix.getEntries( 0 );

    this->insert( maxInserts );

    ASSERT_EQ( m_matrix.getColumns( 0 ).dataIfContiguous(), columns0 );
    ASSERT_EQ( m_matrix.getEntries( 0 ).dataIfContiguous(), entries0 );

    IndexType const numRows = m_matrix.numRows();
    ASSERT_EQ( numRows, m_ref.size());

    for( IndexType row = 0; row < numRows; ++row )
    {
      IndexType const rowNNZ = m_matrix.numNonZeros( row );
      ASSERT_EQ( rowNNZ, m_ref[row].size());

      // The new capacity will be the index of the row.
      IndexType const new_capacity = row;
      if( new_capacity < rowNNZ )
      {
        ColType const * const columns = m_matrix.getColumns( row );
        T const * const entries = m_matrix.getEntries( row );
        m_matrix.setRowCapacity( row, new_capacity );

        ASSERT_EQ( m_matrix.nonZeroCapacity( row ), new_capacity );
        ASSERT_EQ( m_matrix.nonZeroCapacity( row ), m_matrix.numNonZeros( row ));

        // Check that the pointers to the columns and entries haven't changed.
        ASSERT_EQ( m_matrix.getColumns( row ).dataIfContiguous(), columns );
        ASSERT_EQ( m_matrix.getEntries( row ).dataIfContiguous(), entries );

        // Erase the last entries from the reference.
        IndexType const difference = rowNNZ - new_capacity;
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

    COMPARE_TO_REFERENCE
  }

  /**
   * @brief Test the copy constructor of the CRSMatrix.
   */
  void deepCopyTest() const
  {
    CRS_MATRIX copy( m_matrix );

    ASSERT_EQ( m_matrix.numRows(), copy.numRows());
    ASSERT_EQ( m_matrix.numColumns(), copy.numColumns());
    ASSERT_EQ( m_matrix.numNonZeros(), copy.numNonZeros());

    IndexType const totalNNZ = m_matrix.numNonZeros();

    for( IndexType row = 0; row < m_matrix.numRows(); ++row )
    {
      IndexType const nnz = m_matrix.numNonZeros( row );
      ASSERT_EQ( nnz, copy.numNonZeros( row ));

      ColType const * const cols = m_matrix.getColumns( row );
      ColType const * const cols_cpy = copy.getColumns( row );
      ASSERT_NE( cols, cols_cpy );

      T const * const entries = m_matrix.getEntries( row );
      T const * const entries_cpy = copy.getEntries( row );
      ASSERT_NE( entries, entries_cpy );

      // Iterate backwards and remove entries from copy.
      for( IndexType i = nnz - 1; i >= 0; --i )
      {
        EXPECT_EQ( cols[ i ], cols_cpy[ i ] );
        EXPECT_EQ( entries[ i ], entries_cpy[ i ] );

        copy.removeNonZero( row, cols_cpy[ i ] );
      }

      EXPECT_EQ( copy.numNonZeros( row ), 0 );
      EXPECT_EQ( m_matrix.numNonZeros( row ), nnz );
    }

    EXPECT_EQ( copy.numNonZeros(), 0 );
    EXPECT_EQ( m_matrix.numNonZeros(), totalNNZ );

    COMPARE_TO_REFERENCE
  }

  /**
   * @brief Test the copy constructor of the CRSMatrixView.
   */
  void shallowCopyTest() const
  {
    ViewType copy( m_matrix );

    ASSERT_EQ( m_matrix.numRows(), copy.numRows());
    ASSERT_EQ( m_matrix.numColumns(), copy.numColumns());
    ASSERT_EQ( m_matrix.numNonZeros(), copy.numNonZeros());

    for( IndexType row = 0; row < m_matrix.numRows(); ++row )
    {
      IndexType const nnz = m_matrix.numNonZeros( row );
      ASSERT_EQ( nnz, copy.numNonZeros( row ));

      ColType const * const cols = m_matrix.getColumns( row );
      ColType const * const cols_cpy = copy.getColumns( row );
      ASSERT_EQ( cols, cols_cpy );

      T const * const entries = m_matrix.getEntries( row );
      T const * const entries_cpy = copy.getEntries( row );
      ASSERT_EQ( entries, entries_cpy );

      // Iterate backwards and remove entries from copy.
      for( IndexType i = nnz - 1; i >= 0; --i )
      {
        EXPECT_EQ( cols[ i ], cols_cpy[ i ] );
        EXPECT_EQ( entries[ i ], entries_cpy[ i ] );

        copy.removeNonZero( row, cols_cpy[ i ] );
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
  void reserveTest( IndexType const capacityPerRow )
  {
    IndexType const nRows = m_matrix.numRows();

    ASSERT_EQ( m_matrix.nonZeroCapacity(), 0 );

    m_matrix.reserveNonZeros( nRows * capacityPerRow );
    ASSERT_EQ( m_matrix.nonZeroCapacity(), nRows * capacityPerRow );
    ASSERT_EQ( m_matrix.numNonZeros(), 0 );

    for( IndexType row = 0; row < nRows; ++row )
    {
      ASSERT_EQ( m_matrix.nonZeroCapacity( row ), 0 );
      ASSERT_EQ( m_matrix.numNonZeros( row ), 0 );
    }

    ColType const * const columns = m_matrix.getColumns( 0 );
    T const * const entries = m_matrix.getEntries( 0 );

    insert( capacityPerRow / 2 );

    ColType const * const newColumns = m_matrix.getColumns( 0 );
    T const * const newEntries = m_matrix.getEntries( 0 );

    EXPECT_EQ( newColumns, columns );
    EXPECT_EQ( newEntries, entries );
  }

  void compress()
  {
    m_matrix.compress();

    T const * const entries = m_matrix.getEntries( 0 );
    ColType const * const columns = m_matrix.getColumns( 0 );
    IndexType const * const offsets = m_matrix.getOffsets();

    IndexType curOffset = 0;
    for( IndexType row = 0; row < m_matrix.numRows(); ++row )
    {
      // The last row will have all the extra capacity.
      if( row != m_matrix.numRows() - 1 )
      {
        EXPECT_EQ( m_matrix.numNonZeros( row ), m_matrix.nonZeroCapacity( row ));
      }

      ColType const * const rowColumns = m_matrix.getColumns( row );
      EXPECT_EQ( rowColumns, columns + curOffset );

      T const * const rowEntries = m_matrix.getEntries( row );
      EXPECT_EQ( rowEntries, entries + curOffset );

      EXPECT_EQ( offsets[row], curOffset );
      curOffset += m_matrix.numNonZeros( row );
    }

    COMPARE_TO_REFERENCE
  }

protected:

  ArrayOfArraysT< ColType > createArrayOfColumns( IndexType const maxCols, bool const oneAtATime )
  {
    IndexType const numRows = m_matrix.numRows();

    // Create an array of the columns.
    ArrayOfArraysT< ColType > columns;

    for( IndexType row = 0; row < numRows; ++row )
    {
      IndexType const nCols = rand( maxCols );
      columns.appendArray( nCols );
      for( IndexType i = 0; i < nCols; ++i )
      {
        columns( row, i ) = rand( m_matrix.numColumns() - 1 );
      }

      if( !oneAtATime && nCols != 0 )
      {
        IndexType const numUnique = sortedArrayManipulation::makeSortedUnique( &columns( row, 0 ), &columns( row, 0 ) + nCols );
        columns.resizeArray( row, numUnique );
      }
    }

    return columns;
  }

  void insertIntoRef( ArrayOfArraysViewT< ColType const > const & columnsToInsert )
  {
    for( IndexType i = 0; i < columnsToInsert.size(); ++i )
    {
      for( IndexType j = 0; j < columnsToInsert.sizeOfArray( i ); ++j )
      {
        ColType const col = columnsToInsert( i, j );
        m_ref[ i ][ col ] = T( col );
      }
    }
  }

  ArrayOfArraysT< T > createArrayOfValues( ArrayOfArraysViewT< ColType const > const & columns ) const
  {
    ArrayOfArraysT< T > values;

    IndexType const numRows = m_matrix.numRows();
    for( IndexType row = 0; row < numRows; ++row )
    {
      IndexType const nCols = columns.sizeOfArray( row );
      values.appendArray( nCols );
      for( IndexType i = 0; i < nCols; ++i )
      {
        values( row, i ) = T( columns( row, i ) );
      }
    }

    return values;
  }

  IndexType rand( IndexType const max )
  {
    return std::uniform_int_distribution< IndexType >( 0, max )( m_gen );
  }

  std::mt19937_64 m_gen;

  CRS_MATRIX m_matrix;

  std::vector< std::map< ColType, T > > m_ref;
};

using CRSMatrixTestTypes = ::testing::Types<
  CRSMatrix< int, int, std::ptrdiff_t, MallocBuffer >
  , CRSMatrix< Tensor, int, std::ptrdiff_t, MallocBuffer >
  , CRSMatrix< TestString, int, std::ptrdiff_t, MallocBuffer >
#if defined(LVARRAY_USE_CHAI)
  , CRSMatrix< int, int, std::ptrdiff_t, ChaiBuffer >
  , CRSMatrix< Tensor, int, std::ptrdiff_t, ChaiBuffer >
  , CRSMatrix< TestString, int, std::ptrdiff_t, ChaiBuffer >
#endif
  >;
TYPED_TEST_SUITE( CRSMatrixTest, CRSMatrixTestTypes, );

const int DEFAULT_NROWS = 25;
const int DEFAULT_NCOLS = 40;
const int DEFAULT_MAX_INSERTS = DEFAULT_NCOLS / 2;

TYPED_TEST( CRSMatrixTest, construction )
{
  TypeParam m( DEFAULT_NROWS, DEFAULT_NCOLS );
  EXPECT_EQ( m.numRows(), DEFAULT_NROWS );
  EXPECT_EQ( m.numColumns(), DEFAULT_NCOLS );
  EXPECT_EQ( m.numNonZeros(), 0 );
  EXPECT_TRUE( m.empty());

  for( int row = 0; row < DEFAULT_NROWS; ++row )
  {
    EXPECT_EQ( m.numNonZeros( row ), 0 );
    EXPECT_EQ( m.nonZeroCapacity( row ), 0 );
    EXPECT_EQ( m.empty( row ), true );

    auto const * const columns = m.getColumns( row ).dataIfContiguous();
    EXPECT_EQ( columns, nullptr );

    auto const * const entries = m.getEntries( row ).dataIfContiguous();
    EXPECT_EQ( entries, nullptr );
  }
}

TYPED_TEST( CRSMatrixTest, hintConstruction )
{
  TypeParam m( DEFAULT_NROWS, DEFAULT_NCOLS, 5 );
  EXPECT_EQ( m.numRows(), DEFAULT_NROWS );
  EXPECT_EQ( m.numColumns(), DEFAULT_NCOLS );
  EXPECT_EQ( m.numNonZeros(), 0 );
  EXPECT_EQ( m.nonZeroCapacity(), DEFAULT_NROWS * 5 );
  EXPECT_TRUE( m.empty());

  for( int row = 0; row < DEFAULT_NROWS; ++row )
  {
    EXPECT_EQ( m.numNonZeros( row ), 0 );
    EXPECT_EQ( m.nonZeroCapacity( row ), 5 );
    EXPECT_EQ( m.empty( row ), true );

    auto const * const columns = m.getColumns( row ).dataIfContiguous();
    ASSERT_NE( columns, nullptr );

    auto const * const entries = m.getEntries( row ).dataIfContiguous();
    ASSERT_NE( entries, nullptr );
  }
}

TYPED_TEST( CRSMatrixTest, resizeFromRowCapacities )
{
  for( int i = 0; i < 2; ++i )
  {
    this->template resizeFromRowCapacities< serialPolicy >( DEFAULT_NROWS, DEFAULT_NCOLS );
    this->insert( DEFAULT_MAX_INSERTS, true );

#if defined( RAJA_ENABLE_OPENMP )
    this->template resizeFromRowCapacities< parallelHostPolicy >( 2 * DEFAULT_NROWS, DEFAULT_NCOLS / 2 );
    this->insert( DEFAULT_MAX_INSERTS, true );
#endif
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

TYPED_TEST( CRSMatrixTest, compress )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );
  this->insert( DEFAULT_MAX_INSERTS );
  this->compress();
}

// Sphinx start after CRSMatrixViewTest
template< typename CRS_MATRIX_POLICY_PAIR >
class CRSMatrixViewTest : public CRSMatrixTest< typename CRS_MATRIX_POLICY_PAIR::first_type >
// Sphinx end before CRSMatrixViewTest
{
public:

  using CRS_MATRIX = typename CRS_MATRIX_POLICY_PAIR::first_type;
  using POLICY = typename CRS_MATRIX_POLICY_PAIR::second_type;

  using ParentClass = CRSMatrixTest< CRS_MATRIX >;

  using typename ParentClass::T;
  using typename ParentClass::ColType;
  using typename ParentClass::IndexType;
  using typename ParentClass::ViewType;
  using typename ParentClass::ViewTypeConstSizes;
  using typename ParentClass::ViewTypeConst;

  template< typename U >
  using ArrayOfArraysT = typename ParentClass::template ArrayOfArraysT< U >;

  template< typename U >
  using ArrayOfArraysViewT = typename ParentClass::template ArrayOfArraysViewT< U >;

  CRSMatrixViewTest():
    ParentClass()
  {};

  /**
   * @brief Test the CRSMatrixView copy constructor in regards to memory motion.
   */
  void memoryMotionTest()
  {
    memoryMotionInit();

    // Check that the columns and entries have been updated.
    IndexType curIndex = 0;
    ViewType const view = m_matrix.toView();
    forall< serialPolicy >( m_matrix.numRows(),
                            [view, &curIndex]( IndexType row )
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

    this->m_matrix.move( MemorySpace::host );
    IndexType curIndex = 0;
    for( IndexType row = 0; row < m_matrix.numRows(); ++row )
    {
      memoryMotionCheckRow( m_matrix.toViewConst(), row, curIndex );
    }
  }

  /**
   * @brief Test the const capture semantics where ColType is const.
   */
  void memoryMotionConstTest()
  {
    IndexType const numRows = m_matrix.numRows();
    IndexType const numCols = m_matrix.numColumns();

    // Capture a view with const columns on device and update the entries.
    ViewTypeConstSizes const view = m_matrix.toViewConstSizes();
    forall< POLICY >( numRows,
                      [view, numCols] LVARRAY_HOST_DEVICE ( IndexType row )
        {
          ColType const * const columns = view.getColumns( row );
          T * const entries = view.getEntries( row );
          for( IndexType i = 0; i < view.numNonZeros( row ); ++i )
          {
            LVARRAY_ERROR_IF( !arrayManipulation::isPositive( columns[ i ] ) || columns[ i ] >= numCols, "Invalid column." );
            entries[ i ] = T( 2 * i + 7 );
          }
        } );

    // This should copy back the entries but not the columns.
    this->m_matrix.move( MemorySpace::host );
    for( IndexType row = 0; row < numRows; ++row )
    {
      ASSERT_EQ( m_matrix.numNonZeros( row ), this->m_ref[row].size());

      auto it = this->m_ref[row].begin();
      for( IndexType i = 0; i < m_matrix.numNonZeros( row ); ++i )
      {
        // So the columns should be the same as the reference.
        ColType const col = m_matrix.getColumns( row )[ i ];
        EXPECT_EQ( col, it->first );
        EXPECT_EQ( m_matrix.getEntries( row )[ i ], T( 2 * i + 7 ));
        ++it;
      }
    }
  }

  /**
   * @brief Test the const capture semantics where both T and ColType are const.
   * @param [in] maxInserts the maximum number of inserts.
   */
  void memoryMotionConstConstTest( IndexType const maxInserts )
  {
    IndexType const numRows = m_matrix.numRows();
    IndexType const numCols = m_matrix.numColumns();

    // Capture a view with const columns and const values on device.
    ViewTypeConst const view = m_matrix.toViewConst();
    forall< POLICY >( numRows,
                      [view, numCols] LVARRAY_HOST_DEVICE ( IndexType row )
        {
          ColType const * const columns = view.getColumns( row );
          T const * const entries = view.getEntries( row );
          for( IndexType i = 0; i < view.numNonZeros( row ); ++i )
          {
            LVARRAY_ERROR_IF( !arrayManipulation::isPositive( columns[ i ] ) || columns[ i ] >= numCols, "Invalid column." );
            LVARRAY_ERROR_IF( entries[ i ] != T( columns[ i ] ), "Incorrect value." );
          }
        }
                      );

    // Insert entries, this will modify the offsets, sizes, columns and entries.
    this->insert( maxInserts );

    // Move the matrix back to the host, this should copy nothing.
    this->m_matrix.move( MemorySpace::host );

    // And therefore the matrix should still equal the reference.
    COMPARE_TO_REFERENCE
  }

  void setValues()
  {
    T const val = T( this->rand( 100 ) );
    this->m_matrix.template setValues< POLICY >( val );

    for( auto & row : this->m_ref )
    {
      for( auto & kvPair : row )
      {
        kvPair.second = val;
      }
    }

    this->m_matrix.move( MemorySpace::host );
    COMPARE_TO_REFERENCE
  }

  void zero()
  {
    // Kind of a hack, this shouldn't be tested with TestString.
    if( std::is_same< T, TestString >::value )
    { return; }

    m_matrix.move( RAJAHelper< POLICY >::space );
    this->m_matrix.zero();

    for( auto & row : this->m_ref )
    {
      for( auto & kvPair : row )
      {
        kvPair.second = T{};
      }
    }

    this->m_matrix.move( MemorySpace::host );
    COMPARE_TO_REFERENCE
  }

  /**
   * @brief Test the CRSMatrixView insert methods.
   * @param [in] maxInserts the maximum number of inserts.
   * @param [in] oneAtATime If true the entries are inserted one at a time.
   */
  void insertIntoView( IndexType const maxInserts, bool const oneAtATime=false )
  {
    IndexType const numRows = this->m_matrix.numRows();

    // Create an Array of the columns and values to insert into each row.
    ArrayOfArraysT< ColType > const columnsToInsert = this->createArrayOfColumns( maxInserts, oneAtATime );
    ArrayOfArraysT< T > const valuesToInsert = this->createArrayOfValues( columnsToInsert.toViewConst() );

    // Reserve space for the inserts
    for( IndexType row = 0; row < numRows; ++row )
    {
      this->m_matrix.reserveNonZeros( row, this->m_matrix.numNonZeros( row ) + columnsToInsert.sizeOfArray( row ) );
    }

    // Insert the entries into the reference.
    this->insertIntoRef( columnsToInsert.toViewConst() );

    ViewType const view = this->m_matrix.toView();
    ArrayOfArraysViewT< ColType const > const columnsView = columnsToInsert.toViewConst();
    ArrayOfArraysViewT< T const > const valuesView = valuesToInsert.toViewConst();
    if( oneAtATime )
    {
      forall< POLICY >( numRows,
                        [view, columnsView] LVARRAY_HOST_DEVICE ( IndexType const row )
          {
            IndexType const initialNNZ = view.numNonZeros( row );
            IndexType nInserted = 0;
            for( IndexType i = 0; i < columnsView.sizeOfArray( row ); ++i )
            {
              ColType const col = columnsView( row, i );
              nInserted += view.insertNonZero( row, col, T( col ) );
            }

            PORTABLE_EXPECT_EQ( view.numNonZeros( row ) - initialNNZ, nInserted );
          }
                        );
    }
    else
    {
      forall< POLICY >( numRows,
                        [view, columnsView, valuesView] LVARRAY_HOST_DEVICE ( IndexType const row )
          {
            IndexType const initialNNZ = view.numNonZeros( row );
            IndexType const nInserted = view.insertNonZeros( row,
                                                             columnsView[row],
                                                             valuesView[row],
                                                             columnsView.sizeOfArray( row ) );
            PORTABLE_EXPECT_EQ( view.numNonZeros( row ) - initialNNZ, nInserted );
          }
                        );
    }

    this->m_matrix.move( MemorySpace::host );
    COMPARE_TO_REFERENCE
  }

  /**
   * @brief Test the CRSMatrixView remove method.
   * @param [in] maxRemoves the maximum number of removes.
   * @param [in] oneAtATime If true the columns are removed one at a time.
   */
  void removeFromView( IndexType const maxRemoves, bool const oneAtATime=false )
  {
    IndexType const numRows = m_matrix.numRows();

    // Create an Array of the columns and values to insert into each row.
    ArrayOfArraysT< ColType > const columnsToRemove = this->createArrayOfColumns( maxRemoves, oneAtATime );

    // Insert the entries into the reference.
    for( IndexType i = 0; i < columnsToRemove.size(); ++i )
    {
      for( IndexType j = 0; j < columnsToRemove.sizeOfArray( i ); ++j )
      {
        ColType const col = columnsToRemove( i, j );
        this->m_ref[ i ].erase( col );
      }
    }

    ViewType const view = m_matrix.toView();
    ArrayOfArraysViewT< ColType const > const columnsView = columnsToRemove.toViewConst();
    if( oneAtATime )
    {
      forall< POLICY >( numRows,
                        [view, columnsView] LVARRAY_HOST_DEVICE ( IndexType row )
          {
            IndexType const initialNNZ = view.numNonZeros( row );
            IndexType nRemoved = 0;
            for( IndexType i = 0; i < columnsView.sizeOfArray( row ); ++i )
            {
              nRemoved += view.removeNonZero( row, columnsView( row, i ) );
            }

            PORTABLE_EXPECT_EQ( initialNNZ - view.numNonZeros( row ), nRemoved );
          } );
    }
    else
    {
      forall< POLICY >( numRows,
                        [view, columnsView] LVARRAY_HOST_DEVICE ( IndexType const row )
          {
            IndexType const initialNNZ = view.numNonZeros( row );
            IndexType const nRemoved = view.removeNonZeros( row, columnsView[row], columnsView.sizeOfArray( row ) );
            PORTABLE_EXPECT_EQ( initialNNZ - view.numNonZeros( row ), nRemoved );
          } );
    }

    this->m_matrix.move( MemorySpace::host );
    COMPARE_TO_REFERENCE
  }

  /**
   * @brief Test the empty method of the CRSMatrixView on device.
   */
  void emptyTest() const
  {
    IndexType const numRows = m_matrix.numRows();

    // Initialize each row to only contain even numbered columns.
    for( IndexType row = 0; row < numRows; ++row )
    {
      ColType const * const columns = m_matrix.getColumns( row );
      ColType * const columnsNC = const_cast< ColType * >(columns);
      for( IndexType i = 0; i < m_matrix.numNonZeros( row ); ++i )
      {
        columnsNC[ i ] = ColType( 2 * i );
      }
    }

    // Check that each row contains the even columns and no odd columns.
    ViewTypeConst const view = m_matrix.toViewConst();
    forall< POLICY >( numRows,
                      [view] LVARRAY_HOST_DEVICE ( IndexType const row )
        {
          for( IndexType i = 0; i < view.numNonZeros( row ); ++i )
          {
            PORTABLE_EXPECT_EQ( view.empty( row, ColType( 2 * i ) ), false );
            PORTABLE_EXPECT_EQ( view.empty( row, ColType( 2 * i + 1 ) ), true );
          }
        } );
  }

  /**
   * @brief Test the toSparsityPatternView method of the CRSMatrixView.
   */
  void sparsityPatternViewTest() const
  {
    IndexType const numRows = m_matrix.numRows();

    auto const & spView = m_matrix.toSparsityPatternView();

    // Check that each row contains the even columns and no odd columns on device.
    ViewTypeConst const view = m_matrix.toViewConst();
    forall< POLICY >( numRows,
                      [spView, view] LVARRAY_HOST_DEVICE ( IndexType const row )
        {
          IndexType const nnz = view.numNonZeros( row );
          PORTABLE_EXPECT_EQ( nnz, spView.numNonZeros( row ) );

          ColType const * const columns = view.getColumns( row );
          ColType const * const spColumns = spView.getColumns( row );
          PORTABLE_EXPECT_EQ( columns, spColumns );

          for( IndexType i = 0; i < nnz; ++i )
          {
            PORTABLE_EXPECT_EQ( columns[ i ], spColumns[ i ] );
            PORTABLE_EXPECT_EQ( view.empty( row, columns[ i ] ), spView.empty( row, columns[ i ] ) );
            PORTABLE_EXPECT_EQ( view.empty( row, ColType( i ) ), spView.empty( row, ColType( i ) ) );
          }
        } );
  }

  // This should be private but you can't have device lambdas need to be in public methods :(
  void memoryMotionInit() const
  {
    IndexType const numRows = m_matrix.numRows();

    // Set the columns and entries. The const casts here isn't necessary, we could remove and insert
    // into the view instead, but this is quicker.
    IndexType curIndex = 0;
    for( IndexType row = 0; row < numRows; ++row )
    {
      ColType const * const columns = m_matrix.getColumns( row );
      ColType * const columnsNC = const_cast< ColType * >(columns);
      T * const entries = m_matrix.getEntries( row );
      for( IndexType i = 0; i < m_matrix.numNonZeros( row ); ++i )
      {
        columnsNC[ i ] = curIndex;
        entries[ i ] = T( curIndex++ );
      }
    }

    // Capture the view on device and set the entries. Here the const cast is necessary
    // because we don't want to test the insert/remove methods on device yet.
    ViewType const view = m_matrix.toView();
    forall< POLICY >( numRows,
                      [view] LVARRAY_HOST_DEVICE ( IndexType const row )
        {
          ColType const * const columns = view.getColumns( row );
          ColType * const columnsNC = const_cast< ColType * >(columns);
          T * const entries = view.getEntries( row );
          for( IndexType i = 0; i < view.numNonZeros( row ); ++i )
          {
            columnsNC[ i ] += columns[ i ];
            entries[ i ] += entries[ i ];
          }
        }
                      );
  }

protected:

  static void memoryMotionCheckRow( ViewTypeConst const & view,
                                    IndexType const row, IndexType & curIndex )
  {
    for( IndexType i = 0; i < view.numNonZeros( row ); ++i )
    {
      EXPECT_EQ( ColType( curIndex ) * 2, view.getColumns( row )[ i ] );
      T expectedEntry( curIndex );
      expectedEntry += T( curIndex );
      EXPECT_EQ( expectedEntry, view.getEntries( row )[ i ] );
      ++curIndex;
    }
  }

  using ParentClass::m_matrix;
};

// Sphinx start after CRSMatrixViewTestTypes
using CRSMatrixViewTestTypes = ::testing::Types<
  std::pair< CRSMatrix< int, int, std::ptrdiff_t, MallocBuffer >, serialPolicy >
// Sphinx end before CRSMatrixViewTestTypes
  , std::pair< CRSMatrix< Tensor, int, std::ptrdiff_t, MallocBuffer >, serialPolicy >
  , std::pair< CRSMatrix< TestString, int, std::ptrdiff_t, MallocBuffer >, serialPolicy >
#if defined(LVARRAY_USE_CHAI)
  , std::pair< CRSMatrix< int, int, std::ptrdiff_t, ChaiBuffer >, serialPolicy >
  , std::pair< CRSMatrix< Tensor, int, std::ptrdiff_t, ChaiBuffer >, serialPolicy >
  , std::pair< CRSMatrix< TestString, int, std::ptrdiff_t, ChaiBuffer >, serialPolicy >
#endif

#if ( defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP) ) && defined(LVARRAY_USE_CHAI)
  , std::pair< CRSMatrix< int, int, std::ptrdiff_t, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< CRSMatrix< Tensor, int, std::ptrdiff_t, ChaiBuffer >, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( CRSMatrixViewTest, CRSMatrixViewTestTypes, );

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

TYPED_TEST( CRSMatrixViewTest, setValues )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );
  this->insert( DEFAULT_MAX_INSERTS );
  this->setValues();
}

TYPED_TEST( CRSMatrixViewTest, zero )
{
  this->resize( DEFAULT_NROWS, DEFAULT_NCOLS );
  this->insert( DEFAULT_MAX_INSERTS );
  this->zero();
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

template< typename CRS_MATRIX_POLICY_PAIR >
class CRSMatrixViewAtomicTest : public CRSMatrixViewTest< CRS_MATRIX_POLICY_PAIR >
{
public:
  using CRS_MATRIX = typename CRS_MATRIX_POLICY_PAIR::first_type;
  using POLICY = typename CRS_MATRIX_POLICY_PAIR::second_type;
  using AtomicPolicy = typename RAJAHelper< POLICY >::AtomicPolicy;

  using ParentClass = CRSMatrixViewTest< CRS_MATRIX_POLICY_PAIR >;

  using typename ParentClass::T;
  using typename ParentClass::IndexType;
  using typename ParentClass::ColType;
  using typename ParentClass::ViewType;
  using typename ParentClass::ViewTypeConstSizes;
  using typename ParentClass::ViewTypeConst;

  template< typename U >
  using ArrayOfArraysT = typename ParentClass::template ArrayOfArraysT< U >;

  template< typename U >
  using ArrayOfArraysViewT = typename ParentClass::template ArrayOfArraysViewT< U >;

  void addToRow( AddType const type, IndexType const nThreads )
  {
    IndexType const numRows = m_matrix.numRows();

    for( IndexType row = 0; row < numRows; ++row )
    {
      IndexType const nColsInRow = m_matrix.numNonZeros( row );

      // Array that holds the columns each thread will add to in the current row.
      ArrayOfArraysT< ColType > columns;
      for( IndexType threadID = 0; threadID < nThreads; ++threadID )
      {
        IndexType const nColsToAddTo = this->rand( nColsInRow );
        columns.appendArray( nColsToAddTo );

        for( IndexType i = 0; i < nColsToAddTo; ++i )
        {
          IndexType const pos = this->rand( nColsInRow - 1 );
          columns( threadID, i ) = m_matrix.getColumns( row )[ pos ];
        }
      }

      if( type != AddType::UNSORTED_BINARY )
      {
        for( IndexType threadID = 0; threadID < nThreads; ++threadID )
        {
          IndexType const numUniqueCols =
            sortedArrayManipulation::makeSortedUnique( columns[ threadID ].begin(), columns[ threadID ].end() );
          columns.resizeArray( threadID, numUniqueCols );
        }
      }

      // Array that holds the values each thread will add to in the current row.
      ArrayOfArraysT< T > values;
      for( IndexType threadID = 0; threadID < nThreads; ++threadID )
      {
        IndexType const nColsToAddTo = columns.sizeOfArray( threadID );
        values.appendArray( nColsToAddTo );

        for( IndexType i = 0; i < nColsToAddTo; ++i )
        {
          ColType const column = columns( threadID, i );
          values( threadID, i ) = T( columns( threadID, i ) );
          this->m_ref[ row ][ column ] += values( threadID, i );
        }
      }

      ViewTypeConstSizes const view = m_matrix.toViewConstSizes();
      ArrayOfArraysViewT< ColType const > colView = columns.toViewConst();
      ArrayOfArraysViewT< T const > valView = values.toViewConst();

      forall< POLICY >( nThreads, [type, row, view, colView, valView] LVARRAY_HOST_DEVICE ( IndexType const threadID )
          {
            if( type == AddType::LINEAR_SEARCH )
            {
              view.template addToRowLinearSearch< AtomicPolicy >( row,
                                                                  colView[ threadID ],
                                                                  valView[ threadID ],
                                                                  colView.sizeOfArray( threadID ) );
            }
            if( type == AddType::BINARY_SEARCH )
            {
              view.template addToRowBinarySearch< AtomicPolicy >( row,
                                                                  colView[ threadID ],
                                                                  valView[ threadID ],
                                                                  colView.sizeOfArray( threadID ) );
            }
            if( type == AddType::HYBRID )
            {
              view.template addToRow< AtomicPolicy >( row,
                                                      colView[ threadID ],
                                                      valView[ threadID ],
                                                      colView.sizeOfArray( threadID ) );
            }
            if( type == AddType::UNSORTED_BINARY )
            {
              view.template addToRowBinarySearchUnsorted< AtomicPolicy >( row,
                                                                          colView[ threadID ],
                                                                          valView[ threadID ],
                                                                          colView.sizeOfArray( threadID ) );
            }
          } );
    }

    this->m_matrix.move( MemorySpace::host );
    COMPARE_TO_REFERENCE
  }

protected:
  using CRSMatrixViewTest< CRS_MATRIX_POLICY_PAIR >::m_matrix;
};

using CRSMatrixViewAtomicTestTypes = ::testing::Types<
  std::pair< CRSMatrix< int, int, std::ptrdiff_t, MallocBuffer >, serialPolicy >
  , std::pair< CRSMatrix< Tensor, int, std::ptrdiff_t, MallocBuffer >, serialPolicy >
  , std::pair< CRSMatrix< TestString, int, std::ptrdiff_t, MallocBuffer >, serialPolicy >
#if defined(LVARRAY_USE_CHAI)
  , std::pair< CRSMatrix< int, int, std::ptrdiff_t, ChaiBuffer >, serialPolicy >
  , std::pair< CRSMatrix< Tensor, int, std::ptrdiff_t, ChaiBuffer >, serialPolicy >
  , std::pair< CRSMatrix< TestString, int, std::ptrdiff_t, ChaiBuffer >, serialPolicy >
#endif

#if defined(RAJA_ENABLE_OPENMP)
  , std::pair< CRSMatrix< int, int, std::ptrdiff_t, MallocBuffer >, parallelHostPolicy >
  , std::pair< CRSMatrix< double, int, std::ptrdiff_t, MallocBuffer >, parallelHostPolicy >
#endif
#if defined(RAJA_ENABLE_OPENMP) && defined(LVARRAY_USE_CHAI)
  , std::pair< CRSMatrix< int, int, std::ptrdiff_t, ChaiBuffer >, parallelHostPolicy >
  , std::pair< CRSMatrix< double, int, std::ptrdiff_t, ChaiBuffer >, parallelHostPolicy >
#endif

#if ( defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP) ) && defined(LVARRAY_USE_CHAI)
  , std::pair< CRSMatrix< int, int, std::ptrdiff_t, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< CRSMatrix< double, int, std::ptrdiff_t, ChaiBuffer >, parallelDevicePolicy< 32 > >
#endif
  >;

TYPED_TEST_SUITE( CRSMatrixViewAtomicTest, CRSMatrixViewAtomicTestTypes, );

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
