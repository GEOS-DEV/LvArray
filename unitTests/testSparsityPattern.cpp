/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "SparsityPattern.hpp"
#include "CRSMatrix.hpp"
#include "testUtils.hpp"
#include "Array.hpp"
#include "MallocBuffer.hpp"

// TPL includes
#include <gtest/gtest.h>

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

template< typename U, typename T >
struct ToArray1D
{};

template< typename U, typename T, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
struct ToArray1D< U, SparsityPattern< T, INDEX_TYPE, BUFFER_TYPE > >
{
  using array = Array< U, 1, RAJA::PERM_I, INDEX_TYPE, BUFFER_TYPE >;
  using view = ArrayView< U, 1, 0, INDEX_TYPE, BUFFER_TYPE >;
};

template< typename SPARSITY_PATTERN >
class SparsityPatternTest : public ::testing::Test
{
public:
  using ColType = typename SPARSITY_PATTERN::ColType;
  using IndexType = typename SPARSITY_PATTERN::IndexType;

  using ViewType = typeManipulation::ViewType< SPARSITY_PATTERN >;
  using ViewTypeConst = typeManipulation::ViewTypeConst< SPARSITY_PATTERN >;

  void compareToReference( ViewTypeConst const & v )
  {
    IndexType const numRows = v.numRows();
    ASSERT_EQ( numRows, IndexType( m_ref.size()));

    IndexType ref_nnz = 0;
    for( IndexType row = 0; row < numRows; ++row )
    {
      IndexType const rowNNZ = v.numNonZeros( row );
      IndexType const refRowNNZ = m_ref[row].size();
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
      ColType const * const columns = v.getColumns( row );
      for( IndexType i = 0; i < v.numNonZeros( row ); ++i )
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

  #define COMPARE_TO_REFERENCE { SCOPED_TRACE( "" ); this->compareToReference( m_sp.toViewConst() ); \
  }

  void resize( IndexType const nRows,
               IndexType const nCols,
               IndexType initialRowCapacity=0 )
  {
    m_sp.resize( nRows, nCols, initialRowCapacity );
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

    m_sp.template resizeFromRowCapacities< POLICY >( numRows, numCols, newCapacities.data() );

    EXPECT_EQ( m_sp.numRows(), numRows );
    EXPECT_EQ( m_sp.numColumns(), numCols );
    for( IndexType row = 0; row < m_sp.numRows(); ++row )
    {
      EXPECT_EQ( m_sp.numNonZeros( row ), 0 );
      EXPECT_EQ( m_sp.nonZeroCapacity( row ), newCapacities[ row ] );
    }

    m_ref.clear();
    m_ref.resize( numRows );
    COMPARE_TO_REFERENCE;
  }

  void appendRow( IndexType const nRows, IndexType const maxInserts )
  {
    COMPARE_TO_REFERENCE

    std::vector< ColType > columnsToAppend( maxInserts );

    for( IndexType i = 0; i < nRows; ++i )
    {
      IndexType const nCols = randCol();
      columnsToAppend.resize( nCols );

      for( IndexType j = 0; j < nCols; ++j )
      {
        columnsToAppend[ j ] = randCol();
      }

      m_sp.appendRow( nCols );
      EXPECT_EQ( nCols, m_sp.nonZeroCapacity( m_sp.numRows() - 1 ) );

      for( ColType const & col : columnsToAppend )
      {
        m_sp.insertNonZero( m_sp.numRows() - 1, col );
      }

      m_ref.push_back( std::set< ColType >( columnsToAppend.begin(), columnsToAppend.end() ) );
    }

    COMPARE_TO_REFERENCE
  }

  /**
   * @brief Test the insert method of the SparsityPattern.
   * @param [in] maxInserts the number of times to call insert.
   */
  void insertTest( IndexType const maxInserts )
  {
    IndexType const numRows = m_sp.numRows();
    ASSERT_EQ( numRows, IndexType( m_ref.size()));

    for( IndexType row = 0; row < numRows; ++row )
    {
      std::set< ColType > & refRow = m_ref[row];
      ASSERT_EQ( m_sp.numNonZeros( row ), IndexType( refRow.size()));

      for( IndexType i = 0; i < maxInserts; ++i )
      {
        ColType const col = randCol();
        ASSERT_EQ( refRow.insert( col ).second, m_sp.insertNonZero( row, col ));
      }
    }

    COMPARE_TO_REFERENCE
  }

  /**
   * @brief Test the insertSorted method of the SparsityPattern.
   * @param [in] maxInserts the number of values to insert at a time.
   */
  void insertMultipleTest( IndexType const maxInserts )
  {
    IndexType const numRows = m_sp.numRows();
    ASSERT_EQ( numRows, IndexType( m_ref.size() ) );

    for( IndexType row = 0; row < numRows; ++row )
    {
      IndexType const nCols = rand( maxInserts / 2 );

      // Insert using a std::vector
      {
        std::vector< ColType > columnsToInsert( nCols );

        for( IndexType j = 0; j < nCols; ++j )
        { columnsToInsert[ j ] = randCol(); }

        IndexType const numUnique = sortedArrayManipulation::makeSortedUnique( columnsToInsert.begin(), columnsToInsert.end() );
        columnsToInsert.resize( numUnique );

        IndexType const numInserted = m_sp.insertNonZeros( row, columnsToInsert.begin(), columnsToInsert.end() );
        EXPECT_EQ( numInserted, insertIntoRef( row, columnsToInsert ) );
      }

      /// Insert using a std::set
      {
        std::set< ColType > columnsToInsert;

        for( IndexType j = 0; j < nCols; ++j )
        { columnsToInsert.insert( randCol() ); }

        IndexType const numInserted = m_sp.insertNonZeros( row, columnsToInsert.begin(), columnsToInsert.end() );
        EXPECT_EQ( numInserted, insertIntoRef( row, columnsToInsert ) );
      }
    }

    COMPARE_TO_REFERENCE
  }

  /**
   * @brief Test the remove method of the SparsityPatternView.
   * @param [in] maxRemoves the number of times to call remove.
   */
  void removeTest( IndexType const maxRemoves )
  {
    IndexType const numRows = m_sp.numRows();
    ASSERT_EQ( numRows, IndexType( m_ref.size()));

    for( IndexType row = 0; row < numRows; ++row )
    {
      std::set< ColType > & refRow = m_ref[row];
      ASSERT_EQ( m_sp.numNonZeros( row ), ColType( refRow.size()));

      for( IndexType i = 0; i < maxRemoves; ++i )
      {
        ColType const col = randCol();
        ASSERT_EQ( m_sp.removeNonZero( row, col ), refRow.erase( col ));
      }
    }

    COMPARE_TO_REFERENCE
  }

  /**
   * @brief Test the removeSorted method of the SparsityPatternView.
   * @param [in] maxRemoves the number of values to remove at a time.
   */
  void removeMultipleTest( IndexType const maxRemoves )
  {
    IndexType const numRows = m_sp.numRows();
    ASSERT_EQ( numRows, IndexType( m_ref.size()));

    for( IndexType row = 0; row < numRows; ++row )
    {
      IndexType const nCols = rand( maxRemoves / 2 );

      // Insert using a std::vector
      {
        std::vector< ColType > columnsToRemove( nCols );

        for( IndexType j = 0; j < nCols; ++j )
        { columnsToRemove[ j ] = randCol(); }

        IndexType const numUnique = sortedArrayManipulation::makeSortedUnique( columnsToRemove.begin(), columnsToRemove.end() );
        columnsToRemove.resize( numUnique );

        IndexType const numInserted = m_sp.removeNonZeros( row, columnsToRemove.begin(), columnsToRemove.end() );
        EXPECT_EQ( numInserted, removeFromRef( row, columnsToRemove ) );
      }

      /// Insert using a std::set
      {
        std::set< ColType > columnsToRemove;

        for( IndexType j = 0; j < nCols; ++j )
        { columnsToRemove.insert( randCol() ); }

        IndexType const numInserted = m_sp.removeNonZeros( row, columnsToRemove.begin(), columnsToRemove.end() );
        EXPECT_EQ( numInserted, removeFromRef( row, columnsToRemove ) );
      }
    }

    COMPARE_TO_REFERENCE
  }

  /**
   * @brief Test the empty method of the SparsityPatternView.
   */
  void emptyTest() const
  {
    IndexType const numRows = m_sp.numRows();
    ASSERT_EQ( numRows, m_ref.size());

    for( IndexType row = 0; row < numRows; ++row )
    {
      std::set< ColType > const & refRow = m_ref[row];
      IndexType const nnz = m_sp.numNonZeros( row );
      ASSERT_EQ( nnz, refRow.size());

      auto it = refRow.begin();
      for( IndexType i = 0; i < nnz; ++i )
      {
        ColType const col = *it;
        EXPECT_FALSE( m_sp.empty( row, col ));
        EXPECT_NE( m_sp.empty( row, ColType( i )), refRow.count( ColType( i )));
        ++it;
      }
    }
  }

  /**
   * @brief Fill a row of the SparsityPatternView up to its capacity.
   * @param [in] row the row of the SparsityPatternView to fill.
   */
  void fillRow( IndexType const row )
  {
    IndexType nToInsert = m_sp.nonZeroCapacity( row ) - m_sp.numNonZeros( row );
    ColType testColum = ColType( m_sp.numColumns() - 1 );
    ColType const * const columns = m_sp.getColumns( row );

    ASSERT_GE( m_sp.numColumns(), m_sp.nonZeroCapacity( row ));

    while( nToInsert > 0 )
    {
      bool const success = m_ref[ row ].insert( testColum ).second;
      ASSERT_EQ( success, m_sp.insertNonZero( row, testColum ));
      nToInsert -= success;
      testColum -= 1;
    }

    ColType const * const newColumns = m_sp.getColumns( row );
    ASSERT_EQ( columns, newColumns );
  }

  /**
   * @brief Test the setRowCapacity method of SparsityPattern.
   */
  void rowCapacityTest()
  {
    IndexType const numRows = m_sp.numRows();
    ASSERT_EQ( numRows, IndexType( m_ref.size()));

    for( IndexType row = 0; row < numRows; ++row )
    {
      std::set< ColType > & refRow = m_ref[row];
      IndexType const rowNNZ = m_sp.numNonZeros( row );
      ASSERT_EQ( rowNNZ, ColType( refRow.size()));

      // The new capacity will be the index of the row.
      IndexType const new_capacity = row;
      if( new_capacity < rowNNZ )
      {
        ColType const * const columns = m_sp.getColumns( row );
        m_sp.setRowCapacity( row, new_capacity );

        // Erase the last values from the reference.
        IndexType const difference = rowNNZ - new_capacity;
        auto erase_pos = refRow.end();
        std::advance( erase_pos, -difference );
        refRow.erase( erase_pos, refRow.end());

        // Check that the pointer to the row hasn't changed.
        ColType const * const newColumns = m_sp.getColumns( row );
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

    COMPARE_TO_REFERENCE
  }

  /**
   * @brief Test the compress method of SparsityPattern.
   */
  void compressTest()
  {
    m_sp.compress();

    ColType const * const columns = m_sp.getColumns( 0 );
    IndexType const * const offsets = m_sp.getOffsets();

    IndexType curOffset = 0;
    for( IndexType row = 0; row < m_sp.numRows(); ++row )
    {
      // The last row will have all the extra capacity.
      if( row != m_sp.numRows() - 1 )
      {
        ASSERT_EQ( m_sp.numNonZeros( row ), m_sp.nonZeroCapacity( row ));
      }

      ColType const * const rowColumns = m_sp.getColumns( row );
      ASSERT_EQ( rowColumns, columns + curOffset );
      ASSERT_EQ( offsets[row], curOffset );

      curOffset += m_sp.numNonZeros( row );
    }

    COMPARE_TO_REFERENCE
  }

  /**
   * @brief Test the copy constructor of the SparsityPattern.
   */
  void deepCopyTest()
  {
    SPARSITY_PATTERN copy( m_sp );

    ASSERT_EQ( m_sp.numRows(), copy.numRows());
    ASSERT_EQ( m_sp.numColumns(), copy.numColumns());
    ASSERT_EQ( m_sp.numNonZeros(), copy.numNonZeros());

    IndexType const totalNNZ = m_sp.numNonZeros();

    for( IndexType row = 0; row < m_sp.numRows(); ++row )
    {
      IndexType const nnz = m_sp.numNonZeros( row );
      ASSERT_EQ( nnz, copy.numNonZeros( row ));

      ColType const * const cols = m_sp.getColumns( row );
      ColType const * const cols_cpy = copy.getColumns( row );
      ASSERT_NE( cols, cols_cpy );

      // Iterate backwards and remove entries from copy.
      for( IndexType i = nnz - 1; i >= 0; --i )
      {
        EXPECT_EQ( cols[i], cols_cpy[i] );
        copy.removeNonZero( row, cols_cpy[i] );
      }

      EXPECT_EQ( copy.numNonZeros( row ), 0 );
      EXPECT_EQ( m_sp.numNonZeros( row ), nnz );
    }

    EXPECT_EQ( copy.numNonZeros(), 0 );
    EXPECT_EQ( m_sp.numNonZeros(), totalNNZ );

    COMPARE_TO_REFERENCE
  }

  /**
   * @brief Test the copy constructor of the SparsityPatternView.
   */
  void shallowCopyTest() const
  {
    ViewType copy( m_sp.toView() );

    ASSERT_EQ( m_sp.numRows(), copy.numRows());
    ASSERT_EQ( m_sp.numColumns(), copy.numColumns());
    ASSERT_EQ( m_sp.numNonZeros(), copy.numNonZeros());

    for( IndexType row = 0; row < m_sp.numRows(); ++row )
    {
      IndexType const nnz = m_sp.numNonZeros( row );
      ASSERT_EQ( nnz, copy.numNonZeros( row ));

      ColType const * const cols = m_sp.getColumns( row );
      ColType const * const cols_cpy = copy.getColumns( row );
      ASSERT_EQ( cols, cols_cpy );

      // Iterate backwards and remove entries from copy.
      for( IndexType i = nnz - 1; i >= 0; --i )
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
  IndexType insertIntoRef( IndexType const row, CONTAINER const & columns )
  {
    IndexType numInserted = 0;
    for( ColType const & col : columns )
    { numInserted += m_ref[ row ].insert( col ).second; }

    return numInserted;
  }

  template< typename CONTAINER >
  IndexType removeFromRef( IndexType const row, CONTAINER const & columns )
  {
    IndexType numRemoved = 0;
    for( ColType const & col : columns )
    { numRemoved += m_ref[ row ].erase( col ); }

    return numRemoved;
  }

  IndexType rand( IndexType const max )
  { return std::uniform_int_distribution< IndexType >( 0, max )( m_gen ); }

  ColType randCol()
  { return rand( m_sp.numColumns() - 1 ); }

  std::mt19937_64 m_gen;

  SPARSITY_PATTERN m_sp;
  std::vector< std::set< ColType > > m_ref;
};

// When using the XL compiler a couple tests will fail. Built individually they all pass.
// When the SparsityPatternViewTests aren't built all of these will pass. I have no idea what
// is going on.
using SparsityPatternTestTypes = ::testing::Types<
  SparsityPattern< int, std::ptrdiff_t, MallocBuffer >
#if !defined( __ibmxl__ )
  , SparsityPattern< uint, std::ptrdiff_t, MallocBuffer >
#endif

#if defined(LVARRAY_USE_CHAI)
  , SparsityPattern< int, std::ptrdiff_t, ChaiBuffer >
#if !defined( __ibmxl__ )
  , SparsityPattern< uint, std::ptrdiff_t, ChaiBuffer >
#endif
#endif
  >;
TYPED_TEST_SUITE( SparsityPatternTest, SparsityPatternTestTypes, );

std::ptrdiff_t const NROWS = 100;
std::ptrdiff_t const NCOLS = 150;
std::ptrdiff_t const MAX_INSERTS = 75;

TYPED_TEST( SparsityPatternTest, constructionNoHint )
{
  TypeParam sp( NROWS, NCOLS );
  EXPECT_EQ( sp.numRows(), NROWS );
  EXPECT_EQ( sp.numColumns(), NCOLS );
  EXPECT_EQ( sp.numNonZeros(), 0 );
  EXPECT_TRUE( sp.empty());

  for( std::ptrdiff_t row = 0; row < NROWS; ++row )
  {
    EXPECT_EQ( sp.numNonZeros( row ), 0 );
    EXPECT_EQ( sp.nonZeroCapacity( row ), 0 );
    EXPECT_EQ( sp.empty( row ), true );
    auto const * const columns = sp.getColumns( row ).dataIfContiguous();
    EXPECT_EQ( columns, nullptr );
  }
}

TYPED_TEST( SparsityPatternTest, constructionWithHint )
{
  constexpr int SIZE_HINT = 5;

  TypeParam sp( NROWS, NCOLS, SIZE_HINT );
  EXPECT_EQ( sp.numRows(), NROWS );
  EXPECT_EQ( sp.numColumns(), NCOLS );
  EXPECT_EQ( sp.numNonZeros(), 0 );
  EXPECT_EQ( sp.nonZeroCapacity(), NROWS * SIZE_HINT );
  EXPECT_TRUE( sp.empty());

  for( std::ptrdiff_t row = 0; row < NROWS; ++row )
  {
    EXPECT_EQ( sp.numNonZeros( row ), 0 );
    EXPECT_EQ( sp.nonZeroCapacity( row ), SIZE_HINT );
    EXPECT_EQ( sp.empty( row ), true );

    auto const * const columns = sp.getColumns( row ).dataIfContiguous();
    ASSERT_NE( columns, nullptr );
  }
}

TYPED_TEST( SparsityPatternTest, resizeFromRowCapacities )
{
  for( int i = 0; i < 2; ++i )
  {
    this->template resizeFromRowCapacities< serialPolicy >( 100, 75 );
    this->insertTest( MAX_INSERTS );

#if defined( RAJA_ENABLE_OPENMP )
    this->template resizeFromRowCapacities< parallelHostPolicy >( 150, 200 );
    this->insertTest( MAX_INSERTS );
#endif
  }
}

TYPED_TEST( SparsityPatternTest, appendRow )
{
  this->resize( 0, NCOLS );
  this->appendRow( NROWS, MAX_INSERTS );
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

  for( std::ptrdiff_t row = 0; row < NROWS; ++row )
  {
    ASSERT_EQ( this->m_sp.nonZeroCapacity( row ), 0 );
    ASSERT_EQ( this->m_sp.numNonZeros( row ), 0 );
  }

  auto const * const columns = this->m_sp.getColumns( 0 ).dataIfContiguous();
  this->insertTest( MAX_INSERTS );
  ASSERT_EQ( &this->m_sp.getColumns( 0 )[ 0 ], columns );
}

TYPED_TEST( SparsityPatternTest, rowCapacity )
{
  this->resize( NROWS, NCOLS, MAX_INSERTS );

  auto const * const columns = this->m_sp.getColumns( 0 ).dataIfContiguous();
  this->insertTest( MAX_INSERTS );
  ASSERT_EQ( &this->m_sp.getColumns( 0 )[ 0 ], columns );

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

template< typename SPARSITY_PATTERN_POLICY_PAIR >
class SparsityPatternViewTest : public SparsityPatternTest< typename SPARSITY_PATTERN_POLICY_PAIR::first_type >
{
public:
  using SPARSITY_PATTERN = typename SPARSITY_PATTERN_POLICY_PAIR::first_type;
  using POLICY = typename SPARSITY_PATTERN_POLICY_PAIR::second_type;

  using ParentClass = SparsityPatternTest< SPARSITY_PATTERN >;

  using typename ParentClass::ColType;
  using typename ParentClass::IndexType;

  using typename ParentClass::ViewType;
  using typename ParentClass::ViewTypeConst;

  template< typename T >
  using Array1D = typename ToArray1D< T, SPARSITY_PATTERN >::array;

  template< typename T >
  using ArrayView1D = typename ToArray1D< T, SPARSITY_PATTERN >::view;

  /**
   * @brief Test the SparsityPatternView copy constructor in regards to memory motion.
   */
  void memoryMotionTest() const
  {
    IndexType const numRows = m_sp.numRows();

    // Set the columns. The const cast here isn't necessary, we could remove and insert
    // into the SparsityPattern instead, but this is quicker.
    IndexType curIndex = 0;
    for( IndexType row = 0; row < numRows; ++row )
    {
      ColType const * const columns = m_sp.getColumns( row );
      ColType * const columnsNC = const_cast< ColType * >(columns);
      for( IndexType i = 0; i < m_sp.numNonZeros( row ); ++i )
      {
        columnsNC[i] = ColType( curIndex++ );
      }
    }

    // Capture the view on device and set the values. Here the const cast is necessary
    // because we don't want to test the insert/remove methods on device yet.
    ViewType const view = m_sp.toView();
    forall< POLICY >( numRows,
                      [view] LVARRAY_HOST_DEVICE ( IndexType row )
        {
          ColType const * const columns = view.getColumns( row );
          ColType * const columnsNC = const_cast< ColType * >(columns);
          for( IndexType i = 0; i < view.numNonZeros( row ); ++i )
          {
            columnsNC[i] *= columns[i];
          }
        } );

    // Check that the columns have been updated.
    curIndex = 0;
    forall< serialPolicy >( numRows,
                            [view, &curIndex]( IndexType row )
    {
      for( IndexType i = 0; i < view.numNonZeros( row ); ++i )
      {
        ColType val = ColType( curIndex ) * ColType( curIndex );
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
    IndexType const numRows = m_sp.numRows();

    ViewType const view = m_sp.toView();

    IndexType curIndex = 0;
    for( IndexType row = 0; row < numRows; ++row )
    {
      ColType const * const columns = view.getColumns( row );
      ColType * const columnsNC = const_cast< ColType * >(columns);
      for( IndexType i = 0; i < view.numNonZeros( row ); ++i )
      {
        columnsNC[i] = ColType( curIndex++ );
      }
    }

    forall< POLICY >( numRows,
                      [=] LVARRAY_HOST_DEVICE ( IndexType row )
        {
          ColType const * const columns = view.getColumns( row );
          ColType * const columnsNC = const_cast< ColType * >(columns);
          for( IndexType i = 0; i < view.numNonZeros( row ); ++i )
          {
            columnsNC[i] *= columns[i];
          }
        } );

    m_sp.move( MemorySpace::host );
    curIndex = 0;
    for( IndexType row = 0; row < numRows; ++row )
    {
      for( IndexType i = 0; i < m_sp.numNonZeros( row ); ++i )
      {
        ColType val = ColType( curIndex ) * ColType( curIndex );
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
    IndexType const numRows = m_sp.numRows();
    IndexType const numCols = m_sp.numColumns();

    // Create a view const and capture it on device.
    ViewTypeConst const constView = m_sp.toViewConst();
    forall< POLICY >( numRows,
                      [constView, numCols] LVARRAY_HOST_DEVICE ( IndexType row )
        {
          ColType const * const columns = constView.getColumns( row );
          for( IndexType i = 0; i < constView.numNonZeros( row ); ++i )
          {
            PORTABLE_EXPECT_EQ( arrayManipulation::isPositive( columns[ i ] ) && columns[ i ] < numCols, true );
          }
        } );

    // Fill in the rows on host.
    for( IndexType row = 0; row < numRows; ++row )
    {
      this->fillRow( row );
    }

    // Move the SparsityPattern back from device, this should be a no-op.
    m_sp.move( MemorySpace::host );
    COMPARE_TO_REFERENCE
  }

  void insertViewTest()
  {
    COMPARE_TO_REFERENCE

    Array1D< Array1D< ColType > > toInsert = createColumns( true, false );
    ArrayView1D< ArrayView1D< ColType const > const > const toInsertView = toInsert.toNestedViewConst();

    ViewType const view = m_sp.toView();
    forall< POLICY >( m_sp.numRows(), [view, toInsertView] LVARRAY_HOST_DEVICE ( IndexType const row )
        {
          for( IndexType j = 0; j < toInsertView[ row ].size(); ++j )
          {
            view.insertNonZero( row, toInsertView[ row ][ j ] );
          }
        } );

    m_sp.move( MemorySpace::host );
    COMPARE_TO_REFERENCE
  }

  void insertMultipleViewTest()
  {
    COMPARE_TO_REFERENCE

    Array1D< Array1D< ColType > > const toInsert = createColumns( true, true );
    ArrayView1D< ArrayView1D< ColType const > const > const toInsertView = toInsert.toNestedViewConst();

    ViewType const view = m_sp.toView();
    forall< POLICY >( m_sp.numRows(), [view, toInsertView] LVARRAY_HOST_DEVICE ( IndexType const row )
        {
          view.insertNonZeros( row, toInsertView[ row ].begin(), toInsertView[ row ].end() );
        } );

    m_sp.move( MemorySpace::host );
    COMPARE_TO_REFERENCE
  }

  void removeViewTest()
  {
    COMPARE_TO_REFERENCE

    Array1D< Array1D< ColType > > const toRemove = createColumns( false, false );
    ArrayView1D< ArrayView1D< ColType const > const > const toRemoveView = toRemove.toNestedViewConst();

    ViewType const view = m_sp.toView();
    forall< POLICY >( m_sp.numRows(), [view, toRemoveView] LVARRAY_HOST_DEVICE ( IndexType const row )
        {
          for( IndexType j = 0; j < toRemoveView[ row ].size(); ++j )
          {
            view.removeNonZero( row, toRemoveView[ row ][ j ] );
          }
        } );

    m_sp.move( MemorySpace::host );
    COMPARE_TO_REFERENCE
  }

  void removeMultipleViewTest()
  {
    COMPARE_TO_REFERENCE

    Array1D< Array1D< ColType > > const toRemove = createColumns( false, true );
    ArrayView1D< ArrayView1D< ColType const > const > const toRemoveView = toRemove.toNestedViewConst();

    ViewType const view = m_sp.toView();
    forall< POLICY >( m_sp.numRows(), [view, toRemoveView] LVARRAY_HOST_DEVICE ( IndexType const row )
        {
          view.removeNonZeros( row, toRemoveView[ row ].begin(), toRemoveView[ row ].end() );
        } );

    m_sp.move( MemorySpace::host );
    COMPARE_TO_REFERENCE
  }

  /**
   * @brief Test the empty method of the SparsityPatternView on device.
   * @param [in] v the SparsityPatternView to test.
   */
  void emptyViewTest() const
  {
    IndexType const numRows = m_sp.numRows();

    // Initialize each row to only contain even numbered columns.
    for( IndexType row = 0; row < numRows; ++row )
    {
      ColType const * const columns = m_sp.getColumns( row );
      ColType * const columnsNC = const_cast< ColType * >(columns);
      for( IndexType i = 0; i < m_sp.numNonZeros( row ); ++i )
      {
        columnsNC[i] = ColType( 2 * i );
      }
    }

    // Check that each row contains the even columns and no odd columns on device.
    ViewTypeConst const view = m_sp.toViewConst();
    forall< POLICY >( numRows,
                      [view] LVARRAY_HOST_DEVICE ( IndexType row )
        {
          for( IndexType i = 0; i < view.numNonZeros( row ); ++i )
          {
            PORTABLE_EXPECT_EQ( view.empty( row, ColType( 2 * i ) ), false );
            PORTABLE_EXPECT_EQ( view.empty( row, ColType( 2 * i + 1 ) ), true );
          }
        } );
  }

protected:

  Array1D< Array1D< ColType > > createColumns( bool const insert, bool const sortedUnique )
  {
    IndexType const numRows = m_sp.numRows();

    Array1D< Array1D< ColType > > columns( numRows );

    for( IndexType i = 0; i < numRows; ++i )
    {
      IndexType const nnz = m_sp.numNonZeros( i );
      IndexType const capacity = m_sp.nonZeroCapacity( i );

      IndexType const upperBound = insert ? capacity - nnz : capacity;
      IndexType const nCols = rand( upperBound );

      columns[ i ].resize( nCols );

      for( IndexType j = 0; j < nCols; ++j )
      {
        ColType const column = randCol();
        columns[ i ][ j ] = column;

        if( insert )
        { m_ref[ i ].insert( column ); }
        else
        { m_ref[ i ].erase( column ); }
      }

      if( sortedUnique )
      {
        IndexType const numUnique = sortedArrayManipulation::makeSortedUnique( columns[ i ].begin(), columns[ i ].end() );
        columns[ i ].resize( numUnique );
      }
    }

    return columns;
  }

  using ParentClass::m_sp;
  using ParentClass::m_ref;
  using ParentClass::rand;
  using ParentClass::randCol;
};

using SparsityPatternViewTestTypes = ::testing::Types<
  std::pair< SparsityPattern< int, std::ptrdiff_t, MallocBuffer >, serialPolicy >
#if !defined( __ibmxl__ )
  , std::pair< SparsityPattern< uint, std::ptrdiff_t, MallocBuffer >, serialPolicy >
#endif
#if defined(LVARRAY_USE_CHAI)
  , std::pair< SparsityPattern< int, std::ptrdiff_t, ChaiBuffer >, serialPolicy >
#if !defined( __ibmxl__ )
  , std::pair< SparsityPattern< uint, std::ptrdiff_t, ChaiBuffer >, serialPolicy >
#endif
#endif

#if ( defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP) ) && defined(LVARRAY_USE_CHAI)
  , std::pair< SparsityPattern< int, std::ptrdiff_t, ChaiBuffer >, parallelDevicePolicy< 32 > >
#if !defined( __ibmxl__ )
  , std::pair< SparsityPattern< uint, std::ptrdiff_t, ChaiBuffer >, parallelDevicePolicy< 32 > >
#endif
#endif
  >;

TYPED_TEST_SUITE( SparsityPatternViewTest, SparsityPatternViewTestTypes, );

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

template< typename >
struct CRSMatrixToSparsityPattern;

template< typename T,
          typename ColType,
          typename IndexType,
          template< typename > class BUFFER_TYPE >
struct CRSMatrixToSparsityPattern< CRSMatrix< T, ColType, IndexType, BUFFER_TYPE > >
{
  using type = SparsityPattern< ColType, IndexType, BUFFER_TYPE >;
};

template< typename CRS_MATRIX_POLICY_PAIR >
class CRSMatrixTest : public SparsityPatternTest< typename CRSMatrixToSparsityPattern< typename CRS_MATRIX_POLICY_PAIR::first_type >::type >
{
public:
  using CRS_MATRIX = typename CRS_MATRIX_POLICY_PAIR::first_type;
  using POLICY = typename CRS_MATRIX_POLICY_PAIR::second_type;
  using ParentClass = SparsityPatternTest< typename CRSMatrixToSparsityPattern< CRS_MATRIX >::type >;

  using T = typename CRS_MATRIX::EntryType;
  using typename ParentClass::ColType;
  using typename ParentClass::IndexType;

  void assimilate()
  {
    CRS_MATRIX matrix;
    matrix.template assimilate< POLICY >( std::move( this->m_sp ) );
    matrix.move( MemorySpace::host );
    this->compareToReference( matrix.toSparsityPatternView() );

    EXPECT_EQ( this->m_sp.numRows(), 0 );
    EXPECT_EQ( this->m_sp.numColumns(), 0 );
    EXPECT_EQ( this->m_sp.numNonZeros(), 0 );

    IndexType const numRows = matrix.numRows();
    ASSERT_EQ( numRows, IndexType( this->m_ref.size() ) );

    IndexType ref_nnz = 0;
    for( IndexType row = 0; row < numRows; ++row )
    {
      IndexType const rowNNZ = matrix.numNonZeros( row );
      IndexType const refRowNNZ = this->m_ref[row].size();
      ref_nnz += refRowNNZ;

      ASSERT_EQ( rowNNZ, refRowNNZ );

      if( rowNNZ == 0 )
      {
        ASSERT_TRUE( matrix.empty( row ) );
      }
      else
      {
        ASSERT_FALSE( matrix.empty( row ) );
      }

      auto it = this->m_ref[ row ].begin();
      ColType const * const columns = matrix.getColumns( row );
      T const * const entries = matrix.getEntries( row );
      for( IndexType i = 0; i < matrix.numNonZeros( row ); ++i )
      {
        EXPECT_FALSE( matrix.empty( row, columns[ i ] ));
        EXPECT_EQ( columns[ i ], *it );
        EXPECT_EQ( entries[ i ], T() );
        it++;
      }
    }
  }
};

using CRSMatrixTestTypes = ::testing::Types<
  std::pair< CRSMatrix< int, int, std::ptrdiff_t, MallocBuffer >, serialPolicy >
  , std::pair< CRSMatrix< Tensor, int, std::ptrdiff_t, MallocBuffer >, serialPolicy >
  , std::pair< CRSMatrix< TestString, int, std::ptrdiff_t, MallocBuffer >, serialPolicy >
#if ( defined(LVARRAY_USE_CUDA) || defined(LVARRAY_USE_HIP) ) && defined(LVARRAY_USE_CHAI)
  , std::pair< CRSMatrix< int, int, std::ptrdiff_t, ChaiBuffer >, parallelDevicePolicy< 32 > >
  , std::pair< CRSMatrix< Tensor, int, std::ptrdiff_t, ChaiBuffer >, parallelDevicePolicy< 32 > >
#endif
  >;
TYPED_TEST_SUITE( CRSMatrixTest, CRSMatrixTestTypes, );

TYPED_TEST( CRSMatrixTest, assimilate )
{
  this->resize( NROWS, NCOLS );
  for( int i = 0; i < 2; ++i )
  {
    this->insertMultipleTest( MAX_INSERTS );
  }
  this->assimilate();
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
