/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file CRSMatrix.hpp
 * @brief Contains the implementation of LvArray::CRSMatrix.
 */

#pragma once

#include "CRSMatrixView.hpp"
#include "arrayManipulation.hpp"

namespace LvArray
{

template< typename COL_TYPE, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
class SparsityPattern;

/**
 * @tparam T the type of the entries in the matrix.
 * @tparam COL_TYPE the integer used to enumerate the columns.
 * @tparam INDEX_TYPE the integer to use for indexing.
 * @class CRSMatrix
 * @brief This class implements a compressed row storage matrix.
 */
template< typename T,
          typename COL_TYPE,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class CRSMatrix : protected CRSMatrixView< T, COL_TYPE, INDEX_TYPE, BUFFER_TYPE >
{
  using ColTypeNC = std::remove_const_t< COL_TYPE >;
  /// An alias for the parent class.
  using ParentClass = CRSMatrixView< T, COL_TYPE, INDEX_TYPE, BUFFER_TYPE >;

public:

  using typename ParentClass::EntryType;
  using typename ParentClass::ColType;
  using typename ParentClass::IndexType;

  /**
   * @name Constructors and the destructor
   */
  ///@{

  /**
   * @brief Constructor.
   * @param nrows the number of rows in the matrix.
   * @param ncols the number of columns in the matrix.
   * @param initialRowCapacity the initial non zero capacity of each row.
   */
  CRSMatrix( INDEX_TYPE const nrows=0,
             ColTypeNC const ncols=0,
             INDEX_TYPE const initialRowCapacity=0 ):
    ParentClass( true )
  {
    resize( nrows, ncols, initialRowCapacity );
    setName( "" );
  }

  /**
   * @brief Copy constructor, performs a deep copy.
   * @param src the CRSMatrix to copy.
   */
  inline
  CRSMatrix( CRSMatrix const & src ):
    ParentClass( true )
  { *this = src; }

  /**
   * @brief Default move constructor, performs a shallow copy.
   */
  inline
  CRSMatrix( CRSMatrix && ) = default;

  /**
   * @brief Destructor, frees the entries, values (columns), sizes and offsets Buffers.
   */
  ~CRSMatrix()
  { ParentClass::free( this->m_entries ); }

  ///@}

  /**
   * @name Methods to construct the matrix from scratch.
   */
  ///@{

  /**
   * @brief Copy assignment operator, performs a deep copy.
   * @param src the CRSMatrix to copy.
   * @return *this.
   */
  inline
  CRSMatrix & operator=( CRSMatrix const & src )
  {
    this->m_numCols = src.m_numCols;
    ParentClass::setEqualTo( src.m_numArrays,
                             src.m_offsets[ src.m_numArrays ],
                             src.m_offsets,
                             src.m_sizes,
                             src.m_values,
                             typename ParentClass::template PairOfBuffers< T >( this->m_entries, src.m_entries ) );
    return *this;
  }

  /**
   * @brief Default move assignment operator, performs a shallow copy.
   * @param src The CRSMatrix to be moved from.
   * @return *this.
   */
  inline
  CRSMatrix & operator=( CRSMatrix && src )
  {
    ParentClass::free( this->m_entries );
    ParentClass::operator=( std::move( src ) );
    return *this;
  }

  /**
   * @brief Steal the resources from a SparsityPattern.
   * @param src the SparsityPattern to convert.
   * @note All entries are default initialized. For numeric types this is zero initialization.
   */
  template< typename POLICY >
  inline
  void assimilate( SparsityPattern< COL_TYPE, INDEX_TYPE, BUFFER_TYPE > && src )
  {
    // Destroy the current entries.
    if( !std::is_trivially_destructible< T >::value )
    {
      CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE > const view = toViewConstSizes();
      RAJA::forall< POLICY >( RAJA::TypedRangeSegment< INDEX_TYPE >( 0, numRows() ),
                              [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const row )
        {
          INDEX_TYPE const nnz = view.numNonZeros( row );
          T * const entries = view.getEntries( row );
          arrayManipulation::destroy( entries, nnz );
        } );
    }

    // Reallocate to the appropriate length
    bufferManipulation::reserve( this->m_entries, 0, MemorySpace::host, src.nonZeroCapacity() );

    ParentClass::assimilate( reinterpret_cast< SparsityPatternView< COL_TYPE, INDEX_TYPE, BUFFER_TYPE > && >( src ) );

    CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE > const view = toViewConstSizes();
    RAJA::forall< POLICY >( RAJA::TypedRangeSegment< INDEX_TYPE >( 0, numRows() ),
                            [view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const row )
      {
        T * const rowEntries = view.getEntries( row );
        for( INDEX_TYPE i = 0; i < view.numNonZeros( row ); ++i )
        {
          new ( rowEntries + i ) T();
        }
      } );

    setName( "" );
  }

  /**
   * @tparam POLICY The RAJA policy used to convert @p rowCapacities into the offsets array.
   *   Should NOT be a device policy.
   * @brief Clears the matrix and creates a new matrix with the given number of rows and columns.
   * @param nRows The new number of rows.
   * @param nCols The new number of columns.
   * @param rowCapacities A pointer to an array of length @p nRows containing the capacity
   *   of each new row.
   */
  template< typename POLICY >
  void resizeFromRowCapacities( INDEX_TYPE const nRows, ColTypeNC const nCols, INDEX_TYPE const * const rowCapacities )
  {
    LVARRAY_ERROR_IF( !arrayManipulation::isPositive( nCols ), "nCols must be positive." );
    LVARRAY_ERROR_IF( nCols - 1 > std::numeric_limits< COL_TYPE >::max(),
                      "COL_TYPE must be able to hold the range of columns: [0, " << nCols - 1 << "]." );

    this->m_numCols = nCols;
    ParentClass::template resizeFromCapacities< POLICY >( nRows, rowCapacities, this->m_entries );
  }

  ///@}

  /**
   * @name CRSMatrixView and SparsityPatternView creation methods
   */
  ///@{

  /**
   * @copydoc ParentClass::toView
   * @note This is just a wrapper around the CRSMatrixView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  constexpr inline
  CRSMatrixView< T, COL_TYPE, INDEX_TYPE const, BUFFER_TYPE >
  toView() const &
  { return ParentClass::toView(); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null CRSMatrixView.
   * @note This cannot be called on a rvalue since the @c CRSMatrixView would
   *   contain the buffers of the current @c CRSMatrix that is about to be destroyed.
   *   This overload prevents that from happening.
   */
  constexpr inline
  CRSMatrixView< T, COL_TYPE, INDEX_TYPE const, BUFFER_TYPE >
  toView() const && = delete;

  /**
   * @copydoc ParentClass::toViewConstSizes
   * @note This is just a wrapper around the CRSMatrixView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >
  toViewConstSizes() const &
  { return ParentClass::toViewConstSizes(); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null CRSMatrixView.
   * @note This cannot be called on a rvalue since the @c CRSMatrixView would
   *   contain the buffers of the current @c CRSMatrix that is about to be destroyed.
   *   This overload prevents that from happening.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >
  toViewConstSizes() const && = delete;

  /**
   * @copydoc ParentClass::toViewConst
   * @note This is just a wrapper around the CRSMatrixView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >
  toViewConst() const &
  { return ParentClass::toViewConst(); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null CRSMatrixView.
   * @note This cannot be called on a rvalue since the @c CRSMatrixView would
   *   contain the buffers of the current @c CRSMatrix that is about to be destroyed.
   *   This overload prevents that from happening.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >
  toViewConst() const && = delete;

  using ParentClass::toSparsityPatternView;

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null CRSMatrixView.
   * @note This cannot be called on a rvalue since the @c SparsityPatternView would
   *   contain the buffers of the current @c CRSMatrix that is about to be destroyed.
   *   This overload prevents that from happening.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  SparsityPatternView< COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >
  toSparsityPatternView() const && = delete;

  ///@}

  /**
   * @name Attribute querying methods
   */
  ///@{

  using ParentClass::numRows;
  using ParentClass::numColumns;
  using ParentClass::numNonZeros;
  using ParentClass::nonZeroCapacity;
  using ParentClass::empty;

  ///@}

  /**
   * @name Methods that provide access to the data
   */
  ///@{

  using ParentClass::getOffsets;
  using ParentClass::getColumns;
  using ParentClass::getEntries;

  ///@}

  /**
   * @name Methods to change the capacity
   */
  ///@{

  /**
   * @brief Reserve space to hold at least the given total number of non zero entries.
   * @param nnz the number of no zero entries to reserve space for.
   */
  inline
  void reserveNonZeros( INDEX_TYPE const nnz )
  { ParentClass::reserveValues( nnz, this->m_entries ); }

  /**
   * @brief Reserve space to hold at least the given number of non zero entries in the given row.
   * @param row the row to reserve space in.
   * @param nnz the number of no zero entries to reserve space for.
   */
  inline
  void reserveNonZeros( INDEX_TYPE const row, INDEX_TYPE const nnz )
  {
    if( nonZeroCapacity( row ) >= nnz ) return;
    setRowCapacity( row, nnz );
  }

  /**
   * @brief Set the non zero capacity of the given row.
   * @param row the row to modify.
   * @param newCapacity the new capacity of the row.
   *
   * @note If the given capacity is less than the current number of non zero entries
   *       the entries are truncated.
   * @note Since a row can hold at most numColumns() entries the resulting capacity is
   *       min(newCapacity, numColumns()).
   */
  inline
  void setRowCapacity( INDEX_TYPE const row, INDEX_TYPE newCapacity )
  {
    if( newCapacity > numColumns() ) newCapacity = numColumns();
    ParentClass::setCapacityOfArray( row, newCapacity, this->m_entries );
  }

  /**
   * @brief Set the dimensions of the matrix.
   * @param nRows the new number of rows.
   * @param nCols the new number of columns.
   * @param initialRowCapacity the default capacity for each new array.
   * @note When shrinking the number of columns this method doesn't get rid of any existing entries.
   *   This can leave the matrix in an invalid state where a row has more columns than the matrix
   *   or where a specific column is greater than the number of columns in the matrix.
   *   If you will be constructing the matrix from scratch it is reccomended to clear it first.
   */
  void resize( INDEX_TYPE const nRows, ColTypeNC const nCols, INDEX_TYPE const initialRowCapacity )
  { ParentClass::resize( nRows, nCols, initialRowCapacity, this->m_entries ); }

  /**
   * @brief Compress the CRSMatrix so that the non-zeros and values of each row
   *        are contiguous with no extra capacity in between.
   * @note This method doesn't free any memory.
   */
  inline
  void compress()
  { ParentClass::compress( this->m_entries ); }

  ///@}

  /**
   * @name Methods that insert or remove entries from a row.
   */
  ///@{

  /**
   * @brief Insert a non-zero entry at the given position.
   * @param row the row to insert in.
   * @param col the column to insert at.
   * @param entry the entry to insert.
   * @return True iff the entry was inserted (the entry was zero before).
   */
  inline
  bool insertNonZero( INDEX_TYPE const row, ColTypeNC const col, T const & entry )
  { return ParentClass::insertIntoSetImpl( row, col, CallBacks( *this, row, &entry ) ); }

  /**
   * @brief Insert a non-zero entries into the given row.
   * @param row the row to insert into.
   * @param cols the columns to insert at, of length ncols.
   * @param entriesToInsert the entries to insert, of length ncols.
   * @param ncols the number of columns/entries to insert.
   * @return The number of entries inserted.
   *
   * @note The range [ entriesToInsert, entriesToInsert + ncols ) must be sorted and contain no duplicates.
   */
  inline
  INDEX_TYPE insertNonZeros( INDEX_TYPE const row,
                             ColTypeNC const * const cols,
                             T const * const entriesToInsert,
                             ColTypeNC const ncols )
  { return ParentClass::insertIntoSetImpl( row, cols, cols + ncols, CallBacks( *this, row, entriesToInsert ) ); }

  using ParentClass::removeNonZero;
  using ParentClass::removeNonZeros;

  ///@}

  /**
   * @name Methods that modify the entries of the matrix
   */
  ///@{

  /**
   * @copydoc ParentClass::setValues
   * @note This is not brought in with a @c using statement because it breaks doxygen.
   */
  template< typename POLICY >
  inline
  void setValues( T const & value ) const
  { ParentClass::template setValues< POLICY >( value ); }

  using ParentClass::zero;

  /**
   * @copydoc ParentClass::addToRow
   * @note This is not brought in with a @c using statement because it breaks doxygen.
   */
  template< typename AtomicPolicy >
  inline
  void addToRow( INDEX_TYPE const row,
                 ColTypeNC const * const LVARRAY_RESTRICT cols,
                 T const * const LVARRAY_RESTRICT vals,
                 ColTypeNC const nCols ) const
  { ParentClass::template addToRow< AtomicPolicy >( row, cols, vals, nCols ); }

  /**
   * @copydoc ParentClass::addToRowBinarySearch
   * @note This is not brought in with a @c using statement because it breaks doxygen.
   */
  template< typename AtomicPolicy >
  inline
  void addToRowBinarySearch( INDEX_TYPE const row,
                             ColTypeNC const * const LVARRAY_RESTRICT cols,
                             T const * const LVARRAY_RESTRICT vals,
                             ColTypeNC const nCols ) const
  { ParentClass::template addToRowBinarySearch< AtomicPolicy >( row, cols, vals, nCols ); }

  /**
   * @copydoc ParentClass::addToRowLinearSearch
   * @note This is not brought in with a @c using statement because it breaks doxygen.
   */
  template< typename AtomicPolicy >
  inline
  void addToRowLinearSearch( INDEX_TYPE const row,
                             ColTypeNC const * const LVARRAY_RESTRICT cols,
                             T const * const LVARRAY_RESTRICT vals,
                             ColTypeNC const nCols ) const
  { ParentClass::template addToRowLinearSearch< AtomicPolicy >( row, cols, vals, nCols ); }

  /**
   * @copydoc ParentClass::addToRowBinarySearchUnsorted
   * @note This is not brought in with a @c using statement because it breaks doxygen.
   */
  template< typename AtomicPolicy >
  inline
  void addToRowBinarySearchUnsorted( INDEX_TYPE const row,
                                     ColTypeNC const * const LVARRAY_RESTRICT cols,
                                     T const * const LVARRAY_RESTRICT vals,
                                     ColTypeNC const nCols ) const
  { ParentClass::template addToRowBinarySearchUnsorted< AtomicPolicy >( row, cols, vals, nCols ); }

  ///@}

  /**
   * @name Methods dealing with memory spaces
   */
  ///@{

  /**
   * @copydoc CRSMatrixView::move
   * @note This is just a wrapper around the CRSMatrixView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  void move( MemorySpace const space, bool const touch=true ) const
  { return ParentClass::move( space, touch ); }

  ///@}

  /**
   * @brief Set the name associated with this CRSMatrix which is used in the chai callback.
   * @param name the of the CRSMatrix.
   */
  void setName( std::string const & name )
  { ParentClass::template setName< decltype( *this ) >( name ); }

private:

  /**
   * @brief Increase the capacity of a row to accommodate at least the given number of
   *        non zero entries.
   * @param row the row to increase the capacity of.
   * @param newNNZ the new number of non zero entries.
   * @note This method over-allocates so that subsequent calls to insert don't have to reallocate.
   */
  void dynamicallyGrowRow( INDEX_TYPE const row, INDEX_TYPE const newNNZ )
  { setRowCapacity( row, 2 * newNNZ ); }

  /**
   * @class CallBacks
   * @brief This class provides the callbacks for the ArrayManipulation sorted routines.
   */
  class CallBacks
  {
public:

    /**
     * @brief Constructor.
     * @param crsM the CRSMatrix this CallBacks is associated with.
     * @param row the row this CallBacks is associated with.
     * @param entriesToInsert pointer to the entries to insert.
     */
    CallBacks( CRSMatrix & crsM,
               INDEX_TYPE const row, T const * const entriesToInsert ):
      m_crsM( crsM ),
      m_row( row ),
      m_rowNNZ( crsM.numNonZeros( row ) ),
      m_rowCapacity( crsM.nonZeroCapacity( row ) ),
      m_entries( nullptr ),
      m_entriesToInsert( entriesToInsert )
    {}

    /**
     * @brief Callback signaling that the size of the row has increased.
     * @param curPtr the current pointer to the array.
     * @param nToAdd the increase in the size.
     * @return a pointer to the rows columns.
     * @note This method doesn't actually change the size, but it does potentially
     *   do an allocation.
     */
    inline
    COL_TYPE * incrementSize( COL_TYPE * const curPtr, INDEX_TYPE const nToAdd )
    {
      LVARRAY_UNUSED_VARIABLE( curPtr );
      if( m_rowNNZ + nToAdd > m_rowCapacity )
      {
        m_crsM.dynamicallyGrowRow( m_row, m_rowNNZ + nToAdd );
      }

      m_entries = m_crsM.getEntries( m_row );
      return m_crsM.getSetValues( m_row );
    }

    /**
     * @brief Used with sortedArrayManipulation::insert routine this callback signals
     *        that the column was inserted at the given position. This means we also
     *        need to insert the entry at the same position.
     * @param pos the position the column was inserted at.
     */
    inline
    void insert( INDEX_TYPE const pos ) const
    { arrayManipulation::emplace( m_entries, m_rowNNZ, pos, m_entriesToInsert[0] ); }

    /**
     * @brief Used with the sortedArrayManipulation::insertSorted routine this callback
     *        signals that the given position was set to the column at the other position.
     *        This means we need to perform the same operation on the entries.
     * @param pos the position that was set.
     * @param colPos the position of the column.
     */
    inline
    void set( INDEX_TYPE const pos, INDEX_TYPE const colPos ) const
    { new (&m_entries[pos]) T( m_entriesToInsert[colPos] ); }

    /**
     * @brief Used with the sortedArrayManipulation::insertSorted routine this callback
     *        signals that the given column was inserted at the given position. Further information
     *        is provided in order to make the insertion efficient. This means that we need to perform
     *        the same operation on the entries.
     * @param nLeftToInsert the number of insertions that occur after this one.
     * @param colPos the position of the column that was inserted.
     * @param pos the position the column was inserted at.
     * @param prevPos the position the previous column was inserted at or m_rowNNZ if it is
     *             the first insertion.
     */
    inline
    void insert( INDEX_TYPE const nLeftToInsert,
                 INDEX_TYPE const colPos,
                 INDEX_TYPE const pos,
                 INDEX_TYPE const prevPos ) const
    {
      arrayManipulation::shiftUp( m_entries, prevPos, pos, nLeftToInsert );
      new (&m_entries[pos + nLeftToInsert - 1]) T( m_entriesToInsert[colPos] );
    }

private:
    /// The CRSMatrix the call back is associated with.
    CRSMatrix & m_crsM;

    /// The row the call back is assocated with.
    INDEX_TYPE const m_row;

    /// The number of non zeros in the row.
    INDEX_TYPE const m_rowNNZ;

    /// The non zero capacity of the row.
    INDEX_TYPE const m_rowCapacity;

    /// A pointer to the entries of the row.
    T * m_entries;

    /// A pointer to the entries to insert.
    T const * const m_entriesToInsert;
  };
};

} /* namespace LvArray */
