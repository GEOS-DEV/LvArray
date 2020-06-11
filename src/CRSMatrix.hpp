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

/**
 * @file CRSMatrix.hpp
 */

#ifndef CRSMATRIX_HPP_
#define CRSMATRIX_HPP_

#include "CRSMatrixView.hpp"
#include "arrayManipulation.hpp"

namespace LvArray
{

template< typename COL_TYPE, typename INDEX_TYPE >
class SparsityPattern;

/**
 * @tparam T the type of the entries in the matrix.
 * @tparam COL_TYPE the integer used to enumerate the columns.
 * @tparam INDEX_TYPE the integer to use for indexing.
 * @class CRSMatrix
 * @brief This class implements a compressed row storage matrix.
 */
template< typename T, typename COL_TYPE=int, typename INDEX_TYPE=std::ptrdiff_t >
class CRSMatrix : protected CRSMatrixView< T, COL_TYPE, INDEX_TYPE >
{

  /// An alias for the parent class.
  using ParentClass = CRSMatrixView< T, COL_TYPE, INDEX_TYPE >;

public:

  // Aliasing public methods of CRSMatrixView.
  using ParentClass::numRows;
  using ParentClass::numColumns;
  using ParentClass::numNonZeros;
  using ParentClass::nonZeroCapacity;
  using ParentClass::empty;
  using ParentClass::getColumns;
  using ParentClass::getOffsets;
  using ParentClass::toSparsityPatternView;
  using ParentClass::getEntries;
  using ParentClass::insertNonZero;
  using ParentClass::insertNonZeros;
  using ParentClass::removeNonZero;
  using ParentClass::removeNonZeros;
  using ParentClass::setValues;
  using ParentClass::addToRow;

  /**
   * @brief Constructor.
   * @param nrows the number of rows in the matrix.
   * @param ncols the number of columns in the matrix.
   * @param initialRowCapacity the initial non zero capacity of each row.
   */
  CRSMatrix( INDEX_TYPE const nrows=0,
             INDEX_TYPE const ncols=0,
             INDEX_TYPE const initialRowCapacity=0 ):
    ParentClass()
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
    ParentClass()
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
  { ParentClass::free( m_entries ); }

  /**
   * @brief Copy assignment operator, performs a deep copy.
   * @param src the CRSMatrix to copy.
   * @return *this.
   */
  inline
  CRSMatrix & operator=( CRSMatrix const & src ) LVARRAY_RESTRICT_THIS
  {
    m_numCols = src.m_numCols;
    internal::PairOfBuffers< T > entriesPair( m_entries, src.m_entries );
    ParentClass::setEqualTo( src.m_numArrays,
                             src.m_offsets[ src.m_numArrays ],
                             src.m_offsets,
                             src.m_sizes,
                             src.m_values,
                             entriesPair );
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
    ParentClass::free( m_entries );
    ParentClass::operator=( std::move( src ) );
    return *this;
  }

  /**
   * @brief Steal the resources from a SparsityPattern.
   * @param src the SparsityPattern to convert.
   */
  inline
  void stealFrom( SparsityPattern< COL_TYPE, INDEX_TYPE > && src )
  {
    // Destroy the current entries.
    for( INDEX_TYPE row = 0; row < numRows(); ++row )
    {
      INDEX_TYPE const offset = m_offsets[ row ];
      INDEX_TYPE const nnz = numNonZeros( row );
      arrayManipulation::destroy( &m_entries[ offset ], nnz );
    }

    // Reallocate to the appropriate length
    bufferManipulation::reserve( m_entries, 0, src.nonZeroCapacity() );

    ParentClass::stealFrom( reinterpret_cast< SparsityPatternView< COL_TYPE, INDEX_TYPE > && >( src ) );

    for( INDEX_TYPE row = 0; row < numRows(); ++row )
    {
      T * const rowEntries = getEntries( row );
      for( INDEX_TYPE i = 0; i < numNonZeros( row ); ++i )
      {
        new ( rowEntries + i ) T();
      }
    }

    setName( "" );
  }

  /**
   * @brief @return A reference to *this reinterpreted as a CRSMatrixView< T, COL_TYPE, INDEX_TYPE const >.
   */
  constexpr inline
  CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > const & toView() const LVARRAY_RESTRICT_THIS
  { return reinterpret_cast< CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > const & >( *this ); }

  /**
   * @brief @return A reference to *this reinterpreted as a CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const >.
   * @note Duplicated for SFINAE needs.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > const & toViewConstSizes() const LVARRAY_RESTRICT_THIS
  { return ParentClass::toViewConstSizes(); }

  /**
   * @brief @return A reference to *this reinterpreted as a CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const >.
   * @note Duplicated for SFINAE needs.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const & toViewConst() const LVARRAY_RESTRICT_THIS
  { return ParentClass::toViewConst(); }

  /**
   * @brief @return A reference to *this reinterpreted as a SparsityPatternView< COL_TYPE const, INDEX_TYPE const >.
   * @note Duplicated for SFINAE needs.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  SparsityPatternView< COL_TYPE const, INDEX_TYPE const > const & toSparsityPatternView() const
  { return ParentClass::toSparsityPatternView(); }

  /**
   * @brief Reserve space to hold at least the given total number of non zero entries.
   * @param nnz the number of no zero entries to reserve space for.
   */
  inline
  void reserveNonZeros( INDEX_TYPE const nnz ) LVARRAY_RESTRICT_THIS
  {
    SparsityPatternView< COL_TYPE, INDEX_TYPE >::reserveValues( nnz, m_entries );
  }

  /**
   * @brief Reserve space to hold at least the given number of non zero entries in the given row.
   * @param row the row to reserve space in.
   * @param nnz the number of no zero entries to reserve space for.
   */
  inline
  void reserveNonZeros( INDEX_TYPE row, INDEX_TYPE nnz ) LVARRAY_RESTRICT_THIS
  {
    if( nonZeroCapacity( row ) >= nnz ) return;
    setRowCapacity( row, nnz );
  }

  /**
   * @brief Insert a non-zero entry at the given position.
   * @param row the row to insert in.
   * @param col the column to insert at.
   * @param entry the entry to insert.
   * @return True iff the entry was inserted (the entry was zero before).
   */
  inline
  bool insertNonZero( INDEX_TYPE const row, COL_TYPE const col, T const & entry ) LVARRAY_RESTRICT_THIS
  { return ParentClass::insertIntoSetImpl( row, col, CallBacks( *this, row, &entry )); }

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
                             COL_TYPE const * const cols,
                             T const * const entriesToInsert,
                             INDEX_TYPE const ncols ) LVARRAY_RESTRICT_THIS
  { return ParentClass::insertIntoSetImpl( row, cols, cols + ncols, CallBacks( *this, row, entriesToInsert ) ); }

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
    ParentClass::setCapacityOfArray( row, newCapacity, m_entries );
  }

  /**
   * @brief Compress the CRSMatrix so that the non-zeros and values of each row
   *        are contiguous with no extra capacity in between.
   * @note This method doesn't free any memory.
   */
  inline
  void compress() LVARRAY_RESTRICT_THIS
  { ParentClass::compress( m_entries ); }

  /**
   * @brief Set the dimensions of the matrix.
   * @param nRows the new number of rows.
   * @param nCols the new number of columns.
   * @param initialRowCapacity the default capacity for each new array.
   * @note When shrinking the number of columns this method doesn't get rid of any existing entries.
   *   This can leave the matrix in an invalid state where a row has more columns than the matrix
   *   or where a specific column is greater than the number of columns in the matrix.
   *   If you will be constructing the matrix from scratch it is reccomended to clear it first.
   * TODO: Add tests.
   */
  void resize( INDEX_TYPE const nRows, INDEX_TYPE const nCols, INDEX_TYPE const initialRowCapacity ) LVARRAY_RESTRICT_THIS
  { ParentClass::resize( nRows, nCols, initialRowCapacity, m_entries ); }

  /**
   * @brief Set the name associated with this CRSMatrix which is used in the chai callback.
   * @param name the of the CRSMatrix.
   */
  void setName( std::string const & name )
  { ParentClass::template setName< decltype( *this ) >( name ); }

  /**
   * @brief Move this SparsityPattern to the given memory space and touch the values, sizes and offsets.
   * @param space the memory space to move to.
   * @param touch If true touch the values, sizes and offsets in the new space.
   * @note When moving to the GPU since the offsets can't be modified on device they are not touched.
   * @note Duplicated for SFINAE needs.
   */
  void move( chai::ExecutionSpace const space, bool const touch=true ) const
  { ParentClass::move( space, touch ); }

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
    CallBacks( CRSMatrix< T, COL_TYPE, INDEX_TYPE > & crsM,
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
     *       do an allocation.
     */
    inline
    COL_TYPE * incrementSize( COL_TYPE * const LVARRAY_UNUSED_ARG( curPtr ),
                              INDEX_TYPE const nToAdd )
    {
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
    { arrayManipulation::insert( m_entries, m_rowNNZ, pos, m_entriesToInsert[0] ); }

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
    CRSMatrix< T, COL_TYPE, INDEX_TYPE > & m_crsM;
    INDEX_TYPE const m_row;
    INDEX_TYPE const m_rowNNZ;
    INDEX_TYPE const m_rowCapacity;
    T * m_entries;
    T const * const m_entriesToInsert;
  };

  // Aliasing protected members of CRSMatrixView.
  using ParentClass::m_numCols;
  using ParentClass::m_offsets;
  using ParentClass::m_sizes;
  using ParentClass::m_values;
  using ParentClass::m_entries;
};

} /* namespace LvArray */

#endif /* CRSMATRIX_HPP_ */
