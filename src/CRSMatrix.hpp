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

/**
 * @tparam T the type of the entries in the matrix.
 * @tparam COL_TYPE the integer used to enumerate the columns.
 * @tparam INDEX_TYPE the integer to use for indexing.
 * @class CRSMatrix
 * @brief This class implements a compressed row storage matrix.
 */
template< class T, class COL_TYPE=int, class INDEX_TYPE=std::ptrdiff_t >
class CRSMatrix : protected CRSMatrixView< T, COL_TYPE, INDEX_TYPE >
{
public:

  // Aliasing public methods of CRSMatrixView.
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::numRows;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::numColumns;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::numNonZeros;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::nonZeroCapacity;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::empty;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::getColumns;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::getOffsets;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::toViewC;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::toViewCC;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::toSparsityPatternView;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::getEntries;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::insertNonZero;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::insertNonZeros;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::removeNonZero;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::removeNonZeros;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::removeNonZerosSorted;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::setValues;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::addToRow;

  /**
   * @brief Constructor.
   * @param [in] nrows the number of rows in the matrix.
   * @param [in] ncols the number of columns in the matrix.
   * @param [in] initialRowCapacity the initial non zero capacity of each row.
   */
  CRSMatrix( INDEX_TYPE const nrows=0,
             INDEX_TYPE const ncols=0,
             INDEX_TYPE const initialRowCapacity=0 ):
    CRSMatrixView< T, COL_TYPE, INDEX_TYPE >()
  {
    resize( nrows, ncols, initialRowCapacity );
    setName( "" );
  }

  /**
   * @brief Copy constructor, performs a deep copy.
   * @param [in] src the CRSMatrix to copy.
   */
  inline
  CRSMatrix( CRSMatrix const & src ):
    CRSMatrixView< T, COL_TYPE, INDEX_TYPE >()
  { *this = src; }

  /**
   * @brief Default move constructor, performs a shallow copy.
   * @param [in/out] src the CRSMatrix to be moved from.
   */
  inline
  CRSMatrix( CRSMatrix && ) = default;

  /**
   * @brief Destructor, frees the entries, values (columns), sizes and offsets Buffers.
   */
  ~CRSMatrix()
  { CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::free( m_entries ); }

  /**
   * @brief Conversion operator to CRSMatrixView<T, COL_TYPE, INDEX_TYPE const>.
   */
  CONSTEXPRFUNC inline
  operator CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > const &
  () const restrict_this
  { return reinterpret_cast< CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > const & >(*this); }

  /**
   * @brief Method to convert to CRSMatrixView<T, COL_TYPE, INDEX_TYPE const>. Use this method when
   *        the above UDC isn't invoked, this usually occurs with template argument deduction.
   */
  CONSTEXPRFUNC inline
  CRSMatrixView< T, COL_TYPE, INDEX_TYPE const > const & toView() const restrict_this
  { return *this; }

  /**
   * @brief Conversion operator to CRSMatrixView<T, COL_TYPE const, INDEX_TYPE const>.
   *        Although CRSMatrixView defines this operator nvcc won't let us alias it so
   *        it is redefined here.
   */
  CONSTEXPRFUNC inline
  operator CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > const &
  () const restrict_this
  { return toViewC(); }

  /**
   * @brief Conversion operator to CRSMatrixView<T const, COL_TYPE const, INDEX_TYPE const>.
   *        Although CRSMatrixView defines this operator nvcc won't let us alias it so
   *        it is redefined here.
   */
  CONSTEXPRFUNC inline
  operator CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const &
  () const restrict_this
  { return toViewCC(); }

  /**
   * @brief Conversion operator to SparsityPatternView<COL_TYPE const, INDEX_TYPE const>.
   *        Although CRSMatrixView defines this operator nvcc won't let us alias it so
   *        it is redefined here.
   */
  CONSTEXPRFUNC inline
  operator SparsityPatternView< COL_TYPE const, INDEX_TYPE const > const &
  () const restrict_this
  { return toSparsityPatternView(); }

  /**
   * @brief Copy assignment operator, performs a deep copy.
   * @param [in] src the CRSMatrix to copy.
   */
  inline
  CRSMatrix & operator=( CRSMatrix const & src ) restrict_this
  {
    m_numCols = src.m_numCols;
    internal::PairOfBuffers< T > entriesPair( m_entries, src.m_entries );
    SparsityPatternView< COL_TYPE, INDEX_TYPE >::setEqualTo( src.m_numArrays,
                                                             src.m_offsets[ src.m_numArrays ],
                                                             src.m_offsets,
                                                             src.m_sizes,
                                                             src.m_values,
                                                             entriesPair );
    return *this;
  }

  /**
   * @brief Default move assignment operator, performs a shallow copy.
   * @param [in] src the CRSMatrix to be moved from.
   */
  inline
  CRSMatrix & operator=( CRSMatrix && src ) = default;

  /**
   * @brief Move to the given memory space, optionally touching it.
   * @param [in] space the memory space to move to.
   * @param [in] touch whether to touch the memory in the space or not.
   */
  void move( chai::ExecutionSpace const space, bool const touch=true ) restrict_this
  { SparsityPatternView< COL_TYPE, INDEX_TYPE >::move( space, touch, m_entries ); }

  /**
   * @brief Touch in the given memory space.
   * @param [in] space the memory space to touch.
   */
  void registerTouch( chai::ExecutionSpace const space ) restrict_this
  { SparsityPatternView< COL_TYPE, INDEX_TYPE >::registerTouch( space, m_entries ); }

  /**
   * @brief Reserve space to hold at least the given total number of non zero entries.
   * @param [in] nnz the number of no zero entries to reserve space for.
   */
  inline
  void reserveNonZeros( INDEX_TYPE const nnz ) restrict_this
  {
    SparsityPatternView< COL_TYPE, INDEX_TYPE >::reserveValues( nnz, m_entries );
  }

  /**
   * @brief Reserve space to hold at least the given number of non zero entries in the given row.
   * @param [in] row the row to reserve space in.
   * @param [in] nnz the number of no zero entries to reserve space for.
   */
  inline
  void reserveNonZeros( INDEX_TYPE row, INDEX_TYPE nnz ) restrict_this
  {
    if( nonZeroCapacity( row ) >= nnz ) return;
    setRowCapacity( row, nnz );
  }

  /**
   * @brief Insert a non-zero entry at the given position.
   * @param [in] row the row to insert in.
   * @param [in] col the column to insert at.
   * @param [in] entry the entry to insert.
   * @return True iff the entry was inserted (the entry was zero before).
   */
  inline
  bool insertNonZero( INDEX_TYPE const row, COL_TYPE const col, T const & entry ) restrict_this
  { return CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::insertIntoSetImpl( row, col, CallBacks( *this, row, &entry )); }

  /**
   * @brief Insert a non-zero entries into the given row.
   * @param [in] row the row to insert into.
   * @param [in] cols the columns to insert at, of length ncols.
   * @param [in] entriesToInsert the entries to insert, of length ncols.
   * @param [in] ncols the number of columns/entries to insert.
   * @return The number of entries inserted.
   *
   * @note If possible sort cols and entriesToInsert first by calling sortedArrayManipulation::dualSort(cols, cols +
   * ncols, entriesToInsert)
   *       and then call insertNonZerosSorted, this will be substantially faster.
   */
  inline
  INDEX_TYPE insertNonZeros( INDEX_TYPE const row,
                             COL_TYPE const * const cols,
                             T const * const entriesToInsert,
                             INDEX_TYPE const ncols ) restrict_this
  { return CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::insertNonZerosImpl( row, cols, entriesToInsert, ncols, *this ); }

  /**
   * @brief Insert a non-zero entries into the given row.
   * @param [in] row the row to insert into.
   * @param [in] cols the columns to insert at, of length ncols. Must be sorted.
   * @param [in] entriesToInsert the entries to insert, of length ncols.
   * @param [in] ncols the number of columns/entries to insert.
   * @return The number of entries inserted.
   */
  inline
  INDEX_TYPE insertNonZerosSorted( INDEX_TYPE const row,
                                   COL_TYPE const * const cols,
                                   T const * const entriesToInsert,
                                   INDEX_TYPE const ncols ) restrict_this
  { return CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::insertSortedIntoSetImpl( row, cols, ncols, CallBacks( *this, row, entriesToInsert )); }

  /**
   * @brief Set the non zero capacity of the given row.
   * @param [in] row the row to modify.
   * @param [in] newCapacity the new capacity of the row.
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
    CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::setCapacityOfArray( row, newCapacity, m_entries );
  }

  /**
   * @brief Compress the CRSMatrix so that the non-zeros and values of each row
   *        are contiguous with no extra capacity in between.
   * @note This method doesn't free any memory.
   */
  inline
  void compress() restrict_this
  { CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::compress( m_entries ); }

  /**
   * @brief Set the dimensions of the matrix.
   * @param [in] nRows the new number of rows.
   * @param [in] nCols the new number of columns.
   * @param [in] defaultSetCapacity the default capacity for each new array.
   * @note When shrinking the number of columns this method doesn't get rid of any existing entries.
   *       This can leave the matrix in an invalid state where a row has more columns than the matrix
   *       or where a specific column is greater than the number of columns in the matrix.
   *       If you will be constructing the matrix from scratch it is reccomended to clear it first.
   * TODO: Add tests.
   */
  void resize( INDEX_TYPE const nRows, INDEX_TYPE const nCols, INDEX_TYPE const initialRowCapacity ) restrict_this
  {
    CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::resize( nRows, nCols, initialRowCapacity, m_entries );
  }

  /**
   * @brief Set the name associated with this CRSMatrix which is used in the chai callback.
   * @param name the of the CRSMatrix.
   */
  void setName( std::string const & name )
  {
    CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::template setName< decltype( *this ) >( name );
  }

private:

  /**
   * @brief Increase the capacity of a row to accommodate at least the given number of
   *        non zero entries.
   * @param [in] row the row to increase the capacity of.
   * @param [in] newNNZ the new number of non zero entries.
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
     * @param [in/out] crsM the CRSMatrix this CallBacks is associated with.
     * @param [in] row the row this CallBacks is associated with.
     * @param [in] entriesToInsert pointer to the entries to insert.
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
     * @param [in] nToAdd the increase in the size.
     * @return a pointer to the rows columns.
     * @note This method doesn't actually change the size, but it does potentially
     *       do an allocation.
     */
    inline
    COL_TYPE * incrementSize( INDEX_TYPE const nToAdd )
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
     * @param [in] pos the position the column was inserted at.
     */
    inline
    void insert( INDEX_TYPE const pos ) const
    { arrayManipulation::insert( m_entries, m_rowNNZ, pos, m_entriesToInsert[0] ); }

    /**
     * @brief Used with the sortedArrayManipulation::insertSorted routine this callback
     *        signals that the given position was set to the column at the other position.
     *        This means we need to perform the same operation on the entries.
     * @param [in] pos the position that was set.
     * @param [in] colPos the position of the column.
     */
    inline
    void set( INDEX_TYPE const pos, INDEX_TYPE const colPos ) const
    { new (&m_entries[pos]) T( m_entriesToInsert[colPos] ); }

    /**
     * @brief Used with the sortedArrayManipulation::insertSorted routine this callback
     *        signals that the given column was inserted at the given position. Further information
     *        is provided in order to make the insertion efficient. This means that we need to perform
     *        the same operation on the entries.
     * @param [in] nLeftToInsert the number of insertions that occur after this one.
     * @param [in] colPos the position of the column that was inserted.
     * @param [in] pos the position the column was inserted at.
     * @param [in] prevPos the position the previous column was inserted at or m_rowNNZ if it is
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
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::m_numCols;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::m_offsets;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::m_sizes;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::m_values;
  using CRSMatrixView< T, COL_TYPE, INDEX_TYPE >::m_entries;
};

} /* namespace LvArray */

#endif /* CRSMATRIX_HPP_ */
