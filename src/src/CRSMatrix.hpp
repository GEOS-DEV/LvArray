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
#include "ArrayManipulation.hpp"

namespace LvArray
{

/**
 * @class CRSMatrix
 * @brief This class implements a compressed row storage matrix.
 * @tparam T the type of the entries in the matrix.
 * @tparam COL_TYPE the integer used to enumerate the columns.
 * @tparam INDEX_TYPE the integer to use for indexing.
 *
 * @note CRSMatrixView is a protected base class of CRSMatrix. This is to control the
 * conversion to CRSMatrixView so that when using a View the INDEX_TYPE is always const.
 * However the CRSMatrixView interface is reproduced here.
 */
template <class T, class COL_TYPE=unsigned int, class INDEX_TYPE=std::ptrdiff_t>
class CRSMatrix : protected CRSMatrixView<T, COL_TYPE, INDEX_TYPE>
{
public:

  // Aliasing public methods of SparsityPatternView via CRSMatrixView.
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::numRows;
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::numColumns;
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::numNonZeros;
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::nonZeroCapacity;
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::empty;
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::getColumns;

  // Aliasing public methods of CRSMatrixView.
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::toViewC;
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::toViewCC;
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::toSparsityPatternView;
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::getValues;
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::insertNonZero;
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::insertNonZeros;
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::removeNonZero;
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::removeNonZeros;

  /**
   * @brief Constructor.
   * @param [in] nrows the number of rows in the matrix.
   * @param [in] ncols the number of columns in the matrix.
   * @param [in] initialRowCapacity the initial non zero capacity of each row.
   */
  CRSMatrix( INDEX_TYPE nrows, INDEX_TYPE ncols, INDEX_TYPE initialRowCapacity=0 ):
    CRSMatrixView<T, COL_TYPE, INDEX_TYPE>()
  {
    m_num_columns = ncols;
    m_offsets.resize( nrows + 1, 0 );
    m_sizes.resize( nrows, 0 );

    if( initialRowCapacity != 0 )
    {
      m_columns.resize( nrows * initialRowCapacity, COL_TYPE( -1 ));
      m_values.resize( nrows * initialRowCapacity );
      for( INDEX_TYPE row = 1 ; row < nrows + 1 ; ++row )
      {
        m_offsets[row] = row * initialRowCapacity;
      }
    }
  }

  /**
   * @brief Copy constructor, performs a deep copy.
   * @param [in] src the CRSMatrix to copy.
   */
  inline
  CRSMatrix( CRSMatrix const & src ):
    CRSMatrixView<T, COL_TYPE, INDEX_TYPE>()
  { *this = src; }

  /**
   * @brief Default move constructor, performs a shallow copy.
   * @param [in/out] src the CRSMatrix to be moved from.
   */
  inline
  CRSMatrix( CRSMatrix && ) = default;

  /**
   * @brief Destructor, frees the values, columns, sizes and offsets ChaiVectors.
   */
  ~CRSMatrix()
  {
    clear();
    m_values.releaseAllocation();

    m_columns.free();
    m_sizes.free();
    m_offsets.free();
  }

  /**
   * @brief Conversion operator to CRSMatrixView<T, COL_TYPE, INDEX_TYPE const>.
   */
  CONSTEXPRFUNC inline
  operator CRSMatrixView<T, COL_TYPE, INDEX_TYPE const> const &
  () const restrict_this
  { return reinterpret_cast<CRSMatrixView<T, COL_TYPE, INDEX_TYPE const> const &>(*this); }

  /**
   * @brief Method to convert to CRSMatrixView<T, COL_TYPE, INDEX_TYPE const>. Use this method when
   * the above UDC isn't invoked, this usually occurs with template argument deduction.
   */
  CONSTEXPRFUNC inline
  CRSMatrixView<T, COL_TYPE, INDEX_TYPE const> const & toView() const restrict_this
  { return *this; }

  /**
   * @brief Conversion operator to CRSMatrixView<T, COL_TYPE const, INDEX_TYPE const>.
   * Although CRSMatrixView defines this operator nvcc won't let us alias it so
   * it is redefined here.
   */
  CONSTEXPRFUNC inline
  operator CRSMatrixView<T, COL_TYPE const, INDEX_TYPE const> const &
  () const restrict_this
  { return toViewC(); }

  /**
   * @brief Conversion operator to CRSMatrixView<T const, COL_TYPE const, INDEX_TYPE const>.
   * Although CRSMatrixView defines this operator nvcc won't let us alias it so
   * it is redefined here.
   */
  CONSTEXPRFUNC inline
  operator CRSMatrixView<T const, COL_TYPE const, INDEX_TYPE const> const &
  () const restrict_this
  { return toViewCC(); }

  /**
   * @brief Conversion operator to SparsityPatternView<COL_TYPE const, INDEX_TYPE const>.
   * Although CRSMatrixView defines this operator nvcc won't let us alias it so
   * it is redefined here.
   */
  CONSTEXPRFUNC inline
  operator SparsityPatternView<COL_TYPE const, INDEX_TYPE const> const &
  () const restrict_this
  { return toSparsityPatternView(); }

  /**
   * @brief Copy assignment operator, performs a deep copy.
   * @param [in] src the CRSMatrix to copy.
   */
  CRSMatrix & operator=( CRSMatrix const & src )
  {
    clear();

    INDEX_TYPE const newNNZ = src.m_columns.size();
    m_values.reserve( newNNZ );
    m_values.setSize( newNNZ );

    m_num_columns = src.m_num_columns;
    src.m_offsets.copy_into( m_offsets );
    src.m_sizes.copy_into( m_sizes );
    src.m_columns.copy_into( m_columns );

    INDEX_TYPE const nRows = numRows();
    for( INDEX_TYPE row = 0 ; row < nRows ; ++row )
    {
      T * const values = getValues( row );
      T const * const srcValues = src.getValues( row );

      INDEX_TYPE const rowNNZ = numNonZeros( row );
      for( INDEX_TYPE i = 0 ; i < rowNNZ ; ++i )
      {
        new (&values[i]) T( srcValues[i] );
      }
    }

    return *this;
  }

  /**
   * @brief Default move assignment operator, performs a shallow copy.
   * @param [in] src the CRSMatrix to be moved from.
   */
  inline
  CRSMatrix & operator=( CRSMatrix && src ) = default;

#ifdef USE_CHAI
  /**
   * @brief Moves the CRSMatrix to the given execution space.
   * @param [in] space the space to move to.
   */
  void move( chai::ExecutionSpace const space ) restrict_this
  {
    m_offsets.move( space );
    m_sizes.move( space );
    m_columns.move( space );
    m_values.move( space );
  }
#endif

  /**
   * @brief Reserve space to hold at least the given total number of non zero entries without reallocation.
   * @param [in] nnz the number of no zero entries to reserve space for.
   */
  inline
  void reserveNonZeros( INDEX_TYPE const nnz ) restrict_this
  {
    m_columns.reserve( nnz );
    m_values.reserve( nnz );
  }

  /**
   * @brief Reserve space to hold at least the given number of non zero entries in the given row without
   * either reallocation or shifting the row offsets.
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
   * @param [in] value the value to insert.
   * @return True iff the value was inserted (the entry was zero before).
   */
  inline
  bool insertNonZero( INDEX_TYPE const row, COL_TYPE const col, T const & value ) restrict_this
  {
    INDEX_TYPE const rowNNZ = numNonZeros( row );
    INDEX_TYPE const rowCapacity = nonZeroCapacity( row );
    T * const values = getValues( row );
    return insertNonZeroImpl( row, col, CallBacks( *this, row, rowNNZ, rowCapacity, &value ));
  }

  /**
   * @brief Insert a non-zero entries into the given row.
   * @param [in] row the row to insert into.
   * @param [in] cols the columns to insert at, of length ncols.
   * @param [in] valuesToInsert the values to insert, of length ncols.
   * @param [in] ncols the number of columns/values to insert.
   * @return The number of values inserted.
   */
  inline
  INDEX_TYPE insertNonZeros( INDEX_TYPE const row,
                             COL_TYPE const * const cols,
                             T const * const valuesToInsert,
                             INDEX_TYPE const ncols ) restrict_this
  {
    INDEX_TYPE const rowNNZ = numNonZeros( row );
    INDEX_TYPE const rowCapacity = nonZeroCapacity( row );
    return insertNonZerosImpl( row, cols, ncols, CallBacks( *this, row, rowNNZ, rowCapacity, valuesToInsert ));
  }

  /**
   * @brief Set the non zero capacity of the given row.
   * @param [in] row the row to modify.
   * @param [in] newCapacity the new capacity of the row.
   * @note If the given capacity is less than the current number of non zero entries
   * the entries are truncated.
   * @note Since a row can hold at most numColumns() entries the resulting capacity is
   * min(newCapacity, numColumns()).
   */
  inline
  void setRowCapacity( INDEX_TYPE row, INDEX_TYPE newCapacity )
  {
    SPARSITYPATTERN_CHECK_BOUNDS( row );
    GEOS_ASSERT( newCapacity >= 0 );

    if( newCapacity > numColumns())
    {
      newCapacity = numColumns();
    }

    INDEX_TYPE const row_offset = m_offsets[row];
    INDEX_TYPE const rowCapacity = nonZeroCapacity( row );
    INDEX_TYPE const capacity_difference = newCapacity - rowCapacity;
    if( capacity_difference == 0 ) return;

    INDEX_TYPE const nRows = numRows();
    for( INDEX_TYPE i = row + 1 ; i < nRows + 1 ; ++i )
    {
      m_offsets[i] += capacity_difference;
    }

    if( capacity_difference > 0 )
    {
      m_columns.shiftUp( row_offset + rowCapacity, capacity_difference );
      m_values.shiftUp( row_offset + rowCapacity, capacity_difference );
    }
    else
    {
      INDEX_TYPE const prev_row_size = numNonZeros( row );
      if( newCapacity < prev_row_size )
      {
        m_sizes[row] = newCapacity;
      }

      m_columns.erase( row_offset + rowCapacity + capacity_difference, -capacity_difference );
      m_values.erase( row_offset + rowCapacity + capacity_difference, -capacity_difference );
    }
  }

private:

  void clear()
  {
    INDEX_TYPE const nRows = numRows();
    for( INDEX_TYPE row = 0 ; row < nRows ; ++row )
    {
      INDEX_TYPE const rowNNZ = numNonZeros( row );
      T * const values = getValues( row );
      for( INDEX_TYPE i = 0 ; i < rowNNZ ; ++i )
      {
        values[i].~T();
      }
    }
  }

  /**
   * @brief Increase the capacity of a row to accommodate at least the given number of
   * non zero entries.
   * @param [in] row the row to increase the capacity of.
   * @param [in] newNNZ the new number of non zero entries.
   * @note This method over-allocates so that subsequent calls to insert don't have to reallocate.
   */
  void dynamicallyGrowRow( INDEX_TYPE row, INDEX_TYPE newNNZ )
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
     * @param [in] rowNNZ the number of non zeros in the row.
     * @param [in] rowCapacity the non zero capacity of the row.
     * @param [in] valuesToInsert pointer to the values to insert.
     */
    CallBacks( CRSMatrix<T, COL_TYPE, INDEX_TYPE> & crsM,
               INDEX_TYPE const row,
               INDEX_TYPE const rowNNZ,
               INDEX_TYPE const rowCapacity,
               T const * const valuesToInsert ):
      m_crsM( crsM ),
      m_row( row ),
      m_rowNNZ( rowNNZ ),
      m_rowCapacity( rowCapacity ),
      m_values( nullptr ),
      m_valuesToInsert( valuesToInsert )
    {}

    /**
     * @brief Callback signaling that the size of the row has increased.
     * @param [in] nToAdd the increase in the size.
     * @note This method doesn't actually change the size, but it does potentially
     * do an allocation.
     * @return a pointer to the rows columns.
     */
    inline
    COL_TYPE * incrementSize( INDEX_TYPE const nToAdd )
    {
      if( m_rowNNZ + nToAdd > m_rowCapacity )
      {
        m_crsM.dynamicallyGrowRow( m_row, m_rowNNZ + nToAdd );
      }

      m_values = m_crsM.getValues( m_row );
      return m_crsM.getColumnsProtected( m_row );
    }

    /**
     * @brief Used with ArrayManipulation::insertSorted routine this callback signals
     * that the column was inserted at the given position. This means we also need to insert
     * the value at the same position.
     * @param [in] pos the position the column was inserted at.
     */
    inline
    void insert( INDEX_TYPE const pos ) const
    { ArrayManipulation::insert( m_values, m_rowNNZ, pos, m_valuesToInsert[0] ); }

    /**
     * @brief Used with the ArrayManipulation::insertSorted multiple routine this callback
     * signals that the given position was set to the column at the other position.
     * This means we need to perform the same operation on the values.
     * @param [in] pos the position that was set.
     * @param [in] colPos the position of the column.
     */
    inline
    void set( INDEX_TYPE const pos, INDEX_TYPE const colPos ) const
    { new (&m_values[pos]) T( m_valuesToInsert[colPos] ); }

    /**
     * @brief Used with the ArrayManipulation::insertSorted multiple routine this callback
     * signals that the given column was inserted at the given position. Further information
     * is provided in order to make the insertion efficient. This means that we need to perform
     * the same operation on the values.
     * @param [in] nLeftToInsert the number of insertions that occur after this one.
     * @param [in] colPos the position of the column that was inserted.
     * @param [in] pos the position the column was inserted at.
     * @param [in] prevPos the position the previous column was inserted at or m_rowNNZ
     * if it is the first insertion.
     */
    inline
    void insert( INDEX_TYPE const nLeftToInsert,
                 INDEX_TYPE const colPos,
                 INDEX_TYPE const pos,
                 INDEX_TYPE const prevPos ) const
    {
      ArrayManipulation::shiftUp( m_values, prevPos, pos, nLeftToInsert );
      new (&m_values[pos + nLeftToInsert - 1]) T( m_valuesToInsert[colPos] );
    }

private:
    CRSMatrix<T, COL_TYPE, INDEX_TYPE> & m_crsM;
    INDEX_TYPE const m_row;
    INDEX_TYPE const m_rowNNZ;
    INDEX_TYPE const m_rowCapacity;
    T * m_values;
    T const * const m_valuesToInsert;
  };

  // Aliasing protected methods of SparsityPatternView.
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::insertNonZeroImpl;
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::insertNonZerosImpl;

  // Aliasing protected methods of CRSMatrixView.
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::getColumnsProtected;

  // Aliasing protected members of SparsityPatternView
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::m_num_columns;
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::m_offsets;
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::m_sizes;
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::m_columns;

  // Aliasing protected members of CRSMatrixView.
  using CRSMatrixView<T, COL_TYPE, INDEX_TYPE>::m_values;
};

} /* namespace LvArray */

#endif /* CRSMATRIX_HPP_ */
