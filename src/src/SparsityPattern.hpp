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
 * @file SparsityPatternView.hpp
 */

#ifndef SPARSITYPATTERN_HPP_
#define SPARSITYPATTERN_HPP_

#include "SparsityPatternView.hpp"

namespace LvArray
{

/**
 * @class SparsityPattern
 * @brief This class implements a compressed row storage sparsity pattern.
 * @tparam COL_TYPE the integer used to enumerate the columns.
 * @tparam INDEX_TYPE the integer to use for indexing.
 *
 * @note SparsityPatternView is a protected base class of SparsityPattern. This is to control
 * the conversion to SparsityPatternView so that when using a View the INDEX_TYPE is always const.
 * However the SparsityPatternView interface is reproduced here.
 */
template <class COL_TYPE=unsigned int, typename INDEX_TYPE=std::ptrdiff_t>
class SparsityPattern : protected SparsityPatternView<COL_TYPE, INDEX_TYPE>
{
public:

  // Aliasing public methods of SparsityPatternView.
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::toViewC;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::numRows;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::numColumns;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::numNonZeros;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::nonZeroCapacity;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::empty;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::getColumns;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::insertNonZero;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::insertNonZeros;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::removeNonZero;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::removeNonZeros;

  /**
   * @brief Constructor.
   * @param [in] nrows the number of rows in the matrix.
   * @param [in] ncols the number of columns in the matrix.
   * @param [in] initialRowCapacity the initial non zero capacity of each row.
   */
  inline
  SparsityPattern( INDEX_TYPE const nrows, INDEX_TYPE const ncols, INDEX_TYPE initialRowCapacity=0 ) restrict_this:
    SparsityPatternView<COL_TYPE, INDEX_TYPE>()
  {
    GEOS_ERROR_IF( !ArrayManipulation::isPositive( nrows ), "nrows must be positive." );
    GEOS_ERROR_IF( !ArrayManipulation::isPositive( ncols ), "ncols must be positive." );
    GEOS_ERROR_IF( ncols - 1 > std::numeric_limits<COL_TYPE>::max(),
                   "COL_TYPE must be able to hold the range of columns: [0, " << ncols - 1 << "]." );

    if( initialRowCapacity > ncols )
    {
      GEOS_WARNING_IF( initialRowCapacity > ncols, "Number of non-zeros per row cannot exceed the the number of columns." );
      initialRowCapacity = ncols;
    }

    m_num_columns = ncols;
    m_offsets.resize( nrows + 1, 0 );
    m_sizes.resize( nrows, 0 );

    if( initialRowCapacity != 0 )
    {
      m_columns.resize( nrows * initialRowCapacity, COL_TYPE( -1 ));
      for( INDEX_TYPE row = 1 ; row < nrows + 1 ; ++row )
      {
        m_offsets[row] = row * initialRowCapacity;
      }
    }
  }

  /**
   * @brief Copy constructor, performs a deep copy.
   * @param [in] src the SparsityPattern to copy.
   */
  inline
  SparsityPattern( SparsityPattern const & src ) restrict_this:
    SparsityPatternView<COL_TYPE, INDEX_TYPE>()
  { *this = src; }

  /**
   * @brief Default move constructor, performs a shallow copy.
   * @param [in/out] src the SparsityPattern to be moved from.
   */
  inline
  SparsityPattern( SparsityPattern && src ) = default;

  /**
   * @brief Destructor, frees the columns, sizes and offsets ChaiVectors.
   */
  inline
  ~SparsityPattern() restrict_this
  {
    m_columns.free();
    m_sizes.free();
    m_offsets.free();
  }

  /**
   * @brief Conversion operator to SparsityPatternView<COL_TYPE, INDEX_TYPE const>.
   */
  CONSTEXPRFUNC inline
  operator SparsityPatternView<COL_TYPE, INDEX_TYPE const> const &
  () const restrict_this
  { return reinterpret_cast<SparsityPatternView<COL_TYPE, INDEX_TYPE const> const &>(*this); }

  /**
   * @brief Method to convert to SparsityPatternView<COL_TYPE, INDEX_TYPE const>. Use this method when
   * the above UDC isn't invoked, this usually occurs with template argument deduction.
   */
  inline
  SparsityPatternView<COL_TYPE, INDEX_TYPE const> const & toView() const restrict_this
  { return *this; }

  /**
   * @brief Conversion operator to SparsityPatternView<COL_TYPE const, INDEX_TYPE const>.
   * Although SparsityPatternView defines this operator nvcc won't let us alias it so
   * it is redefined here.
   */
  CONSTEXPRFUNC inline
  operator SparsityPatternView<COL_TYPE const, INDEX_TYPE const> const &
  () const restrict_this
  { return toViewC(); }

  /**
   * @brief Copy assignment operator, performs a deep copy.
   * @param [in] src the SparsityPattern to copy.
   */
  inline
  SparsityPattern & operator=( SparsityPattern const & src ) restrict_this
  {
    m_num_columns = src.m_num_columns;
    src.m_offsets.copy_into( m_offsets );
    src.m_sizes.copy_into( m_sizes );
    src.m_columns.copy_into( m_columns );

    return *this;
  }

  /**
   * @brief Default move assignment operator, performs a shallow copy.
   * @param [in] src the SparsityPattern to be moved from.
   */
  inline
  SparsityPattern & operator=( SparsityPattern && src ) = default;

#ifdef USE_CHAI
  /**
   * @brief Moves the SparsityPattern to the given execution space.
   * @param [in] space the space to move to.
   */
  void move( chai::ExecutionSpace const space ) restrict_this
  {
    m_offsets.move( space );
    m_sizes.move( space );
    m_columns.move( space );
  }
#endif

  /**
   * @brief Reserve space to hold at least the given total number of non zero entries without reallocation.
   * @param [in] nnz the number of no zero entries to reserve space for.
   */
  inline
  void reserveNonZeros( INDEX_TYPE const nnz ) restrict_this
  { m_columns.reserve( nnz ); }

  /**
   * @brief Reserve space to hold at least the given number of non zero entries in the given row without
   * either reallocation or shifting the row offsets.
   * @param [in] row the row to reserve space in.
   * @param [in] nnz the number of no zero entries to reserve space for.
   */
  inline
  void reserveNonZeros( INDEX_TYPE const row, INDEX_TYPE const nnz ) restrict_this
  {
    if( nonZeroCapacity( row ) >= nnz ) return;
    setRowCapacity( row, nnz );
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
  void setRowCapacity( INDEX_TYPE const row, INDEX_TYPE newCapacity ) restrict_this
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
    }
    else
    {
      INDEX_TYPE const prev_row_size = numNonZeros( row );
      if( newCapacity < prev_row_size )
      {
        m_sizes[row] = newCapacity;
      }

      m_columns.erase( row_offset + rowCapacity + capacity_difference, -capacity_difference );
    }
  }

  /**
   * @brief Insert a non zero entry in the entry (row, col).
   * @param [in] row the row of the entry to insert.
   * @param [in] col the column of the entry to insert.
   * @return True iff the entry was previously empty.
   */
  inline
  bool insertNonZero( INDEX_TYPE const row, COL_TYPE const col ) restrict_this
  {
    INDEX_TYPE const rowNNZ = numNonZeros( row );
    INDEX_TYPE const rowCapacity = nonZeroCapacity( row );
    return insertNonZeroImpl( row, col, CallBacks( *this, row, rowNNZ, rowCapacity ));
  }

  /**
   * @brief Insert non zeros in the given columns of the given row.
   * @param [in] row the row to insert into.
   * @param [in] cols the columns to insert.
   * @param [in] ncols the number of columns to insert.
   * @return The number of columns actually inserted.
   */
  inline
  INDEX_TYPE insertNonZeros( INDEX_TYPE const row, COL_TYPE const * const cols, INDEX_TYPE const ncols ) restrict_this
  {
    INDEX_TYPE const rowNNZ = numNonZeros( row );
    INDEX_TYPE const rowCapacity = nonZeroCapacity( row );
    return insertNonZerosImpl( row, cols, ncols, CallBacks( *this, row, rowNNZ, rowCapacity ));
  }

private:

  /**
   * @brief Increase the capacity of a row to accommodate at least the given number of
   * non zero entries.
   * @param [in] row the row to increase the capacity of.
   * @param [in] newNNZ the new number of non zero entries.
   * @note This method over-allocates so that subsequent calls to insert don't have to reallocate.
   */
  inline
  void dynamicallyGrowRow( INDEX_TYPE const row, INDEX_TYPE const newNNZ ) restrict_this
  { setRowCapacity( row, newNNZ * 2 ); }

  /**
   * @class CallBacks
   * @brief This class provides the callbacks for the ArrayManipulation sorted routines.
   */
  class CallBacks
  {
public:

    /**
     * @brief Constructor.
     * @param [in/out] sp the SparsityPattern this CallBacks is associated with.
     * @param [in] row the row in the SparsityPattern this CallBacks is associated with.
     * @param [in] rowNNZ the number of non zeros in the given row.
     * @param [in] the capacity of the given row.
     */
    inline
    CallBacks( SparsityPattern<COL_TYPE, INDEX_TYPE> & sp,
               INDEX_TYPE const row,
               INDEX_TYPE const rowNNZ,
               INDEX_TYPE const rowCapacity ):
      m_sp( sp ),
      m_row( row ),
      m_rowNNZ( rowNNZ ),
      m_rowCapacity( rowCapacity )
    {}

    /**
     * @brief Callback signaling that the number of non zeros of the associated row has increased.
     * @param [in] nToAdd the number of non zeros added.
     * @note This method doesn't actually change the size but it can do reallocation.
     * @return a pointer to the columns of the associated row.
     */
    inline
    COL_TYPE * incrementSize( INDEX_TYPE const nToAdd ) const restrict_this
    {
      INDEX_TYPE const newNNZ = m_rowNNZ + nToAdd;
      if( newNNZ > m_rowCapacity )
      {
        m_sp.dynamicallyGrowRow( m_row, newNNZ );
      }

      return m_sp.getColumnsProtected( m_row );
    }

    /**
     * @brief These methods are placeholder and are no-ops.
     */
    /// @{
    inline
    void set( INDEX_TYPE, INDEX_TYPE ) const restrict_this
    {}

    inline
    void insert( INDEX_TYPE ) const restrict_this
    {}

    inline
    void insert( INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE ) const restrict_this
    {}

    inline
    void remove( INDEX_TYPE ) const restrict_this
    {}

    inline
    void remove( INDEX_TYPE, INDEX_TYPE, INDEX_TYPE ) const restrict_this
    {}
    /// @}

private:
    SparsityPattern<COL_TYPE, INDEX_TYPE> & m_sp;
    INDEX_TYPE const m_row;
    INDEX_TYPE const m_rowNNZ;
    INDEX_TYPE const m_rowCapacity;
  };

  // Aliasing protected methods of SparsityPatternView.
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::getColumnsProtected;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::insertNonZeroImpl;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::insertNonZerosImpl;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::removeNonZeroImpl;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::removeNonZerosImpl;

  // Aliasing protected members of SparsityPatternView.
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::m_offsets;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::m_num_columns;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::m_sizes;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::m_columns;
};

} /* namespace LvArray */

#endif /* SPARSITYPATTERN_HPP_ */
