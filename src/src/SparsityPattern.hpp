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
 * @file SparsityPattern.hpp
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
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::getOffsets;
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
    SparsityPatternView<COL_TYPE, INDEX_TYPE>::resize(nrows, ncols, initialRowCapacity);
    setUserCallBack( "" );
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
   * @brief Destructor, frees the values, sizes and offsets Buffers.
   */
  inline
  ~SparsityPattern() restrict_this
  { SparsityPatternView<COL_TYPE, INDEX_TYPE>::free(); }

  /**
   * @brief Conversion operator to SparsityPatternView<COL_TYPE, INDEX_TYPE const>.
   */
  CONSTEXPRFUNC inline
  operator SparsityPatternView<COL_TYPE, INDEX_TYPE const> const &
  () const restrict_this
  { return reinterpret_cast<SparsityPatternView<COL_TYPE, INDEX_TYPE const> const &>(*this); }

  /**
   * @brief Method to convert to SparsityPatternView<COL_TYPE, INDEX_TYPE const>. Use this method when
   *        the above UDC isn't invoked, this usually occurs with template argument deduction.
   */
  inline
  SparsityPatternView<COL_TYPE, INDEX_TYPE const> const & toView() const restrict_this
  { return *this; }

  /**
   * @brief Conversion operator to SparsityPatternView<COL_TYPE const, INDEX_TYPE const>.
   *        Although SparsityPatternView defines this operator nvcc won't let us alias it so
   *        it is redefined here.
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
    SparsityPatternView<COL_TYPE, INDEX_TYPE>::setEqualTo( src.m_numArrays,
                                                           src.m_offsets[ src.m_numArrays ],
                                                           src.m_offsets,
                                                           src.m_sizes,
                                                           src.m_values);
    return *this;
  }

  /**
   * @brief Default move assignment operator, performs a shallow copy.
   * @param [in] src the SparsityPattern to be moved from.
   */
  inline
  SparsityPattern & operator=( SparsityPattern && src ) = default;

  /**
   * @brief Move to the given memory space, optionally touching it.
   * @param [in] space the memory space to move to.
   * @param [in] touch whether to touch the memory in the space or not.
   */
  void move(chai::ExecutionSpace const space, bool const touch=true) restrict_this
  { SparsityPatternView<COL_TYPE, INDEX_TYPE>::move(space, touch); }

  
  void registerTouch(chai::ExecutionSpace const space) restrict_this
  { SparsityPatternView<COL_TYPE, INDEX_TYPE>::registerTouch(space); }

  /**
   * @brief Reserve space to hold at least the given total number of non zero entries without reallocation.
   * @param [in] nnz the number of no zero entries to reserve space for.
   */
  inline
  void reserveNonZeros( INDEX_TYPE const nnz ) restrict_this
  { SparsityPatternView<COL_TYPE, INDEX_TYPE>::reserveValues( nnz ); }

  /**
   * @brief Reserve space to hold at least the given number of non zero entries in the given row without
   *        either reallocation or shifting the row offsets.
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
   *
   * @note If the given capacity is less than the current number of non zero entries
   *       the entries are truncated.
   * @note Since a row can hold at most numColumns() entries the resulting capacity is
   *       min(newCapacity, numColumns()).
   */
  inline
  void setRowCapacity( INDEX_TYPE const row, INDEX_TYPE newCapacity ) restrict_this
  {
    if( newCapacity > numColumns() ) newCapacity = numColumns();
    SparsityPatternView<COL_TYPE, INDEX_TYPE>::setCapacityOfArray(row, newCapacity);
  }

  /**
   * @brief Compress the SparsityPattern so that the non-zeros of each row
   *        are contiguous with no extra capacity in between.
   * @note This method doesn't free any memory.
   */
  inline
  void compress() restrict_this
  { SparsityPatternView<COL_TYPE, INDEX_TYPE>::compress(); }

  /**
   * @brief Insert a non zero entry in the entry (row, col).
   * @param [in] row the row of the entry to insert.
   * @param [in] col the column of the entry to insert.
   * @return True iff the entry was previously empty.
   */
  inline
  bool insertNonZero( INDEX_TYPE const row, COL_TYPE const col ) restrict_this
  { return SparsityPatternView<COL_TYPE, INDEX_TYPE>::insertIntoSetImpl( row, col, CallBacks( *this, row ) ); }

  /**
   * @brief Insert non zeros in the given columns of the given row.
   * @param [in] row the row to insert into.
   * @param [in] cols the columns to insert.
   * @param [in] ncols the number of columns to insert.
   * @return The number of columns actually inserted.
   *
   * @note If possible sort cols first by calling sortedArrayManipulation::makeSorted(cols, ncols)
   *       and then call insertNonZerosSorted, this will be substantially faster.
   */
  inline
  INDEX_TYPE insertNonZeros( INDEX_TYPE const row, COL_TYPE const * const cols, INDEX_TYPE const ncols ) restrict_this
  { return SparsityPatternView<COL_TYPE, INDEX_TYPE>::insertIntoSetImpl( row, cols, ncols, CallBacks( *this, row ) ); }

  /**
   * @brief Insert non zeros in the given columns of the given row.
   * @param [in] row the row to insert into.
   * @param [in] cols the columns to insert, must be sorted.
   * @param [in] ncols the number of columns to insert.
   * @return The number of columns actually inserted.
   */
  inline
  INDEX_TYPE insertNonZerosSorted( INDEX_TYPE const row, COL_TYPE const * const cols, INDEX_TYPE const ncols ) restrict_this
  { return SparsityPatternView<COL_TYPE, INDEX_TYPE>::insertSortedIntoSetImpl( row, cols, ncols, CallBacks( *this, row ) ); }

  void setUserCallBack( std::string const & name )
  {
    SparsityPatternView< COL_TYPE, INDEX_TYPE >::template setUserCallBack< decltype( *this ) >( name );
  }

private:

  /**
   * @brief Increase the capacity of a row to accommodate at least the given number of
   *        non zero entries.
   * @param [in] row the row to increase the capacity of.
   * @param [in] newNNZ the new number of non zero entries.
   * @note This method over-allocates so that subsequent calls to insert don't have to reallocate.
   */
  inline
  void dynamicallyGrowRow( INDEX_TYPE const row, INDEX_TYPE const newNNZ ) restrict_this
  { setRowCapacity( row, newNNZ * 2 ); }

  /**
   * @class CallBacks
   * @brief This class provides the callbacks for the sortedArrayManipulation routines.
   */
  class CallBacks : public sortedArrayManipulation::CallBacks<COL_TYPE, INDEX_TYPE>
  {
public:

    /**
     * @brief Constructor.
     * @param [in/out] sp the SparsityPattern this CallBacks is associated with.
     * @param [in] row the row in the SparsityPattern this CallBacks is associated with.
     */
    inline
    CallBacks( SparsityPattern<COL_TYPE, INDEX_TYPE> & sp, INDEX_TYPE const row ):
      m_sp( sp ),
      m_row( row )
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
      INDEX_TYPE const newNNZ = m_sp.numNonZeros( m_row ) + nToAdd;
      if( newNNZ > m_sp.nonZeroCapacity( m_row ) )
      {
        m_sp.dynamicallyGrowRow( m_row, newNNZ );
      }

      return m_sp.getSetValues( m_row );
    }

private:
    SparsityPattern<COL_TYPE, INDEX_TYPE> & m_sp;
    INDEX_TYPE const m_row;
  };

  // Aliasing protected members of SparsityPatternView.
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::m_numArrays;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::m_offsets;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::m_sizes;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::m_values;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::m_num_columns;
};

} /* namespace LvArray */

#endif /* SPARSITYPATTERN_HPP_ */
