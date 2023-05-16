/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file SparsityPattern.hpp
 * @brief Contains the implementation of LvArray:SparsityPattern.
 */

#pragma once

#include "SparsityPatternView.hpp"

namespace LvArray
{

/**
 * @class SparsityPattern
 * @brief This class implements a compressed row storage sparsity pattern.
 * @tparam COL_TYPE the integer used to enumerate the columns.
 * @tparam INDEX_TYPE the integer to use for indexing.
 */
template< typename COL_TYPE,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class SparsityPattern : protected SparsityPatternView< COL_TYPE, INDEX_TYPE, BUFFER_TYPE >
{
  /// An alias for the parent class.
  using ParentClass = SparsityPatternView< COL_TYPE, INDEX_TYPE, BUFFER_TYPE >;

public:

  using typename ParentClass::ColType;
  using typename ParentClass::ColTypeNC;
  using typename ParentClass::IndexType;

  /**
   * @name Constructors and the destructor.
   */
  ///@{

  /**
   * @brief Constructor.
   * @param nrows the number of rows in the matrix.
   * @param ncols the number of columns in the matrix.
   * @param initialRowCapacity the initial non zero capacity of each row.
   */
  inline
  SparsityPattern( INDEX_TYPE const nrows=0,
                   ColTypeNC const ncols=0,
                   INDEX_TYPE initialRowCapacity=0 ):
    ParentClass( true )
  {
    resize( nrows, ncols, initialRowCapacity );
    setName( "" );
  }

  /**
   * @brief Copy constructor, performs a deep copy.
   * @param src the SparsityPattern to copy.
   */
  inline
  SparsityPattern( SparsityPattern const & src ):
    ParentClass( true )
  { *this = src; }

  /**
   * @brief Default move constructor, performs a shallow copy.
   */
  inline
  SparsityPattern( SparsityPattern && ) = default;

  /**
   * @brief Destructor, frees the values, sizes and offsets Buffers.
   */
  inline
  ~SparsityPattern()
  { ParentClass::free(); }

  ///@}

  /**
   * @name Methods to construct the matrix from scratch.
   */
  ///@{

  /**
   * @brief Copy assignment operator, performs a deep copy.
   * @param src the SparsityPattern to copy.
   * @return *this.
   */
  inline
  SparsityPattern & operator=( SparsityPattern const & src )
  {
    this->m_numCols = src.m_numCols;
    ParentClass::setEqualTo( src.m_numArrays,
                             src.m_offsets[ src.m_numArrays ],
                             src.m_offsets,
                             src.m_sizes,
                             src.m_values );
    return *this;
  }

  /**
   * @brief Default move assignment operator, performs a shallow copy.
   * @param src The SparsityPattern to be moved from.
   * @return *this.
   */
  inline
  SparsityPattern & operator=( SparsityPattern && src )
  {
    ParentClass::free();
    ParentClass::operator=( std::move( src ) );
    return *this;
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
    ParentClass::template resizeFromCapacities< POLICY >( nRows, rowCapacities );
  }

  ///@}

  /**
   * @name SparsityPatternView creation methods
   */
  ///@{

  /**
   * @copydoc ParentClass::toView
   * @note This is just a wrapper around the SparsityPatternView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  constexpr inline
  SparsityPatternView< COL_TYPE, INDEX_TYPE const, BUFFER_TYPE >
  toView() const &
  { return ParentClass::toView(); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null SparsityPatternView.
   * @note This cannot be called on a rvalue since the @c SparsityPatternView would
   *   contain the buffers of the current @c SparsityPattern that is about to be destroyed.
   *   This overload prevents that from happening.
   */
  constexpr inline
  SparsityPatternView< COL_TYPE, INDEX_TYPE const, BUFFER_TYPE >
  toView() const && = delete;

  /**
   * @copydoc ParentClass::toViewConst
   * @note This is just a wrapper around the SparsityPatternView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  SparsityPatternView< COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >
  toViewConst() const &
  { return ParentClass::toViewConst(); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null SparsityPatternView.
   * @note This cannot be called on a rvalue since the @c SparsityPatternView would
   *   contain the buffers of the current @c SparsityPattern that is about to be destroyed.
   *   This overload prevents that from happening.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  SparsityPatternView< COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >
  toViewConst() const && = delete;

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

  using ParentClass::getColumns;
  using ParentClass::getOffsets;

  ///@}

  /**
   * @name Methods to change the capacity
   */
  ///@{

  /**
   * @brief Reserve space for the given number of rows.
   * @param rowCapacity The new minimum capacity for the number of rows.
   */
  inline
  void reserveRows( INDEX_TYPE const rowCapacity )
  { ParentClass::reserve( rowCapacity ); }

  /**
   * @brief Reserve space to hold at least the given total number of non zero entries without reallocation.
   * @param nnz the number of no zero entries to reserve space for.
   */
  inline
  void reserveNonZeros( INDEX_TYPE const nnz )
  { ParentClass::reserveValues( nnz ); }

  /**
   * @brief Reserve space to hold at least the given number of non zero entries in the given row without
   *        either reallocation or shifting the row offsets.
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
    ParentClass::setCapacityOfArray( row, newCapacity );
  }

  /**
   * @brief Compress the SparsityPattern so that the non-zeros of each row
   *        are contiguous with no extra capacity in between.
   * @note This method doesn't free any memory.
   */
  inline
  void compress()
  { ParentClass::compress(); }

  ///@}

  /**
   * @name Methods to resize the matrix.
   */
  ///@{

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
  { ParentClass::resize( nRows, nCols, initialRowCapacity ); }


  /**
   * @brief Append a row with the given capacity.
   * @param nzCapacity The non zero capacity of the row.
   */
  inline
  void appendRow( INDEX_TYPE const nzCapacity=0 )
  {
    INDEX_TYPE const maxOffset = this->m_offsets[ this->m_numArrays ];
    bufferManipulation::emplaceBack( this->m_offsets, this->m_numArrays + 1, maxOffset );
    bufferManipulation::emplaceBack( this->m_sizes, this->m_numArrays, 0 );
    ++this->m_numArrays;

    setRowCapacity( this->m_numArrays - 1, nzCapacity );
  }

  ///@}

  /**
   * @name Methods to modify a row
   */
  ///@{

  /**
   * @brief Insert a non zero entry in the entry (row, col).
   * @param row the row of the entry to insert.
   * @param col the column of the entry to insert.
   * @return True iff the entry was previously empty.
   */
  inline
  bool insertNonZero( INDEX_TYPE const row, COL_TYPE const col )
  { return ParentClass::insertIntoSetImpl( row, col, CallBacks( *this, row ) ); }

  /**
   * @tparam ITER An iterator type.
   * @brief Inserts multiple non-zero entries into the given row.
   * @param row The row to insert into.
   * @param first An iterator to the first column to insert.
   * @param last An iterator to the end of the columns to insert.
   * @return The number of columns inserted.
   * @note The columns to insert [ @p first, @p last ) must be sorted and contain no duplicates.
   */
  template< typename ITER >
  inline
  INDEX_TYPE insertNonZeros( INDEX_TYPE const row, ITER const first, ITER const last )
  { return ParentClass::insertIntoSetImpl( row, first, last, CallBacks( *this, row ) ); }

  using ParentClass::removeNonZero;

  /**
   * @copydoc ParentClass::removeNonZeros
   * @note This is not brought in with a @c using statement because it breaks doxygen.
   */
  template< typename ITER >
  inline INDEX_TYPE
  removeNonZeros( INDEX_TYPE const row, ITER const first, ITER const last ) const
  { return ParentClass::removeNonZeros( row, first, last ); }

  ///@}

  /**
   * @name Methods dealing with memory spaces
   */
  ///@{

  /**
   * @copydoc SparsityPatternView::move
   * @note This is just a wrapper around the SparsityPatternView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  void move( MemorySpace const space, bool const touch=true ) const
  { return ParentClass::move( space, touch ); }

  ///@}

  /**
   * @brief Set the name associated with this SparsityPattern which is used in the chai callback.
   * @param name the of the SparsityPattern.
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
  inline
  void dynamicallyGrowRow( INDEX_TYPE const row, INDEX_TYPE const newNNZ )
  { setRowCapacity( row, newNNZ * 2 ); }

  /**
   * @class CallBacks
   * @brief This class provides the callbacks for the sortedArrayManipulation routines.
   */
  class CallBacks : public sortedArrayManipulation::CallBacks< COL_TYPE >
  {
public:

    /**
     * @brief Constructor.
     * @param sp the SparsityPattern this CallBacks is associated with.
     * @param row the row in the SparsityPattern this CallBacks is associated with.
     */
    inline
    CallBacks( SparsityPattern & sp, INDEX_TYPE const row ):
      m_sp( sp ),
      m_row( row )
    {}

    /**
     * @brief Callback signaling that the number of non zeros of the associated row has increased.
     * @param curPtr the current pointer to the array.
     * @param nToAdd the number of non zeros added.
     * @note This method doesn't actually change the size but it can do reallocation.
     * @return a pointer to the columns of the associated row.
     */
    inline
    COL_TYPE * incrementSize( COL_TYPE * const curPtr, INDEX_TYPE const nToAdd ) const
    {
      LVARRAY_UNUSED_VARIABLE( curPtr );
      INDEX_TYPE const newNNZ = m_sp.numNonZeros( m_row ) + nToAdd;
      if( newNNZ > m_sp.nonZeroCapacity( m_row ) )
      {
        m_sp.dynamicallyGrowRow( m_row, newNNZ );
      }

      return m_sp.getSetValues( m_row );
    }

private:
    /// The SparsityPattern the call back is associated with.
    SparsityPattern & m_sp;

    /// The row the call back is associated with.
    INDEX_TYPE const m_row;
  };
};

} /* namespace LvArray */
