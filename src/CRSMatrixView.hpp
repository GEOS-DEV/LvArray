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
 * @file CRSMatrixView.hpp
 */

#ifndef CRSMATRIXVIEW_HPP_
#define CRSMATRIXVIEW_HPP_

#include "SparsityPatternView.hpp"
#include "arrayManipulation.hpp"
#include "ArraySlice.hpp"

namespace LvArray
{

// Wrapper around RAJA::atomicAdd that allows for non primitive types with RAJA::seq_atomic
namespace
{

template< typename POLICY, typename T >
LVARRAY_HOST_DEVICE inline
void atomicAdd( POLICY, T * const acc, T const & val )
{ RAJA::atomicAdd< POLICY >( acc, val ); }

DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE constexpr inline
void atomicAdd( RAJA::seq_atomic, T * acc, T const & val )
{ *acc += val; }

}

/**
 * @class CRSMatrixView
 * @brief This class provides a view into a compressed row storage matrix.
 *
 * @tparam T the type of the entries of the matrix.
 * @tparam COL_TYPE the integer used to enumerate the columns.
 * @tparam INDEX_TYPE the integer to use for indexing.
 *
 * @note When INDEX_TYPE is const m_offsets is not copied back from the device. INDEX_TYPE should always be const
 *       since CRSMatrixView is not allowed to modify the offsets.
 * @note When COL_TYPE is const and INDEX_TYPE is const you cannot insert or remove from the View
 *       and. You can however modify the entries of the matrix which is copied back from the device.
 * @note When T, COL_TYPE and INDEX_TYPE are const you cannot modify the matrix in any way
 *       and nothing is copied back from the device.
 */
template< class T, class COL_TYPE=unsigned int, class INDEX_TYPE=std::ptrdiff_t >
class CRSMatrixView : protected SparsityPatternView< COL_TYPE, INDEX_TYPE >
{

  /// An alias for the parent class.
  using ParentClass = SparsityPatternView< COL_TYPE, INDEX_TYPE >;

public:
  static_assert( !std::is_const< T >::value ||
                 (std::is_const< COL_TYPE >::value && std::is_const< INDEX_TYPE >::value),
                 "When T is const COL_TYPE and INDEX_TYPE must also be const." );

  /// An alias for the non const index type.
  using typename ParentClass::INDEX_TYPE_NC;

  // Aliasing public methods of SparsityPatternView.
  using ParentClass::numRows;
  using ParentClass::numColumns;
  using ParentClass::numNonZeros;
  using ParentClass::nonZeroCapacity;
  using ParentClass::empty;
  using ParentClass::getColumns;
  using ParentClass::getOffsets;

  /**
   * @brief Default copy constructor. Performs a shallow copy and calls the
   *        chai::ManagedArray copy constructor.
   */
  CRSMatrixView( CRSMatrixView const & ) = default;

  /**
   * @brief Default move constructor.
   */
  inline
  CRSMatrixView( CRSMatrixView && ) = default;

  /**
   * @brief Default copy assignment operator, this does a shallow copy.
   * @return *this.
   */
  inline
  CRSMatrixView & operator=( CRSMatrixView const & ) = default;

  /**
   * @brief @return Return *this.
   * @brief This is included for SFINAE needs.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  CRSMatrixView< T, COL_TYPE, INDEX_TYPE > const & toView() const LVARRAY_RESTRICT_THIS
  { return *this; }

  /**
   * @brief @return A reference to *this reinterpreted as a CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const >.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > const & toViewConstSizes() const LVARRAY_RESTRICT_THIS
  { return reinterpret_cast< CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > const & >( *this ); }

  /**
   * @brief @return A reference to *this reinterpreted as a CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const >.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const & toViewConst() const LVARRAY_RESTRICT_THIS
  { return reinterpret_cast< CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const & >(*this); }

  /**
   * @brief @return A reference to *this reinterpreted as a SparsityPatternView< COL_TYPE const, INDEX_TYPE const >.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  SparsityPatternView< COL_TYPE const, INDEX_TYPE const > const & toSparsityPatternView() const
  { return reinterpret_cast< SparsityPatternView< COL_TYPE const, INDEX_TYPE const > const & >(*this); }

  /**
   * @brief @return Return an ArraySlice1d to the matrix entries of the given row.
   * @param row The row to access.
   */
  LVARRAY_HOST_DEVICE inline
  ArraySlice< T, 1, 0, INDEX_TYPE_NC > getEntries( INDEX_TYPE const row ) const LVARRAY_RESTRICT_THIS
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( row );
    return ArraySlice< T, 1, 0, INDEX_TYPE_NC >( m_entries.data() + m_offsets[row], &m_sizes[row], nullptr );
  }

  /**
   * @brief Insert a non-zero entry at the given position.
   * @param row The row to insert in.
   * @param col The column to insert at.
   * @param entry The entry to insert.
   * @return True iff the entry was inserted (the entry was zero before).
   * @pre Since the CRSMatrixView can't do reallocation or shift the offsets it is
   *   up to the user to ensure that the given row has enough space for the new entry.
   */
  LVARRAY_HOST_DEVICE inline
  bool insertNonZero( INDEX_TYPE const row, COL_TYPE const col, T const & entry ) const LVARRAY_RESTRICT_THIS
  { return ParentClass::insertIntoSetImpl( row, col, CallBacks( *this, row, &entry ) ); }

  /**
   * @brief Insert a non-zero entries into the given row.
   * @param row The row to insert into.
   * @param cols The columns to insert at, of length ncols. Must be sorted.
   * @param entriesToInsert The entries to insert, of length ncols.
   * @param ncols The number of columns/entries to insert.
   * @return The number of entries inserted.
   * @pre The range [ @p cols, @p cols + @p ncols ) must be sorted and contain no duplicates.
   * @pre Since the CRSMatrixView can't do reallocation or shift the offsets it is
   *   up to the user to ensure that the given row has enough space for the new entries.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC insertNonZeros( INDEX_TYPE const row,
                                COL_TYPE const * const LVARRAY_RESTRICT cols,
                                T const * const LVARRAY_RESTRICT entriesToInsert,
                                INDEX_TYPE const ncols ) const LVARRAY_RESTRICT_THIS
  { return ParentClass::insertIntoSetImpl( row, cols, cols + ncols, CallBacks( *this, row, entriesToInsert ) ); }

  /**
   * @brief Remove a non-zero entry at the given position.
   * @param row The row to remove from.
   * @param col The column to remove.
   * @return True iff the entry was removed (the entry was non-zero before).
   */
  LVARRAY_HOST_DEVICE inline
  bool removeNonZero( INDEX_TYPE const row, COL_TYPE const col ) const LVARRAY_RESTRICT_THIS
  { return ParentClass::removeFromSetImpl( row, col, CallBacks( *this, row, nullptr )); }

  /**
   * @brief Remove non-zero entries from the given row.
   * @param row The row to remove from.
   * @param cols The columns to remove, of length ncols.
   * @param ncols The number of columns to remove.
   * @return True iff the entry was removed (the entry was non-zero before).
   * @pre The range [ @p cols, @p cols + @p ncols ) must be sorted and contain no duplicates.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC removeNonZeros( INDEX_TYPE const row,
                                COL_TYPE const * const LVARRAY_RESTRICT cols,
                                INDEX_TYPE const ncols ) const LVARRAY_RESTRICT_THIS
  {
    T * const entries = getEntries( row );
    INDEX_TYPE const rowNNZ = numNonZeros( row );
    INDEX_TYPE const nRemoved = ParentClass::removeFromSetImpl( row, cols, cols + ncols, CallBacks( *this, row, nullptr ) );

    arrayManipulation::destroy( entries + rowNNZ - nRemoved, nRemoved );

    return nRemoved;
  }

  /**
   * @brief Set all the values in the matrix to the given value.
   * @param value the value to set values in the matrix to.
   */
  inline
  void setValues( T const & value ) const
  {
    for( INDEX_TYPE_NC row = 0; row < numRows(); ++row )
    {
      INDEX_TYPE const nnz = numNonZeros( row );
      T * const entries = getEntries( row );

      for( INDEX_TYPE_NC i = 0; i < nnz; ++i )
      {
        entries[ i ] = value;
      }
    }
  }

  /// @cond DO_NOT_DOCUMENT
  // This method breaks uncrustify, and it has something to do with the overloading. If it's called something
  // else it works just fine.

  /**
   * @brief Add to the given entries, the entries must already exist in the matrix.
   *        The columns must be sorted.
   * @tparam AtomicPolicy the policy to use when adding to the values.
   * @param row The row to access.
   * @param cols The columns to add to, must be sorted, unique and of length nCols.
   * @param vals The values to add, of length nCols.
   * @param nCols The number of columns to add to.
   * @pre The range [ @p cols, @p cols + @p ncols ) must be sorted and contain no duplicates.
   * TODO: Use benchmarks of addToRowBinarySearch and addToRowLinearSearch
   *       to develop a better heuristic.
   */
  template< typename AtomicPolicy >
  LVARRAY_HOST_DEVICE inline
  void addToRow( INDEX_TYPE const row,
                 COL_TYPE const * const LVARRAY_RESTRICT cols,
                 T const * const LVARRAY_RESTRICT vals,
                 INDEX_TYPE const nCols ) const
  {
    INDEX_TYPE const nnz = numNonZeros( row );
    if( nCols < nnz / 4 && nnz > 64 )
    {
      addToRowBinarySearch< AtomicPolicy >( row, cols, vals, nCols );
    }
    else
    {
      addToRowLinearSearch< AtomicPolicy >( row, cols, vals, nCols );
    }
  }

  /// @endcond DO_NOT_DOCUMENT

  /**
   * @brief Add to the given entries, the entries must already exist in the matrix.
   * @details This method uses a binary search to find the entries in the row to add to.
   *   This makes the method O( nCols * log( numNonZeros( @p row ) ) ) and is therefore best
   *   to use when @p nCols is much less than numNonZeros( @p row ).
   * @tparam AtomicPolicy the policy to use when adding to the values.
   * @param row The row to access.
   * @param cols The columns to add to, must be sorted, unique and of length @p nCols.
   * @param vals The values to add, of length @p nCols.
   * @param nCols The number of columns to add to.
   * @pre The range [ @p cols, @p cols + @p ncols ) must be sorted and contain no duplicates.
   */
  template< typename AtomicPolicy >
  LVARRAY_HOST_DEVICE inline
  void addToRowBinarySearch( INDEX_TYPE const row,
                             COL_TYPE const * const LVARRAY_RESTRICT cols,
                             T const * const LVARRAY_RESTRICT vals,
                             INDEX_TYPE const nCols ) const
  {
    LVARRAY_ASSERT( sortedArrayManipulation::isSortedUnique( cols, cols + nCols ) );

    INDEX_TYPE const nnz = numNonZeros( row );
    COL_TYPE const * const columns = getColumns( row );
    T * const entries = getEntries( row );

    INDEX_TYPE_NC curPos = 0;
    for( INDEX_TYPE_NC i = 0; i < nCols; ++i )
    {
      INDEX_TYPE const pos = sortedArrayManipulation::find( columns + curPos, nnz - curPos, cols[ i ] ) + curPos;
      LVARRAY_ASSERT_GT( nnz, pos );
      LVARRAY_ASSERT_EQ( columns[ pos ], cols[ i ] );

      atomicAdd( AtomicPolicy{}, entries + pos, vals[ i ] );
      curPos = pos + 1;
    }
  }

  /**
   * @brief Add to the given entries, the entries must already exist in the matrix.
   * @brief This method uses a linear search to find the entries in the row to add to.
   *   This makes the method O( numNonZeros( @p row ) ) and is therefore best to use when
   *   @p nCols is similar to numNonZeros( @p row ).
   * @tparam AtomicPolicy the policy to use when adding to the values.
   * @param row The row to access.
   * @param cols The columns to add to, must be sorted, unique and of length @p nCols.
   * @param vals The values to add, of length @p nCols.
   * @param nCols The number of columns to add to.
   * @pre The range [ @p cols, @p cols + @p ncols ) must be sorted and contain no duplicates.
   */
  template< typename AtomicPolicy >
  LVARRAY_HOST_DEVICE inline
  void addToRowLinearSearch( INDEX_TYPE const row,
                             COL_TYPE const * const LVARRAY_RESTRICT cols,
                             T const * const LVARRAY_RESTRICT vals,
                             INDEX_TYPE const nCols ) const
  {
    LVARRAY_ASSERT( sortedArrayManipulation::isSortedUnique( cols, cols + nCols ) );

    INDEX_TYPE const nnz = numNonZeros( row );
    COL_TYPE const * const columns = getColumns( row );
    T * const entries = getEntries( row );

    INDEX_TYPE_NC curPos = 0;
    for( INDEX_TYPE_NC i = 0; i < nCols; ++i )
    {
      for( INDEX_TYPE_NC j = curPos; j < nnz; ++j )
      {
        if( columns[ j ] == cols[ i ] )
        {
          curPos = j;
          break;
        }
      }
      LVARRAY_ASSERT_EQ( columns[ curPos ], cols[ i ] );
      atomicAdd( AtomicPolicy{}, entries + curPos, vals[ i ] );
      ++curPos;
    }
  }

  /**
   * @brief Add to the given entries, the entries must already exist in the matrix.
   * @details This method uses a binary search to find the entries in the row to add to.
   *   This makes the method O( nCols * log( numNonZeros( @p row ) ) ) and is therefore best
   *   to use when @p nCols is much less than numNonZeros( @p row ).
   * @tparam AtomicPolicy the policy to use when adding to the values.
   * @param row The row to access.
   * @param cols The columns to add to, must be sorted, unique and of length @p nCols.
   * @param vals The values to add, of length @p nCols.
   * @param nCols The number of columns to add to.
   */
  template< typename AtomicPolicy >
  LVARRAY_HOST_DEVICE inline
  void addToRowBinarySearchUnsorted( INDEX_TYPE const row,
                                     COL_TYPE const * const LVARRAY_RESTRICT cols,
                                     T const * const LVARRAY_RESTRICT vals,
                                     INDEX_TYPE const nCols ) const
  {
    INDEX_TYPE const nnz = numNonZeros( row );
    COL_TYPE const * const columns = getColumns( row );
    T * const entries = getEntries( row );

    for( INDEX_TYPE_NC i = 0; i < nCols; ++i )
    {
      INDEX_TYPE const pos = sortedArrayManipulation::find( columns, nnz, cols[ i ] );
      LVARRAY_ASSERT_GT( nnz, pos );
      LVARRAY_ASSERT_EQ( columns[ pos ], cols[ i ] );

      atomicAdd( AtomicPolicy{}, entries + pos, vals[ i ] );
    }
  }

  /**
   * @brief Move this SparsityPattern to the given memory space and touch the values, sizes and offsets.
   * @param space the memory space to move to.
   * @param touch If true touch the values, sizes and offsets in the new space.
   * @note  When moving to the GPU since the offsets can't be modified on device they are not touched.
   */
  void move( chai::ExecutionSpace const space, bool const touch=true ) const
  {
    ParentClass::move( space, touch );
    m_entries.move( space, touch );
  }


protected:

  /**
   * @brief Default constructor. Made protected since every CRSMatrixView should
   *        either be the base of a CRSMatrix or copied from another CRSMatrixView.
   */
  CRSMatrixView():
    ParentClass(),
    m_entries( true )
  {}

  /**
   * @tparam U The type of the owning object.
   * @brief Set the name to be displayed whenever the underlying Buffer's user call back is called.
   * @param name the name to display.
   */
  template< typename U >
  void setName( std::string const & name )
  {
    ParentClass::template setName< U >( name );
    m_entries.template setName< U >( name + "/entries" );
  }

  // Aliasing protected members of SparsityPatternView.
  using ParentClass::m_numCols;
  using ParentClass::m_offsets;
  using ParentClass::m_sizes;
  using ParentClass::m_values;

  /// Holds the entries of the matrix, of length numNonZeros().
  NewChaiBuffer< T > m_entries;

private:

  /**
   * @class CallBacks
   * @brief This class provides the callbacks for the sortedArrayManipulation routines.
   */
  class CallBacks
  {
public:

    /**
     * @brief Constructor.
     * @param crsMV the CRSMatrixView this CallBacks is associated with.
     * @param row the row this CallBacks is associated with.
     * @param entriesToInsert pointer to the entries to insert.
     */
    LVARRAY_HOST_DEVICE inline
    CallBacks( CRSMatrixView< T, COL_TYPE, INDEX_TYPE > const & crsMV,
               INDEX_TYPE const row, T const * const entriesToInsert ):
      m_crsMV( crsMV ),
      m_row( row ),
      m_rowNNZ( crsMV.numNonZeros( row ) ),
      m_rowCapacity( crsMV.nonZeroCapacity( row ) ),
      m_entries( crsMV.getEntries( row ) ),
      m_entriesToInsert( entriesToInsert )
    {}

    /**
     * @brief Callback signaling that the size of the row has increased.
     * @param curPtr the current pointer to the array.
     * @param nToAdd the increase in the size.
     * @note This method doesn't actually change the size, it just checks that the new
     *       size doesn't exceed the capacity since the CRSMatrixView can't do allocation.
     * @return a pointer to the rows columns.
     */
    LVARRAY_HOST_DEVICE inline
    COL_TYPE * incrementSize( COL_TYPE * const LVARRAY_UNUSED_ARG( curPtr ),
                              INDEX_TYPE const nToAdd ) const
    {
#ifdef USE_ARRAY_BOUNDS_CHECK
      LVARRAY_ERROR_IF_GT_MSG( m_rowNNZ + nToAdd, m_rowCapacity, "CRSMatrixView cannot do reallocation." );
#else
      LVARRAY_DEBUG_VAR( nToAdd );
#endif
      return m_crsMV.getSetValues( m_row );
    }

    /**
     * @brief Used with sortedArrayManipulation::insert routine this callback signals
     *        that the column was inserted at the given position. This means we also need to insert
     *        the entry at the same position.
     * @param insertPos the position the column was inserted at.
     */
    LVARRAY_HOST_DEVICE inline
    void insert( INDEX_TYPE const insertPos ) const
    { arrayManipulation::insert( m_entries, m_rowNNZ, insertPos, m_entriesToInsert[0] ); }

    /**
     * @brief Used with the sortedArrayManipulation::insertSorted routine this callback
     *        signals that the given position was set to the column at the other position.
     *        This means we need to perform the same operation on the entries.
     * @param pos the position that was set.
     * @param colPos the position of the column.
     */
    DISABLE_HD_WARNING
    LVARRAY_HOST_DEVICE inline
    void set( INDEX_TYPE const pos, INDEX_TYPE const colPos ) const
    { new (&m_entries[pos]) T( m_entriesToInsert[colPos] ); }

    /**
     * @brief Used with the sortedArrayManipulation::insertSorted routine this callback
     *        signals that the given column was inserted at the given position. Further information
     *        is provided in order to make the insertion efficient. This means that we need to perform
     *        the same operation on the entries.
     * @param nLeftToInsert the number of insertions that occur after this one.
     * @param colPos the position of the column that was inserted.
     * @param insertPos the position the column was inserted at.
     * @param prevPos the position the previous column was inserted at or m_rowNNZ
     *             if it is the first insertion.
     */
    DISABLE_HD_WARNING
    LVARRAY_HOST_DEVICE inline
    void insert( INDEX_TYPE const nLeftToInsert,
                 INDEX_TYPE const colPos,
                 INDEX_TYPE const pos,
                 INDEX_TYPE const prevPos ) const
    {
      arrayManipulation::shiftUp( m_entries, prevPos, pos, nLeftToInsert );
      new (&m_entries[pos + nLeftToInsert - 1]) T( m_entriesToInsert[colPos] );
    }

    /**
     * @brief Used with sortedArrayManipulation::remove routine this callback signals
     *        that the column was removed from the given position. This means we also need to remove
     *        the entry from the same position.
     * @param removePos the position the column was removed from.
     */
    LVARRAY_HOST_DEVICE inline
    void remove( INDEX_TYPE_NC removePos ) const
    { arrayManipulation::erase( m_entries, m_rowNNZ, removePos ); }

    /**
     * @brief Used with the sortedArrayManipulation::removeSorted routine this callback
     *        signals that the given column was removed from the given position. Further information
     *        is provided in order to make the removal efficient. This means that we need to perform
     *        the same operation on the entries.
     * @param nRemoved the number of columns removed, starts at 1.
     * @param curPos the position the column was removed at.
     * @param nextPos the position the next column will be removed at or m_rowNNZ
     *             if this was the last column removed.
     */
    LVARRAY_HOST_DEVICE inline
    void remove( INDEX_TYPE const nRemoved,
                 INDEX_TYPE const curPos,
                 INDEX_TYPE const nextPos ) const
    { arrayManipulation::shiftDown( m_entries, nextPos, curPos + 1, nRemoved ); }

private:
    /// A reference to the associated CRSMatrixView.
    CRSMatrixView< T, COL_TYPE, INDEX_TYPE > const & m_crsMV;

    /// The associated row.
    INDEX_TYPE const m_row;

    /// The number of non zero entries in the row.
    INDEX_TYPE const m_rowNNZ;

    /// The non zero capacity of the row.
    INDEX_TYPE const m_rowCapacity;

    /// A pointer to the entries in the row.
    T * const m_entries;

    /// A pointer to the entries to insert.
    T const * const m_entriesToInsert;
  };

};

} /* namespace LvArray */

#endif /* CRSMATRIXVIEW_HPP_ */
