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
public:
  static_assert( !std::is_const< T >::value ||
                 (std::is_const< COL_TYPE >::value && std::is_const< INDEX_TYPE >::value),
                 "When T is const COL_TYPE and INDEX_TYPE must also be const." );

  using ParentClass = SparsityPatternView< COL_TYPE, INDEX_TYPE >;

  // Aliasing public typedefs of SparsityPatternView.
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
   * @param [in] src the CRSMatrixView to be copied.
   */
  CRSMatrixView( CRSMatrixView const & src ) = default;

  /**
   * @brief Default move constructor.
   * @param [in/out] src the CRSMatrixView to be moved from.
   */
  inline
  CRSMatrixView( CRSMatrixView && ) = default;

  /**
   * @brief User defined conversion to convert to CRSMatrixView<T, COL_TYPE const, INDEX_TYPE const>.
   *        This prevents you from inserting or removing but still allows modification of the matrix entries.
   */
  template< class U=T, class CTYPE=COL_TYPE >
  LVARRAY_HOST_DEVICE constexpr inline
  operator typename std::enable_if< !std::is_const< U >::value && !std::is_const< CTYPE >::value,
                                    CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > const & >::type
    () const LVARRAY_RESTRICT_THIS
  { return reinterpret_cast< CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > const & >(*this); }

  /**
   * @brief Method to convert to CRSMatrixView<T, COL_TYPE const, INDEX_TYPE const>. Use this method
   *        when the above UDC isn't invoked, this usually occurs with template argument deduction.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const > const & toViewSemiConst() const LVARRAY_RESTRICT_THIS
  { return *this; }

  /**
   * @brief User defined conversion to convert to CRSMatrixView<T const, COL_TYPE const, INDEX_TYPE const>.
   *        This prevents you from inserting, removing or modifying the matrix entries.
   */
  template< class U=T >
  LVARRAY_HOST_DEVICE constexpr inline
  operator typename std::enable_if< !std::is_const< U >::value, CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const & >::type
    () const LVARRAY_RESTRICT_THIS
  { return reinterpret_cast< CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const & >(*this); }

  /**
   * @brief Method to convert to CRSMatrixView<T const, COL_TYPE const, INDEX_TYPE const>. Use this method
   *        when the above UDC isn't invoked, this usually occurs with template argument deduction.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const > const & toViewConst() const LVARRAY_RESTRICT_THIS
  { return *this; }

  /**
   * @brief Method to convert to SparsityPatternView<COL_TYPE const, INDEX_TYPE const>.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  SparsityPatternView< COL_TYPE const, INDEX_TYPE const > const & toSparsityPatternView() const
  { return reinterpret_cast< SparsityPatternView< COL_TYPE const, INDEX_TYPE const > const & >(*this); }

  /**
   * @brief Default copy assignment operator, this does a shallow copy.
   * @param [in] src the CRSMatrixView to be copied from.
   */
  inline
  CRSMatrixView & operator=( CRSMatrixView const & src ) = default;

  /**
   * @brief Return an ArraySlice1d (pointer) to the matrix entries of the given row.
   *        This array has length numNonZeros(row).
   * @param [in] row the row to access.
   */
  LVARRAY_HOST_DEVICE inline
  ArraySlice< T, 1, 0, INDEX_TYPE_NC > getEntries( INDEX_TYPE const row ) const LVARRAY_RESTRICT_THIS
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( row );
    return ArraySlice< T, 1, 0, INDEX_TYPE_NC >( m_entries.data() + m_offsets[row], &m_sizes[row], nullptr );
  }

  /**
   * @brief Insert a non-zero entry at the given position.
   * @param [in] row the row to insert in.
   * @param [in] col the column to insert at.
   * @param [in] entry the entry to insert.
   * @return True iff the entry was inserted (the entry was zero before).
   *
   * @note Since the CRSMatrixView can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entry.
   *       If USE_ARRAY_BOUNDS_CHECK is defined a lack of space will result in an error,
   *       otherwise the values in the subsequent row will be overwritten.
   */
  LVARRAY_HOST_DEVICE inline
  bool insertNonZero( INDEX_TYPE const row, COL_TYPE const col, T const & entry ) const LVARRAY_RESTRICT_THIS
  { return ParentClass::insertIntoSetImpl( row, col, CallBacks( *this, row, &entry ) ); }

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
   * @note Since the CRSMatrixView can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entries.
   *       If USE_ARRAY_BOUNDS_CHECK is defined a lack of space will result in an error,
   *       otherwise the values in the subsequent row will be overwritten.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC insertNonZeros( INDEX_TYPE const row,
                                COL_TYPE const * const LVARRAY_RESTRICT cols,
                                T const * const LVARRAY_RESTRICT entriesToInsert,
                                INDEX_TYPE const ncols ) const LVARRAY_RESTRICT_THIS
  { return insertNonZerosImpl( row, cols, entriesToInsert, ncols, *this ); }

  /**
   * @brief Insert a non-zero entries into the given row.
   * @param [in] row the row to insert into.
   * @param [in] cols the columns to insert at, of length ncols. Must be sorted.
   * @param [in] entriesToInsert the entries to insert, of length ncols.
   * @param [in] ncols the number of columns/entries to insert.
   * @return The number of entries inserted.
   *
   * @note Since the CRSMatrixView can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entries.
   *       If USE_ARRAY_BOUNDS_CHECK is defined a lack of space will result in an error,
   *       otherwise the values in the subsequent row will be overwritten.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC insertNonZerosSorted( INDEX_TYPE const row,
                                      COL_TYPE const * const LVARRAY_RESTRICT cols,
                                      T const * const LVARRAY_RESTRICT entriesToInsert,
                                      INDEX_TYPE const ncols ) const LVARRAY_RESTRICT_THIS
  { return ParentClass::insertSortedIntoSetImpl( row, cols, ncols, CallBacks( *this, row, entriesToInsert ) ); }

  /**
   * @brief Remove a non-zero entry at the given position.
   * @param [in] row the row to remove from.
   * @param [in] col the column to remove.
   * @return True iff the entry was removed (the entry was non-zero before).
   */
  LVARRAY_HOST_DEVICE inline
  bool removeNonZero( INDEX_TYPE const row, COL_TYPE const col ) const LVARRAY_RESTRICT_THIS
  { return ParentClass::removeFromSetImpl( row, col, CallBacks( *this, row, nullptr )); }

  /**
   * @brief Remove non-zero entries from the given row.
   * @param [in] row the row to remove from.
   * @param [in] cols the columns to remove, of length ncols.
   * @param [in] ncols the number of columns to remove.
   * @return True iff the entry was removed (the entry was non-zero before).
   *
   * @note If possible sort cols first by calling sortedArrayManipulation::makeSorted(cols, cols + ncols)
   *       and then call removeNonZerosSorted, this will be substantially faster.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC removeNonZeros( INDEX_TYPE const row,
                                COL_TYPE const * const LVARRAY_RESTRICT cols,
                                INDEX_TYPE const ncols ) const LVARRAY_RESTRICT_THIS
  {
    T * const entries = getEntries( row );
    INDEX_TYPE const rowNNZ = numNonZeros( row );
    INDEX_TYPE const nRemoved = ParentClass::removeFromSetImpl( row, cols, ncols, CallBacks( *this, row, nullptr ));

    for( INDEX_TYPE_NC i = rowNNZ - nRemoved; i < rowNNZ; ++i )
    {
      entries[i].~T();
    }

    return nRemoved;
  }

  /**
   * @brief Remove non-zero entries from the given row.
   * @param [in] row the row to remove from.
   * @param [in] cols the columns to remove, of length ncols. Must be sorted.
   * @param [in] ncols the number of columns to remove.
   * @return True iff the entry was removed (the entry was non-zero before).
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC removeNonZerosSorted( INDEX_TYPE const row,
                                      COL_TYPE const * const LVARRAY_RESTRICT cols,
                                      INDEX_TYPE const ncols ) const LVARRAY_RESTRICT_THIS
  {
    T * const entries = getEntries( row );
    INDEX_TYPE const rowNNZ = numNonZeros( row );
    INDEX_TYPE const nRemoved = ParentClass::removeSortedFromSetImpl( row, cols, ncols, CallBacks( *this, row, nullptr ));

    for( INDEX_TYPE_NC i = rowNNZ - nRemoved; i < rowNNZ; ++i )
    {
      entries[i].~T();
    }

    return nRemoved;
  }

  /**
   * @brief Set all the values in the matrix to the given value.
   * @param [in] value the value to set values in the matrix to.
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

  /**
   * @brief Add to the given entries, the entries must already exist in the matrix.
   *        The columns must be sorted.
   * @tparam AtomicPolicy the policy to use when adding to the values.
   * @param [in] row the row to access.
   * @param [in] cols the columns to add to, must be sorted, unique and of length nCols.
   * @param [in] vals the values to add, of length nCols.
   * @param [in] nCols the number of columns to add to.
   * TODO: Use benchmarks of addToRowBinarySearch and addToRowLinearSearch
   *       to develop a better heuristic.
   */
  template< typename AtomicPolicy=RAJA::seq_atomic >
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

  /**
   * @brief Add to the given entries, the entries must already exist in the matrix.
   *        The columns must be sorted and this method uses a binary search to find
   *        the entries in the row to add to. This makes the method O( nCols * log( numNonZeros( row ) ) )
   *        and is therefore best to use when nCols is much less than numNonZeros( row ).
   * @tparam AtomicPolicy the policy to use when adding to the values.
   * @param [in] row the row to access.
   * @param [in] cols the columns to add to, must be sorted, unique and of length nCols.
   * @param [in] vals the values to add, of length nCols.
   * @param [in] nCols the number of columns to add to.
   */
  template< typename AtomicPolicy=RAJA::seq_atomic >
  LVARRAY_HOST_DEVICE inline
  void addToRowBinarySearch( INDEX_TYPE const row,
                             COL_TYPE const * const LVARRAY_RESTRICT cols,
                             T const * const LVARRAY_RESTRICT vals,
                             INDEX_TYPE const nCols ) const
  {
    LVARRAY_ASSERT( sortedArrayManipulation::isSorted( cols, nCols ) );
    LVARRAY_ASSERT( sortedArrayManipulation::allUnique( cols, nCols ) );

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
   *        The columns must be sorted and this method uses a linear search to find
   *        the entries in the row to add to. This makes the method O( numNonZeros( row ) )
   *        and is therefore best to use when nCols is similar to numNonZeros( row ).
   * @tparam AtomicPolicy the policy to use when adding to the values.
   * @param [in] row the row to access.
   * @param [in] cols the columns to add to, must be sorted, unique and of length nCols.
   * @param [in] vals the values to add, of length nCols.
   * @param [in] nCols the number of columns to add to.
   */
  template< typename AtomicPolicy=RAJA::seq_atomic >
  LVARRAY_HOST_DEVICE inline
  void addToRowLinearSearch( INDEX_TYPE const row,
                             COL_TYPE const * const LVARRAY_RESTRICT cols,
                             T const * const LVARRAY_RESTRICT vals,
                             INDEX_TYPE const nCols ) const
  {
    LVARRAY_ASSERT( sortedArrayManipulation::isSorted( cols, nCols ) );
    LVARRAY_ASSERT( sortedArrayManipulation::allUnique( cols, nCols ) );

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
   * @brief Move this SparsityPattern to the given memory space and touch the values, sizes and offsets.
   * @param [in] space the memory space to move to.
   * @param [in] touch If true touch the values, sizes and offsets in the new space.
   * @note  When moving to the GPU since the offsets can't be modified on device they are not touched.
   */
  void move( chai::ExecutionSpace const space, bool const touch=true )
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
   * @brief Helper function to insert non zeros into the given row.
   * @tparam DERIVED type of the class this method is called from, must be a derived class of CRSMatrixView.
   * @param [in] row the row to insert into.
   * @param [in] cols the columns to insert.
   * @param [in] entriesToInsert the entries to insert.
   * @param [in] ncols the number of columns to insert.
   * @param [in/out] obj the derived object this method is being called from.
   * @return The number of entries inserted.
   */
  DISABLE_HD_WARNING
  template< class DERIVED >
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC insertNonZerosImpl( INDEX_TYPE const row,
                                    COL_TYPE const * const cols,
                                    T const * const entriesToInsert,
                                    INDEX_TYPE const ncols,
                                    DERIVED & obj ) const LVARRAY_RESTRICT_THIS
  {
    static_assert( std::is_base_of< CRSMatrixView, DERIVED >::value, "DERIVED must be derived from CRSMatrixView!" );

    constexpr int LOCAL_SIZE = 16;
    COL_TYPE localColumnBuffer[LOCAL_SIZE];
    T localEntryBuffer[LOCAL_SIZE];

    COL_TYPE * const columnBuffer = sortedArrayManipulation::createTemporaryBuffer( cols, ncols, localColumnBuffer );
    T * const entryBuffer = sortedArrayManipulation::createTemporaryBuffer( entriesToInsert, ncols, localEntryBuffer );
    sortedArrayManipulation::dualSort( columnBuffer, columnBuffer + ncols, entryBuffer );

    INDEX_TYPE const nInserted = obj.insertNonZerosSorted( row, columnBuffer, entryBuffer, ncols );

    sortedArrayManipulation::freeTemporaryBuffer( columnBuffer, ncols, localColumnBuffer );
    sortedArrayManipulation::freeTemporaryBuffer( entryBuffer, ncols, localEntryBuffer );
    return nInserted;
  }

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

  // Holds the entries of the matrix, of length numNonZeros().
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
     * @param [in/out] crsMV the CRSMatrixView this CallBacks is associated with.
     * @param [in] row the row this CallBacks is associated with.
     * @param [in] entriesToInsert pointer to the entries to insert.
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
     * @param [in] curPtr the current pointer to the array.
     * @param [in] nToAdd the increase in the size.
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
     * @param [in] insertPos the position the column was inserted at.
     */
    LVARRAY_HOST_DEVICE inline
    void insert( INDEX_TYPE const insertPos ) const
    { arrayManipulation::insert( m_entries, m_rowNNZ, insertPos, m_entriesToInsert[0] ); }

    /**
     * @brief Used with the sortedArrayManipulation::insertSorted routine this callback
     *        signals that the given position was set to the column at the other position.
     *        This means we need to perform the same operation on the entries.
     * @param [in] pos the position that was set.
     * @param [in] colPos the position of the column.
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
     * @param [in] nLeftToInsert the number of insertions that occur after this one.
     * @param [in] colPos the position of the column that was inserted.
     * @param [in] insertPos the position the column was inserted at.
     * @param [in] prevPos the position the previous column was inserted at or m_rowNNZ
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
     * @param [in] removePos the position the column was removed from.
     */
    LVARRAY_HOST_DEVICE inline
    void remove( INDEX_TYPE_NC removePos ) const
    { arrayManipulation::erase( m_entries, m_rowNNZ, removePos ); }

    /**
     * @brief Used with the sortedArrayManipulation::removeSorted routine this callback
     *        signals that the given column was removed from the given position. Further information
     *        is provided in order to make the removal efficient. This means that we need to perform
     *        the same operation on the entries.
     * @param [in] nRemoved the number of columns removed, starts at 1.
     * @param [in] curPos the position the column was removed at.
     * @param [in] nextPos the position the next column will be removed at or m_rowNNZ
     *             if this was the last column removed.
     */
    LVARRAY_HOST_DEVICE inline
    void remove( INDEX_TYPE const nRemoved,
                 INDEX_TYPE const curPos,
                 INDEX_TYPE const nextPos ) const
    { arrayManipulation::shiftDown( m_entries, nextPos, curPos + 1, nRemoved ); }

private:
    CRSMatrixView< T, COL_TYPE, INDEX_TYPE > const & m_crsMV;
    INDEX_TYPE const m_row;
    INDEX_TYPE const m_rowNNZ;
    INDEX_TYPE const m_rowCapacity;
    T * const m_entries;
    T const * const m_entriesToInsert;
  };

};

} /* namespace LvArray */

#endif /* CRSMATRIXVIEW_HPP_ */
