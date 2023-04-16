/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file CRSMatrixView.hpp
 * @brief Contains the implementation of LvArray::CRSMatrixView.
 */

#pragma once

#include "SparsityPatternView.hpp"
#include "arrayManipulation.hpp"
#include "ArraySlice.hpp"
#include "umpireInterface.hpp"
#include "math.hpp"

namespace LvArray
{

namespace internal
{

/**
 * @brief Wrapper around RAJA::atomicAdd.
 * @tparam POLICY The RAJA atomic policy to use.
 * @tparam T The type of the value to add to.
 * @param acc A pointer to the value to add to.
 * @param val The value to add.
 */
template< typename POLICY, typename T >
LVARRAY_HOST_DEVICE inline
void atomicAdd( POLICY, T * const acc, T const & val )
{ RAJA::atomicAdd< POLICY >( acc, val ); }

/**
 * @brief Wrapper around RAJA::atomicAdd.
 * @tparam T The type of the value to add to.
 * @param acc A pointer to the value to add to.
 * @param val The value to add.
 * @note An overload for RAJA::seq_atomic that doesn't use
 *   a volatile pointer. Necessary for non-primative types.
 */
DISABLE_HD_WARNING
template< typename T >
LVARRAY_HOST_DEVICE constexpr inline
void atomicAdd( RAJA::seq_atomic, T * acc, T const & val )
{ *acc += val; }

} // namespace internal

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
template< typename T,
          typename COL_TYPE,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE
          >
class CRSMatrixView : protected SparsityPatternView< COL_TYPE, INDEX_TYPE, BUFFER_TYPE >
{
  using ColTypeNC = std::remove_const_t< COL_TYPE >;

  /// An alias for the parent class.
  using ParentClass = SparsityPatternView< COL_TYPE, INDEX_TYPE, BUFFER_TYPE >;

  /// An alias for the non const index type.
  using typename ParentClass::INDEX_TYPE_NC;

  using typename ParentClass::SIZE_TYPE;

public:
  static_assert( !std::is_const< T >::value ||
                 (std::is_const< COL_TYPE >::value && std::is_const< INDEX_TYPE >::value),
                 "When T is const COL_TYPE and INDEX_TYPE must also be const." );

  /// The type of the entries in the matrix.
  using EntryType = T;
  using typename ParentClass::ColType;
  using typename ParentClass::IndexType;

  /**
   * @name Constructors, destructor and assignment operators
   */
  ///@{

  /**
   * @brief A constructor to create an uninitialized CRSMatrixView.
   * @note An uninitialized CRSMatrixView should not be used until it is assigned to.
   */
  LVARRAY_HOST_DEVICE_HIP
  CRSMatrixView() = default;

  /**
   * @brief Default copy constructor.
   */
  LVARRAY_HOST_DEVICE_HIP
  CRSMatrixView( CRSMatrixView const & ) = default;

  /**
   * @brief Default move constructor.
   */
  LVARRAY_HOST_DEVICE_HIP
  CRSMatrixView( CRSMatrixView && ) = default;

  /**
   * @brief Construct a new CRSMatrixView from the given buffers.
   * @param nRows The number of rows.
   * @param nCols The number of columns
   * @param offsets The offsets buffer, of size @p nRows + 1.
   * @param nnz The buffer containing the number of non zeros in each row, of size @p nRows.
   * @param columns The columns buffer, of size @p offsets[ nRows ].
   * @param entries The entries buffer, of size @p offsets[ nRows ].
   */
  LVARRAY_HOST_DEVICE constexpr inline
  CRSMatrixView( INDEX_TYPE const nRows,
                 ColTypeNC const nCols,
                 BUFFER_TYPE< INDEX_TYPE > const & offsets,
                 BUFFER_TYPE< SIZE_TYPE > const & nnz,
                 BUFFER_TYPE< COL_TYPE > const & columns,
                 BUFFER_TYPE< T > const & entries ):
    ParentClass( nRows, nCols, offsets, nnz, columns ),
    m_entries( entries )
  {}

  /**
   * @brief Default copy assignment operator.
   * @return *this.
   */
  inline
  CRSMatrixView & operator=( CRSMatrixView const & ) = default;

  /**
   * @brief Default move assignment operator.
   * @return *this.
   */
  inline
  CRSMatrixView & operator=( CRSMatrixView && ) = default;

  /**
   * @name CRSMatrixView and SparsityPatternView creation methods
   */
  ///@{

  /**
   * @return A new CRSMatrixView< T, COL_TYPE, INDEX_TYPE const >.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  CRSMatrixView< T, COL_TYPE, INDEX_TYPE const, BUFFER_TYPE >
  toView() const
  {
    return CRSMatrixView< T, COL_TYPE, INDEX_TYPE const, BUFFER_TYPE >( numRows(),
                                                                        numColumns(),
                                                                        this->m_offsets,
                                                                        this->m_sizes,
                                                                        this->m_values,
                                                                        this->m_entries );
  }

  /**
   * @return A new CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const >.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >
  toViewConstSizes() const
  {
    return CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >( numRows(),
                                                                              numColumns(),
                                                                              this->m_offsets,
                                                                              this->m_sizes,
                                                                              this->m_values,
                                                                              this->m_entries );
  }

  /**
   * @return A new CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const >.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >
  toViewConst() const
  {
    return CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >( numRows(),
                                                                                    numColumns(),
                                                                                    this->m_offsets,
                                                                                    this->m_sizes,
                                                                                    this->m_values,
                                                                                    this->m_entries );
  }

  /**
   * @return A reference to *this reinterpreted as a SparsityPatternView< COL_TYPE const, INDEX_TYPE const >.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  SparsityPatternView< COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >
  toSparsityPatternView() const &
  {
    return SparsityPatternView< COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >( numRows(),
                                                                                 numColumns(),
                                                                                 this->m_offsets,
                                                                                 this->m_sizes,
                                                                                 this->m_values );
  }

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

  /**
   * @return Return an ArraySlice1d to the matrix entries of the given row.
   * @param row The row to access.
   */
  LVARRAY_HOST_DEVICE inline
  ArraySlice< T, 1, 0, INDEX_TYPE_NC > getEntries( INDEX_TYPE const row ) const
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( row );
    return ArraySlice< T, 1, 0, INDEX_TYPE_NC >( m_entries.data() + this->m_offsets[ row ],
                                                 &this->m_sizes[ row ],
                                                 nullptr );
  }

  /**
   * @return A pointer to the values of m_entires(data) of the matrix.
   */
  LVARRAY_HOST_DEVICE inline
  T const * getEntries() const
  {
    return m_entries.data();
  }


  template< typename POLICY >
  inline
  bool NaNCheck( ) const
  {
    RAJA::ReduceMax< typename RAJAHelper< POLICY >::ReducePolicy, int > anyNaN( 0 );
    CRSMatrixView< T const, COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE > const view = toViewConst();
    RAJA::forall< POLICY >( RAJA::TypedRangeSegment< INDEX_TYPE >( 0, numRows() ),
                            [view, anyNaN] LVARRAY_HOST_DEVICE ( INDEX_TYPE const row )
      {
        forValuesInSlice( view[row], [&anyNaN] ( T const & value )
      {
        anyNaN.max( std::isnan( value ) ? 1 : 0 );
      } );
      } );
    return anyNaN.get() == 1;
  }


  using ParentClass::getColumns;
  using ParentClass::getOffsets;
  using ParentClass::getSizes;

  ///@}

  /**
   * @name Methods that insert or remove entries from a row.
   */
  ///@{

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
  bool insertNonZero( INDEX_TYPE const row, ColTypeNC const col, T const & entry ) const
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
                                ColTypeNC const * const LVARRAY_RESTRICT cols,
                                T const * const LVARRAY_RESTRICT entriesToInsert,
                                ColTypeNC const ncols ) const
  { return ParentClass::insertIntoSetImpl( row, cols, cols + ncols, CallBacks( *this, row, entriesToInsert ) ); }

  /**
   * @brief Remove a non-zero entry at the given position.
   * @param row The row to remove from.
   * @param col The column to remove.
   * @return True iff the entry was removed (the entry was non-zero before).
   */
  LVARRAY_HOST_DEVICE inline
  bool removeNonZero( INDEX_TYPE const row, ColTypeNC const col ) const
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
                                ColTypeNC const * const LVARRAY_RESTRICT cols,
                                ColTypeNC const ncols ) const
  {
    T * const entries = getEntries( row );
    INDEX_TYPE const rowNNZ = numNonZeros( row );
    INDEX_TYPE const nRemoved = ParentClass::removeFromSetImpl( row, cols, cols + ncols, CallBacks( *this, row, nullptr ) );

    arrayManipulation::destroy( entries + rowNNZ - nRemoved, nRemoved );

    return nRemoved;
  }

  ///@}

  /**
   * @name Methods that modify the entires of the matrix
   */
  ///@{

  /**
   * @brief Set all the entries in the matrix to the given value.
   * @tparam POLICY The kernel launch policy to use.
   * @param value The value to set entries in the matrix to.
   */
  template< typename POLICY >
  inline void setValues( T const & value ) const
  {
    CRSMatrixView< T, COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE > const view = toViewConstSizes();
    RAJA::forall< POLICY >( RAJA::TypedRangeSegment< INDEX_TYPE >( 0, numRows() ),
                            [view, value] LVARRAY_HOST_DEVICE ( INDEX_TYPE const row )
      {
        INDEX_TYPE const nnz = view.numNonZeros( row );
        ArraySlice< T, 1, 0, INDEX_TYPE_NC > const entries = view.getEntries( row );

        for( INDEX_TYPE_NC i = 0; i < nnz; ++i )
        { entries[ i ] = value; }
      } );
  }

  /**
   * @brief Use memset to set all the values in the matrix to 0.
   * @details This is preferred over setValues< POLICY >( 0 ) for numeric types since it is much faster in most cases.
   *   If the buffer is allocated using Umpire then the Umpire ResouceManager is used, otherwise
   *   std::memset is used. This zeroes out all of the memory spanned by m_entries including values that are
   *   part of overallocated capacity so if you have a small matrix with a huge capacity it won't be as quick as
   *   you might like.
   * @note The memset occurs in the last space the matrix was used in and the matrix is moved and the entries are
   *   touched in that space.
   */
  inline void zero() const
  {
  #if !defined( LVARRAY_USE_UMPIRE )
    LVARRAY_ERROR_IF_NE_MSG( m_entries.getPreviousSpace(), MemorySpace::host, "Without Umpire only host memory is supported." );
  #endif

    if( m_entries.capacity() > 0 )
    {
      ParentClass::move( m_entries.getPreviousSpace(), false );
      m_entries.move( m_entries.getPreviousSpace(), true );

      size_t const numBytes = m_entries.capacity() * sizeof( T );

      umpireInterface::memset( m_entries.data(), 0, numBytes );

    }
  }

  /**
   * @brief Add to the given entries, the entries must already exist in the matrix.
   *   The columns must be sorted.
   * @tparam AtomicPolicy the policy to use when adding to the values.
   * @param row The row to access.
   * @param cols The columns to add to, must be sorted, unique and of length nCols.
   * @param vals The values to add, of length nCols.
   * @param nCols The number of columns to add to.
   * @pre The range [ @p cols, @p cols + @p ncols ) must be sorted and contain no duplicates.
   * TODO: Use benchmarks of addToRowBinarySearch and addToRowLinearSearch
   *   to develop a better heuristic.
   */
  template< typename AtomicPolicy >
  LVARRAY_HOST_DEVICE inline
  void addToRow( INDEX_TYPE const row,
                 ColTypeNC const * const LVARRAY_RESTRICT cols,
                 T const * const LVARRAY_RESTRICT vals,
                 ColTypeNC const nCols ) const
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
                             ColTypeNC const * const LVARRAY_RESTRICT cols,
                             T const * const LVARRAY_RESTRICT vals,
                             ColTypeNC const nCols ) const
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

      internal::atomicAdd( AtomicPolicy{}, entries + pos, vals[ i ] );
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
                             ColTypeNC const * const LVARRAY_RESTRICT cols,
                             T const * const LVARRAY_RESTRICT vals,
                             ColTypeNC const nCols ) const
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
      internal::atomicAdd( AtomicPolicy{}, entries + curPos, vals[ i ] );
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
   * @param cols The columns to add to, unsorted, of length @p nCols.
   * @param vals The values to add, of length @p nCols.
   * @param nCols The number of columns to add to.
   */
  template< typename AtomicPolicy >
  LVARRAY_HOST_DEVICE inline
  void addToRowBinarySearchUnsorted( INDEX_TYPE const row,
                                     ColTypeNC const * const LVARRAY_RESTRICT cols,
                                     T const * const LVARRAY_RESTRICT vals,
                                     ColTypeNC const nCols ) const
  {
    INDEX_TYPE const nnz = numNonZeros( row );
    COL_TYPE const * const columns = getColumns( row );
    T * const entries = getEntries( row );

    for( INDEX_TYPE_NC i = 0; i < nCols; ++i )
    {
      INDEX_TYPE const pos = sortedArrayManipulation::find( columns, nnz, cols[ i ] );
      LVARRAY_ASSERT_GT( nnz, pos );
      LVARRAY_ASSERT_EQ( columns[ pos ], cols[ i ] );

      internal::atomicAdd( AtomicPolicy{}, entries + pos, vals[ i ] );
    }
  }

  ///@}

  /**
   * @name Methods dealing with memory spaces
   */
  ///@{

  /**
   * @brief Move this matrix to the given memory space and touch the values, sizes and offsets.
   * @param space the memory space to move to.
   * @param touch If true touch the values, sizes and offsets in the new space.
   * @note  When moving to the GPU since the offsets can't be modified on device they are not touched.
   */
  void move( MemorySpace const space, bool const touch=true ) const
  {
    ParentClass::move( space, touch );
    m_entries.move( space, touch );
  }

  ///@}

protected:

  /**
   * @brief Protected constructor to be used by the CRSMatrix class.
   * @note The unused boolean parameter is to distinguish this from the default constructor.
   */
  CRSMatrixView( bool ):
    ParentClass( true ),
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

  /// Holds the entries of the matrix, of length numNonZeros().
  BUFFER_TYPE< T > m_entries;

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
    CallBacks( CRSMatrixView const & crsMV,
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
     *   size doesn't exceed the capacity since the CRSMatrixView can't do allocation.
     * @return a pointer to the rows columns.
     */
    LVARRAY_HOST_DEVICE inline
    ColTypeNC * incrementSize( ColTypeNC * const curPtr, INDEX_TYPE const nToAdd ) const
    {
      LVARRAY_UNUSED_VARIABLE( curPtr );
#ifdef LVARRAY_BOUNDS_CHECK
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
    { arrayManipulation::emplace( m_entries, m_rowNNZ, insertPos, m_entriesToInsert[0] ); }

    /**
     * @brief Used with the sortedArrayManipulation::insertSorted routine this callback
     *        signals that the given position was set to the column at the other position.
     *        This means we need to perform the same operation on the entries.
     * @param pos the position that was set.
     * @param colPos the position of the column.
     */
    DISABLE_HD_WARNING
    LVARRAY_HOST_DEVICE inline
    void set( INDEX_TYPE const pos, ColTypeNC const colPos ) const
    { new (&m_entries[ pos ]) T( m_entriesToInsert[ colPos ] ); }

    /**
     * @brief Used with the sortedArrayManipulation::insertSorted routine this callback signals that the
     *   given column was inserted at the given position. Further information is provided in order to make
     *   the insertion efficient. This means that we need to perform the same operation on the entries.
     * @param nLeftToInsert the number of insertions that occur after this one.
     * @param colPos the position of the column that was inserted.
     * @param pos The position the column was inserted at.
     * @param prevPos The position the previous column was inserted at or m_rowNNZ if it is the first insertion.
     */
    DISABLE_HD_WARNING
    LVARRAY_HOST_DEVICE inline
    void insert( INDEX_TYPE const nLeftToInsert,
                 ColTypeNC const colPos,
                 ColTypeNC const pos,
                 ColTypeNC const prevPos ) const
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
    void remove( ColTypeNC removePos ) const
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
                 ColTypeNC const curPos,
                 ColTypeNC const nextPos ) const
    { arrayManipulation::shiftDown( m_entries, nextPos, curPos + 1, nRemoved ); }

private:
    /// A reference to the associated CRSMatrixView.
    CRSMatrixView const & m_crsMV;

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
