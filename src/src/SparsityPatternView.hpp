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

#ifndef SPARSITYPATTERNVIEW_HPP_
#define SPARSITYPATTERNVIEW_HPP_

#include "ChaiVector.hpp"
#include "arrayManipulation.hpp"
#include "sortedArrayManipulation.hpp"
#include "ArraySlice.hpp"

#ifdef USE_ARRAY_BOUNDS_CHECK

#undef CONSTEXPRFUNC
#define CONSTEXPRFUNC

#define SPARSITYPATTERN_CHECK_BOUNDS( row ) \
  GEOS_ERROR_IF( !arrayManipulation::isPositive( row ) || row >= numRows(), \
                 "Bounds Check Failed: row=" << row << " numRows()=" << numRows())

#define SPARSITYPATTERN_CHECK_BOUNDS2( row, col ) \
  GEOS_ERROR_IF( !arrayManipulation::isPositive( row ) || row >= numRows() || \
                 !arrayManipulation::isPositive( col ) || col >= numColumns(), \
                 "Bounds Check Failed: row=" << row << " numRows()=" << numRows() \
                                             << " col=" << col << " numColumns=" << numColumns())

#else // USE_ARRAY_BOUNDS_CHECK

#define SPARSITYPATTERN_CHECK_BOUNDS( index )
#define SPARSITYPATTERN_CHECK_BOUNDS2( index0, index1 )

#endif // USE_ARRAY_BOUNDS_CHECK

namespace LvArray
{

/**
 * @class SparsityPatternView
 * @brief This class provides a view into a compressed row storage sparsity pattern.
 * @tparam COL_TYPE the integer used to enumerate the columns.
 * @tparam INDEX_TYPE the integer to use for indexing.
 *
 * @note When INDEX_TYPE is const m_offsets is not copied back from the device. INDEX_TYPE should always be const
 * since SparsityPatternView is not allowed to modify the offsets.
 * @note When COL_TYPE is const and INDEX_TYPE is const you cannot insert or remove from the View
 * and neither the offsets, sizes, or columns are copied back from the device.
 */
template <class COL_TYPE=unsigned int, class INDEX_TYPE=std::ptrdiff_t>
class SparsityPatternView
#ifdef USE_CHAI
  : public chai::CHAICopyable
#endif
{
public:
  static_assert( std::is_integral<COL_TYPE>::value, "COL_TYPE must be integral." );
  static_assert( std::is_integral<INDEX_TYPE>::value, "INDEX_TYPE must be integral." );
  static_assert( std::numeric_limits<INDEX_TYPE>::max() >= std::numeric_limits<COL_TYPE>::max(),
                 "INDEX_TYPE must be able to hold values at least as large as COL_TYPE." );

  using INDEX_TYPE_NC = typename std::remove_const<INDEX_TYPE>::type;

  /**
   * @brief Default copy constructor. Performs a shallow copy and calls the
   * chai::ManagedArray copy constructor.
   * @param [in] src the SparsityPatternView to be copied.
   */
  inline
  SparsityPatternView( SparsityPatternView const & src ) = default;

  /**
   * @brief Default move constructor.
   * @param [in/out] src the SparsityPatternView to be moved from.
   */
  inline
  SparsityPatternView( SparsityPatternView && src ) = default;

  /**
   * @brief User defined conversion to move from T to T const.
   */
  template<class CTYPE=COL_TYPE>
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  operator typename std::enable_if<!std::is_const<CTYPE>::value,
                                   SparsityPatternView<COL_TYPE const, INDEX_TYPE_NC const> const &>::type
    () const restrict_this
  { return reinterpret_cast<SparsityPatternView<COL_TYPE const, INDEX_TYPE_NC const> const &>(*this); }

  /**
   * @brief Method to convert T to T const. Use this method when the above UDC
   * isn't invoked, this usually occurs with template argument deduction.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  SparsityPatternView<COL_TYPE const, INDEX_TYPE_NC const> const & toViewC() const restrict_this
  { return *this; }

  /**
   * @brief Default copy assignment operator, this does a shallow copy.
   * @param [in] src the SparsityPatternView to be copied from.
   */
  inline
  SparsityPatternView & operator=( SparsityPatternView const & src ) = default;

  /**
   * @brief Default move assignment operator, this does a shallow copy.
   * @param [in/out] src the SparsityPatternView to be moved from.
   */
  inline
  SparsityPatternView & operator=( SparsityPatternView && src ) = default;

  /**
   * @brief Return the number of rows in the matrix.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE_NC numRows() const restrict_this
  { return m_sizes.size(); }

  /**
   * @brief Return the number of columns in the matrix.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE_NC numColumns() const restrict_this
  { return m_num_columns; }

  /**
   * @brief Return the total number of non zero entries in the matrix.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC numNonZeros() const restrict_this
  {
    INDEX_TYPE_NC nnz = 0;
    for( INDEX_TYPE_NC row = 0 ; row < numRows() ; ++row )
    {
      nnz += numNonZeros( row );
    }

    return nnz;
  }

  /**
   * @brief Return the total number of non zero entries in the given row.
   * @param [in] row the row to query.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE_NC numNonZeros( INDEX_TYPE_NC const row ) const restrict_this
  {
    SPARSITYPATTERN_CHECK_BOUNDS( row );
    return m_sizes[row];
  }

  /**
   * @brief Return the total number of non zero entries able to be stored without a reallocation.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE_NC nonZeroCapacity() const restrict_this
  { return m_columns.capacity(); }

  /**
   * @brief Return the total number of non zero entries able to be stored in a given row without shifting
   * subsequent rows and possibly reallocating.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE_NC nonZeroCapacity( INDEX_TYPE_NC const row ) const restrict_this
  {
    SPARSITYPATTERN_CHECK_BOUNDS( row );
    return m_offsets[row + 1] - m_offsets[row];
  }

  /**
   * @brief Return true iff the matrix is all zeros.
   */
  LVARRAY_HOST_DEVICE inline
  bool empty() const restrict_this
  { return numNonZeros() == 0; }

  /**
   * @brief Return true iff the given row is all zeros.
   * @param [in] row the row to query.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  bool empty( INDEX_TYPE_NC const row ) const restrict_this
  {
    SPARSITYPATTERN_CHECK_BOUNDS( row );
    return numNonZeros( row ) == 0;
  }

  /**
   * @brief Return true iff the given entry is zero.
   * @param [in] row the row to query.
   * @param [in] col the col to query.
   */
  LVARRAY_HOST_DEVICE inline
  bool empty( INDEX_TYPE_NC const row, COL_TYPE const col ) const restrict_this
  {
    SPARSITYPATTERN_CHECK_BOUNDS2( row, col );

    INDEX_TYPE_NC const nnz = numNonZeros( row );
    COL_TYPE const * columns = getColumns( row );

    return !sortedArrayManipulation::contains( columns, nnz, col );
  }

  /**
   * @brief Return an ArraySlice1d (pointer) to the columns of the given row.
   * This array has length numNonZeros(row).
   * @param [in] row the row to access.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  ArraySlice1d_rval<COL_TYPE const, INDEX_TYPE_NC> getColumns( INDEX_TYPE_NC const row ) const restrict_this
  {
    SPARSITYPATTERN_CHECK_BOUNDS( row );
    return createArraySlice1d<COL_TYPE const, INDEX_TYPE_NC>( m_columns.data() + m_offsets[row], &m_sizes[row], nullptr );
  }

  /**
   * @brief Insert a non-zero entry at the given position.
   * @param [in] row the row to insert in.
   * @param [in] col the column to insert.
   * @return True iff the column was inserted (the entry was zero before).
   */
  LVARRAY_HOST_DEVICE inline
  bool insertNonZero( INDEX_TYPE_NC const row, COL_TYPE const col ) const restrict_this
  {
    INDEX_TYPE_NC const rowNNZ = numNonZeros( row );
    INDEX_TYPE_NC const rowCapacity = nonZeroCapacity( row );
    return insertNonZeroImpl( row, col, CallBacks( *this, row, rowNNZ, rowCapacity ));
  }

  /**
   * @brief Insert non-zero entries into the given row.
   * @param [in] row the row to insert in.
   * @param [in] cols the columns to insert, of length ncols.
   * @param [in] ncols the number of columns to insert.
   * @return The number of columns inserted.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC insertNonZeros( INDEX_TYPE_NC const row, COL_TYPE const * cols, INDEX_TYPE_NC const ncols ) const restrict_this
  {
    INDEX_TYPE_NC const rowNNZ = numNonZeros( row );
    INDEX_TYPE_NC const rowCapacity = nonZeroCapacity( row );
    return insertNonZerosImpl( row, cols, ncols, CallBacks( *this, row, rowNNZ, rowCapacity ));
  }

  /**
   * @brief Remove a non-zero entry at the given position.
   * @param [in] row the row to remove from.
   * @param [in] col the column to remove.
   * @return True iff the column was removed (the entry was non zero before).
   */
  LVARRAY_HOST_DEVICE inline
  bool removeNonZero( INDEX_TYPE_NC const row, COL_TYPE const col ) const restrict_this
  {
    INDEX_TYPE_NC const rowNNZ = numNonZeros( row );
    INDEX_TYPE_NC const rowCapacity = nonZeroCapacity( row );
    return removeNonZeroImpl( row, col, CallBacks( *this, row, rowNNZ, rowCapacity ));
  }

  /**
   * @brief Remove non-zero entries from the given row.
   * @param [in] row the row to remove from.
   * @param [in] cols the columns to remove, of length ncols.
   * @param [in] ncols the number of columns to remove.
   * @return The number of columns removed.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC removeNonZeros( INDEX_TYPE_NC const row, COL_TYPE const * const cols, INDEX_TYPE_NC const ncols ) const restrict_this
  {
    INDEX_TYPE_NC const rowNNZ = numNonZeros( row );
    INDEX_TYPE_NC const rowCapacity = nonZeroCapacity( row );
    return removeNonZerosImpl( row, cols, ncols, CallBacks( *this, row, rowNNZ, rowCapacity ));
  }

protected:

  /**
   * @brief Default constructor. Made protected since every SparsityPatternView should
   * either be the base of a SparsityPattern or copied from another SparsityPatternView.
   */
  SparsityPatternView() = default;

  /**
   * @brief Return an ArraySlice1d (pointer) to the columns of the given row.
   * This array has length numNonZeros(row).
   * @param [in] row the row to access.
   * @note The returned array is mutable so this method is private.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  ArraySlice1d_rval<COL_TYPE, INDEX_TYPE_NC> getColumnsProtected( INDEX_TYPE_NC const row ) const restrict_this
  {
    SPARSITYPATTERN_CHECK_BOUNDS( row );
    return createArraySlice1d<COL_TYPE, INDEX_TYPE_NC>( m_columns.data() + m_offsets[row], &m_sizes[row], nullptr );
  }

  /**
   * @brief Insert a non-zero entry at the given position.
   * @param [in] row the row to insert in.
   * @param [in] col the column to insert.
   * @param [in/out] cbacks class that defines a set of call-back methods to be used by
   * the sortedArrayManipulation routines.
   * @return True iff the column was inserted (the entry was zero before).
   */
  template <class CALLBACKS>
  LVARRAY_HOST_DEVICE inline
  bool insertNonZeroImpl( INDEX_TYPE_NC const row, COL_TYPE const col, CALLBACKS && cbacks ) const restrict_this
  {
    SPARSITYPATTERN_CHECK_BOUNDS2( row, col );

    INDEX_TYPE_NC const rowNNZ = numNonZeros( row );
    COL_TYPE const * const columns = getColumns( row );

    bool const success = sortedArrayManipulation::insert( columns, rowNNZ, col, std::move( cbacks ));
    m_sizes[row] += success;
    return success;
  }

  /**
   * @brief Insert non-zero entries into the given row.
   * @param [in] row the row to insert in.
   * @param [in] cols the columns to insert.
   * @param [in] ncols the number of columns to insert.
   * @param [in/out] cbacks class that defines a set of call-back methods to be used by
   * the sortedArrayManipulation routines.
   * @return The number of columns inserted.
   */
  template <class CALLBACKS>
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC insertNonZerosImpl( INDEX_TYPE_NC const row,
                                    COL_TYPE const * const cols,
                                    INDEX_TYPE_NC const ncols,
                                    CALLBACKS && cbacks ) const restrict_this
  {
    SPARSITYPATTERN_CHECK_BOUNDS( row );
    GEOS_ASSERT( cols != nullptr );
    GEOS_ASSERT( ncols >= 0 );
    GEOS_ASSERT( sortedArrayManipulation::isSorted( cols, ncols ));

    if( ncols == 0 ) return true;

    for( INDEX_TYPE_NC i = 0 ; i < ncols ; ++i )
    {
      SPARSITYPATTERN_CHECK_BOUNDS2( row, cols[i] );
    }

    INDEX_TYPE_NC const rowNNZ = numNonZeros( row );
    COL_TYPE const * const columns = getColumns( row );

    INDEX_TYPE_NC const nInserted = sortedArrayManipulation::insertSorted( columns, rowNNZ, cols, ncols, std::move( cbacks ));
    m_sizes[row] += nInserted;
    return nInserted;
  }

  /**
   * @brief Remove a non-zero entry at the given position.
   * @param [in] row the row to remove from.
   * @param [in] col the column to remove.
   * @param [in/out] cbacks class that defines a set of call-back methods to be used by
   * the arrayManipulation routines.
   * @return True iff the column was removed (the entry was non zero before).
   */
  template <class CALLBACKS>
  LVARRAY_HOST_DEVICE inline
  bool removeNonZeroImpl( INDEX_TYPE_NC const row, COL_TYPE const col, CALLBACKS && cbacks ) const restrict_this
  {
    SPARSITYPATTERN_CHECK_BOUNDS2( row, col );

    INDEX_TYPE_NC const rowNNZ = numNonZeros( row );
    COL_TYPE * const columns = getColumnsProtected( row );

    bool const success = sortedArrayManipulation::remove( columns, rowNNZ, col, std::move( cbacks ));
    m_sizes[row] -= success;
    return success;
  }

  /**
   * @brief Remove non-zero entries from the given row.
   * @param [in] row the row to remove from.
   * @param [in] cols the columns to remove.
   * @param [in] ncols the number of columns to remove.
   * @param [in/out] cbacks class that defines a set of call-back methods to be used by
   * the arrayManipulation routines.
   * @return The number of columns removed.
   */
  template <class CALLBACKS>
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC removeNonZerosImpl( INDEX_TYPE_NC const row,
                                    COL_TYPE const * const cols,
                                    INDEX_TYPE_NC const ncols,
                                    CALLBACKS && cbacks ) const restrict_this
  {
    SPARSITYPATTERN_CHECK_BOUNDS( row );
    GEOS_ASSERT( cols != nullptr );
    GEOS_ASSERT( ncols >= 0 );
    GEOS_ASSERT( sortedArrayManipulation::isSorted( cols, ncols ));

    if( ncols == 0 ) return true;

    for( INDEX_TYPE_NC i = 0 ; i < ncols ; ++i )
    {
      SPARSITYPATTERN_CHECK_BOUNDS2( row, cols[i] );
    }

    INDEX_TYPE_NC const rowNNZ = numNonZeros( row );
    COL_TYPE * const columns = getColumnsProtected( row );

    INDEX_TYPE_NC const nRemoved = sortedArrayManipulation::removeSorted( columns, rowNNZ, cols, ncols, std::move( cbacks ));
    m_sizes[row] -= nRemoved;
    return nRemoved;
  }

  // The number of columns in the matrix.
  INDEX_TYPE m_num_columns;

  // Holds the offset of each row, of length numRows() + 1. Row i begins at
  // m_offsets[i] and has capacity m_offsets[i+1] - m_offsets[i].
  ChaiVector<INDEX_TYPE> m_offsets;

  // Holds the size of each row, of length numRows().
  using SIZE_TYPE = std::conditional_t<std::is_const<COL_TYPE>::value, INDEX_TYPE_NC const, INDEX_TYPE_NC>;
  ChaiVector<SIZE_TYPE> m_sizes;

  // Holds the columns of each row, of length numNonZeros().
  ChaiVector<COL_TYPE> m_columns;

private:

  /**
   * @class CallBacks
   * @brief This class provides the callbacks for the arrayManipulation sorted routines.
   */
  class CallBacks
  {
public:

    /**
     * @brief Constructor.
     * @param [in/out] spv the SparsityPatternView this CallBacks is associated with.
     * @param [in] row the row this CallBacks is associated with.
     * @param [in] rowNNZ the number of non zeros in the row.
     * @param [in] rowCapacity the non zero capacity of the row.
     */
    LVARRAY_HOST_DEVICE inline
    CallBacks( SparsityPatternView<COL_TYPE, INDEX_TYPE> const & spv,
               INDEX_TYPE_NC const row,
               INDEX_TYPE_NC const rowNNZ,
               INDEX_TYPE_NC const rowCapacity ):
      m_spv( spv ),
      m_row( row ),
      m_rowNNZ( rowNNZ ),
      m_rowCapacity( rowCapacity )
    {}

    /**
     * @brief Callback signaling that the size of the row has increased.
     * @param [in] nToAdd the increase in the size.
     * @note This method doesn't actually change the size, it just checks that the new
     * size doesn't exceed the capacity since the SparsityPatternView can't do allocation.
     * @return a pointer to the rows columns.
     */
    LVARRAY_HOST_DEVICE inline
    COL_TYPE * incrementSize( INDEX_TYPE_NC const nToAdd ) const restrict_this
    {
#ifdef USE_ARRAY_BOUNDS_CHECK
      GEOS_ERROR_IF( m_rowNNZ + nToAdd > m_rowCapacity,
                     "SparsityPatternView cannot do reallocation." );
#endif
      return m_spv.getColumnsProtected( m_row );
    }

    /**
     * @brief These methods are placeholder and are no-ops.
     */
    /// @{
    LVARRAY_HOST_DEVICE inline
    void insert( INDEX_TYPE ) const restrict_this
    {}

    LVARRAY_HOST_DEVICE inline
    void set( INDEX_TYPE, INDEX_TYPE ) const restrict_this
    {}

    LVARRAY_HOST_DEVICE inline
    void insert( INDEX_TYPE, INDEX_TYPE, INDEX_TYPE, INDEX_TYPE ) const restrict_this
    {}

    LVARRAY_HOST_DEVICE inline
    void remove( INDEX_TYPE ) const restrict_this
    {}

    LVARRAY_HOST_DEVICE inline
    void remove( INDEX_TYPE, INDEX_TYPE, INDEX_TYPE ) const restrict_this
    {}
    /// @}

private:
    SparsityPatternView<COL_TYPE, INDEX_TYPE> const & m_spv;
    INDEX_TYPE_NC const m_row;
    INDEX_TYPE_NC const m_rowNNZ;
    INDEX_TYPE_NC const m_rowCapacity;
  };

};

} /* namespace LvArray */

#endif /* SPARSITYPATTERNVIEW_HPP_ */
