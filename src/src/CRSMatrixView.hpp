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
#include "ChaiVector.hpp"
#include "arrayManipulation.hpp"
#include "ArraySlice.hpp"

namespace LvArray
{

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
 *       and. You can however modify the values of the matrix which is copied back from the device.
 * @note When T, COL_TYPE and INDEX_TYPE are const you cannot modify the matrix in any way
 *       and nothing is copied back from the device.
 * @note SparsityPatternView is a protected base class of CRSMatrixView. This is to control the
 *       conversion to SparsityPatternView so as to only provide an immutable SparsityPatternView.
 *       However the SparsityPatternView interface is reproduced here.
 */
template <class T, class COL_TYPE=unsigned int, class INDEX_TYPE=std::ptrdiff_t>
class CRSMatrixView : protected SparsityPatternView<COL_TYPE, INDEX_TYPE>
{
public:

  // Aliasing public typedefs of SparsityPatternView.
  using typename SparsityPatternView<COL_TYPE, INDEX_TYPE>::INDEX_TYPE_NC;

  // Aliasing public methods of SparsityPatternView.
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::numRows;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::numColumns;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::numNonZeros;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::nonZeroCapacity;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::empty;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::getColumns;

  /**
   * @brief Default copy constructor. Performs a shallow copy and calls the
   *        chai::ManagedArray copy constructor.
   * @param [in] src the CRSMatrixView to be copied.
   */
  inline
  CRSMatrixView( CRSMatrixView const & ) = default;

  /**
   * @brief Default move constructor.
   * @param [in/out] src the CRSMatrixView to be moved from.
   */
  inline
  CRSMatrixView( CRSMatrixView && ) = default;

  /**
   * @brief User defined conversion to convert to CRSMatrixView<T, COL_TYPE const, INDEX_TYPE const>.
   *        This prevents you from inserting or removing but still allows modification of the matrix values.
   */
  template <class U=T, class CTYPE=COL_TYPE>
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  operator typename std::enable_if<!std::is_const<U>::value && !std::is_const<CTYPE>::value,
                                   CRSMatrixView<T, COL_TYPE const, INDEX_TYPE_NC const> const &>::type
    () const restrict_this
  { return reinterpret_cast<CRSMatrixView<T, COL_TYPE const, INDEX_TYPE_NC const> const &>(*this); }

  /**
   * @brief Method to convert to CRSMatrixView<T, COL_TYPE const, INDEX_TYPE const>. Use this method
   *        when the above UDC isn't invoked, this usually occurs with template argument deduction.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  CRSMatrixView<T, COL_TYPE const, INDEX_TYPE_NC const> const & toViewC() const restrict_this
  { return *this; }

  /**
   * @brief User defined conversion to convert to CRSMatrixView<T const, COL_TYPE const, INDEX_TYPE const>.
   *        This prevents you from inserting, removing or modifying the matrix values.
   */
  template<class U=T>
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  operator typename std::enable_if<!std::is_const<U>::value, CRSMatrixView<T const, COL_TYPE const, INDEX_TYPE_NC const> const &>::type
    () const restrict_this
  { return reinterpret_cast<CRSMatrixView<T const, COL_TYPE const, INDEX_TYPE_NC const> const &>(*this); }

  /**
   * @brief Method to convert to CRSMatrixView<T const, COL_TYPE const, INDEX_TYPE const>. Use this method
   *        when the above UDC isn't invoked, this usually occurs with template argument deduction.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  CRSMatrixView<T const, COL_TYPE const, INDEX_TYPE_NC const> const & toViewCC() const restrict_this
  { return *this; }

  /**
   * @brief Method to convert to SparsityPatternView<COL_TYPE const, INDEX_TYPE const>.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  SparsityPatternView<COL_TYPE const, INDEX_TYPE_NC const> const & toSparsityPatternView() const
  { return reinterpret_cast<SparsityPatternView<COL_TYPE const, INDEX_TYPE_NC const> const &>(*this); }

  /**
   * @brief Default copy assignment operator, this does a shallow copy.
   * @param [in] src the CRSMatrixView to be copied from.
   */
  inline
  CRSMatrixView & operator=( CRSMatrixView const & src ) = default;

  /**
   * @brief Return an ArraySlice1d (pointer) to the matrix values of the given row.
   *        This array has length numNonZeros(row).
   * @param [in] row the row to access.
   */
  LVARRAY_HOST_DEVICE inline
  ArraySlice1d_rval<T, INDEX_TYPE_NC> getValues( INDEX_TYPE_NC const row ) const restrict_this
  {
    SPARSITYPATTERN_CHECK_BOUNDS( row );
    return createArraySlice1d<T, INDEX_TYPE_NC>( m_values.data() + m_offsets[row], &m_sizes[row], nullptr );
  }

  /**
   * @brief Insert a non-zero entry at the given position.
   * @param [in] row the row to insert in.
   * @param [in] col the column to insert at.
   * @param [in] value the value to insert.
   * @return True iff the value was inserted (the entry was zero before).
   *
   * @note Since the CRSMatrixView can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entry.
   *       If USE_ARRAY_BOUNDS_CHECK is defined a lack of space will result in an error,
   *       otherwise the values in the subsequent row will be overwritten.
   */
  LVARRAY_HOST_DEVICE inline
  bool insertNonZero( INDEX_TYPE_NC const row, COL_TYPE const col, T const & value ) const restrict_this
  {
    INDEX_TYPE_NC const rowNNZ = numNonZeros( row );
    INDEX_TYPE_NC const rowCapacity = nonZeroCapacity( row );
    T * const values = getValues( row );
    return insertNonZeroImpl( row, col, CallBacks( *this, row, rowNNZ, rowCapacity, values, &value ));
  }

  /**
   * @brief Insert a non-zero entries into the given row.
   * @param [in] row the row to insert into.
   * @param [in] cols the columns to insert at, of length ncols.
   * @param [in] valuesToInsert the values to insert, of length ncols.
   * @param [in] ncols the number of columns/values to insert.
   * @return The number of values inserted.
   *
   * @note If possible sort cols and valuesToInsert first by calling sortedArrayManipulation::dualSort(cols, cols + ncols, valuesToInsert)
   *       and then call insertNonZerosSorted, this will be substantially faster.
   * @note Since the CRSMatrixView can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entries.
   *       If USE_ARRAY_BOUNDS_CHECK is defined a lack of space will result in an error,
   *       otherwise the values in the subsequent row will be overwritten.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC insertNonZeros( INDEX_TYPE_NC const row,
                                COL_TYPE const * const cols,
                                T const * const valuesToInsert,
                                INDEX_TYPE_NC const ncols ) const restrict_this
  {
    constexpr int LOCAL_SIZE = 16;
    COL_TYPE localColumnBuffer[LOCAL_SIZE];
    T localValueBuffer[LOCAL_SIZE];

    COL_TYPE * const columnBuffer = sortedArrayManipulation::createTemporaryBuffer(cols, ncols, localColumnBuffer);
    T * const valueBuffer = sortedArrayManipulation::createTemporaryBuffer(valuesToInsert, ncols, localValueBuffer);
    sortedArrayManipulation::dualSort(columnBuffer, columnBuffer + ncols, valueBuffer);

    INDEX_TYPE_NC const nInserted = insertNonZerosSorted(row, columnBuffer, valueBuffer, ncols);

    sortedArrayManipulation::freeTemporaryBuffer(columnBuffer, ncols, localColumnBuffer);
    sortedArrayManipulation::freeTemporaryBuffer(valueBuffer, ncols, localValueBuffer);
    return nInserted;
  }

  /**
   * @brief Insert a non-zero entries into the given row.
   * @param [in] row the row to insert into.
   * @param [in] cols the columns to insert at, of length ncols. Must be sorted.
   * @param [in] valuesToInsert the values to insert, of length ncols.
   * @param [in] ncols the number of columns/values to insert.
   * @return The number of values inserted.
   *
   * @note Since the CRSMatrixView can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entries.
   *       If USE_ARRAY_BOUNDS_CHECK is defined a lack of space will result in an error,
   *       otherwise the values in the subsequent row will be overwritten.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC insertNonZerosSorted( INDEX_TYPE_NC const row,
                                      COL_TYPE const * const cols,
                                      T const * const valuesToInsert,
                                      INDEX_TYPE_NC const ncols ) const restrict_this
  {
    INDEX_TYPE_NC const rowNNZ = numNonZeros( row );
    INDEX_TYPE_NC const rowCapacity = nonZeroCapacity( row );
    T * const values = getValues( row );
    return insertNonZerosSortedImpl( row, cols, ncols, CallBacks( *this, row, rowNNZ, rowCapacity, values, valuesToInsert ));
  }

  /**
   * @brief Remove a non-zero entry at the given position.
   * @param [in] row the row to remove from.
   * @param [in] col the column to remove.
   * @return True iff the entry was removed (the entry was non-zero before).
   */
  LVARRAY_HOST_DEVICE inline
  bool removeNonZero( INDEX_TYPE_NC const row, COL_TYPE const col ) const restrict_this
  {
    INDEX_TYPE_NC const rowNNZ = numNonZeros( row );
    INDEX_TYPE_NC const rowCapacity = nonZeroCapacity( row );
    T * const values = getValues( row );
    bool const success = removeNonZeroImpl( row, col, CallBacks( *this, row, rowNNZ, rowCapacity, values, nullptr ));

    return success;
  }

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
  INDEX_TYPE_NC removeNonZeros( INDEX_TYPE_NC const row,
                                COL_TYPE const * const cols,
                                INDEX_TYPE_NC const ncols ) const restrict_this
  {
    constexpr int LOCAL_SIZE = 16;
    COL_TYPE localColumnBuffer[LOCAL_SIZE];

    COL_TYPE * const columnBuffer = sortedArrayManipulation::createTemporaryBuffer(cols, ncols, localColumnBuffer);
    sortedArrayManipulation::makeSorted(columnBuffer, columnBuffer + ncols);

    INDEX_TYPE_NC const nRemoved = removeNonZerosSorted(row, columnBuffer, ncols);

    sortedArrayManipulation::freeTemporaryBuffer(columnBuffer, ncols, localColumnBuffer);
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
  INDEX_TYPE_NC removeNonZerosSorted( INDEX_TYPE_NC const row,
                                      COL_TYPE const * const cols,
                                      INDEX_TYPE_NC const ncols ) const restrict_this
  {
    INDEX_TYPE_NC const rowNNZ = numNonZeros( row );
    INDEX_TYPE_NC const rowCapacity = nonZeroCapacity( row );
    T * const values = getValues( row );
    INDEX_TYPE_NC const nRemoved = removeNonZerosSortedImpl( row, cols, ncols, CallBacks( *this, row, rowNNZ, rowCapacity, values, nullptr ));

    for( INDEX_TYPE_NC i = rowNNZ - nRemoved ; i < rowNNZ ; ++i )
    {
      values[i].~T();
    }

    return nRemoved;
  }

protected:

  /**
   * @brief Default constructor. Made protected since every CRSMatrixView should
   * either be the base of a CRSMatrix or copied from another CRSMatrixView.
   */
  CRSMatrixView() = default;

  // Aliasing protected methods of SparsityPatternView.
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::getColumnsProtected;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::insertNonZeroImpl;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::insertNonZerosSortedImpl;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::removeNonZeroImpl;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::removeNonZerosSortedImpl;

  // Aliasing protected members of SparsityPatternView.
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::m_num_columns;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::m_offsets;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::m_sizes;
  using SparsityPatternView<COL_TYPE, INDEX_TYPE>::m_columns;

  // Holds the values of the matrix, of length numNonZeros().
  ChaiVector<T> m_values;

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
     * @param [in] rowNNZ the number of non zeros in the row.
     * @param [in] rowCapacity the non zero capacity of the row.
     * @param [in/out] values pointer to the values of row. Of length rownNNZ and capacity
     * rowCapacity.
     * @param [in] valuesToInsert pointer to the values to insert.
     */
    LVARRAY_HOST_DEVICE inline
    CallBacks( CRSMatrixView<T, COL_TYPE, INDEX_TYPE> const & crsMV,
               INDEX_TYPE_NC const row,
               INDEX_TYPE_NC const rowNNZ,
               INDEX_TYPE_NC const rowCapacity,
               T * const values,
               T const * const valuesToInsert ):
      m_crsMV( crsMV ),
      m_row( row ),
      m_rowNNZ( rowNNZ ),
      m_rowCapacity( rowCapacity ),
      m_values( values ),
      m_valuesToInsert( valuesToInsert )
    {}

    /**
     * @brief Callback signaling that the size of the row has increased.
     * @param [in] nToAdd the increase in the size.
     * @note This method doesn't actually change the size, it just checks that the new
     * size doesn't exceed the capacity since the CRSMatrixView can't do allocation.
     * @return a pointer to the rows columns.
     */
    LVARRAY_HOST_DEVICE inline
    COL_TYPE * incrementSize( INDEX_TYPE_NC const nToAdd ) const
    {
#ifdef USE_ARRAY_BOUNDS_CHECK
      GEOS_ERROR_IF( m_rowNNZ + nToAdd > m_rowCapacity, "CRSMatrixView cannot do reallocation." );
#endif
      return m_crsMV.getColumnsProtected( m_row );
    }

    /**
     * @brief Used with sortedArrayManipulation::insert routine this callback signals
     * that the column was inserted at the given position. This means we also need to insert
     * the value at the same position.
     * @param [in] insertPos the position the column was inserted at.
     */
    LVARRAY_HOST_DEVICE inline
    void insert( INDEX_TYPE_NC const insertPos ) const
    { arrayManipulation::insert( m_values, m_rowNNZ, insertPos, m_valuesToInsert[0] ); }

    /**
     * @brief Used with the sortedArrayManipulation::insertSorted routine this callback
     * signals that the given position was set to the column at the other position.
     * This means we need to perform the same operation on the values.
     * @param [in] pos the position that was set.
     * @param [in] colPos the position of the column.
     */
    LVARRAY_HOST_DEVICE inline
    void set( INDEX_TYPE_NC const pos, INDEX_TYPE_NC const colPos ) const
    { new (&m_values[pos]) T( m_valuesToInsert[colPos] ); }

    /**
     * @brief Used with the sortedArrayManipulation::insertSorted routine this callback
     * signals that the given column was inserted at the given position. Further information
     * is provided in order to make the insertion efficient. This means that we need to perform
     * the same operation on the values.
     * @param [in] nLeftToInsert the number of insertions that occur after this one.
     * @param [in] colPos the position of the column that was inserted.
     * @param [in] insertPos the position the column was inserted at.
     * @param [in] prevPos the position the previous column was inserted at or m_rowNNZ
     * if it is the first insertion.
     */
    LVARRAY_HOST_DEVICE inline
    void insert( INDEX_TYPE_NC const nLeftToInsert,
                 INDEX_TYPE_NC const colPos,
                 INDEX_TYPE_NC const pos,
                 INDEX_TYPE_NC const prevPos ) const
    {
      arrayManipulation::shiftUp( m_values, prevPos, pos, nLeftToInsert );
      new (&m_values[pos + nLeftToInsert - 1]) T( m_valuesToInsert[colPos] );
    }

    /**
     * @brief Used with sortedArrayManipulation::remove routine this callback signals
     * that the column was removed from the given position. This means we also need to remove
     * the value from the same position.
     * @param [in] removePos the position the column was removed from.
     */
    LVARRAY_HOST_DEVICE inline
    void remove( INDEX_TYPE_NC removePos ) const
    { arrayManipulation::erase( m_values, m_rowNNZ, removePos ); }

    /**
     * @brief Used with the sortedArrayManipulation::removeSorted routine this callback
     * signals that the given column was removed from the given position. Further information
     * is provided in order to make the removal efficient. This means that we need to perform
     * the same operation on the values.
     * @param [in] nRemoved the number of columns removed, starts at 1.
     * @param [in] curPos the position the column was removed at.
     * @param [in] nextPos the position the next column will be removed at or m_rowNNZ
     * if this was the last column removed.
     */
    LVARRAY_HOST_DEVICE inline
    void remove( INDEX_TYPE_NC const nRemoved,
                 INDEX_TYPE_NC const curPos,
                 INDEX_TYPE_NC const nextPos ) const
    { arrayManipulation::shiftDown( m_values, nextPos, curPos + 1, nRemoved ); }

private:
    CRSMatrixView<T, COL_TYPE, INDEX_TYPE> const & m_crsMV;
    INDEX_TYPE_NC const m_row;
    INDEX_TYPE_NC const m_rowNNZ;
    INDEX_TYPE_NC const m_rowCapacity;
    T * const m_values;
    T const * const m_valuesToInsert;
  };

};

} /* namespace LvArray */

#endif /* CRSMATRIXVIEW_HPP_ */
