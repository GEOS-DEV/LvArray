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

#include <limits>
#include "ArrayOfSetsView.hpp"

#ifdef USE_ARRAY_BOUNDS_CHECK

#define SPARSITYPATTERN_COLUMN_CHECK( col ) \
  GEOS_ERROR_IF( !arrayManipulation::isPositive( col ) || col >= this->numColumns(), \
                 "Column Check Failed: col=" << col << " numColumns=" << this->numColumns() )

#else // USE_ARRAY_BOUNDS_CHECK

#define SPARSITYPATTERN_COLUMN_CHECK( col )

#endif

namespace LvArray
{

/**
 * @class SparsityPatternView
 * @brief This class provides a view into a compressed row storage sparsity pattern.
 * @tparam COL_TYPE the integer used to enumerate the columns.
 * @tparam INDEX_TYPE the integer to use for indexing.
 *
 * When INDEX_TYPE is const m_offsets is not copied between memory spaces. INDEX_TYPE should always be const
 * since SparsityPatternView is not allowed to modify the offsets.
 *
 * When COL_TYPE is const and INDEX_TYPE is const you cannot insert or remove from the View
 * and neither the offsets, sizes, or columns are copied between memory spaces.
 */
template< class COL_TYPE=unsigned int, class INDEX_TYPE=std::ptrdiff_t >
class SparsityPatternView : protected ArrayOfSetsView< COL_TYPE, INDEX_TYPE >
{
public:
  static_assert( std::is_integral< COL_TYPE >::value, "COL_TYPE must be integral." );
  static_assert( std::is_integral< INDEX_TYPE >::value, "INDEX_TYPE must be integral." );
  static_assert( std::numeric_limits< INDEX_TYPE >::max() >= std::numeric_limits< COL_TYPE >::max(),
                 "INDEX_TYPE must be able to hold values at least as large as COL_TYPE." );

  using INDEX_TYPE_NC = typename std::remove_const< INDEX_TYPE >::type;

  /**
   * @brief Default copy constructor. Performs a shallow copy and calls the
   *        chai::ManagedArray copy constructor.
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
   * @brief User defined conversion to move from COL_TYPE to COL_TYPE const.
   */
  template< class CTYPE=COL_TYPE >
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  operator typename std::enable_if< !std::is_const< CTYPE >::value,
                                    SparsityPatternView< COL_TYPE const, INDEX_TYPE const > const & >::type
    () const restrict_this
  { return reinterpret_cast< SparsityPatternView< COL_TYPE const, INDEX_TYPE const > const & >(*this); }

  /**
   * @brief Method to convert COL_TYPE to COL_TYPE const. Use this method when the above UDC
   *        isn't invoked, this usually occurs with template argument deduction.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  SparsityPatternView< COL_TYPE const, INDEX_TYPE const > const & toViewC() const restrict_this
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
  { return ArrayOfSetsView< COL_TYPE, INDEX_TYPE >::size(); }

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
  INDEX_TYPE_NC numNonZeros( INDEX_TYPE const row ) const restrict_this
  { return ArrayOfSetsView< COL_TYPE, INDEX_TYPE >::sizeOfSet( row ); }

  /**
   * @brief Return the total number of non zero entries able to be stored without a reallocation.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE_NC nonZeroCapacity() const restrict_this
  { return m_values.capacity(); }

  /**
   * @brief Return the total number of non zero entries able to be stored in a given row without shifting
   *        subsequent rows and possibly reallocating.
   * @param [in] row the row to query.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE_NC nonZeroCapacity( INDEX_TYPE const row ) const restrict_this
  { return ArrayOfSetsView< COL_TYPE, INDEX_TYPE >::capacityOfSet( row ); }

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
  bool empty( INDEX_TYPE const row ) const restrict_this
  { return numNonZeros( row ) == 0; }

  /**
   * @brief Return true iff the given entry is zero.
   * @param [in] row the row to query.
   * @param [in] col the col to query.
   */
  LVARRAY_HOST_DEVICE inline
  bool empty( INDEX_TYPE const row, COL_TYPE const col ) const restrict_this
  { return !ArrayOfSetsView< COL_TYPE, INDEX_TYPE >::contains( row, col ); }

  /**
   * @brief Return an ArraySlice1d (pointer) to the columns of the given row.
   *        This array has length numNonZeros(row).
   * @param [in] row the row to access.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  ArraySlice1d_rval< COL_TYPE const, INDEX_TYPE_NC > getColumns( INDEX_TYPE const row ) const restrict_this
  { return (*this)[row]; }

  /**
   * @brief Return a pointer to the array of offsets.
   *        This array has length numRows() + 1.
   */
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  INDEX_TYPE const * getOffsets() const restrict_this
  { return m_offsets.data(); }

  /**
   * @brief Insert a non-zero entry at the given position.
   * @param [in] row the row to insert in.
   * @param [in] col the column to insert.
   * @return True iff the column was inserted (the entry was zero before).
   *
   * @note Since the SparsityPatternView can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entries.
   *       If USE_ARRAY_BOUNDS_CHECK is defined a lack of space will result in an error,
   *       otherwise the values in the subsequent row will be overwritten.
   */
  LVARRAY_HOST_DEVICE inline
  bool insertNonZero( INDEX_TYPE const row, COL_TYPE const col ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( row );
    SPARSITYPATTERN_COLUMN_CHECK( col );
    return ArrayOfSetsView< COL_TYPE, INDEX_TYPE >::insertIntoSet( row, col );
  }

  /**
   * @brief Insert non-zero entries into the given row.
   * @param [in] row the row to insert in.
   * @param [in] cols the columns to insert, of length ncols.
   * @param [in] ncols the number of columns to insert.
   * @return The number of columns inserted.
   *
   * @note If possible sort cols first by calling sortedArrayManipulation::makeSorted(cols, cols + ncols)
   *       and then call insertNonZerosSorted, this will be substantially faster.
   * @note Since the SparsityPatternView can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entries.
   *       If USE_ARRAY_BOUNDS_CHECK is defined a lack of space will result in an error,
   *       otherwise the values in the subsequent row will be overwritten.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC insertNonZeros( INDEX_TYPE const row, COL_TYPE const * cols, INDEX_TYPE const ncols ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( row );
    GEOS_ASSERT( cols != nullptr || ncols == 0 );
    GEOS_ASSERT( arrayManipulation::isPositive( ncols ) );

    for( INDEX_TYPE_NC i = 0 ; i < ncols ; ++i )
    {
      SPARSITYPATTERN_COLUMN_CHECK( cols[i] );
    }

    return ArrayOfSetsView< COL_TYPE, INDEX_TYPE >::insertIntoSet( row, cols, ncols );
  }

  /**
   * @brief Insert non-zero entries into the given row.
   * @param [in] row the row to insert in.
   * @param [in] cols the columns to insert, of length ncols. Must be sorted.
   * @param [in] ncols the number of columns to insert.
   * @return The number of columns inserted.
   *
   * @note Since the SparsityPatternView can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entries.
   *       If USE_ARRAY_BOUNDS_CHECK is defined a lack of space will result in an error,
   *       otherwise the values in the subsequent row will be overwritten.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC insertNonZerosSorted( INDEX_TYPE const row, COL_TYPE const * cols, INDEX_TYPE const ncols ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( row );
    GEOS_ASSERT( cols != nullptr || ncols == 0 );
    GEOS_ASSERT( arrayManipulation::isPositive( ncols ) );

    for( INDEX_TYPE_NC i = 0 ; i < ncols ; ++i )
    {
      SPARSITYPATTERN_COLUMN_CHECK( cols[i] );
    }

    return ArrayOfSetsView< COL_TYPE, INDEX_TYPE >::insertSortedIntoSet( row, cols, ncols );
  }

  /**
   * @brief Remove a non-zero entry at the given position.
   * @param [in] row the row to remove from.
   * @param [in] col the column to remove.
   * @return True iff the column was removed (the entry was non zero before).
   */
  LVARRAY_HOST_DEVICE inline
  bool removeNonZero( INDEX_TYPE const row, COL_TYPE const col ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( row );
    SPARSITYPATTERN_COLUMN_CHECK( col );
    return ArrayOfSetsView< COL_TYPE, INDEX_TYPE >::removeFromSet( row, col );
  }

  /**
   * @brief Remove non-zero entries from the given row.
   * @param [in] row the row to remove from.
   * @param [in] cols the columns to remove, of length ncols.
   * @param [in] ncols the number of columns to remove.
   * @return The number of columns removed.
   *
   * @note If possible sort cols first by calling sortedArrayManipulation::makeSorted(cols, cols + ncols)
   *       and then call removeNonZerosSorted, this will be substantially faster.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC removeNonZeros( INDEX_TYPE const row, COL_TYPE const * const cols, INDEX_TYPE const ncols ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( row );
    GEOS_ASSERT( cols != nullptr || ncols == 0 );
    GEOS_ASSERT( arrayManipulation::isPositive( ncols ) );

    for( INDEX_TYPE_NC i = 0 ; i < ncols ; ++i )
    {
      SPARSITYPATTERN_COLUMN_CHECK( cols[i] );
    }

    return ArrayOfSetsView< COL_TYPE, INDEX_TYPE >::removeFromSet( row, cols, ncols );
  }

  /**
   * @brief Remove non-zero entries from the given row.
   * @param [in] row the row to remove from.
   * @param [in] cols the columns to remove, of length ncols.
   * @param [in] ncols the number of columns to remove.
   * @return The number of columns removed.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC removeNonZerosSorted( INDEX_TYPE const row, COL_TYPE const * const cols, INDEX_TYPE const ncols ) const restrict_this
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( row );
    GEOS_ASSERT( cols != nullptr || ncols == 0 );
    GEOS_ASSERT( arrayManipulation::isPositive( ncols ) );

    for( INDEX_TYPE_NC i = 0 ; i < ncols ; ++i )
    {
      SPARSITYPATTERN_COLUMN_CHECK( cols[i] );
    }

    return ArrayOfSetsView< COL_TYPE, INDEX_TYPE >::removeSortedFromSet( row, cols, ncols );
  }

protected:

  /**
   * @brief Default constructor. Made protected since every SparsityPatternView should
   *        either be the base of a SparsityPattern or copied from another SparsityPatternView.
   */
  SparsityPatternView() = default;

  template< class ... VECTORS >
  void resize( INDEX_TYPE const nrows, INDEX_TYPE const ncols, INDEX_TYPE_NC initialRowCapacity, VECTORS & ... vectors )
  {
    GEOS_ERROR_IF( !arrayManipulation::isPositive( nrows ), "nrows must be positive." );
    GEOS_ERROR_IF( !arrayManipulation::isPositive( ncols ), "ncols must be positive." );
    GEOS_ERROR_IF( ncols - 1 > std::numeric_limits< COL_TYPE >::max(),
                   "COL_TYPE must be able to hold the range of columns: [0, " << ncols - 1 << "]." );

    if( initialRowCapacity > ncols )
    {
      GEOS_WARNING_IF( initialRowCapacity > ncols, "Number of non-zeros per row cannot exceed the the number of columns." );
      initialRowCapacity = ncols;
    }

    m_num_columns = ncols;
    ArrayOfSetsView< COL_TYPE, INDEX_TYPE >::resize( nrows, initialRowCapacity, vectors ... );
  }

  // Aliasing protected members in ArrayOfSetsView
  using ArrayOfSetsView< COL_TYPE, INDEX_TYPE >::m_offsets;
  using ArrayOfSetsView< COL_TYPE, INDEX_TYPE >::m_sizes;
  using ArrayOfSetsView< COL_TYPE, INDEX_TYPE >::m_values;

  // The number of columns in the matrix.
  INDEX_TYPE m_num_columns;
};

} /* namespace LvArray */

#endif /* SPARSITYPATTERNVIEW_HPP_ */
