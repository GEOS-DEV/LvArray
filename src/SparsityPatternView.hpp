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

/**
 * @brief Check that @p col is a valid column in the matrix.
 * @param col The column to check.
 * @note This macro is only active with USE_ARRAY_BOUNDS_CHECK.
 */
#define SPARSITYPATTERN_COLUMN_CHECK( col ) \
  LVARRAY_ERROR_IF( !arrayManipulation::isPositive( col ) || col >= this->numColumns(), \
                    "Column Check Failed: col=" << col << " numColumns=" << this->numColumns() )

#else // USE_ARRAY_BOUNDS_CHECK

/**
 * @brief Check that @p col is a valid column in the matrix.
 * @param col The column to check.
 * @note This macro is only active with USE_ARRAY_BOUNDS_CHECK.
 */
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

  /// An alias for the parent class.
  using ParentClass = ArrayOfSetsView< COL_TYPE, INDEX_TYPE >;

public:
  static_assert( std::is_integral< COL_TYPE >::value, "COL_TYPE must be integral." );
  static_assert( std::is_integral< INDEX_TYPE >::value, "INDEX_TYPE must be integral." );
  static_assert( std::numeric_limits< INDEX_TYPE >::max() >= std::numeric_limits< COL_TYPE >::max(),
                 "INDEX_TYPE must be able to hold values at least as large as COL_TYPE." );

  /// An alias for the non const index type.
  using INDEX_TYPE_NC = typename ParentClass::INDEX_TYPE_NC;

  /**
   * @brief Default copy constructor. Performs a shallow copy and calls the
   *   chai::ManagedArray copy constructor.
   */
  inline
  SparsityPatternView( SparsityPatternView const & ) = default;

  /**
   * @brief Default move constructor.
   */
  inline
  SparsityPatternView( SparsityPatternView && ) = default;

  /**
   * @brief Default copy assignment operator, this does a shallow copy.
   * @return *this.
   */
  inline
  SparsityPatternView & operator=( SparsityPatternView const & ) = default;

  /**
   * @brief Default move assignment operator, this does a shallow copy.
   * @return *this.
   */
  inline
  SparsityPatternView & operator=( SparsityPatternView && ) = default;

  /**
   * @brief @return Return *this.
   * @brief This is included for SFINAE needs.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  SparsityPatternView< COL_TYPE, INDEX_TYPE > const & toView() const LVARRAY_RESTRICT_THIS
  { return *this; }

  /**
   * @brief @return A reference to *this reinterpreted as a SparsityPatternView< COL_TYPE const, INDEX_TYPE const >.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  SparsityPatternView< COL_TYPE const, INDEX_TYPE const > const & toViewConst() const LVARRAY_RESTRICT_THIS
  { return reinterpret_cast< SparsityPatternView< COL_TYPE const, INDEX_TYPE const > const & >( *this ); }

  /**
   * @brief @return Return the number of rows in the matrix.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE_NC numRows() const LVARRAY_RESTRICT_THIS
  { return ParentClass::size(); }

  /**
   * @brief @return Return the number of columns in the matrix.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE_NC numColumns() const LVARRAY_RESTRICT_THIS
  { return m_numCols; }

  /**
   * @brief @return Return the total number of non zero entries in the matrix.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC numNonZeros() const LVARRAY_RESTRICT_THIS
  {
    INDEX_TYPE_NC nnz = 0;
    for( INDEX_TYPE_NC row = 0; row < numRows(); ++row )
    {
      nnz += numNonZeros( row );
    }

    return nnz;
  }

  /**
   * @brief @return Return the total number of non zero entries in the given row.
   * @param row the row to query.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE_NC numNonZeros( INDEX_TYPE const row ) const LVARRAY_RESTRICT_THIS
  { return ParentClass::sizeOfSet( row ); }

  /**
   * @brief @return Return the total number of non zero entries able to be stored without a reallocation.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE_NC nonZeroCapacity() const LVARRAY_RESTRICT_THIS
  { return m_values.capacity(); }

  /**
   * @brief @return Return the total number of non zero entries able to be stored in a given row without increasing its
   * capacity.
   * @param row the row to query.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE_NC nonZeroCapacity( INDEX_TYPE const row ) const LVARRAY_RESTRICT_THIS
  { return ParentClass::capacityOfSet( row ); }

  /**
   * @brief @return Return true iff the matrix is all zeros.
   */
  LVARRAY_HOST_DEVICE inline
  bool empty() const LVARRAY_RESTRICT_THIS
  { return numNonZeros() == 0; }

  /**
   * @brief @return Return true iff the given row is all zeros.
   * @param row the row to query.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  bool empty( INDEX_TYPE const row ) const LVARRAY_RESTRICT_THIS
  { return numNonZeros( row ) == 0; }

  /**
   * @brief @return Return true iff the given entry is zero.
   * @param row the row to query.
   * @param col the col to query.
   */
  LVARRAY_HOST_DEVICE inline
  bool empty( INDEX_TYPE const row, COL_TYPE const col ) const LVARRAY_RESTRICT_THIS
  { return !ParentClass::contains( row, col ); }

  /**
   * @brief @return Return an ArraySlice1d to the columns of the given row.
   * @param row the row to access.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  ArraySlice< COL_TYPE const, 1, 0, INDEX_TYPE_NC > getColumns( INDEX_TYPE const row ) const LVARRAY_RESTRICT_THIS
  { return (*this)[row]; }

  /**
   * @brief @return Return a pointer to the array of offsets, this array has length numRows() + 1.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE const * getOffsets() const LVARRAY_RESTRICT_THIS
  { return m_offsets.data(); }

  /**
   * @brief Insert a non-zero entry at the given position.
   * @param row the row to insert in.
   * @param col the column to insert.
   * @return True iff the column was inserted (the entry was zero before).
   * @pre Since the SparsityPatternView can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entries.
   */
  LVARRAY_HOST_DEVICE inline
  bool insertNonZero( INDEX_TYPE const row, COL_TYPE const col ) const LVARRAY_RESTRICT_THIS
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( row );
    SPARSITYPATTERN_COLUMN_CHECK( col );
    return ParentClass::insertIntoSet( row, col );
  }

  /// @cond DO_NOT_DOCUMENT
  // This method breaks uncrustify, and it has something to do with the overloading. If it's called something
  // else it works just fine.

  /**
   * @tparam ITER An iterator type.
   * @brief Inserts multiple non-zero entries into the given row.
   * @param row The row to insert into.
   * @param first An iterator to the first column to insert.
   * @param last An iterator to the end of the columns to insert.
   * @return The number of columns inserted.
   * @pre The columns to insert [ @p first, @p last ) must be sorted and contain no duplicates.
   * @pre Since the SparsityPatternview can't do reallocation or shift the offsets it is
   *   up to the user to ensure that the given row has enough space for the new entries.
   */
  template< typename ITER >
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC insertNonZeros( INDEX_TYPE const row, ITER const first, ITER const last ) const LVARRAY_RESTRICT_THIS
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( row );

  #ifdef USE_ARRAY_BOUNDS_CHECK
    for( ITER iter = first; iter != last; ++iter )
    { SPARSITYPATTERN_COLUMN_CHECK( *iter ); }
  #endif

    return ParentClass::insertIntoSet( row, first, last );
  }

  /// @endcond DO_NOT_DOCUMENT

  /**
   * @brief Remove a non-zero entry at the given position.
   * @param row the row to remove from.
   * @param col the column to remove.
   * @return True iff the column was removed (the entry was non zero before).
   */
  LVARRAY_HOST_DEVICE inline
  bool removeNonZero( INDEX_TYPE const row, COL_TYPE const col ) const LVARRAY_RESTRICT_THIS
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( row );
    SPARSITYPATTERN_COLUMN_CHECK( col );
    return ParentClass::removeFromSet( row, col );
  }

  /// @cond DO_NOT_DOCUMENT
  // This method breaks uncrustify, and it has something to do with the overloading. If it's called something
  // else it works just fine.

  /**
   * @tparam ITER An iterator type.
   * @brief Remove multiple non-zero entries from the given row.
   * @param row The row to remove from.
   * @param first An iterator to the first column to remove.
   * @param last An iterator to the end of the columns to remove.
   * @return The number of columns removed.
   * @pre The columns to remove [ @p first, @p last ) must be sorted and contain no duplicates.
   */
  DISABLE_HD_WARNING
  template< typename ITER >
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC removeNonZeros( INDEX_TYPE const row, ITER const first, ITER const last ) const LVARRAY_RESTRICT_THIS
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( row );

  #ifdef USE_ARRAY_BOUNDS_CHECK
    for( ITER iter = first; iter != last; ++iter )
    { SPARSITYPATTERN_COLUMN_CHECK( *iter ); }
  #endif

    return ParentClass::removeFromSet( row, first, last );
  }

  /// @endcond DO_NOT_DOCUMENT

  /**
   * @brief Move this SparsityPattern to the given memory space and touch the values, sizes and offsets.
   * @param space the memory space to move to.
   * @param touch If true touch the values, sizes and offsets in the new space.
   * @note When moving to the GPU since the offsets can't be modified on device they are not touched.
   */
  void move( chai::ExecutionSpace const space, bool const touch=true ) const
  { return ParentClass::move( space, touch ); }

protected:

  /**
   * @brief Default constructor.
   * @note Protected since every SparsityPatternView should either be the base of a
   *   SparsityPattern or copied from another SparsityPatternView.
   */
  SparsityPatternView() = default;

  /**
   * @tparam BUFFERS A variadic pack of buffer types.
   * @brief Resize the SparsityPattern to the given size.
   * @param nrows The number of rows in the matrix.
   * @param ncols The number of columns in the matrix.
   * @param initialRowCapacity The initial capacity of any newly created rows.
   * @param buffers A variadic pack of buffers to resize along with the columns.
   */
  template< class ... BUFFERS >
  void resize( INDEX_TYPE const nrows, INDEX_TYPE const ncols, INDEX_TYPE_NC initialRowCapacity, BUFFERS & ... buffers )
  {
    LVARRAY_ERROR_IF( !arrayManipulation::isPositive( nrows ), "nrows must be positive." );
    LVARRAY_ERROR_IF( !arrayManipulation::isPositive( ncols ), "ncols must be positive." );
    LVARRAY_ERROR_IF( ncols - 1 > std::numeric_limits< COL_TYPE >::max(),
                      "COL_TYPE must be able to hold the range of columns: [0, " << ncols - 1 << "]." );

    if( initialRowCapacity > ncols )
    {
      LVARRAY_WARNING_IF( initialRowCapacity > ncols, "Number of non-zeros per row cannot exceed the the number of columns." );
      initialRowCapacity = ncols;
    }

    m_numCols = ncols;
    ParentClass::resize( nrows, initialRowCapacity, buffers ... );
  }

  // Aliasing protected members in ArrayOfSetsView
  using ParentClass::m_offsets;
  using ParentClass::m_sizes;
  using ParentClass::m_values;

  /// The number of columns in the matrix.
  INDEX_TYPE m_numCols;
};

} /* namespace LvArray */

#endif /* SPARSITYPATTERNVIEW_HPP_ */
