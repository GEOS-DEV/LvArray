/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file SparsityPatternView.hpp
 * @brief Contains the implementation of LvArray:SparsityPatternView.
 */

#pragma once

// Source includes
#include "ArrayOfSetsView.hpp"

// System includes
#include <limits>

#ifdef LVARRAY_BOUNDS_CHECK

/**
 * @brief Check that @p col is a valid column in the matrix.
 * @param col The column to check.
 * @note This macro is only active with LVARRAY_BOUNDS_CHECK.
 */
#define SPARSITYPATTERN_COLUMN_CHECK( col ) \
  LVARRAY_ERROR_IF( !arrayManipulation::isPositive( col ) || col >= this->numColumns(), \
                    "Column Check Failed: col=" << col << " numColumns=" << this->numColumns() )

#else // LVARRAY_BOUNDS_CHECK

/**
 * @brief Check that @p col is a valid column in the matrix.
 * @param col The column to check.
 * @note This macro is only active with LVARRAY_BOUNDS_CHECK.
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
template< typename COL_TYPE,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class SparsityPatternView : protected ArrayOfSetsView< COL_TYPE, INDEX_TYPE, BUFFER_TYPE >
{
protected:
  using COL_TYPE_NC = std::remove_const_t< COL_TYPE >;
  /// An alias for the parent class.
  using ParentClass = ArrayOfSetsView< COL_TYPE, INDEX_TYPE, BUFFER_TYPE >;

  /// An alias for the non const index type.
  using typename ParentClass::INDEX_TYPE_NC;

  using typename ParentClass::SIZE_TYPE;

public:
  static_assert( std::is_integral< COL_TYPE >::value, "COL_TYPE must be integral." );
  static_assert( std::is_integral< INDEX_TYPE >::value, "INDEX_TYPE must be integral." );

  /// The integer type used to enumerate the columns.

  using ColType = COL_TYPE;
  using ColTypeNC = COL_TYPE_NC;
  using typename ParentClass::IndexType;

  /**
   * @name Constructors, destructor and assignment operators
   */
  ///@{

  /**
   * @brief A constructor to create an uninitialized SparsityPatternView.
   * @note An uninitialized SparsityPatternView should not be used until it is assigned to.
   */
  SparsityPatternView() = default;

  /**
   * @brief Default copy constructor.
   * @note The copy constructor will trigger the copy constructor for @tparam BUFFER_TYPE
   */
  inline
  SparsityPatternView( SparsityPatternView const & ) = default;

  /**
   * @brief Move constructor.
   * @param src The SparsityPatternView to move from.
   */
  inline
  SparsityPatternView( SparsityPatternView && src ):
    ParentClass( std::move( src ) ),
    m_numCols( src.m_numCols )
  { src.m_numCols = 0; }

  /**
   * @brief Construct a new CRSMatrixView from the given buffers.
   * @param nRows The number of rows.
   * @param nCols The number of columns
   * @param offsets The offsets buffer, of size @p nRows + 1.
   * @param nnz The buffer containing the number of non zeros in each row, of size @p nRows.
   * @param columns The columns buffer, of size @p offsets[ nRows ].
   */
  LVARRAY_HOST_DEVICE constexpr inline
  SparsityPatternView( INDEX_TYPE const nRows,
                       COL_TYPE const nCols,
                       BUFFER_TYPE< INDEX_TYPE > const & offsets,
                       BUFFER_TYPE< SIZE_TYPE > const & nnz,
                       BUFFER_TYPE< COL_TYPE > const & columns ):
    ParentClass( nRows, offsets, nnz, columns ),
    m_numCols( nCols )
  {}

  /**
   * @brief Default copy assignment operator, this does a shallow copy.
   * @return *this.
   */
  inline
  SparsityPatternView & operator=( SparsityPatternView const & ) = default;

  /**
   * @brief Move assignment operator, this does a shallow copy.
   * @param src The SparsityPatternView to move from.
   * @return *this.
   */
  inline
  SparsityPatternView & operator=( SparsityPatternView && src )
  {
    ParentClass::operator=( std::move( src ) );
    m_numCols = src.m_numCols;
    src.m_numCols = 0;
    return *this;
  }

  ///@}

  /**
   * @name SparsityPatternView creation methods
   */
  ///@{


  /**
   * @return A new SparsityPatternView< COL_TYPE, INDEX_TYPE const >.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  SparsityPatternView< COL_TYPE, INDEX_TYPE const, BUFFER_TYPE >
  toView() const
  {
    return SparsityPatternView< COL_TYPE, INDEX_TYPE const, BUFFER_TYPE >( numRows(),
                                                                           numColumns(),
                                                                           this->m_offsets,
                                                                           this->m_sizes,
                                                                           this->m_values );
  }

  /**
   * @return A new SparsityPatternView< COL_TYPE const, INDEX_TYPE const >.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  SparsityPatternView< COL_TYPE const, INDEX_TYPE const, BUFFER_TYPE >
  toViewConst() const
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

  /**
   * @return Return the number of rows in the matrix.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE_NC const numRows() const
  { return ParentClass::size(); }

  /**
   * @return Return the number of columns in the matrix.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  COL_TYPE const numColumns() const
  { return m_numCols; }

  /**
   * @return Return the total number of non zero entries in the matrix.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE_NC numNonZeros() const
  {
    INDEX_TYPE_NC nnz = 0;
    for( INDEX_TYPE_NC row = 0; row < numRows(); ++row )
    {
      nnz += numNonZeros( row );
    }

    return nnz;
  }

  /**
   * @return Return the total number of non zero entries in the given row.
   * @param row the row to query.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE_NC const numNonZeros( INDEX_TYPE const row ) const
  { return ParentClass::sizeOfSet( row ); }

  /**
   * @return Return the total number of non zero entries able to be stored without a reallocation.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE_NC const nonZeroCapacity() const
  { return ParentClass::valueCapacity(); }

  /**
   * @return Return the total number of non zero entries able to be stored in a given row without increasing its
   * capacity.
   * @param row the row to query.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE_NC const nonZeroCapacity( INDEX_TYPE const row ) const
  { return ParentClass::capacityOfSet( row ); }

  /**
   * @return Return true iff the matrix is all zeros.
   */
  LVARRAY_HOST_DEVICE inline
  bool empty() const
  { return numNonZeros() == 0; }

  /**
   * @return Return true iff the given row is all zeros.
   * @param row the row to query.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  bool empty( INDEX_TYPE const row ) const
  { return numNonZeros( row ) == 0; }

  /**
   * @return Return true iff the given entry is in the matrix.
   * @param row the row to query.
   * @param col the col to query.
   */
  LVARRAY_HOST_DEVICE inline
  bool empty( INDEX_TYPE const row, COL_TYPE const col ) const
  { return !ParentClass::contains( row, col ); }

  ///@}

  /**
   * @name Methods that provide access to the data
   */
  ///@{

  /**
   * @return Return an ArraySlice1d to the columns of the given row.
   * @param row the row to access.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  ArraySlice< COL_TYPE const, 1, 0, INDEX_TYPE_NC >getColumns( INDEX_TYPE const row ) const
  { return (*this)[row]; }

  /**
   * @return Return a pointer to the array of offsets, this array has length numRows() + 1.
   */
  LVARRAY_HOST_DEVICE constexpr inline
  INDEX_TYPE const * getOffsets() const
  { return this->m_offsets.data(); }

  /**
   * @return Return a pointer to the array of values, this array has length numRows().
   */
  LVARRAY_HOST_DEVICE constexpr inline
  COL_TYPE const * getColumns() const
  {
    return this->getValues();
  }

  using ParentClass::getSizes;


  ///@}

  /**
   * @name Methods that insert or remove entries from a row.
   */
  ///@{

  /**
   * @brief Insert a non-zero entry at the given position.
   * @param row the row to insert in.
   * @param col the column to insert.
   * @return True iff the column was inserted (the entry was zero before).
   * @pre Since the SparsityPatternView can't do reallocation or shift the offsets it is
   *       up to the user to ensure that the given row has enough space for the new entries.
   */
  LVARRAY_HOST_DEVICE inline
  bool insertNonZero( INDEX_TYPE const row, COL_TYPE const col ) const
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( row );
    SPARSITYPATTERN_COLUMN_CHECK( col );
    return ParentClass::insertIntoSet( row, col );
  }

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
  INDEX_TYPE_NC insertNonZeros( INDEX_TYPE const row, ITER const first, ITER const last ) const
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( row );

  #ifdef LVARRAY_BOUNDS_CHECK
    for( ITER iter = first; iter != last; ++iter )
    { SPARSITYPATTERN_COLUMN_CHECK( *iter ); }
  #endif

    return ParentClass::insertIntoSet( row, first, last );
  }

  /**
   * @brief Remove a non-zero entry at the given position.
   * @param row the row to remove from.
   * @param col the column to remove.
   * @return True iff the column was removed (the entry was non zero before).
   */
  LVARRAY_HOST_DEVICE inline
  bool removeNonZero( INDEX_TYPE const row, COL_TYPE const col ) const
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( row );
    SPARSITYPATTERN_COLUMN_CHECK( col );
    return ParentClass::removeFromSet( row, col );
  }

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
  INDEX_TYPE_NC removeNonZeros( INDEX_TYPE const row, ITER const first, ITER const last ) const
  {
    ARRAYOFARRAYS_CHECK_BOUNDS( row );

  #ifdef LVARRAY_BOUNDS_CHECK
    for( ITER iter = first; iter != last; ++iter )
    { SPARSITYPATTERN_COLUMN_CHECK( *iter ); }
  #endif

    return ParentClass::removeFromSet( row, first, last );
  }

  ///@}

  /**
   * @name Methods dealing with memory spaces
   */
  ///@{

  /**
   * @copydoc ArrayOfSets::move
   * @note This is just a wrapper around the ArrayOfSetsView method. The reason
   *   it isn't pulled in with a @c using statement is that it is detected using
   *   IS_VALID_EXPRESSION and this fails with NVCC.
   */
  void move( MemorySpace const space, bool const touch=true ) const
  { return ParentClass::move( space, touch ); }

  ///@}

protected:

  /**
   * @brief Protected constructor to be used by parent classes.
   * @note The unused boolean parameter is to distinguish this from the default constructor.
   */
  SparsityPatternView( bool ):
    ParentClass( true )
  {};

  /**
   * @brief Steal the resources of @p src, clearing it in the process.
   * @param src The SparsityPatternView to steal from.
   */
  void assimilate( SparsityPatternView && src )
  {
    ParentClass::free();
    *this = std::move( src );
  }

  /**
   * @tparam BUFFERS A variadic pack of buffer types.
   * @brief Resize the SparsityPattern to the given size.
   * @param nrows The number of rows in the matrix.
   * @param ncols The number of columns in the matrix.
   * @param initialRowCapacity The initial capacity of any newly created rows.
   * @param buffers A variadic pack of buffers to resize along with the columns.
   */
  template< class ... BUFFERS >
  void resize( INDEX_TYPE const nrows,
               COL_TYPE const ncols,
               INDEX_TYPE_NC initialRowCapacity,
               BUFFERS & ... buffers )
  {
    LVARRAY_ERROR_IF( !arrayManipulation::isPositive( nrows ), "nrows must be positive." );
    LVARRAY_ERROR_IF( !arrayManipulation::isPositive( ncols ), "ncols must be positive." );
    LVARRAY_ERROR_IF( ncols - 1 > std::numeric_limits< COL_TYPE >::max(),
                      "COL_TYPE must be able to hold the range of columns: [0, " << ncols - 1 << "]." );

    m_numCols = ncols;
    ParentClass::resizeImpl( nrows, initialRowCapacity, buffers ... );
  }

  /// The number of columns in the matrix.
  COL_TYPE_NC m_numCols;
};

} // namespace LvArray
