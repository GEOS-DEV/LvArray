/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file ArraySlice.hpp
 * @brief Contains the implementation of LvArray::ArraySlice
 */

#pragma once

#if !defined( NDEBUG ) && !defined( __APPLE__ ) && !defined( __ibmxl__ )
/**
 * @brief Add GDB pretty printers the given script.
 * @param script_name The python script that contains the gdb hooks.
 * @note Taken from https://sourceware.org/gdb/onlinedocs/gdb/dotdebug_005fgdb_005fscripts-section.html
 */
#define DEFINE_GDB_PY_SCRIPT( script_name ) \
  asm (".pushsection \".debug_gdb_scripts\", \"MS\",@progbits,1\n \
                .byte 1 /* Python */\n \
                .asciz \"" script_name "\"\n \
                .popsection \n" );
#else
/**
 * @brief Add GDB pretty printers for OSX. This hasn't been done yet.
 * @param script_name The python script that contains the gdb hooks.
 */
#define DEFINE_GDB_PY_SCRIPT( script_name )
#endif

/// Point GDB at the scripts/gdb-printers.py
DEFINE_GDB_PY_SCRIPT( "scripts/gdb-printers.py" )

// Source includes
#include "LvArrayConfig.hpp"
#include "indexing.hpp"
#include "Macros.hpp"

// System includes
#ifndef NDEBUG
  #include "totalview/tv_data_display.h"
  #include "totalview/tv_helpers.hpp"
#endif

#ifdef LVARRAY_BOUNDS_CHECK

/**
 * @brief Check that @p index is a valid index into the first dimension.
 * @param index The index to check.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define ARRAY_SLICE_CHECK_BOUNDS( index ) \
  LVARRAY_ERROR_IF( index < 0 || index >= m_dims[ 0 ], \
                    "Array Bounds Check Failed: index=" << index << " m_dims[0]=" << m_dims[0] )

#else // LVARRAY_BOUNDS_CHECK

/**
 * @brief Check that @p index is a valid index into the first dimension.
 * @param index The index to check.
 * @note This is only active when LVARRAY_BOUNDS_CHECK is defined.
 */
#define ARRAY_SLICE_CHECK_BOUNDS( index )

#endif // LVARRAY_BOUNDS_CHECK


namespace LvArray
{

/**
 * @class ArraySlice
 * @brief This class serves to provide a sliced multidimensional interface to the family of LvArray classes.
 * @tparam T type of data that is contained by the array
 * @tparam NDIM_TPARAM The number of dimensions in array (e.g. NDIM=1->vector, NDIM=2->Matrix, etc. ).
 * @tparam USD The dimension with a unit stride, in an Array with a standard layout
 *   this is the last dimension.
 * @tparam INDEX_TYPE The integer to use for indexing the components of the array.
 * @brief This class serves as a sliced interface to an array. This is a lightweight class that contains
 *   only pointers, and provides an operator[] to create a lower dimensionsal slice and an operator()
 *   to access values given a multidimensional index.
 *   In general, instantiations of ArraySlice should only result either taking a slice of an an Array or
 *   an ArrayView via operator[] or from a direct creation via the toSlice/toSliceConst method.
 */
template< typename T, int NDIM_TPARAM, int USD_TPARAM, typename INDEX_TYPE >
class ArraySlice
{
public:

  static_assert( USD_TPARAM < NDIM_TPARAM, "USD must be less than NDIM." );

  /// The type of the value in the ArraySlice.
  using ValueType = T;

  /// The number of dimensions.
  static constexpr int NDIM = NDIM_TPARAM;

  /// The unit stride dimension.
  static constexpr int USD = USD_TPARAM;

  /// The integer type used for indexing.
  using IndexType = INDEX_TYPE;

  /**
   * @name Constructors, destructor and assignment operators.
   */
  ///@{

  /// deleted default constructor
  ArraySlice() = delete;

  /**
   * @brief Construct a new ArraySlice.
   * @param inputData pointer to the beginning of the data for this slice of the array
   * @param inputDimensions pointer to the beginning of the dimensions for this slice.
   * @param inputStrides pointer to the beginning of the strides for this slice
   */
  LVARRAY_HOST_DEVICE inline explicit CONSTEXPR_WITHOUT_BOUNDS_CHECK
  ArraySlice( T * const LVARRAY_RESTRICT inputData,
              INDEX_TYPE const * const LVARRAY_RESTRICT inputDimensions,
              INDEX_TYPE const * const LVARRAY_RESTRICT inputStrides ) noexcept:
    m_data( inputData ),
    m_dims( inputDimensions ),
    m_strides( inputStrides )
  {
#if defined(LVARRAY_USE_TOTALVIEW_OUTPUT) && !defined(LVARRAY_DEVICE_COMPILE) && defined(LVARRAY_BOUNDS_CHECK)
    ArraySlice::TV_ttf_display_type( nullptr );
#endif
  }

  ///@}

  /**
   * @name ArraySlice creation methods and user defined conversions
   */
  ///@{

  /**
   * @return Return a new immutable slice.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  ArraySlice< T const, NDIM, USD, INDEX_TYPE >
  toSliceConst() const noexcept
  { return ArraySlice< T const, NDIM, USD, INDEX_TYPE >( m_data, m_dims, m_strides ); }

  /**
   * @return Return a new immutable slice.
   */
  template< typename U=T >
  LVARRAY_HOST_DEVICE inline constexpr
  operator std::enable_if_t< !std::is_const< U >::value,
                             ArraySlice< T const, NDIM, USD, INDEX_TYPE > >
    () const noexcept
  { return toSliceConst(); }

  ///@}

  /**
   * @name Attribute querying methods
   */
  ///@{

  /**
   * @return Return the total size of the slice.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  INDEX_TYPE size() const noexcept
  {
  #if defined( __ibmxl__ )
    // Note: This used to be done with a recursive template but XL-release would produce incorrect results.
    // Specifically in exampleArray it would return an "old" size even after being updated, strange.
    INDEX_TYPE val = m_dims[ 0 ];
    for( int i = 1; i < NDIM; ++i )
    { val *= m_dims[ i ]; }

    return val;
  #else
    return indexing::multiplyAll< NDIM >( m_dims );
  #endif
  }

  /**
   * @return Return the length of the given dimension.
   * @param dim the dimension to get the length of.
   */
  LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
  INDEX_TYPE size( int const dim ) const noexcept
  {
#ifdef LVARRAY_BOUNDS_CHECK
    LVARRAY_ERROR_IF_GE( dim, NDIM );
#endif
    return m_dims[ dim ];
  }

  /**
   * @return Return the stride of the given dimension.
   * @param dim the dimension to get the stride of.
   */
  LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
  INDEX_TYPE stride( int const dim ) const noexcept
  {
#ifdef LVARRAY_BOUNDS_CHECK
    LVARRAY_ERROR_IF_GE( dim, NDIM );
#endif
    return m_strides[ dim ];
  }

  /**
   * @brief Check if the slice is contiguous in memory
   * @return @p true if represented slice is contiguous in memory
   */
  LVARRAY_HOST_DEVICE inline constexpr
  bool isContiguous() const
  {
    if( USD < 0 ) return false;
    if( NDIM == 1 && USD == 0 ) return true;

    bool rval = true;
    for( int i = 0; i < NDIM; ++i )
    {
      if( i == USD ) continue;
      INDEX_TYPE prod = 1;
      for( int j = 0; j < NDIM; ++j )
      {
        if( j != i ) prod *= m_dims[j];
      }
      rval &= (m_strides[i] <= prod);
    }
    return rval;
  }

  /**
   * @tparam INDICES A variadic pack of integral types.
   * @return Return the linear index from a multidimensional index.
   * @param indices The indices of the value to get the linear index of.
   */
  template< typename ... INDICES >
  LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
  INDEX_TYPE linearIndex( INDICES... indices ) const
  {
    static_assert( sizeof ... (INDICES) == NDIM, "number of indices does not match NDIM" );
#ifdef LVARRAY_BOUNDS_CHECK
    indexing::checkIndices( m_dims, indices ... );
#endif
    return indexing::getLinearIndex< USD >( m_strides, indices ... );
  }

  ///@}

  /**
   * @name Methods that provide access to the data.
   */
  ///@{

  /**
   * @return A raw pointer.
   * @note This method is only active when NDIM == 1 and USD == 0.
   */
  template< int NDIM_=NDIM, int USD_=USD >
  LVARRAY_HOST_DEVICE constexpr inline
  operator std::enable_if_t< NDIM_ == 1 && USD_ == 0, T * const LVARRAY_RESTRICT >
    () const noexcept
  { return m_data; }

  /**
   * @return Return a lower dimensionsal slice of this ArrayView.
   * @param index The index of the slice to create.
   * @note This method is only active when NDIM > 1.
   */
  template< int U=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
  std::enable_if_t< (U > 1), ArraySlice< T, NDIM - 1, USD - 1, INDEX_TYPE > >
  operator[]( INDEX_TYPE const index ) const noexcept
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return ArraySlice< T, NDIM-1, USD-1, INDEX_TYPE >( m_data + indexing::ConditionalMultiply< USD == 0 >::multiply( index, m_strides[ 0 ] ),
                                                       m_dims + 1,
                                                       m_strides + 1 );
  }

  /**
   * @return Return a reference to the value at the given index.
   * @param index The index of the value to access.
   * @note This method is only active when NDIM == 1.
   */
  template< int U=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
  std::enable_if_t< U == 1, T & >
  operator[]( INDEX_TYPE const index ) const noexcept
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return m_data[ indexing::ConditionalMultiply< USD == 0 >::multiply( index, m_strides[ 0 ] ) ];
  }

  /**
   * @tparam INDICES A variadic pack of integral types.
   * @return Return a reference to the value at the given multidimensional index.
   * @param indices The indices of the value to access.
   */
  template< typename ... INDICES >
  LVARRAY_HOST_DEVICE inline constexpr
  T & operator()( INDICES... indices ) const
  {
    static_assert( sizeof ... (INDICES) == NDIM, "number of indices does not match NDIM" );
    return m_data[ linearIndex( indices ... ) ];
  }

  /**
   * @return Return a pointer to the values.
   * @tparam USD_ Dummy template parameter, do not specify.
   * @pre The slice must be contiguous.
   */
  template< int USD_ = USD >
  LVARRAY_HOST_DEVICE inline
  T * dataIfContiguous() const
  {
    // Note: need both compile-time and runtime checks as USD >= 0 does not guarantee contiguous data.
    static_assert( USD_ >= 0, "Direct data access not supported for non-contiguous slices" );
    LVARRAY_ERROR_IF( !isContiguous(), "Direct data access not supported for non-contiguous slices" );
    return m_data;
  }

  /**
   * @return Return a pointer to the values.
   * @pre The slice must be contiguous.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  T * begin() const
  { return dataIfContiguous(); }

  /**
   * @return Return a pointer to the end values.
   * @pre The slice must be contiguous.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  T * end() const
  { return dataIfContiguous() + size(); }

  ///@}

#if defined(LVARRAY_USE_TOTALVIEW_OUTPUT) && !defined(LVARRAY_DEVICE_COMPILE) && defined(LVARRAY_BOUNDS_CHECK)
  /**
   * @brief Static function that will be used by Totalview to display the array contents.
   * @param av A pointer to the array that is being displayed.
   * @return 0 if everything went OK
   */
  static int TV_ttf_display_type( ArraySlice const * av )
  {
    if( av!=nullptr )
    {
      int constexpr ndim = NDIM;
      //std::cout<<"Totalview using ("<<totalview::format<T,INDEX_TYPE>(NDIM, av->m_dims )<<") for display of
      // m_data;"<<std::endl;
      TV_ttf_add_row( "tv(m_data)", totalview::format< T, INDEX_TYPE >( NDIM, av->m_dims ).c_str(), (av->m_data) );
      TV_ttf_add_row( "m_data", totalview::format< T, INDEX_TYPE >( 1, av->m_dims ).c_str(), (av->m_data) );
      TV_ttf_add_row( "m_dims", totalview::format< INDEX_TYPE, int >( 1, &ndim ).c_str(), (av->m_dims) );
      TV_ttf_add_row( "m_strides", totalview::format< INDEX_TYPE, int >( 1, &ndim ).c_str(), (av->m_strides) );
    }
    return 0;
  }
#endif

protected:
  /// pointer to beginning of data for this array, or sub-array.
  T * const LVARRAY_RESTRICT m_data;

  /// pointer to array of length NDIM that contains the lengths of each array dimension
  INDEX_TYPE const * const LVARRAY_RESTRICT m_dims;

  /// pointer to array of length NDIM that contains the strides of each array dimension
  INDEX_TYPE const * const LVARRAY_RESTRICT m_strides;

};

} // namespace LvArray
