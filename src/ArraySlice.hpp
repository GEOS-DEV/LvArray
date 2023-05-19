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
#include "Layout.hpp"
#include "indexing.hpp"
#include "Macros.hpp"
#include "sortedArrayManipulation.hpp"

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
 *   only pointers, and provides an operator[] to create a lower dimensional slice and an operator()
 *   to access values given a multidimensional index.
 *   In general, instantiations of ArraySlice should only result either taking a slice of an an Array or
 *   an ArrayView via operator[] or from a direct creation via the toSlice/toSliceConst method.
 */
template< typename T, typename LAYOUT >
class ArraySlice
{
public:

  static_assert( is_layout_v< LAYOUT >, "LAYOUT template parameter must be a Layout instantiation" );

  /// The type of layout.
  using LayoutType = LAYOUT;

  /// The type of extent.
  using Extent = typename LayoutType::ExtentType;

  /// The type of stride.
  using Stride = typename LayoutType::StrideType;

  /// The number of dimensions.
  static constexpr int NDIM = LayoutType::NDIM;

  /// The unit-stride dimension index.
  static constexpr int USD = LayoutType::USD;

  /// The type of the value in the ArraySlice.
  using ValueType = T;

  /// The integer type used for indexing.
  using IndexType = typename LayoutType::IndexType;

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
  LVARRAY_HOST_DEVICE explicit CONSTEXPR_WITHOUT_BOUNDS_CHECK
  ArraySlice( T * const LVARRAY_RESTRICT inputData,
               LayoutType const & layout ) noexcept:
    m_data( inputData ),
    m_layout( layout )
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
  LVARRAY_HOST_DEVICE constexpr
  ArraySlice< T const, LayoutType >
  toSliceConst() const noexcept
  { return ArraySlice< T const, LayoutType >( m_data, m_layout ); }

  /**
   * @return Return a new immutable slice.
   */
  template< typename U=T >
  LVARRAY_HOST_DEVICE constexpr
  operator std::enable_if_t< !std::is_const< U >::value,
                             ArraySlice< T const, LayoutType > >
    () const noexcept
  { return toSliceConst(); }

  ///@}

  /**
   * @name Attribute querying methods
   */
  ///@{

  LVARRAY_HOST_DEVICE constexpr
  auto const &
  layout() const noexcept
  {
    return m_layout;
  }

  /**
   * @return Return the total size of the slice.
   */
  LVARRAY_HOST_DEVICE constexpr
  auto
  size() const noexcept
  {
    return m_layout.size();
  }

  /**
   * @return Return the length of the given dimension.
   * @param dim the dimension to get the length of.
   */
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK
  auto
  size( int const dim ) const noexcept
  {
#ifdef LVARRAY_BOUNDS_CHECK
    LVARRAY_ERROR_IF_GE( dim, NDIM );
#endif
    return tupleManipulation::getValue( m_layout.extent(), dim );
  }

  /**
   * @return Return the length of the given dimension.
   * @param N the dimension to get the length of.
   */
  template< int N >
  LVARRAY_HOST_DEVICE constexpr
  IndexType size() const noexcept
  {
    static_assert( N < NDIM, "Dimension index must be less than number of dimensions." );
    return camp::get< N >( m_layout.extent() );
  }

  /**
   * @return Return the stride of the given dimension.
   * @param dim the dimension to get the stride of.
   */
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK
  IndexType stride( int const dim ) const noexcept
  {
#ifdef LVARRAY_BOUNDS_CHECK
    LVARRAY_ERROR_IF_GE( dim, NDIM );
#endif
    return tupleManipulation::getValue( m_layout.stride(), dim );
  }

  /**
   * @return Return the stride of the given dimension.
   * @param N the dimension to get the stride of.
   */
  template< int N >
  LVARRAY_HOST_DEVICE constexpr
  auto stride() const noexcept
  {
    static_assert( N < NDIM, "Dimension index must be less than number of dimensions." );
    return camp::get< N >( m_layout.stride() );
  }

  /**
   * @brief Check if the slice is contiguous in memory
   * @return @p true if represented slice is contiguous in memory
   */
  LVARRAY_HOST_DEVICE constexpr
  bool isContiguous() const
  {
    // Deal with obvious cases
    if( USD < 0 ) return false;
    if( NDIM == 1 && USD == 0 ) return true;

    // Slice is contiguous when strides are an ex-scan of extents.
    // Since they might be permuted, we need to sort them first.
    // Can't sort tuples directly, so decay to arrays.
    // XXX: Is there an easier/faster way?

    auto sortedExtents = tupleManipulation::toArray( m_layout.extent() );
    auto sortedStrides = tupleManipulation::toArray( m_layout.stride() );
    sortedArrayManipulation::dualSort( sortedStrides.data, sortedStrides.data + NDIM, sortedExtents.data );

    IndexType stride = 1;
    for( int i = 0; i < NDIM; ++i )
    {
      if ( stride != sortedStrides[i] )
      {
        return false;
      }
      stride *= sortedExtents[i];
    }
    return true;
  }

  /**
   * @tparam INDICES A variadic pack of integral types.
   * @return Return the linear index from a multidimensional index.
   * @param indices The indices of the value to get the linear index of.
   */
  template< typename ... INDICES >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK
  IndexType linearIndex( INDICES... indices ) const
  {
    static_assert( sizeof ... (INDICES) == NDIM, "number of indices does not match NDIM" );
    return m_layout( indices... );
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
  LVARRAY_HOST_DEVICE constexpr
  operator std::enable_if_t< NDIM_ == 1 && USD_ == 0, T * const LVARRAY_RESTRICT >
    () const noexcept
  { return m_data; }

  /**
   * @return Return a reference to the value at the given index.
   * @param index The index of the value to access.
   * @note This method is only active when NDIM == 1.
   */
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK
  decltype(auto)
  operator[]( IndexType const index ) const noexcept
  {
    return operator()( index );
  }

  /**
   * @tparam INDICES A variadic pack of integral types.
   * @return Return a reference to the value at the given multidimensional index or a lower-dimensional slice.
   * @param indices The indices of the value to access.
   *
   * If the number of indices matches the number of slice dimensions, and none of the indices are a slicer (_{})
   * a value reference is returned. Otherwise the set of indices is applied to the left-most dimensions, and
   * a slice representing the remaining dimension is returned.
   */
  template< typename ... INDICES >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK
  decltype(auto)
  operator()( INDICES... indices ) const
  {
    auto const sliceLayout = m_layout.slice( indices... );
    auto const sliceOffset = m_layout( indices... );
    if constexpr ( decltype( sliceLayout )::NDIM == 0 )
    {
      return m_data[ sliceOffset ];
    }
    else
    {
      return ArraySlice< T, std::remove_cv_t< decltype( sliceLayout ) > >( m_data + sliceOffset, sliceLayout );
    }
  }

  /**
   * @return Return a pointer to the values.
   * @tparam USD_ Dummy template parameter, do not specify.
   * @pre The slice must be contiguous.
   */
  template< int USD_ = USD >
  LVARRAY_HOST_DEVICE
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
  LVARRAY_HOST_DEVICE constexpr
  T * begin() const
  { return dataIfContiguous(); }

  /**
   * @return Return a pointer to the end values.
   * @pre The slice must be contiguous.
   */
  LVARRAY_HOST_DEVICE constexpr
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

  /// layout describing the mapping from coordinates to memory offsets
  LayoutType m_layout;

};

} // namespace LvArray
