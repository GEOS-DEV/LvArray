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
 * @file ArraySlice.hpp
 */

#ifndef ARRAY_SLICE_HPP_
#define ARRAY_SLICE_HPP_

// Add GDB pretty printers
#ifndef NDEBUG

  #ifndef __APPLE__
/* From: https://sourceware.org/gdb/onlinedocs/gdb/dotdebug_005fgdb_005fscripts-section.html */
    #define DEFINE_GDB_PY_SCRIPT( script_name ) \
  asm ("\
    .pushsection \".debug_gdb_scripts\", \"MS\",@progbits,1\n\
    .byte 1 /* Python */\n\
    .asciz \"" script_name "\"\n\
    .popsection \n\
    ")
  #else
    #define DEFINE_GDB_PY_SCRIPT( script_name )
  #endif

DEFINE_GDB_PY_SCRIPT( "scripts/gdb-printers.py" );

#endif

// Source includes
#include "CXX_UtilsConfig.hpp"
#include "arrayHelpers.hpp"
#include "Macros.hpp"

// System includes
#include <cstring>
#include <vector>
#include <iostream>
#include <utility>

#ifndef NDEBUG
  #include "totalview/tv_data_display.h"
  #include "totalview/tv_helpers.hpp"
#endif

#ifdef USE_ARRAY_BOUNDS_CHECK

#undef CONSTEXPRFUNC
#define CONSTEXPRFUNC

#define ARRAY_SLICE_CHECK_BOUNDS( index ) \
  LVARRAY_ERROR_IF( index < 0 || index >= m_dims[0], \
                    "Array Bounds Check Failed: index=" << index << " m_dims[0]=" << m_dims[0] )

#else // USE_ARRAY_BOUNDS_CHECK

#define ARRAY_SLICE_CHECK_BOUNDS( index )

#endif // USE_ARRAY_BOUNDS_CHECK


namespace LvArray
{

/**
 * @class ArraySlice
 * @brief This class serves to provide a sliced multidimensional interface to the family of LvArray
 *        classes.
 * @tparam T type of data that is contained by the array
 * @tparam NDIM number of dimensions in array (e.g. NDIM=1->vector, NDIM=2->Matrix, etc. ).
 * @tparam USD the dimension with a unit stride, in an Array with a standard layout
 *         this is the last dimension.
 * @tparam INDEX_TYPE the integer to use for indexing the components of the array
 *
 * This class serves as a sliced interface to an array. This is a lightweight class that contains
 * only pointers, and provides an operator[] to create a lower dimensionsal slice and an operator()
 * to access values given a multidimensional index.
 *
 * In general, instantiations of ArraySlice should only result either taking a slice of an an Array or
 * an ArrayView via operator[] or from a direct creation via the toSlice/toSliceConst method.
 */
template< typename T, int NDIM, int USD=NDIM-1, typename INDEX_TYPE=std::ptrdiff_t >
class ArraySlice
{
public:

  static_assert( USD < NDIM, "USD must be less than NDIM." );

  /// deleted default constructor
  ArraySlice() = delete;

  /**
   * @brief Construct a new ArraySlice.
   * @param inputData pointer to the beginning of the data for this slice of the array
   * @param inputDimensions pointer to the beginning of the dimensions for this slice.
   * @param inputStrides pointer to the beginning of the strides for this slice
   */
  LVARRAY_HOST_DEVICE inline explicit CONSTEXPRFUNC
  ArraySlice( T * const LVARRAY_RESTRICT inputData,
              INDEX_TYPE const * const LVARRAY_RESTRICT inputDimensions,
              INDEX_TYPE const * const LVARRAY_RESTRICT inputStrides ) noexcept:
    m_data( inputData ),
    m_dims( inputDimensions ),
    m_strides( inputStrides )
  {
#if defined(USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__) && defined(USE_ARRAY_BOUNDS_CHECK)
    ArraySlice::TV_ttf_display_type( nullptr );
#endif
  }

  /**
   * @brief Return a new immutable slice.
   */
  template< typename U=T >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  ArraySlice< T const, NDIM, USD, INDEX_TYPE >
  toSliceConst() const LVARRAY_RESTRICT_THIS noexcept
  {
    return ArraySlice< T const, NDIM, USD, INDEX_TYPE >( m_data, m_dims, m_strides );
  }

  /**
   * @brief User defined conversion to return a new immutable slice.
   */
  template< typename U=T >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  operator std::enable_if_t< !std::is_const< U >::value,
                             ArraySlice< T const, NDIM, USD, INDEX_TYPE > >
    () const LVARRAY_RESTRICT_THIS noexcept
  {
    return toSliceConst();
  }

  /**
   * @brief User defined conversion to convert an ArraySlice<T,1,0> to a raw pointer.
   */
  template< int _NDIM=NDIM, int _USD=USD >
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline
  operator std::enable_if_t< _NDIM == 1 && _USD == 0, T * const LVARRAY_RESTRICT >
    () const noexcept LVARRAY_RESTRICT_THIS
  { return m_data; }

  /**
   * @brief Return a lower dimensionsal slice of this ArrayView.
   * @param index the index of the slice to create.
   * @note This method is only active when NDIM > 1.
   */
  template< int U=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  std::enable_if_t< (U > 1), ArraySlice< T, NDIM - 1, USD - 1, INDEX_TYPE > >
  operator[]( INDEX_TYPE const index ) const noexcept LVARRAY_RESTRICT_THIS
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return ArraySlice< T, NDIM-1, USD-1, INDEX_TYPE >( m_data + ConditionalMultiply< 0, USD >::multiply( index, m_strides[ 0 ] ),
                                                       m_dims + 1,
                                                       m_strides + 1 );
  }

  /**
   * @brief Return a reference to the value at the given index.
   * @param index index of the element in array to access.
   * @note This method is only active when NDIM == 1.
   */
  template< int U=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  std::enable_if_t< U == 1, T & >
  operator[]( INDEX_TYPE const index ) const noexcept LVARRAY_RESTRICT_THIS
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return m_data[ ConditionalMultiply< 0, USD >::multiply( index, m_strides[ 0 ] ) ];
  }

  /**
   * @brief Return a reference to the value at the given multidimensional index.
   * @tparam INDICES variadic template parameters to serve as index arguments.
   * @param indices the indices of access request.
   * @note This is a standard fortran like parentheses interface to array access.
   */
  template< typename ... INDICES >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T & operator()( INDICES... indices ) const
  {
    static_assert( sizeof ... (INDICES) == NDIM, "number of indices does not match NDIM" );
    return m_data[ linearIndex( indices ... ) ];
  }

  /**
   * @brief Calculates the linear index from a multidimensional index.
   * @tparam INDICES variadic template parameters to serve as index arguments.
   * @param indices the indices of access request.
   */
  template< typename ... INDICES >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  INDEX_TYPE linearIndex( INDICES... indices ) const
  {
    static_assert( sizeof ... (INDICES) == NDIM, "number of indices does not match NDIM" );
#ifdef USE_ARRAY_BOUNDS_CHECK
    checkIndices( m_dims, indices ... );
#endif
    return getLinearIndex< USD >( m_strides, indices ... );
  }

  /**
   * @brief Return the allocated size.
   */
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  INDEX_TYPE size() const noexcept
  { return multiplyAll< NDIM >( m_dims ); }

  /**
   * @brief Return the length of the given dimension.
   * @param dim the dimension to get the length of.
   */
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  INDEX_TYPE size( int dim ) const noexcept
  {
#ifdef USE_ARRAY_BOUNDS_CHECK
    LVARRAY_ERROR_IF_GE( dim, NDIM );
#endif
    return m_dims[dim];
  }

  /**
   * @brief Return true iff the given pointer matches the data pointer of this ArraySlice.
   * @param ptr the pointer to check.
   */
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  bool operator==( T const * const ptr ) const
  {
    return m_data == ptr;
  }

  /**
   * @brief Check if the slice is contiguous in memory
   * @tparam _USD dummy template parameter equal to USD; do not replace
   * @return @p true if represented slice is contiguous in memory
   */
  template< int _USD = USD >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  std::enable_if_t< ( _USD >= 0), bool >
  isContiguous() const
  {
    bool rval = true;
    for( int i = 0 ; i < NDIM ; ++i )
    {
      if( i == _USD ) continue;
      INDEX_TYPE prod = 1;
      for( int j = 0 ; j < NDIM ; ++j )
      {
        if( j != i ) prod *= m_dims[j];
      }
      rval &= (m_strides[i] <= prod);
    }
    return rval;
  }

  /**
   * @brief Check if the slice is contiguous in memory
   * @tparam USD_ dummy template parameter equal to USD; do not replace
   * @return @p false, this overload is enabled for slices that
   *         have already lost its unit stride dimension
   */
  template< int USD_ = USD >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  std::enable_if_t< (USD_ < 0), bool >
  isContiguous() const
  {
    return false;
  }

  /**
   * @brief Return a pointer to the values.
   * @pre The slice must be contiguous
   */
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T * data() const
  {
    LVARRAY_ASSERT( isContiguous() );
    return m_data;
  }

#if defined(USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__) && defined(USE_ARRAY_BOUNDS_CHECK)
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

/**
 * @tparam T The type of @p value.
 * @tparam LAMBDA the type of the function @p f to apply.
 * @brief Apply the given function to the given value.
 * @param value the value to pass to the function.
 * @param f the function to apply to the value.
 */
DISABLE_HD_WARNING
template< typename T, typename LAMBDA >
LVARRAY_HOST_DEVICE
void forValuesInSlice( T & value, LAMBDA && f )
{ f( value ); }

/**
 * @tparam T The type of values stored in @p slice.
 * @tparam NDIM the dimension of @p slice.
 * @tparam USD the unit stride dimension of @p slice.
 * @tparam INDEX_TYPE the integer used to index into @p slice.
 * @tparam LAMBDA the type of the function @p f to apply.
 * @brief Iterate over the values in the slice in lexicographic order.
 * @param slice the slice to iterate over.
 * @param f the function to apply to each value.
 */
DISABLE_HD_WARNING
template< typename T, int NDIM, int USD, typename INDEX_TYPE, typename LAMBDA >
LVARRAY_HOST_DEVICE
void forValuesInSlice( ArraySlice< T, NDIM, USD, INDEX_TYPE > const & slice, LAMBDA && f )
{
  for( INDEX_TYPE i = 0 ; i < slice.size( 0 ) ; ++i )
  {
    forValuesInSlice( slice[ i ], f );
  }
}

/**
 * @tparam T The type of values stored in @p slice.
 * @tparam INDICES variadic pack of indices.
 * @tparam LAMBDA the type of the function @p f to apply.
 * @brief Apply the funtion @p f to the value @p value also
 *        passing @p f any indices used to reach @p value.
 * @param slice the slice to iterate over.
 * @param f the lambda to apply to each value.
 * @param indices the previous sliced off indices.
 */
DISABLE_HD_WARNING
template< typename T, typename LAMBDA, typename ... INDICES >
LVARRAY_HOST_DEVICE
void forValuesInSliceWithIndices( T & value, LAMBDA && f, INDICES const ... indices )
{ f( value, indices ... ); }

/**
 * @tparam T The type of values stored in @p slice.
 * @tparam NDIM the dimension of @p slice.
 * @tparam USD the unit stride dimension of @p slice.
 * @tparam INDEX_TYPE the integer used to index into @p slice.
 * @tparam INDICES variadic pack of indices.
 * @tparam LAMBDA the type of the function @p f to apply.
 * @brief Iterate over the values in the slice in lexicographic order, passing the indices as
 *        well as the value to the lambda.
 * @param slice the slice to iterate over.
 * @param f the lambda to apply to each value.
 * @param indices the previous sliced off indices.
 */
DISABLE_HD_WARNING
template< typename T, int NDIM, int USD, typename INDEX_TYPE, typename LAMBDA, typename ... INDICES >
LVARRAY_HOST_DEVICE
void forValuesInSliceWithIndices( ArraySlice< T, NDIM, USD, INDEX_TYPE > const & slice,
                                  LAMBDA && f,
                                  INDICES const ... indices )
{
  for( INDEX_TYPE i = 0 ; i < slice.size( 0 ) ; ++i )
  {
    forValuesInSliceWithIndices( slice[ i ], f, indices ..., i );
  }
}

}

#endif /* SRC_COMPONENTS_CORE_SRC_ARRAY_MULTIDIMENSIONALARRAY_HPP_ */
