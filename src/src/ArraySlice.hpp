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
    #define DEFINE_GDB_PY_SCRIPT(script_name) \
    asm("\
    .pushsection \".debug_gdb_scripts\", \"MS\",@progbits,1\n\
    .byte 1 /* Python */\n\
    .asciz \"" script_name "\"\n\
    .popsection \n\
    ")
  #else
    #define DEFINE_GDB_PY_SCRIPT(script_name)
  #endif

  DEFINE_GDB_PY_SCRIPT("scripts/gdb-printers.py");

#endif

#include <cstring>
#include <vector>
#include <iostream>
#include <utility>
#include "CXX_UtilsConfig.hpp"
#include "helpers.hpp"
#include "Macros.hpp"
#include "Logger.hpp"

#ifndef NDEBUG
  #include "totalview/tv_data_display.h"
  #include "totalview/tv_helpers.hpp"
#endif

#ifdef USE_ARRAY_BOUNDS_CHECK

#undef CONSTEXPRFUNC
#define CONSTEXPRFUNC

#define ARRAY_SLICE_CHECK_BOUNDS( index )                                        \
  GEOS_ERROR_IF( index < 0 || index >= m_dims[0], "Array Bounds Check Failed: index=" << index << " m_dims[0]=" << m_dims[0] )

#else // USE_ARRAY_BOUNDS_CHECK

#define ARRAY_SLICE_CHECK_BOUNDS( index )

#endif // USE_ARRAY_BOUNDS_CHECK


namespace LvArray
{

/**
 * @class ArraySlice
 * @brief This class serves to provide a sliced multidimensional interface to the family of LvArray
 *        classes.
 * @tparam T    type of data that is contained by the array
 * @tparam NDIM number of dimensions in array (e.g. NDIM=1->vector, NDIM=2->Matrix, etc. )
 * @tparam INDEX_TYPE the integer to use for indexing the components of the array
 *
 * This class serves as a sliced interface to an array. This is a lightweight class that contains
 * only pointers, and provides operator[] for indexed array access. This class may be aliased to
 * a pointer for NDIM=1 in cases where range checking is off.
 *
 * In general, instantiations of ArraySlice should only result from taking a slice of an an Array or
 * an ArrayView via operator[] which it inherits from ArraySlice.
 *
 * Key features:
 * 1) ArraySlice provides operator[] array accessesor
 * 2) operator[] are all const and may be called in non-mutable lamdas
 * 3) Conversion operator to go from ArraySlice<T> to ArraySlice<T const>
 * 4) Conversion operator to go from ArraySlice<T,1> to T*
 *
 */
template< typename T, int NDIM, int UNIT_STRIDE_DIM=NDIM-1, typename INDEX_TYPE=std::ptrdiff_t >
class ArraySlice
{
public:

  static_assert( UNIT_STRIDE_DIM < NDIM, "UNIT_STRIDE_DIM must be less than NDIM." );

  /**
   * @param inputData       pointer to the beginning of the data for this slice of the array
   * @param inputDimensions pointer to the beginning of the dimensions[NDIM] c-array for this slice.
   * @param inputStrides    pointer to the beginning of the strides[NDIM] c-array for this slice
   *
   * Base constructor that takes in raw data pointers as initializers for its members.
   */
  LVARRAY_HOST_DEVICE inline explicit CONSTEXPRFUNC
  ArraySlice( T * const restrict inputData,
              INDEX_TYPE const * const restrict inputDimensions,
              INDEX_TYPE const * const restrict inputStrides ) noexcept:
    m_data( inputData ),
    m_dims( inputDimensions ),
    m_strides( inputStrides )
  {
#if defined(USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__) && defined(USE_ARRAY_BOUNDS_CHECK)
    ArraySlice::TV_ttf_display_type( nullptr );
#endif
  }


  /**
   * User Defined Conversion operator to move from an ArraySlice<T> to ArraySlice<T const>&. This is
   * achieved by applying a reinterpret_cast to the this pointer, which is a safe operation as the
   * only difference between the types is a const specifier.
   * @return a reference to an ArraySlice<T const>
   */
  template< typename U = T >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC operator
  typename std::enable_if< !std::is_const<U>::value,
                           ArraySlice<T const, NDIM, UNIT_STRIDE_DIM, INDEX_TYPE> const & >::type
    () const restrict_this noexcept
  {
    return reinterpret_cast<ArraySlice<T const, NDIM, UNIT_STRIDE_DIM, INDEX_TYPE> const &>(*this);
  }

  template< typename U = T >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  typename std::enable_if< !std::is_const<U>::value,
                           ArraySlice<T const, NDIM, UNIT_STRIDE_DIM, INDEX_TYPE> const & >::type
  toSliceConst() const restrict_this noexcept
  {
    return reinterpret_cast<ArraySlice<T const, NDIM, UNIT_STRIDE_DIM, INDEX_TYPE> const &>(*this);
  }

  /**
   * User defined conversion to convert a ArraySlice<T,1,NDIM> to a raw pointer.
   * @return a copy of m_data.
   */
  template< int _NDIM=NDIM, int _UNIT_STRIDE_DIM=UNIT_STRIDE_DIM >
  LVARRAY_HOST_DEVICE CONSTEXPRFUNC inline operator
  typename std::enable_if< _NDIM == 1 && _UNIT_STRIDE_DIM == 0, T * const restrict >::type () const noexcept restrict_this
  { return m_data; }

  /**
   * @brief This function provides a square bracket operator for array access.
   * @param index index of the element in array to access
   * @return a reference to the member m_childInterface, which is of type ArraySlice<T,NDIM-1>.
   *
   * This function creates a new ArraySlice<T,NDIM-1,INDEX_TYPE> by passing the member pointer
   * locations +1. Thus, the returned object has m_data pointing to the beginning of the data
   * associated with its sub-array. If used as a set of nested operators (array[][][]) the compiler
   * typically provides the pointer dereference directly and does not create the new ArraySlice
   * objects.
   */
  template< int U=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC typename
  std::enable_if< (U > 1), ArraySlice<T, NDIM-1, UNIT_STRIDE_DIM-1, INDEX_TYPE> >::type
  operator[]( INDEX_TYPE const index ) const noexcept restrict_this
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return ArraySlice<T, NDIM-1, UNIT_STRIDE_DIM-1, INDEX_TYPE>( m_data + ConditionalMultiply< 0, UNIT_STRIDE_DIM >::multiply( index, m_strides[ 0 ] ),
                                                                 m_dims + 1,
                                                                 m_strides + 1 );
  }

  /**
   * @brief This function provides a square bracket operator for array access for a 1D array.
   * @param index index of the element in array to access
   * @return a reference to the data location indicated by the index.
   *
   * @note This declaration is for NDIM==1 when bounds checking is ON.
   */
  template< int U=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  typename std::enable_if< U==1, T & >::type
  operator[]( INDEX_TYPE const index ) const noexcept restrict_this
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return m_data[ ConditionalMultiply< 0, UNIT_STRIDE_DIM >::multiply( index, m_strides[ 0 ] ) ];
  }

  /**
   * @brief operator() array accessor
   * @tparam INDICES variadic template parameters to serve as index arguments.
   * @param indices the indices of access request (0,3,4)
   * @return reference to the data at the requested indices.
   *
   * This is a standard fortran like parentheses interface to array access.
   */
  template< typename... INDICES >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  T & operator()( INDICES... indices ) const
  {
    static_assert( sizeof ... (INDICES) == NDIM, "number of indices does not match NDIM" );
    return m_data[ linearIndex( indices... ) ];
  }

  /**
   * @brief calculation of offset or linear index from a multidimensional space to a linear space.
   * @tparam INDICES variadic template parameters to serve as index arguments.
   * @param indices the indices of access request (0,3,4)
   */
  template< typename... INDICES >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  INDEX_TYPE linearIndex( INDICES... indices ) const
  {
    static_assert( sizeof ... (INDICES) == NDIM, "number of indices does not match NDIM" );
#ifdef USE_ARRAY_BOUNDS_CHECK
    checkIndices( m_dims, indices... );
#endif
    return getLinearIndex< UNIT_STRIDE_DIM >( m_strides, indices... );
  }

  /**
   * @brief function to return the allocated size
   */
  inline LVARRAY_HOST_DEVICE INDEX_TYPE size() const noexcept
  { return multiplyAll< NDIM >( m_dims ); }

  /**
   * @brief function to get the length of a dimension
   * @param dim the dimension for which to get the length of
   */
  LVARRAY_HOST_DEVICE inline INDEX_TYPE size( int dim ) const noexcept
  {
    GEOS_ASSERT_GT( NDIM, dim );
    return m_dims[dim];
  }

  /// deleted default constructor
  ArraySlice() = delete;

#if defined(USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__) && defined(USE_ARRAY_BOUNDS_CHECK)
  /**
   * @brief Static function that will be used by Totalview to display the array contents.
   * @param av A pointer to the array that is being displayed.
   * @return 0 if everything went OK
   */
  static int TV_ttf_display_type( ArraySlice const * av)
  {
    if( av!=nullptr )
    {
      int constexpr ndim = NDIM;
      //std::cout<<"Totalview using ("<<totalview::format<T,INDEX_TYPE>(NDIM, av->m_dims )<<") for display of m_data;"<<std::endl;
      TV_ttf_add_row("tv(m_data)", totalview::format<T,INDEX_TYPE>(NDIM, av->m_dims ).c_str(), (av->m_data) );
      TV_ttf_add_row("m_data", totalview::format<T,INDEX_TYPE>(1, av->m_dims ).c_str(), (av->m_data) );
      TV_ttf_add_row("m_dims", totalview::format<INDEX_TYPE,int>(1,&ndim).c_str(), (av->m_dims) );
      TV_ttf_add_row("m_strides", totalview::format<INDEX_TYPE,int>(1,&ndim).c_str(), (av->m_strides) );
    }
    return 0;
  }
#endif

protected:
  /// pointer to beginning of data for this array, or sub-array.
  T * const restrict m_data;

  /// pointer to array of length NDIM that contains the lengths of each array dimension
  INDEX_TYPE const * const restrict m_dims;

  /// pointer to array of length NDIM that contains the strides of each array dimension
  INDEX_TYPE const * const restrict m_strides;

};

}

#endif /* SRC_COMPONENTS_CORE_SRC_ARRAY_MULTIDIMENSIONALARRAY_HPP_ */
