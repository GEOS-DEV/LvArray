/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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


#ifndef ARRAY_SLICE_HPP_
#define ARRAY_SLICE_HPP_

#include <cstring>
#include <vector>
#include <iostream>
#include <utility>
#include "Logger.hpp"
#include "CXX_UtilsConfig.hpp"

#if defined(__clang__)

#define restrict __restrict__
#define restrict_this
#define CONSTEXPRFUNC constexpr

#elif defined(__GNUC__)

#if defined(__INTEL_COMPILER)
#define restrict __restrict__
#define restrict_this
#define CONSTEXPRFUNC
#else
#define restrict __restrict__
#define restrict_this
#define CONSTEXPRFUNC constexpr
#endif
#endif


#ifdef USE_ARRAY_BOUNDS_CHECK

#undef CONSTEXPRFUNC
#define CONSTEXPRFUNC

#ifdef USE_CUDA

#include <cassert>
#define ARRAY_SLICE_CHECK_BOUNDS(index)                                        \
assert( index >= 0 && index < m_dims[0] )

#else // USE_CUDA

#define ARRAY_SLICE_CHECK_BOUNDS(index)                                        \
GEOS_ERROR_IF( index < 0 || index >= m_dims[0], "index=" << index << " m_dims[0]=" << m_dims[0] )

#endif // USE_CUDA

#else // USE_ARRAY_BOUNDS_CHECK

#define ARRAY_SLICE_CHECK_BOUNDS(index)

#endif // USE_ARRAY_BOUNDS_CHECK


namespace LvArray
{


/**
 * @tparam T    type of data that is contained by the array
 * @tparam NDIM number of dimensions in array (e.g. NDIM=1->vector,
 * NDIM=2->Matrix, etc. )
 * This class serves as a multidimensional array interface for a chunk of
 * memory. This is a lightweight
 * class that contains only pointers, on integer data member to hold the stride,
 * and another instantiation of type
 * ArrayAccesssor<T,NDIM-1> to represent the sub-array that is passed back upon
 * use of operator[]. Pointers to the
 * data and length of each dimension passed into the constructor, thus the class
 * does not own the data itself, nor does
 * it own the array that defines the shape of the data.
 */
template< typename T, int NDIM, typename INDEX_TYPE = std::int_fast32_t >
class ArraySlice
{
public:

  using size_type = INDEX_TYPE;

  /**
   * @param inputData pointer to the beginning of the data
   * @param inputDimensions pointer to the beginning of an array of dimensions. This array
   *                        has length NDIM
   *
   * Base constructor that takes in raw data pointers, sets member pointers, and
   * calculates stride.
   */
  inline explicit CONSTEXPRFUNC
  ArraySlice( T * const restrict inputData,
              INDEX_TYPE const * const restrict inputDimensions,
              INDEX_TYPE const * const restrict inputStrides ):
    m_data(inputData),
    m_dims(inputDimensions),
    m_strides(inputStrides)
  {}


  /**
   * User Defined Conversion operator to move from an ArraySlice<T> to ArraySlice<T const>
   */
  template< typename U = T >
  CONSTEXPRFUNC inline operator
  typename std::enable_if< !std::is_const<U>::value,
                           ArraySlice<T const,NDIM,INDEX_TYPE> const & >::type
  () const restrict_this
  {
    return reinterpret_cast<ArraySlice<T const,NDIM,INDEX_TYPE> const &>(*this);
  }

  /**
   * User defined conversion to convert a ArraySlice<T,1,NDIM> to a raw pointer.
   */
  template< int U=NDIM >
  CONSTEXPRFUNC inline operator
  typename std::enable_if< U == 1, T * const restrict >::type () const restrict_this
  {
    return m_data;
  }


  /**
   * User defined conversion to convert to a reduced dimension array. For example, converting from
   * a 2d array to a 1d array is valid if the last dimension of the 2d array is 1.
   */
  template< int U=NDIM >
  CONSTEXPRFUNC inline explicit
  operator typename std::enable_if< (U>1) ,
                                    ArraySlice<T,NDIM-1,INDEX_TYPE> >::type () const restrict_this
  {
    GEOS_ASSERT_MSG( m_dims[NDIM-1]==1, "ManagedArray::operator ArraySlice<T,NDIM-1,INDEX_TYPE>" <<
                     " is only valid if last dimension is equal to 1." );
    return ArraySlice<T,NDIM-1,INDEX_TYPE>( m_data, m_dims + 1, m_strides + 1 );
  }

  /**
   * @param index index of the element in array to access
   * @return a reference to the member m_childInterface, which is of type
   * ArraySlice<T,NDIM-1>.
   * This function sets the data pointer for m_childInterface.m_data to the
   * location corresponding to the input
   * parameter "index". Thus, the returned object has m_data pointing to the
   * beginning of the data associated with its
   * sub-array.
   */
  template< int U=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC typename
#ifdef USE_ARRAY_BOUNDS_CHECK
  std::enable_if< U!=1, ArraySlice<T,NDIM-1,INDEX_TYPE> >::type
#else
  std::enable_if< U >= 3, ArraySlice<T,NDIM-1,INDEX_TYPE> >::type
#endif
  operator[](INDEX_TYPE const index) const restrict_this
  {
    ARRAY_SLICE_CHECK_BOUNDS(index);
    return ArraySlice<T,NDIM-1,INDEX_TYPE>( &(m_data[ index*m_strides[0] ] ), m_dims+1, m_strides+1 );
  }

#ifndef USE_ARRAY_BOUNDS_CHECK
  template< int U=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  typename std::enable_if< U==2, T * restrict >::type
  operator[](INDEX_TYPE const index) const restrict_this
  {
    return &(m_data[ index*m_strides[0] ]);
  }
#endif

  template< int U=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPRFUNC
  typename std::enable_if< U==1, T & >::type
  operator[](INDEX_TYPE const index) const restrict_this
  {
    ARRAY_SLICE_CHECK_BOUNDS(index);
    return m_data[ index ];
  }

  /// deleted default constructor
  ArraySlice() = delete;



protected:
  /// pointer to beginning of data for this array, or sub-array.
  T * const restrict m_data;

  /// pointer to array of length NDIM that contains the lengths of each array dimension
  INDEX_TYPE const * const restrict m_dims;

  /// pointer to array of length NDIM that contains the strides of each array dimension
  INDEX_TYPE const * const restrict m_strides;

};

#ifdef USE_ARRAY_BOUNDS_CHECK

template< typename T, typename INDEX_TYPE >
using ArraySlice1d = ArraySlice<T,1,INDEX_TYPE>;

#else

template< typename T, typename INDEX_TYPE = int>
using ArraySlice1d = T * const restrict;

#endif


}

#endif /* SRC_COMPONENTS_CORE_SRC_ARRAY_MULTIDIMENSIONALARRAY_HPP_ */
