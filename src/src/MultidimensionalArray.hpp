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

/*
 * ArrayWrapper.hpp
 *
 *  Created on: Jul 30, 2016
 *      Author: rrsettgast
 */

#ifndef SRC_COMPONENTS_CORE_SRC_ARRAY_MULTIDIMENSIONALARRAY_HPP_
#define SRC_COMPONENTS_CORE_SRC_ARRAY_MULTIDIMENSIONALARRAY_HPP_
#include <vector>
#include <iostream>

#define ARRAY_BOUNDS_CHECK 0

#ifdef __clang__
#define restrict __restrict__
#define restrict_this
#elif __GNUC__
#define restrict __restrict__
#define restrict_this __restrict__
#endif

#if 0
#include "common/DataTypes.hpp"
#else
#include <assert.h>
using  integer_t     = int;
#endif


namespace multidimensionalArray
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
template< typename T, int NDIM >
class ArrayAccessor
{
public:

  /// deleted default constructor
  ArrayAccessor() = delete;

  /**
   *
   * Base constructor that takes in raw data pointers, sets member pointers, and
   * calculates stride.
   */
  ArrayAccessor( T * const restrict inputData, integer_t const * const restrict inputLength ):
    m_data(inputData),
    m_stride(CalulateStride(inputLength)),
    m_lengths(inputLength),
    m_childInterface(ArrayAccessor<T,NDIM-1>( m_data, &(inputLength[1]) ) )
  {}

  /// default destructor
  ~ArrayAccessor() = default;

  /**
   * @param source object to copy
   * copy constructor invokes direct copy of each member in source
   */
  ArrayAccessor( ArrayAccessor const & source ):
    m_data(source.m_data),
    m_stride(source.m_stride),
    m_lengths(source.m_lengths),
    m_childInterface(source.m_childInterface)
  {}

  /**
   * @param source object to move
   * move constructor invokes direct move of each member in source. In the case
   * of data members that are pointers, this
   * is a straight copy. Not really a move, but rather a copy.
   */
  ArrayAccessor( ArrayAccessor && source ):
    m_data( std::move(source.m_data) ),
    m_stride( std::move(source.m_stride) ),
    m_lengths( std::move(source.m_lengths) ),
    m_childInterface( std::move(source.m_childInterface) )
  {}

  /**
   * @param rhs object to copy
   * copy assignment operator invokes direct copy of each member in source.
   */
//  ArrayAccessor & operator=( ArrayAccessor const & rhs )
//  {
//    m_data           = rhs.m_data;
//    m_lengths        = rhs.m_lengths;
//    m_stride         = rhs.m_stride;
//    m_childInterface = rhs.m_childInterface;
//    return *this;
//  }

  /**
   * @param source object to move
   * move constructor invokes direct move of each member in source. In the case
   * of data members that are pointers, this
   * is a straight copy assignment....so this is really just a copy constructor.
   */
//  ArrayAccessor & operator=( ArrayAccessor && rhs )
//  {
//    m_data           = std::move(rhs.m_data);
//    m_lengths        = std::move(rhs.m_lengths);
//    m_stride         = std::move(rhs.m_stride);
//    m_childInterface = std::move(rhs.m_childInterface);
//    return *this;
//  }

  /**
   * @param index index of the element in array to access
   * @return a reference to the member m_childInterface, which is of type
   * ArrayAccessor<T,NDIM-1>.
   * This function sets the data pointer for m_childInterface.m_data to the
   * location corresponding to the input
   * parameter "index". Thus, the returned object has m_data pointing to the
   * beginning of the data associated with its
   * sub-array.
   */
  inline ArrayAccessor<T,NDIM-1> & operator[](integer_t const index) restrict_this
  {
#if ARRAY_BOUNDS_CHECK == 1
    assert( index < m_lengths[0] );
#endif
    m_childInterface.m_data = &(m_data[index*m_stride]);
    return m_childInterface;
  }

  /// make ArrayAccessor classes with one higher order in dimension friends, so
  // that their operator[] can access m_data.
  friend class ArrayAccessor<T,NDIM+1>;

  /**
   *
   * @return number of data entries until the next value
   *
   * function to calculate the stride of this. Used to set m_stride in
   * constructor.
   */
  static integer_t CalulateStride( integer_t const * const restrict lengths )
  {
    integer_t stride = 1;
    for( int a=1 ; a<NDIM ; ++a )
    {
      stride *= lengths[a];
    }
    return stride;
  }

  T * data() { return m_data;}
  integer_t const * lengths() { return m_lengths;}

private:
  /// pointer to beginning of data for this array, or sub-array.
  T * restrict m_data;

  /// the stride, or number of array entires between each iteration of the first
  // index of this array/sub-array
  integer_t const m_stride;

  /// pointer to array of length NDIM that contains the lengths of each array
  // dimension
  integer_t const * const restrict m_lengths;


  /// a child ArrayAccessor to represent the sub-array below the current array
  // for a given value of the first array index
  /// of this array.
  ArrayAccessor<T,NDIM-1> m_childInterface;
};

/**
 * Specialization for the ArrayAccessor<typename T, int NDIM> class template.
 * This specialization defines the lowest
 * level ArrayAccessor in the array hierarchy. Thus this is the only level that
 * actually allows data access, where the
 * other template instantiations only return references to a sub-array. In
 * essence, this is a 1D array.
 */
template< typename T >
class ArrayAccessor<T,1>
{
public:

  /// deleted default constructor
  ArrayAccessor() = delete;

  /**
   *
   * Base constructor that takes in raw data pointers, sets member pointers.
   * Unlike the higher dimensionality arrays,
   * no calculation of stride is necessary for NDIM=1.
   */
  ArrayAccessor( T * const restrict inputData, integer_t const * const restrict inputLength ):
    m_data(inputData),
    m_lengths(inputLength)
  {}

  /// default destructor
  ~ArrayAccessor() = default;


  /**
   * @param source object to copy
   * copy constructor invokes direct copy of each member in source
   */
  ArrayAccessor( ArrayAccessor const & source ):
    m_data(source.m_data),
    m_lengths(source.m_lengths)
  {}

  /**
   * @param source object to move
   * move constructor invokes direct move of each member in source. In the case
   * of data members that are pointers, this
   * is a straight copy. Not really a move, but rather a copy.
   */
  ArrayAccessor( ArrayAccessor && source ):
    m_data(source.m_data),
    m_lengths(source.m_lengths)
  {}

  /**
   * @param rhs object to copy
   * copy assignment operator invokes direct copy of each member in source.
   */
//  ArrayAccessor & operator=( ArrayAccessor const & rhs )
//  {
//    m_data           = rhs.m_data;
//    m_lengths        = rhs.m_lengths;
//    return *this;
//  }

  /**
   * @param source object to move
   * move constructor invokes direct move of each member in source. In the case
   * of data members that are pointers, this
   * is a straight copy assignment....so this is really just a copy constructor.
   */
//  ArrayAccessor & operator=( ArrayAccessor && rhs )
//  {
//    m_data           = rhs.m_data;
//    m_lengths        = rhs.m_lengths;
//    return *this;
//  }


  /**
   * @param index index of the element in array to access
   * @return a reference to the m_data[index], where m_data is a T*.
   * This function simply returns a reference to the pointer deferenced using
   * index.
   */
  inline T& operator[](integer_t const index) restrict_this
  {
#if ARRAY_BOUNDS_CHECK == 1
    assert( index < m_lengths[0] );
#endif
    return m_data[index];
  }


  T * data() { return m_data;}
  integer_t const * lengths() { return m_lengths;}


  /// make ArrayAccessor classes with NDIM=2 friends, so that their operator[]
  // can access m_data.
  friend class ArrayAccessor<T,2>;

private:
  /// pointer to beginning of data for this array, or sub-array.
  T * restrict m_data;

  /// pointer to array of length NDIM that contains the lengths of each array
  // dimension
  integer_t const * const restrict m_lengths;
};



template< typename T, int NDIM, typename memBlock = std::vector<T> >
class Array
{
public:

  using rtype = ArrayAccessor<T,NDIM-1> &;

  Array() = delete;


  template< class U=T>
  Array( integer_t const lengths[NDIM] ):
    m_memory(),
    m_lengths(),
    m_interface( ArrayAccessor<T,NDIM>(nullptr,lengths) )
//  m_interface( ArrayAccessor<T,NDIM>(nullptr,lengths) )
  {
    integer_t size = 1;
    for( int a=0 ; a<NDIM ; ++a )
    {
      m_lengths[a] = lengths[a];
      size *= lengths[a];
    }
    m_memory.resize(size);

    m_interface = ArrayAccessor<T,NDIM>(m_memory.data(),m_lengths);
//    m_interface.m_data    = m_memory.data();
//    m_interface.m_lengths = m_lengths;
  }

  ~Array() = default;

  Array( Array const & ) = delete;
  Array( Array && source ) = delete;

  Array& operator=( Array const & rhs ) = delete;
  Array& operator=( Array && rhs ) = delete;

  //***** Accessors **********************************************************

  inline ArrayAccessor<T,NDIM> Accessor()
  {
    return m_interface;
  }

  inline rtype operator[](integer_t const index)
  {
    return m_interface[index];
  }

private:
  memBlock m_memory;
  integer_t m_lengths[NDIM] = {0};
  ArrayAccessor<T,NDIM> m_interface;

};

} /* namespace arraywrapper */

#endif /* SRC_COMPONENTS_CORE_SRC_ARRAY_MULTIDIMENSIONALARRAY_HPP_ */
