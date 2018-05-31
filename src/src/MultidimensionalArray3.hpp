// Copyright (c) 2018, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-746361. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the GEOSX Simulation Framework.

//
// GEOSX is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
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

template< typename T, int NDIM >
class ArrayAccessor
{
public:

  ArrayAccessor() = delete;

  ArrayAccessor( T * const restrict inputData, integer_t const * const restrict inputLength ):
    m_data(inputData),
    m_stride(CalulateStride(inputLength)),
    m_lengths(inputLength),
    m_childInterface(ArrayAccessor<T,NDIM-1>( m_data, &(inputLength[1]) ) )
  {}

  ~ArrayAccessor() = default;

  ArrayAccessor( ArrayAccessor const & source ):
    m_data(source.m_data),
    m_stride(source.m_stride),
    m_lengths(source.m_lengths),
    m_childInterface(source.m_childInterface)
  {}

  ArrayAccessor( ArrayAccessor && source ):
    m_data( std::move(source.m_data) ),
    m_stride( std::move(source.m_stride) ),
    m_lengths( std::move(source.m_lengths) ),
    m_childInterface( std::move(source.m_childInterface) )
  {}

  ArrayAccessor & operator=( ArrayAccessor const & rhs )
  {
    m_data           = rhs.m_data;
    m_lengths        = rhs.m_lengths;
    m_stride         = rhs.m_stride;
    m_childInterface = rhs.m_childInterface;
    return *this;
  }

  ArrayAccessor & operator=( ArrayAccessor && rhs )
  {
    m_data           = std::move(rhs.m_data);
    m_lengths        = std::move(rhs.m_lengths);
    m_stride         = std::move(rhs.m_stride);
    m_childInterface = std::move(rhs.m_childInterface);
    return *this;
  }

  inline ArrayAccessor<T,NDIM-1> & operator[](integer_t const index) restrict_this
  {
    m_childInterface.m_data = &(m_data[index*m_stride]);
    return m_childInterface;
  }

  friend class ArrayAccessor<T,NDIM+1>;

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
  T * restrict m_data;
  integer_t m_stride;
  integer_t const * restrict m_lengths;
  ArrayAccessor<T,NDIM-1> m_childInterface;
};

template< typename T >
class ArrayAccessor<T,1>
{
public:
  ArrayAccessor() = delete;

  ArrayAccessor( T * const restrict inputData, integer_t const * const restrict inputLength ):
    m_data(inputData),
    m_lengths(inputLength)
  {}

  ~ArrayAccessor() = default;

  ArrayAccessor( ArrayAccessor const & source ):
    m_data(source.m_data),
    m_lengths(source.m_lengths)
  {}

  ArrayAccessor( ArrayAccessor && source ):
    m_data(source.m_data),
    m_lengths(source.m_lengths)
  {}

  ArrayAccessor & operator=( ArrayAccessor const & rhs )
  {
    m_data           = rhs.m_data;
    m_lengths        = rhs.m_lengths;
    return *this;
  }

  ArrayAccessor & operator=( ArrayAccessor && rhs )
  {
    m_data           = rhs.m_data;
    m_lengths        = rhs.m_lengths;
    return *this;
  }

  inline T& operator[](integer_t const index) restrict_this
  {
    return m_data[index];
  }

  T * data() { return m_data;}
  integer_t const * lengths() { return m_lengths;}

  friend class ArrayAccessor<T,2>;

private:
  T * restrict m_data;
  integer_t const * restrict m_lengths;
};



template< typename T, int NDIM, typename memBlock = std::vector<T> >
class Array
{
public:

  using rtype = ArrayAccessor<T,NDIM-1> &;

  Array() = delete;


  Array( integer_t const lengths[NDIM] ):
    m_memory(),
    m_lengths(),
    m_interface( ArrayAccessor<T,NDIM>(nullptr,lengths) )
  {
    integer_t size = 1;
    for( int a=0 ; a<NDIM ; ++a )
    {
      m_lengths[a] = lengths[a];
      size *= lengths[a];
    }
    m_memory.resize(size);

    m_interface = ArrayAccessor<T,NDIM>(m_memory.data(),m_lengths);
  }

  ~Array() = default;

  Array( Array const & source ) = delete;


  Array( Array && source ) = delete;

  Array& operator=( Array const & rhs ) = delete;
  Array& operator=( Array && rhs ) = delete;

  inline ArrayAccessor<T,NDIM-1>& operator[](integer_t const index)
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
