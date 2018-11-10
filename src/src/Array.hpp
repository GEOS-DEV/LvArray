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

/**
 * @file Array.hpp
 */

#ifndef ARRAY_HPP_
#define ARRAY_HPP_


#include <iostream>
#include <limits>
#include <vector>
#include <iterator>

#include "ArrayView.hpp"
#include "ChaiVector.hpp"


template< typename T >
struct is_integer
{
  constexpr static bool value = std::is_same<T,int>::value ||
                                std::is_same<T,unsigned int>::value ||
                                std::is_same<T,long int>::value ||
                                std::is_same<T,unsigned long int>::value ||
                                std::is_same<T,long long int>::value ||
                                std::is_same<T,unsigned long long int>::value;
};


namespace LvArray
{




template< typename T, int NDIM, typename INDEX_TYPE=std::int_fast32_t > class Array;

namespace detail
{
template<typename>
struct is_array : std::false_type {};

template< typename T, int NDIM, typename INDEX_TYPE >
struct is_array< Array<T,NDIM,INDEX_TYPE> > : std::true_type{};
}

/**
 * @class Array
 * @brief This class provides a multi-dimensional array interface to a vector type object.
 * @tparam T    type of data that is contained by the array
 * @tparam NDIM number of dimensions in array (e.g. NDIM=1->vector, NDIM=2->Matrix, etc. )
 * @tparam INDEX_TYPE the integer to use for indexing the components of the array
 */
template< typename T, int NDIM, typename INDEX_TYPE >
class Array : public ArrayView<T, NDIM, INDEX_TYPE>
{
public:
  using isArray = std::true_type;
  using ArrayView<T, NDIM, INDEX_TYPE>::m_dataVector;
  using ArrayView<T, NDIM, INDEX_TYPE>::m_dimsMem;
  using ArrayView<T, NDIM, INDEX_TYPE>::m_stridesMem;
  using ArrayView<T, NDIM, INDEX_TYPE>::m_singleParameterResizeIndex;

  using ArrayView<T, NDIM, INDEX_TYPE>::size;  
  using ArrayView<T, NDIM, INDEX_TYPE>::setDataPtr;
  using ArrayView<T, NDIM, INDEX_TYPE>::data;
  using ArrayView<T, NDIM, INDEX_TYPE>::begin;
  using ArrayView<T, NDIM, INDEX_TYPE>::end;
  using ArrayView<T, NDIM, INDEX_TYPE>::operator[];
  using ArrayView<T, NDIM, INDEX_TYPE>::operator();

  using value_type = T;
  using typename ArrayView<T, NDIM, INDEX_TYPE>::pointer;
  using typename ArrayView<T, NDIM, INDEX_TYPE>::const_pointer;
  using typename ArrayView<T, NDIM, INDEX_TYPE>::iterator;
  using typename ArrayView<T, NDIM, INDEX_TYPE>::const_iterator;


  /**
   * @brief default constructor
   */
  inline Array():
    ArrayView<T, NDIM, INDEX_TYPE>()
  {
    CalculateStrides();
  }

  /**
   * @brief constructor that takes in the dimensions as a variadic parameter list
   * @param dims the dimensions of the array in form ( n0, n1,..., n(NDIM-1) )
   */
  template< typename... DIMS >
  inline explicit Array( DIMS... dims ):
    Array()
  {
    static_assert( is_integer<INDEX_TYPE>::value, "Error: std::is_integral<INDEX_TYPE> is false" );
    static_assert( sizeof ... (DIMS) == NDIM, "Error: calling Array::Array with incorrect number of arguments.");
    static_assert( check_dim_type<0, DIMS...>::value, "arguments to constructor of geosx::Array( DIMS... dims ) are incompatible with INDEX_TYPE" );

    resize( dims... );
  }

  /**
   * @brief copy constructor
   * @param source object to copy
   *
   * Performs a deep copy of source
   */
  Array( Array const & source ):
    ArrayView<T,NDIM,INDEX_TYPE>()
  {
    // This DOES NOT trigger Chai::ManagedArray CC
    *this = source;
  }

  /**
   * @brief move constructor
   * @param source object to move
   */
  Array( Array&& source ):
    ArrayView<T,NDIM,INDEX_TYPE>( std::move( static_cast<ArrayView<T,NDIM,INDEX_TYPE>& >(source) ) )
  {
    setSingleParameterResizeIndex( source.getSingleParameterResizeIndex() );
    source.clear();
  }

  /**
   * destructor
   */
  ~Array()
  {
    m_dataVector.free();
    setDataPtr();
  }

  /**
   * User Defined Conversion operator to move from an Array<T> to Array<T const>
   */
  template< typename U = T >
  operator
  typename std::enable_if< !std::is_const<U>::value, Array<T const,NDIM,INDEX_TYPE> const & >::type
  () const
  {
    return reinterpret_cast<Array<T const,NDIM,INDEX_TYPE> const &>(*this);
  }

  /**
   * @brief assignment operator
   * @param rhs source for the assignment
   * @return *this
   *
   * The assignment operator performs a deep copy of the rhs.
   */
  Array & operator=( Array const& rhs )
  {
    resize(NDIM, rhs.m_dimsMem);

    INDEX_TYPE const length = size();
    T * const data_ptr = data();
    T const * const rhs_data_ptr = rhs.data();
    for( INDEX_TYPE a = 0 ; a < length ; ++a )
    {
      data_ptr[a] = rhs_data_ptr[a];
    }

    setSingleParameterResizeIndex( rhs.getSingleParameterResizeIndex() );
    return *this;
  }

  /**
   * @brief move constructor
   * @param rhs source for the construction
   * @return *this
   *
   * Move constructor takes data from rhs without triggering a copy construction.
   */
  Array & operator=( Array&& rhs )
  {
    m_dataVector = std::move(rhs.m_dataVector);
    setDataPtr();
    m_singleParameterResizeIndex = rhs.m_singleParameterResizeIndex;

    for( int a=0 ; a<NDIM ; ++a )
    {
      m_dimsMem[a]     = rhs.m_dimsMem[a];
      m_stridesMem[a]  = rhs.m_stridesMem[a];
    }

    return *this;
  }

  int numDimensions() const
  { return NDIM; }

  /**
   * \defgroup stl container interface
   * @{
   */

  /**
   * @brief This function provides a resize or reallocation of the array
   * @param numDims the number of dims in the dims parameter
   * @param dims the new size of the dimensions
   */
  void resize( int const numDims, INDEX_TYPE const * const dims )
  {
    GEOS_ERROR_IF( numDims != NDIM, "Dimension mismatch: " << numDims );
    this->setDims(dims);
    CalculateStrides();
    resize();
  }

  /**
   * @brief function to resize/reallocate the array
   * @tparam DIMS variadic pack containing the dimension types
   * @param newdims the new dimensions
   */
  template< typename... DIMS >
  void resize( DIMS... newdims )
  {
    static_assert( sizeof ... (DIMS) == NDIM,
                   "Error: calling template< typename... DIMS > Array::resize(DIMS...newdims) with incorrect number of arguments.");
    static_assert( check_dim_type<0,DIMS...>::value, "arguments to Array::resize(DIMS...newdims) are incompatible with INDEX_TYPE" );

    dim_unpack<0,DIMS...>::f( m_dimsMem, newdims...);
    CalculateStrides();
    resize();
  }

  /**
   * @brief function to resize/reallocate the array using a single dimension
   * @param newdim the new dimension
   *
   * This resize function will use set the dim[m_singleParameterResizeIndex]=newdim, and reallocate
   * based on that new size.
   */
  template< typename TYPE >
  void resize( TYPE newdim )
  {
    static_assert( is_valid_indexType<TYPE>::value, "arguments to Array::resize(DIMS...newdims) are incompatible with INDEX_TYPE" );

    m_dimsMem[m_singleParameterResizeIndex] = newdim;
    CalculateStrides();
    resize();
  }

  /**
   *
   * @param newLength
   */
  void reserve( INDEX_TYPE newLength )
  {
    m_dataVector.reserve(newLength);
    setDataPtr();
  }

  INDEX_TYPE capacity() const
  { return integer_conversion<INDEX_TYPE>(m_dataVector.capacity()); }

  void clear()
  {
    m_dataVector.clear();
    setDataPtr();

    for( int i = 0; i < NDIM; ++i )
    {
      m_dimsMem[i] = 1;
    }
    m_dimsMem[getSingleParameterResizeIndex()] = 0;

    CalculateStrides();
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type 
  push_back( T const & newValue )
  {
    m_dataVector.push_back(newValue);
    setDataPtr();
    m_dimsMem[0] = integer_conversion<INDEX_TYPE>(m_dataVector.size());
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type 
  push_back( T && newValue )
  {
    m_dataVector.push_back(std::move(newValue));
    setDataPtr();
    m_dimsMem[0] = integer_conversion<INDEX_TYPE>(m_dataVector.size());
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type
  pop_back()
  {
    m_dataVector.pop_back();
    m_dimsMem[0] = integer_conversion<INDEX_TYPE>(m_dataVector.size());
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type 
  insert(iterator pos, T const& value)
  {
    m_dataVector.insert( pos, value );
    setDataPtr();
    m_dimsMem[0] = integer_conversion<INDEX_TYPE>(m_dataVector.size());
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type 
  insert(iterator pos, T && value)
  {
    m_dataVector.insert( pos, std::move(value) );
    setDataPtr();
    m_dimsMem[0] = integer_conversion<INDEX_TYPE>(m_dataVector.size());
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type 
  insert(INDEX_TYPE pos, T const& value)
  {
    m_dataVector.insert( pos, value );
    setDataPtr();
    m_dimsMem[0] = integer_conversion<INDEX_TYPE>(m_dataVector.size());
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type 
  insert(INDEX_TYPE pos, T && value)
  {
    m_dataVector.insert( pos, std::move(value) );
    setDataPtr();
    m_dimsMem[0] = integer_conversion<INDEX_TYPE>(m_dataVector.size());
  }

  template<int N=NDIM, typename InputIt>
  typename std::enable_if< N==1, void >::type 
  insert(iterator pos, InputIt first, InputIt last)
  {
    m_dataVector.insert( pos, first, last );
    setDataPtr();
    m_dimsMem[0] = integer_conversion<INDEX_TYPE>(m_dataVector.size());
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type erase( iterator pos )
  {
    m_dataVector.erase( pos ) ;
    m_dimsMem[0] = integer_conversion<INDEX_TYPE>(m_dataVector.size());
  }

  /**@}*/

  inline int getSingleParameterResizeIndex() const
  {
    return m_singleParameterResizeIndex;
  }

  inline void setSingleParameterResizeIndex( int const index )
  {
    m_singleParameterResizeIndex = index;
  }

private:

  void CalculateStrides()
  {
    m_stridesMem[NDIM-1] = 1;
    for( int a=NDIM-2 ; a>=0 ; --a )
    {
      m_stridesMem[a] = m_dimsMem[a+1] * m_stridesMem[a+1];
    }
  }

  void resize()
  {
    INDEX_TYPE length = 1;
    for( int a=0 ; a<NDIM ; ++a )
    {
      length *= m_dimsMem[a];
    }

    m_dataVector.resize( length );
    this->setDataPtr();
  }

  template< typename CANDIDATE_INDEX_TYPE >
  struct is_valid_indexType
  {
    constexpr static bool value = std::is_same<CANDIDATE_INDEX_TYPE,INDEX_TYPE>::value ||
                                  ( is_integer<CANDIDATE_INDEX_TYPE>::value &&
                                    ( sizeof(CANDIDATE_INDEX_TYPE)<=sizeof(INDEX_TYPE) ) );
  };
  template< int INDEX, typename DIM0, typename... DIMS >
  struct check_dim_type
  {
    constexpr static bool value =  is_valid_indexType<DIM0>::value && check_dim_type<INDEX+1, DIMS...>::value;
  };

  template< typename DIM0, typename... DIMS >
  struct check_dim_type<NDIM-1,DIM0,DIMS...>
  {
    constexpr static bool value = is_valid_indexType<DIM0>::value;
  };

  template< int INDEX, typename DIM0, typename... DIMS >
  struct dim_unpack
  {
    constexpr static int f( INDEX_TYPE m_dims[NDIM], DIM0 dim0, DIMS... dims )
    {
      m_dims[INDEX] = dim0;
      dim_unpack< INDEX+1, DIMS...>::f( m_dims, dims... );
      return 0;
    }
  };

  template< typename DIM0, typename... DIMS >
  struct dim_unpack<NDIM-1,DIM0,DIMS...>
  {
    constexpr static int f( INDEX_TYPE m_dims[NDIM], DIM0 dim0, DIMS... )
    {
      m_dims[NDIM-1] = dim0;
      return 0;
    }
  };

};

} /* namespace LvArray */

#endif /* ARRAY_HPP_ */
