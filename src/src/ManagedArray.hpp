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
 * GEOSX is a free software; you can redistrubute it and/or modify it under
 * the terms of the GNU Lesser General Public Liscense (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

/*
 * ArrayWrapper.hpp
 *
 *  Created on: Jul 30, 2016
 *      Author: rrsettgast
 */

#ifndef MANAGED_ARRAY_HPP_
#define MANAGED_ARRAY_HPP_

#ifdef __clang__
#define restrict __restrict__
#define restrict_this
#elif __GNUC__
#define restrict __restrict__
#define restrict_this __restrict__
#endif

#include <iostream>
#include <limits>

#include "chai/ManagedArray.hpp"
#include "../../../core/src/common/Logger.hpp"
#include "ArrayView.hpp"

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


namespace multidimensionalArray
{

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"


template< typename RTYPE, typename T >
inline typename std::enable_if< std::is_unsigned<T>::value && std::is_signed<RTYPE>::value, RTYPE >::type
integer_conversion( T input )
{
  static_assert( std::numeric_limits<T>::is_integer, "input is not an integer type" );
  static_assert( std::numeric_limits<RTYPE>::is_integer, "requested conversion is not an integer type" );

  if( input > std::numeric_limits<RTYPE>::max()  )
  {
    abort();
  }
  return static_cast<RTYPE>(input);
}

template< typename RTYPE, typename T >
inline typename std::enable_if< std::is_signed<T>::value && std::is_unsigned<RTYPE>::value, RTYPE >::type
integer_conversion( T input )
{
  static_assert( std::numeric_limits<T>::is_integer, "input is not an integer type" );
  static_assert( std::numeric_limits<RTYPE>::is_integer, "requested conversion is not an integer type" );

  if( input > std::numeric_limits<RTYPE>::max() ||
      input < 0 )
  {
    abort();
  }
  return static_cast<RTYPE>(input);
}


template< typename RTYPE, typename T >
inline typename std::enable_if< ( std::is_signed<T>::value && std::is_signed<RTYPE>::value ) ||
                         ( std::is_unsigned<T>::value && std::is_unsigned<RTYPE>::value ), RTYPE >::type
integer_conversion( T input )
{
  static_assert( std::numeric_limits<T>::is_integer, "input is not an integer type" );
  static_assert( std::numeric_limits<RTYPE>::is_integer, "requested conversion is not an integer type" );

  if( input > std::numeric_limits<RTYPE>::max() ||
      input < std::numeric_limits<RTYPE>::lowest() )
  {
    abort();
  }
  return static_cast<RTYPE>(input);
}



#pragma GCC diagnostic pop


template< typename T, int NDIM, typename INDEX_TYPE=std::int_fast32_t >
class ManagedArray
{
public:

  using value_type = T;
  using index_type = INDEX_TYPE;

  using pointer = T*;
  using const_pointer = T const *;
  using reference = T&;
  using const_reference = T const &;

  using iterator = pointer;
  using const_iterator = const_pointer;

  using size_type = INDEX_TYPE;

  inline ManagedArray():
    dataVector(),
    m_data(nullptr),
    m_dims{0},
    m_strides{0},
    m_length(0),
    m_singleParameterResizeIndex(0)
  {}

  template< typename... DIMS >
  inline explicit ManagedArray( DIMS... dims ):
    dataVector(),
    m_data(),
    m_dims{ static_cast<INDEX_TYPE>(dims) ...},
    m_strides(),
    m_length(0),
    m_singleParameterResizeIndex(0)
  {
    static_assert( is_integer<INDEX_TYPE>::value, 
                   "Error: std::is_integral<INDEX_TYPE> is false" );
    static_assert( sizeof ... (DIMS) == NDIM, 
                   "Error: calling ManagedArray::ManagedArray with incorrect number of arguments.");
    static_assert( check_dim_type<0,DIMS...>::value, 
                   "arguments to constructor of geosx::ManagedArray( DIMS... dims ) are incompatible with INDEX_TYPE" );
    CalculateStrides();

    resize( dims...);
  }


  ManagedArray( ManagedArray const & source ):
    dataVector(source.dataVector),
    m_data(&dataVector[0]),
    m_dims(),
    m_strides(),
    m_length(source.m_length),
    m_singleParameterResizeIndex(source.m_singleParameterResizeIndex)
  {
    for( int a = 0; a < NDIM; ++a )
    {
      m_dims[a] = source.m_dims[a];
      m_strides[a] = source.m_strides[a];
    }
  }


  operator ArrayView<T const, NDIM, INDEX_TYPE>() const
  {
    return ArrayView<T const, NDIM, INDEX_TYPE>( const_cast<const_pointer>(m_data), m_dims, 
                                                 m_strides );
  }


  operator ArrayView<T, NDIM, INDEX_TYPE>()
  {
    return ArrayView<T, NDIM, INDEX_TYPE>( m_data, m_dims, m_strides );
  }


  ManagedArray & operator=( ManagedArray const & source )
  {
    dataVector = source.dataVector;
    m_data = &dataVector[0];
    m_length = source.m_length;
    m_singleParameterResizeIndex = source.m_singleParameterResizeIndex;

    for( int a = 0; a < NDIM; ++a )
    {
      m_dims[a] = source.m_dims[a];
      m_strides[a] = source.m_strides[a];
    }

    return *this;
  }


  ManagedArray & operator=( T const & rhs )
  {
    for( INDEX_TYPE a = 0; a < m_length; ++a )
    {
      m_data[a] = rhs;
    }
    return *this;
  }


  void CalculateStrides()
  {
    m_strides[NDIM - 1] = 1;
    for( int a = NDIM - 2; a >= 0; --a )
    {
      m_strides[a] = m_dims[a + 1] * m_strides[a + 1];
    }
  }


  /**
   * \defgroup stl container interface
   * @{
   */


  void resize( int const numDims, INDEX_TYPE const * const dims )
  {
    if( numDims != NDIM )
    {
      abort();
    }

    for( int i = 0; i < NDIM; ++i )
    {
      m_dims[i] = dims[i];
    }

    CalculateStrides();
    resize();
  }

  void resize( int const numDims, INDEX_TYPE * const dims )
  {
    resize( numDims, const_cast<INDEX_TYPE const *>(dims) );
  }



  template< typename... DIMS >
  void resize( DIMS... newdims )
  {
    static_assert( sizeof ... (DIMS) == NDIM,
                   "Error: calling template< typename... DIMS > ManagedArray::resize(DIMS...newdims) with incorrect number of arguments.");
    static_assert( check_dim_type<0, DIMS...>::value, 
                   "arguments to ManagedArray::resize(DIMS...newdims) are incompatible with INDEX_TYPE" );

    dim_unpack<0,DIMS...>::f( m_dims, newdims... );
    CalculateStrides();
    resize();
  }


  void resize(int n_dims, long long const * const dims)
  {
    if ( n_dims != NDIM )
    {
      GEOS_ERROR("Dimensions mismatch: " << n_dims << " != " << NDIM);
    }

    for (int i = 0; i < NDIM; i++)
    {
      m_dims[i] = integer_conversion<INDEX_TYPE>(dims[i]);
    }

    CalculateStrides();
    resize();
  }


  template< typename TYPE >
  void resize( TYPE newdim )
  {
    static_assert( is_valid_indexType<TYPE>::value, "arguments to ManagedArray::resize(DIMS...newdims) are incompatible with INDEX_TYPE" );

    m_dims[m_singleParameterResizeIndex] = newdim;
    CalculateStrides();
    resize();
  }


  void reserve( INDEX_TYPE newLength )
  {
    if( newLength > dataVector.size() )
    {
      dataVector.reallocate(integer_conversion<uint>(newLength));
      m_data = &dataVector[0];
    }
  }


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"

  /**
   * @return total length of the array across all dimensions
   */
  INDEX_TYPE size() const
  {
    return m_length;
  }


  /**
   *
   * @param dim dimension for which the size is requested
   * @return length of a single dimensions specified by dim
   *
   */
  INDEX_TYPE size( int dim ) const
  {
    return m_dims[dim];
  }
  
#pragma GCC diagnostic pop

  int numDimensions() const
  { return NDIM; }

  bool empty() const
  { return m_length == 0; }

  void clear()
  {
    dataVector.free();
    m_data = nullptr;
    m_length = 0;

    for( int i = 0; i < NDIM; ++i )
    {
      m_dims[i] = 1;
    }
    m_dims[getSingleParameterResizeIndex()] = 0;

    CalculateStrides();
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type erase( iterator index )
  {
    while( index != end() )
    {
      *index = *(index + 1);
      index++;
    }

    m_dims[0]--;
    m_length--;
  }

  reference front() { return m_data[0]; }
  const_reference front() const { return m_data[0]; }

  reference back() { return m_data[m_length]; }
  const_reference back() const { return m_data[m_length]; }

  pointer data() { return m_data; }
  const_pointer data() const { return m_data; }

  inline const_pointer data(INDEX_TYPE const index) const
  {
    return &m_data[index * m_strides[0]];
  }

  inline pointer data(INDEX_TYPE const index)
  {
    return &m_data[index * m_strides[0]];
  }


  iterator begin() { return m_data; }
  const_iterator begin() const { return m_data; }

  iterator end() { return m_data + m_length; }
  const_iterator end() const { return m_data + m_length; }


  template<int N=NDIM>
  typename std::enable_if< N == 1, void >::type push_back( T const & newValue )
  {
    m_length++;
    m_dims[0]++;
    if( m_length > dataVector.size() )
    {
      reserve(2 * m_length);
    }

    m_data[m_length - 1] = newValue;
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type pop_back()
  { erase(end()); }

  template< int N=NDIM, class InputIt >
  typename std::enable_if< N==1, void >::type insert(iterator pos, InputIt first, InputIt last)
  {
    const INDEX_TYPE n_elems_to_insert = last - first;
    const INDEX_TYPE new_length = m_length + n_elems_to_insert;
    if( new_length > dataVector.size() )
    {
      reserve(2 * new_length);
    }

    for( iterator i = end() - 1; i != pos - 1; i-- )
    {
      *(i + n_elems_to_insert) = *i;
    }

    for( iterator i = pos; i != pos + n_elems_to_insert; ++i )
    {
      *i = *first;
      first++;
    }
  }

  /**@}*/


  inline ArrayView<T,NDIM,INDEX_TYPE> View() const
  {
    return ArrayView<T, NDIM, INDEX_TYPE>(m_data, m_dims, m_strides);
  }

  inline ArrayView<T,NDIM,INDEX_TYPE> View()
  {
    return ArrayView<T, NDIM, INDEX_TYPE>(m_data, m_dims, m_strides);
  }


  /**
   * @param index index of the element in array to access
   * @return a reference to the member m_childInterface, which is of type
   * ArrayView<T,NDIM-1>.
   * This function sets the data pointer for m_childInterface.m_data to the
   * location corresponding to the input
   * parameter "index". Thus, the returned object has m_data pointing to the
   * beginning of the data associated with its
   * sub-array.
   */

#if ARRAY_BOUNDS_CHECK == 1
  template< int U=NDIM >
  inline typename std::enable_if< U != 1, ArrayView<T, NDIM - 1, INDEX_TYPE> const >::type
  operator[](INDEX_TYPE const index) const
  {
    assert( index < m_dims[0] );
    return ArrayView<T, NDIM - 1, INDEX_TYPE>( &m_data[index * m_strides[0]], m_dims + 1, m_strides + 1 );
  }

  template< int U=NDIM >
  inline typename std::enable_if< U == 1, T const & >::type
  operator[](INDEX_TYPE const index) const
  {
    assert( index < m_dims[0] );
    return m_data[index];
  }
#else
  template< int U=NDIM >
  inline typename std::enable_if< U >= 3, ArrayView<T, NDIM - 1, INDEX_TYPE> const >::type
  operator[](INDEX_TYPE const index) const
  {
    return ArrayView<T, NDIM - 1, INDEX_TYPE>( &m_data[index * m_strides[0]], m_dims + 1, m_strides + 1 );
  }

  template< int U=NDIM >
  inline typename std::enable_if< U == 2, T const * restrict >::type
  operator[](INDEX_TYPE const index) const
  {
    return &m_data[index * m_strides[0]];
  }

  template< int U=NDIM >
  inline typename std::enable_if< U == 1, T const & >::type
  operator[](INDEX_TYPE const index) const
  {
    return m_data[index];
  }

#endif



#if ARRAY_BOUNDS_CHECK == 1
  template< int U=NDIM >
  inline typename std::enable_if< U != 1, ArrayView<T, NDIM - 1, INDEX_TYPE> >::type
  operator[](INDEX_TYPE const index)
  {
    assert( index < m_dims[0] );
    return ArrayView<T, NDIM - 1, INDEX_TYPE>(&m_data[index * m_strides[0]], m_dims + 1, m_strides + 1 );
  }

  template< int U=NDIM >
  inline typename std::enable_if< U == 1, T& >::type
  operator[](INDEX_TYPE const index)
  {
    assert( index < m_dims[0] );
    return m_data[index];
  }
#else
  template< int U=NDIM >
  inline typename std::enable_if< U >= 3, ArrayView<T, NDIM - 1, INDEX_TYPE> >::type
  operator[](INDEX_TYPE const index)
  {
    return ArrayView<T, NDIM - 1, INDEX_TYPE>( &m_data[index * m_strides[0]], m_dims + 1, m_strides + 1 );
  }

  template< int U=NDIM >
  inline typename std::enable_if< U == 2, T * restrict >::type
  operator[](INDEX_TYPE const index)
  {
    return m_data[index * m_strides[0]];
  }

  template< int U=NDIM >
  inline typename std::enable_if< U == 1, T & >::type
  operator[](INDEX_TYPE const index)
  {
    return m_data[index];
  }

#endif



  template< typename... DIMS >
  inline T & operator()( DIMS... dims ) const
  {
    return m_data[linearIndex(dims...)];
  }

  template< typename... DIMS >
  inline INDEX_TYPE linearIndex( DIMS... dims ) const
  {
    return index_helper<NDIM, DIMS...>::f(m_strides, dims...);
  }


  inline INDEX_TYPE const * dims() const
  { return m_dims; }

  inline INDEX_TYPE const * strides() const
  { return m_strides; }

  inline int getSingleParameterResizeIndex() const
  { return m_singleParameterResizeIndex; }

  inline void setSingleParameterResizeIndex( int const index )
  { m_singleParameterResizeIndex = index; }

private:
  chai::ManagedArray<T> dataVector;

  pointer m_data;

  /// pointer to array of length NDIM that contains the lengths of each array
  // dimension
  INDEX_TYPE m_dims[NDIM];

  INDEX_TYPE m_strides[NDIM];

  INDEX_TYPE m_length;

  int m_singleParameterResizeIndex = 0;


  void resize()
  {
    INDEX_TYPE length = 1;
    for( int a = 0; a < NDIM; ++a )
    {
      length *= m_dims[a];
    }

    if( length > dataVector.size() )
    {
      dataVector.reallocate(integer_conversion<uint>(length));
      m_data = &dataVector[0];
    }

    m_length = length;
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



  template< int DIM, typename INDEX, typename... REMAINING_INDICES >
  struct index_helper
  {
    inline constexpr static INDEX_TYPE f( INDEX_TYPE const * const restrict strides,
                                          INDEX index,
                                          REMAINING_INDICES... indices )
    {
      return index*strides[0] + index_helper<DIM-1,REMAINING_INDICES...>::f(strides+1,indices...);
    }
  };

  template< typename INDEX, typename... REMAINING_INDICES >
  struct index_helper<1,INDEX,REMAINING_INDICES...>
  {
    inline constexpr static INDEX_TYPE f( INDEX_TYPE const * const restrict dims,
                                          INDEX index,
                                          REMAINING_INDICES... indices )
    {
      return index;
    }
  };



  template< int INDEX, typename DIM0, typename... DIMS >
  struct dim_unpack
  {
    constexpr static void f( INDEX_TYPE m_dims[NDIM], DIM0 dim0, DIMS... dims )
    {
      m_dims[INDEX] = dim0;
      dim_unpack< INDEX+1, DIMS...>::f( m_dims, dims... );
    }
  };

  template< typename DIM0, typename... DIMS >
  struct dim_unpack<NDIM-1,DIM0,DIMS...>
  {
    constexpr static void f( INDEX_TYPE m_dims[NDIM], DIM0 dim0, DIMS... dims )
    {
      m_dims[NDIM-1] = dim0;
    }

  };

};



} /* namespace arraywrapper */

#endif /* SRC_COMPONENTS_CORE_SRC_ARRAY_MULTIDIMENSIONALARRAY_HPP_ */
