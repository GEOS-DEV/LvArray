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

#ifndef ARRAY_HPP_
#define ARRAY_HPP_


#include <iostream>
#include <limits>
#include <vector>
#include <iterator>

#include "Logger.hpp"
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


template< typename T, int NDIM, typename INDEX_TYPE=std::int_fast32_t > class Array;

namespace detail
{
template<typename>
struct is_array : std::false_type {};

template< typename T, int NDIM, typename INDEX_TYPE >
struct is_array< multidimensionalArray::Array<T,NDIM,INDEX_TYPE> > : std::true_type{};
}


template< typename T, int NDIM, typename INDEX_TYPE >
class Array
{
public:
  using ArrayType = ChaiVector<T>;

  using value_type = T;
  using index_type = INDEX_TYPE;
  using iterator = typename ArrayType::iterator;
  using const_iterator = typename ArrayType::const_iterator;


  using pointer = T*;
  using const_pointer = T const *;
  using reference = T&;
  using const_reference = T const &;

  using size_type = INDEX_TYPE;

  using isArray = std::true_type;

  inline Array():
    ArrayView(),
    m_singleParameterResizeIndex(0)
  {
    CalculateStrides();
  }

  template< typename... DIMS >
  inline explicit Array( DIMS... dims ):
    Array()
  {
    static_assert( is_integer<INDEX_TYPE>::value, "Error: std::is_integral<INDEX_TYPE> is false" );
    static_assert( sizeof ... (DIMS) == NDIM, "Error: calling Array::Array with incorrect number of arguments.");
    static_assert( check_dim_type<0, DIMS...>::value, "arguments to constructor of geosx::Array( DIMS... dims ) are incompatible with INDEX_TYPE" );

    resize( dims... );
  }

  Array( Array const & source ):
    m_singleParameterResizeIndex(source.m_singleParameterResizeIndex)
  {
    // This DOES NOT trigger Chai::ManagedArray CC

    for( int a=0 ; a<NDIM ; ++a )
    {
      m_dims[a]     = source.m_dims[a];
      m_strides[a]  = source.m_strides[a];
    }
  }

  Array( Array&& source ):
    ArrayView( std::move( source ) ),
    m_singleParameterResizeIndex(source.m_singleParameterResizeIndex)
  {}



  /**
   * User Defined Conversion operator to move from an ArrayView<T> to T *
   */
  template< int U = NDIM,
            typename std::enable_if<U==1, int>::type = 0>
  inline
  operator T *()
  {
    return m_data;
  }

  template< int U = NDIM,
            typename std::enable_if<U==1, int>::type = 0>
  inline
  operator T const *() const
  {
    return m_data;
  }

  operator ArrayView<T const,NDIM,INDEX_TYPE>() const
  {
    return ArrayView<T const,NDIM,INDEX_TYPE>( const_cast<T const *>(m_data),
                                               m_dims,
                                               m_strides );
  }

  operator ArrayView<T,NDIM,INDEX_TYPE>()
  {
    return ArrayView<T,NDIM,INDEX_TYPE>( m_data,
                                         m_dims,
                                         m_strides );
  }

  /**
   * User defined conversion to convert to a reduced dimension array. For example, converting from
   * a 2d array to a 1d array is valid if the last dimension of the 2d array is 1.
   */
  template< int U=NDIM >
  operator typename std::enable_if< (U>1) ,ArrayView<T,NDIM-1,INDEX_TYPE> >::type ()

  {
    assert(m_dims[NDIM-1]==1);//,
//                "Array::operator ArrayView<T,NDIM-1,INDEX_TYPE> is only valid if last "
//                "dimension is equal to 1.")
    return ArrayView<T,NDIM-1,INDEX_TYPE>( m_data,
                                           m_dims,
                                           m_strides );
  }

  Array & operator=( Array const & source )
  {
    m_dataVector = source.m_dataVector;
    m_data = m_dataVector.data();

    for( int a=0 ; a<NDIM ; ++a )
    {
      m_dims[a] = source.m_dims[a];
      m_strides[a] = source.m_strides[a];
    }

    return *this;
  }

  Array & operator=( Array&& source )
  {
    m_dataVector = std::move(source.m_dataVector);
    m_data = m_dataVector.data();
    m_singleParameterResizeIndex = source.m_singleParameterResizeIndex;

    for( int a=0 ; a<NDIM ; ++a )
    {
      m_dims[a]     = source.m_dims[a];
      m_strides[a]  = source.m_strides[a];
    }

    return *this;
  }

  Array & operator=( T const & rhs )
  {
    INDEX_TYPE const length = size();
    for( INDEX_TYPE a=0 ; a<length ; ++a )
    {
      m_data[a] = rhs;
    }
    return *this;
  }


  void CalculateStrides()
  {
    m_strides[NDIM-1] = 1;
    for( int a=NDIM-2 ; a>=0 ; --a )
    {
      m_strides[a] = m_dims[a+1] * m_strides[a+1];
    }
  }

  template< int U=NDIM >
  inline  typename std::enable_if< U>=3, void >::type
  copy( INDEX_TYPE const destIndex, INDEX_TYPE const sourceIndex )
  {
    assert(false);
  }

  template< int U=NDIM >
  inline  typename std::enable_if< U==2, void >::type
  copy( INDEX_TYPE const destIndex, INDEX_TYPE const sourceIndex )
  {
    for( INDEX_TYPE a=0 ; a<size(1) ; ++a )
    {
      m_data[destIndex*m_strides[0]+a] = m_data[sourceIndex*m_strides[0]+a];
    }
  }

  template< int U=NDIM >
  inline typename std::enable_if< U==1, void >::type
  copy( INDEX_TYPE const destIndex, INDEX_TYPE const sourceIndex )
  {
    m_data[ destIndex ] = m_data[ sourceIndex ];
  }

  bool isCopy() const
  { return m_dataVector.isCopy(); }

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

    INDEX_TYPE length = 1;

    for( int i=0 ; i<NDIM ; ++i )
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
                   "Error: calling template< typename... DIMS > Array::resize(DIMS...newdims) with incorrect number of arguments.");
    static_assert( check_dim_type<0,DIMS...>::value, "arguments to Array::resize(DIMS...newdims) are incompatible with INDEX_TYPE" );

    INDEX_TYPE length = 1;

    dim_unpack<0,DIMS...>::f( m_dims, newdims...);
    CalculateStrides();
    resize();
  }


  void resize(int n_dims, long long const * const dims)
  {
    assert( n_dims == NDIM );

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
    static_assert( is_valid_indexType<TYPE>::value, "arguments to Array::resize(DIMS...newdims) are incompatible with INDEX_TYPE" );

    m_dims[m_singleParameterResizeIndex] = newdim;
    CalculateStrides();
    resize();
  }


  void reserve( INDEX_TYPE newLength )
  {
    m_dataVector.reserve(newLength);
    m_data = m_dataVector.data();
  }



  INDEX_TYPE capacity() const
  { return m_dataVector.capacity(); }
  


  bool empty() const
  { return size() == 0; }

  void clear()
  {
    m_dataVector.clear();
    m_data = nullptr;

    for( int i = 0; i < NDIM; ++i )
    {
      m_dims[i] = 1;
    }
    m_dims[getSingleParameterResizeIndex()] = 0;

    CalculateStrides();
  }


  reference       front()       { return m_dataVector.front(); }
  const_reference front() const { return m_dataVector.front(); }

  reference       back()       { return m_dataVector.back(); }
  const_reference back() const { return m_dataVector.back(); }

  T *       data()       {return m_data;}
  T const * data() const {return m_data;}



  inline T const *
  data(INDEX_TYPE const index) const
  {
    return &(m_data[ index*m_strides[0] ]);
  }

  inline T *
  data(INDEX_TYPE const index)
  {
    return &(m_data[ index*m_strides[0] ]);
  }

  iterator begin() {return m_dataVector.begin();}
  const_iterator begin() const {return m_dataVector.begin();}

  iterator end() {return m_dataVector.end();}
  const_iterator end() const {return m_dataVector.end();}

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type 
  push_back( T const & newValue )
  {
    m_dataVector.push_back(newValue);
    m_data = m_dataVector.data();
    m_dims[0] = integer_conversion<INDEX_TYPE>(m_dataVector.size());
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type pop_back()
  {
    m_dataVector.pop_back();
    m_dims[0] = integer_conversion<INDEX_TYPE>(m_dataVector.size());
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type 
  insert(iterator pos, T const& value)
  {
    m_dataVector.insert( pos, value );
    m_data = m_dataVector.data();
    m_dims[0] = integer_conversion<INDEX_TYPE>(m_dataVector.size());
  }

  template<int N=NDIM, typename InputIt>
  typename std::enable_if< N==1, void >::type 
  insert(iterator pos, InputIt first, InputIt last)
  {
    m_dataVector.insert( pos, first, last );
    m_data = m_dataVector.data();
    m_dims[0] = integer_conversion<INDEX_TYPE>(m_dataVector.size());
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type erase( iterator index )
  {
    m_dataVector.erase( index ) ;
    m_dims[0] = integer_conversion<INDEX_TYPE>(m_dataVector.size());
  }

  /**@}*/


  inline ArrayView<T,NDIM,INDEX_TYPE> View() const
  {
    return ArrayView<T,NDIM,INDEX_TYPE>(this->m_data,
                                        this->m_dims,
                                        this->m_strides);
  }

  inline ArrayView<T,NDIM,INDEX_TYPE> View()
  {
    return ArrayView<T,NDIM,INDEX_TYPE>(this->m_data,
                                        this->m_dims,
                                        this->m_strides);
  }


  inline int getSingleParameterResizeIndex() const
  {
    return m_singleParameterResizeIndex;
  }

  inline void setSingleParameterResizeIndex( int const index )
  {
    m_singleParameterResizeIndex = index;
  }

  friend std::ostream& operator<< (std::ostream& stream, ManagedArray const & array )
  {
    stream<<"{ "<<array.m_data[0];
    for( INDEX_TYPE a=1 ; a<array.size() ; ++a )
    {
      stream<<", "<<array.m_data[a];
    }
    stream<<" }";
    return stream;
  }



private:

  int m_singleParameterResizeIndex = 0;

  Array( ChaiVector<T>&& source, const INDEX_TYPE* dims, int resize_index ) :
    m_dataVector(std::move(source)),
    m_data(m_dataVector.data()),
    m_dims(),
    m_strides(),
    m_singleParameterResizeIndex(resize_index)
  {  
    for( int a=0 ; a<NDIM ; ++a )
    {
      m_dims[a] = dims[a];
    }

    CalculateStrides();
  }

  void resize()
  {
    INDEX_TYPE length = 1;
    for( int a=0 ; a<NDIM ; ++a )
    {
      length *= m_dims[a];
    }

    m_dataVector.resize( length );
    setDataPointer();
  }

  void setDataPointer()
  {
    const_cast<T *>(m_data) = m_dataVector.data();
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

}

#endif /* ARRAY_HPP_ */
