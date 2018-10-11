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

#ifndef ARRAYVIEW_HPP_
#define ARRAYVIEW_HPP_


#include "ArraySlice.hpp"
#include "ChaiVector.hpp"


namespace LvArray
{

template< typename T, int NDIM, typename INDEX_TYPE >
class Array;

template< typename T, int NDIM, typename INDEX_TYPE = std::int_fast32_t >
class ArrayView : public ArraySlice<T, NDIM, INDEX_TYPE >
{
public:

  using ArraySlice<T,NDIM,INDEX_TYPE>::m_data;
  using ArraySlice<T,NDIM,INDEX_TYPE>::m_dims;
  using ArraySlice<T,NDIM,INDEX_TYPE>::m_strides;

  using ArraySlice<T,NDIM,INDEX_TYPE>::size;
  using ArraySlice<T,NDIM,INDEX_TYPE>::data;
  using ArraySlice<T,NDIM,INDEX_TYPE>::copy;

  using typename ArraySlice<T,NDIM,INDEX_TYPE>::size_type;

  using ArrayType = ChaiVector<T>;
  using iterator = typename ArrayType::iterator;
  using const_iterator = typename ArrayType::const_iterator;

  ArrayView():
    ArraySlice<T,NDIM,INDEX_TYPE>( nullptr, m_dimsMem, m_stridesMem ),
    m_dimsMem{0},
    m_stridesMem{0},
    m_dataVector()
  {}

 // ArrayView( T * const data, INDEX_TYPE const * const dims, INDEX_TYPE const * const strides ):
 //   ArraySlice<T,NDIM,INDEX_TYPE>( data , dims, strides )
 // {

 // }


  //This triggers Chai::ManagedArray CC
  ArrayView( ArrayView const & source ):
    ArraySlice<T,NDIM,INDEX_TYPE>( nullptr, m_dimsMem, m_stridesMem ),
    m_dataVector(source.m_dataVector)
  {
    setDataPtr();
    setDims(source.m_dimsMem);
    setStrides(source.m_stridesMem);
  }

  template< typename U=T  >
  ArrayView( typename std::enable_if< std::is_same< Array<U,NDIM,INDEX_TYPE>,
                                                    Array<T,NDIM,INDEX_TYPE> >::value,
                                      Array<U,NDIM,INDEX_TYPE> >::type const & ):
    ArraySlice<T,NDIM,INDEX_TYPE>( nullptr, m_dimsMem, m_stridesMem )
  {
    static_assert( !std::is_same< Array<U,NDIM,INDEX_TYPE>, Array<T,NDIM,INDEX_TYPE> >::value, "construction of ArrayView from Array is not allowed");
//    static_assert( false, "construction of ArrayView from Array is not allowed");
  }

  ArrayView( ArrayView && source ):
    ArraySlice<T,NDIM,INDEX_TYPE>( nullptr, m_dimsMem, m_stridesMem ),
    m_dataVector( std::move( source.m_dataVector ) )
  {
    setDataPtr();
    setDims(source.m_dimsMem);
    setStrides(source.m_stridesMem);
  }

  ArrayView & operator=( ArrayView const & rhs )
  {
    ArraySlice<T,NDIM,INDEX_TYPE>::operator=(rhs);
    return *this;
  }

  ArrayView & operator=( ArrayView && ) = delete;

  /**
   * User Defined Conversion operator to move from an ArrayView<T> to ArrayView<T const>
   */
  // template< typename U = T >
  // explicit
  // operator typename std::enable_if< !std::is_const<U>::value ,ArrayView<T const,NDIM,INDEX_TYPE> >::type () const
  // {
  //   return ArrayView<T const,NDIM,INDEX_TYPE>( const_cast<T const *>(m_data),
  //                                              m_dims,
  //                                              m_strides );
  // }

  /**
   * User defined conversion to convert to a reduced dimension array. For example, converting from
   * a 2d array to a 1d array is valid if the last dimension of the 2d array is 1.
   */

  template< int U=NDIM >
  explicit
  operator typename std::enable_if< (U>1) ,ArrayView<T,NDIM-1,INDEX_TYPE> >::type ()
  {
    GEOS_ERROR_IF( m_dims[NDIM-1]!=1,
                   "Array::operator ArrayView<T,NDIM-1,INDEX_TYPE> is only valid if last "
                   "dimension is equal to 1." );

    return ArrayView<T,NDIM-1,INDEX_TYPE>( m_data, m_dims, m_strides );
  }

  INDEX_TYPE size() const
  {
    return m_dataVector.size();
  }

  bool empty() const
  {
    return m_dataVector.empty();
  }

  T& front() 
  { 
    return m_dataVector.front();
  }

  T const& front() const 
  { 
    return m_dataVector.front();
  }

  T& back()
  { 
    return m_dataVector.back();
  }

  T const& back() const
  { 
    return m_dataVector.back();
  }

  iterator begin() 
  {
    return m_dataVector.begin();
  }

  const_iterator begin() const 
  {
    return m_dataVector.begin();
  }

  iterator end() 
  {
    return m_dataVector.end();
  }

  const_iterator end() const 
  {
    return m_dataVector.end();
  }

protected:

  void setDataPtr()
  {
    T*& dataPtr = const_cast<T*&>(this->m_data);
    dataPtr = m_dataVector.data() ;
  }

  void setDims( INDEX_TYPE const dims[NDIM] )
  {
    for( int a=0 ; a<NDIM ; ++a )
    {
      this->m_dimsMem[a] = dims[a];
    }
  }

  void setStrides( INDEX_TYPE const strides[NDIM] )
  {
    for( int a=0 ; a<NDIM ; ++a )
    {
      this->m_stridesMem[a] = strides[a];
    }
  }

  INDEX_TYPE m_dimsMem[NDIM];

  INDEX_TYPE m_stridesMem[NDIM];

  ArrayType m_dataVector;


};

} /* namespace LvArray */

#endif /* ARRAYVIEW_HPP_ */
