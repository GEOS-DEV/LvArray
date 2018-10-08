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
class ArrayView : public ArraySlice<T, NDIM, INDEX_TYPE >
{
public:

  using ArrayType = ChaiVector<T>;
  using iterator = typename ArrayType::iterator;
  using const_iterator = typename ArrayType::const_iterator;

  using ArraySlice<T,NDIM,INDEX_TYPE>::m_data;
  using ArraySlice<T,NDIM,INDEX_TYPE>::m_dims;
  using ArraySlice<T,NDIM,INDEX_TYPE>::m_strides;
  using ArraySlice<T,NDIM,INDEX_TYPE>::size;

  explicit
  ArrayView():
    ArraySlice<T,NDIM,INDEX_TYPE>( nullptr, m_dimsMem, m_stridesMem ),
    m_dimsMem{0},
    m_stridesMem{0},
    m_dataVector()
  {}

//  ArrayView( T * const data, INDEX_TYPE const * const dims, INDEX_TYPE const * const strides ):
//    ArraySlice<T,NDIM,INDEX_TYPE>( data , dims, strides )
//  {
//
//  }


  explicit
  ArrayView( ArrayView const & source ):
    ArraySlice<T,NDIM,INDEX_TYPE>( nullptr, m_dimsMem, m_stridesMem ),
    m_dataVector(source.m_dataVector)
  {
    //This triggers Chai::ManagedArray CC

    setDataPtr();
  }

  explicit
  ArrayView( ArrayView && source ):
    ArraySlice<T,NDIM,INDEX_TYPE>( nullptr, m_dimsMem, m_stridesMem ),
    m_dataVector( std::move( source.m_dataVector ) )
  {
    this->setDataPtr();
    setDims(source.m_dimsMem);
    setStrides(source.m_stridesMem);

  }

  ArrayView & operator=( ArrayView const & rhs )
  {
    INDEX_TYPE const rhsLength = rhs.size();
    INDEX_TYPE const length = size();

    for( int a=0 ; a<NDIM ; ++a )
    {
      GEOS_ERROR_IF( m_dimsMem[a] != rhs.m_dimsMem[a],
                     "rhs.m_dims["<<a<<"] of ("<<rhs.m_dimsMem[a]<<") does not equal "
                     "this->m_dims["<<a<<"] of ("<<m_dimsMem[a]<<")");
      GEOS_ERROR_IF( m_stridesMem[a] != rhs.m_stridesMem[a],
                     "rhs.m_stridesMem["<<a<<"] of ("<<rhs.m_stridesMem[a]<<") does not equal "
                     "this->m_stridesMem["<<a<<"] of ("<<m_stridesMem[a]<<")");

    }

    for( INDEX_TYPE a=0 ; a<length ; ++a )
    {
      m_data[a] = rhs[a];
    }
    return *this;
  }

  ArrayView & operator=( T const & rhs )
  {
    INDEX_TYPE const length = size();
    for( INDEX_TYPE a=0 ; a<length ; ++a )
    {
      m_data[a] = rhs;
    }
    return *this;
  }

  ArrayView & operator=( ArrayView && ) = delete;


  /**
   * User Defined Conversion operator to move from an ArrayView<T> to ArrayView<T const>
   */
  template< typename U = T >
  explicit
  operator typename std::enable_if< !std::is_const<U>::value ,ArrayView<T const,NDIM,INDEX_TYPE> >::type () const
  {
    return ArrayView<T const,NDIM,INDEX_TYPE>( const_cast<T const *>(m_data),
                                               m_dims,
                                               m_strides );
  }

  /**
   * User defined conversion to convert to a reduced dimension array. For example, converting from
   * a 2d array to a 1d array is valid if the last dimension of the 2d array is 1.
   */

  template< int U=NDIM >
  explicit
  operator typename std::enable_if< (U>1) ,ArrayView<T,NDIM-1,INDEX_TYPE> >::type ()
  {
    GEOS_ERROR_IF( m_dims[NDIM-1]==1,
                   "Array::operator ArrayView<T,NDIM-1,INDEX_TYPE> is only valid if last "
                   "dimension is equal to 1." );

    return ArrayView<T,NDIM-1,INDEX_TYPE>( m_data,
                                           m_dims,
                                           m_strides );
  }



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


protected:
  ArrayType m_dataVector;

  INDEX_TYPE m_dimsMem[NDIM];

  INDEX_TYPE m_stridesMem[NDIM];



};
}

#endif /* ARRAYVIEW_HPP_ */
