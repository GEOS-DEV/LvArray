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

  ArrayView():
    ArraySlice<T,NDIM,INDEX_TYPE>( nullptr, m_dimsMem, m_stridesMem ),
    m_dimsMem{0},
    m_stridesMem{0},
    m_dataVector()
  {}

  ArrayView( T * const data, INDEX_TYPE const * const dims, INDEX_TYPE const * const strides ):
    ArraySlice<T,NDIM,INDEX_TYPE>( data , dims, strides )
  {

  }


  ArrayView( ArrayView const & source ):
    m_dataVector(source.m_dataVector)
  {
    //This triggers Chai::ManagedArray CC

    setDataPtr();
    setDimsPtr();
    setStridesPtr();
  }

  ArrayView( ArrayView && source ):
    ArraySlice<T,NDIM,INDEX_TYPE>( std::move( source ) ),
    m_dataVector( std::move( source.m_dataVector ) )
  {

  }

  ArrayView & operator=( ArrayView const & rhs )
  {
    INDEX_TYPE const rhsLength = rhs.size();
    INDEX_TYPE const length = size();

    GEOS_ERROR_IF( rhsLength != length,
                   "rhs length of ("<<rhsLength<<") does not equal this->size() of ("<<length<<")");

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

  void setDimsPtr()
  {
    INDEX_TYPE*& dimsPtr = const_cast<INDEX_TYPE*&>(this->m_dims);
    dimsPtr = m_dimsMem;
  }

  void setStridesPtr()
  {
    INDEX_TYPE*& stridesPtr = const_cast<INDEX_TYPE*&>(this->m_strides);
    stridesPtr = m_stridesMem;
  }

protected:
  ArrayType m_dataVector;

  INDEX_TYPE m_dimsMem[NDIM];

  INDEX_TYPE m_stridesMem[NDIM];



};
}

#endif /* ARRAYVIEW_HPP_ */
