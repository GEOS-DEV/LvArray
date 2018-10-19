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
    ArraySlice<T,NDIM,INDEX_TYPE>( nullptr, nullptr, nullptr )
  {
    static_assert( !std::is_same< Array<U,NDIM,INDEX_TYPE>, Array<T,NDIM,INDEX_TYPE> >::value, "construction of ArrayView from Array is not allowed");
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
#ifdef GEOSX_USE_ARRAY_BOUNDS_CHECK
    for ( int dim = 0; dim < NDIM; ++dim )
    {
      GEOS_ERROR_IF( rhs.size( dim ) != size( dim ) );
    }
#endif

    INDEX_TYPE const length = size();
    for( INDEX_TYPE a = 0 ; a < length ; ++a )
    {
      m_data[a] = rhs.m_data[a];
    }
    return *this;
  }

  ArrayView & operator=( T const & rhs )
  {
    INDEX_TYPE const length = size();
    for( INDEX_TYPE a = 0 ; a < length ; ++a )
    {
      m_data[a] = rhs;
    }
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
  operator typename std::enable_if< (U > 1), ArrayView<T, NDIM - 1, INDEX_TYPE>& >::type ()
  {
    GEOS_ERROR_IF( m_dims[NDIM - 1] != 1,
                   "Array::operator ArrayView<T,NDIM-1,INDEX_TYPE> is only valid if last "
                   "dimension is equal to 1." );

    return static_cast<ArrayView<T, NDIM - 1, INDEX_TYPE>&>( *this );
  }

  INDEX_TYPE size() const
  {
#ifdef GEOSX_USE_ARRAY_BOUNDS_CHECK
    GEOS_ERROR_IF( size_helper<0>::f(m_dims) == m_dataVector.size(), "Size mismatch" );
#endif
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

    template< typename... INDICES >
  inline CONSTEXPRFUNC T & operator()( INDICES... indices ) const
  {
    return m_data[ linearIndex(indices...) ];
  }

  template< typename... INDICES >
  inline CONSTEXPRFUNC INDEX_TYPE linearIndex( INDICES... indices ) const
  {
#ifdef GEOSX_USE_ARRAY_BOUNDS_CHECK
    index_checker<NDIM, INDICES...>::f(m_dims, indices...);
#endif
    return index_helper<NDIM,INDICES...>::f(m_strides,indices...);
  }

  T * data() 
  {
    return m_data;
  }

  T const * data() const 
  {
    return m_data;
  }

    inline T const * data(INDEX_TYPE const index) const
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return &(m_data[ index*m_strides[0] ]);
  }

  inline T * data(INDEX_TYPE const index)
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return &(m_data[ index*m_strides[0] ]);
  }

  void copy( INDEX_TYPE const destIndex, INDEX_TYPE const sourceIndex )
  {
    ARRAY_SLICE_CHECK_BOUNDS( destIndex );
    ARRAY_SLICE_CHECK_BOUNDS( sourceIndex );

    operator[](destIndex) = operator[](sourceIndex);
  }

  INDEX_TYPE size( int dim ) const
  {

    return m_dims[dim];
  }

  inline INDEX_TYPE const * dims() const
  {
    return m_dims;
  }

  inline INDEX_TYPE const * strides() const
  {
    return m_strides;
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

  template< int DIM, typename INDEX, typename... REMAINING_INDICES >
  struct index_helper
  {
    inline CONSTEXPRFUNC static INDEX_TYPE f( INDEX_TYPE const * const restrict strides,
                                INDEX index, REMAINING_INDICES... indices )
    {
      return index*strides[0] + index_helper<DIM-1,REMAINING_INDICES...>::f(strides+1,indices...);
    }
  };

  template< typename INDEX, typename... REMAINING_INDICES >
  struct index_helper<1,INDEX,REMAINING_INDICES...>
  {
    inline CONSTEXPRFUNC static INDEX_TYPE f( INDEX_TYPE const * const restrict,
                                INDEX index )
    {
      return index;
    }
  };

  #ifdef GEOSX_USE_ARRAY_BOUNDS_CHECK
  template< int DIM, typename INDEX, typename... REMAINING_INDICES >
  struct index_checker
  {
    inline CONSTEXPRFUNC static void f( INDEX_TYPE const * const restrict dims,
                                        INDEX index, REMAINING_INDICES... indices )
    {
      GEOS_ERROR_IF( index < 0 || index > dims[0], "index=" << index << ", m_dims[" <<
                     (NDIMS - DIM) << "]=" << dims[0] );
      index_checker<DIM-1,REMAINING_INDICES...>::f(dims + 1,indices...);
    }
  };

  template< typename INDEX, typename... REMAINING_INDICES >
  struct index_checker<1,INDEX,REMAINING_INDICES...>
  {
    inline CONSTEXPRFUNC static INDEX_TYPE f( INDEX_TYPE const * const restrict dims,
                                              INDEX index )
    {
      GEOS_ERROR_IF( index < 0 || index > dims[0], "index=" << index << ", m_dims[" <<
                     (NDIMS - 1) << "]=" << dims[0] );
    }
  };
#endif

  template< int DIM >
  struct size_helper
  {
    template< int INDEX=DIM >
    CONSTEXPRFUNC static typename std::enable_if<INDEX!=NDIM-1,INDEX_TYPE>::type
    f( INDEX_TYPE const * const restrict dims )
    {
      return dims[INDEX] * size_helper<INDEX+1>::f(dims);
    }

    template< int INDEX=DIM >
    CONSTEXPRFUNC static typename std::enable_if<INDEX==NDIM-1,INDEX_TYPE>::type
    f( INDEX_TYPE const * const restrict dims )
    {
      return dims[INDEX];
    }

  };

  INDEX_TYPE m_dimsMem[NDIM];

  INDEX_TYPE m_stridesMem[NDIM];

  ArrayType m_dataVector;


};

} /* namespace LvArray */

#endif /* ARRAYVIEW_HPP_ */
