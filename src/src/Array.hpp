/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2019, Lawrence Livermore National Security, LLC.
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

#include "ArrayView.hpp"
#include "helpers.hpp"
#include "ChaiVector.hpp"
#include "SFINAE_Macros.hpp"



namespace LvArray
{


/**
 * @class Array
 * @brief This class provides a multi-dimensional array interface to a vector type object.
 * @tparam T    type of data that is contained by the array
 * @tparam NDIM number of dimensions in array (e.g. NDIM=1->vector, NDIM=2->Matrix, etc. )
 * @tparam INDEX_TYPE the integer to use for indexing the components of the array
 */
template< typename T, int NDIM, typename INDEX_TYPE, typename DATA_VECTOR_TYPE >
class Array : public ArrayView< T,
                                NDIM,
                                INDEX_TYPE,
                                DATA_VECTOR_TYPE >
{
public:
  using isArray = std::true_type;
  using value_type = T;
  static constexpr int ndim = NDIM;

  using ArrayView<T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE>::m_dataVector;
  using ArrayView<T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE>::m_dims;
  using ArrayView<T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE>::m_strides;
  using ArrayView<T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE>::m_singleParameterResizeIndex;

  using ArrayView<T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE>::size;
  using ArrayView<T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE>::setDataPtr;
  using ArrayView<T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE>::data;
  using ArrayView<T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE>::operator[];
  using ArrayView<T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE>::operator();

  using typename ArrayView<T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE>::pointer;
  using typename ArrayView<T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE>::const_pointer;

  using asView = typename detail::to_arrayView< Array< T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE > >::type;
  using asViewConst = typename detail::to_arrayViewConst< Array< T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE > >::type;

  /**
   * @brief default constructor
   */
  inline Array():
    ArrayView<T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE>()
  {
    CalculateStrides();
#ifndef NDEBUG
    Array::TV_ttf_display_type( nullptr );
#endif

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
    static_assert( sizeof ... (DIMS) == NDIM, "Error: calling Array::Array with incorrect number of arguments." );
    static_assert( check_dim_type<INDEX_TYPE, DIMS...>::value, "arguments to constructor of geosx::Array( DIMS... dims ) are incompatible with INDEX_TYPE" );

    resize( dims... );
  }

  /**
   * @brief copy constructor
   * @param source object to copy
   *
   * Performs a deep copy of source
   */
  Array( Array const & source ):
    ArrayView<T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE>()
  {
    // This DOES NOT trigger Chai::ManagedArray CC
    *this = source;
  }

  Array( Array && source ) = default;

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
  operator typename std::enable_if< !std::is_const<U>::value,
                                    Array<T const, NDIM, INDEX_TYPE> const & >::type
    () const
  {
    return reinterpret_cast<Array<T const, NDIM, INDEX_TYPE> const &>(*this);
  }

  /**
   * Convert this Array and any nested Array to ArrayView const.
   */
  asView & toView() const
  {
    return reinterpret_cast<asView &>(*this);
  }

  /**
   * Convert this Array and any nested Array to ArrayView const.
   * In addition the data held by the innermost array will be const.
   */
  template< typename U = T >
  typename std::enable_if< !std::is_const<U>::value, asViewConst & >::type
  toViewConst() const
  {
    return reinterpret_cast<asViewConst &>(*this);
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
    resize( NDIM, rhs.m_dims );

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
   * @brief Function to assign entire array to single value
   * @param rhs value to set array
   * @return *this
   */
  Array & operator=( T const & rhs )
  {
    INDEX_TYPE const length = size();
    for( INDEX_TYPE a = 0 ; a < length ; ++a )
    {
      this->m_data[a] = rhs;
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
  template< typename DIMS_TYPE >
  void resize( int const numDims, DIMS_TYPE const * const dims )
  {
    GEOS_ERROR_IF( numDims != NDIM, "Dimension mismatch: " << numDims );
    INDEX_TYPE const oldLength = size();
    this->setDims( dims );
    CalculateStrides();
    resizePrivate( oldLength );
  }

  /**
   * @brief This function provides a resize or reallocation of the array
   * @param numDims the number of dims in the dims parameter
   * @param dims the new size of the dimensions
   *
   * @note this is required due to an issue where some compilers may prefer the full variadic
   *       parameter pack template<typename ... DIMS> resize( DIMS... newdims)
   */
  template< typename DIMS_TYPE >
  void resize( int const numDims, DIMS_TYPE * const dims )
  {
    resize( numDims, const_cast<DIMS_TYPE const *>(dims) );
  }

  /**
   * @brief function to resize/reallocate the array
   * @tparam DIMS variadic pack containing the dimension types
   * @param newdims the new dimensions
   */
  template< typename... DIMS,
            typename ENABLE = std::enable_if_t<sizeof ... (DIMS) == NDIM> >
  void resize( DIMS... newdims )
  {
    static_assert( check_dim_type<INDEX_TYPE, DIMS...>::value, "arguments to Array::resize(DIMS...newdims) are incompatible with INDEX_TYPE" );

    INDEX_TYPE const oldLength = size();
    dim_unpack<INDEX_TYPE, NDIM, NDIM, DIMS...>::f( const_cast<INDEX_TYPE *>(m_dims), newdims... );
    CalculateStrides();
    resizePrivate( oldLength );
  }

  /**
   * @brief function to resize/reallocate the array using a single dimension
   * @param newdim the new dimension
   *
   * This resize function will use set the dim[m_singleParameterResizeIndex]=newdim, and reallocate
   * based on that new size.
   */
  void resize( INDEX_TYPE const newdim )
  {
    INDEX_TYPE const oldLength = size();
    const_cast<INDEX_TYPE *>(m_dims)[m_singleParameterResizeIndex] = newdim;
    CalculateStrides();
    resizePrivate( oldLength );
  }

  void resizeDefault( INDEX_TYPE const newdim, T const & defaultValue = T() )
  {
    INDEX_TYPE const oldLength = size();
    const_cast<INDEX_TYPE *>(m_dims)[m_singleParameterResizeIndex] = newdim;
    CalculateStrides();
    resizePrivate( oldLength, defaultValue );
  }

  template <class ...ARGS>
  void resizeWithArgs( INDEX_TYPE const newdim, ARGS &&... args )
  {
    INDEX_TYPE const oldLength = size();
    const_cast<INDEX_TYPE *>(m_dims)[m_singleParameterResizeIndex] = newdim;
    CalculateStrides();
    resizePrivate( oldLength, std::forward<ARGS>(args)... );
  }

  template < INDEX_TYPE... INDICES, typename... DIMS>
  void resizeDimension( DIMS... newdims )
  {
    static_assert(sizeof ... (INDICES) <= NDIM, "Too many arguments provided.");
    static_assert(sizeof ... (INDICES) == sizeof ... (DIMS), "The number of indices must match the number of dimensions.");
    static_assert( check_dim_type<INDEX_TYPE, DIMS...>::value, "arguments to Array::resizeDimension(DIMS...newdims) are incompatible with INDEX_TYPE" );
    static_assert( check_dim_indices<INDEX_TYPE, NDIM, INDICES...>::value, "invalid dimension indices in Array::resizeDimension(DIMS...newdims)" );

    INDEX_TYPE const oldLength = size();
    dim_index_unpack<INDEX_TYPE, NDIM>( const_cast<INDEX_TYPE *>(m_dims),
                                        std::integer_sequence<INDEX_TYPE, INDICES...>(),
                                        newdims... );

    CalculateStrides();
    resizePrivate( oldLength );
  }

  /**
   *
   * @param newLength
   */
  void reserve( INDEX_TYPE newLength )
  {
    m_dataVector.reserve( newLength );
    setDataPtr();
  }

  INDEX_TYPE capacity() const
  { return integer_conversion<INDEX_TYPE>( m_dataVector.capacity()); }

  void clear()
  {
    m_dataVector.clear();
    setDataPtr();

    for( int i = 0 ; i < NDIM ; ++i )
    {
      const_cast<INDEX_TYPE *>(m_dims)[i] = 1;
    }
    const_cast<INDEX_TYPE *>(m_dims)[getSingleParameterResizeIndex()] = 0;

    CalculateStrides();
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type
  push_back( T const & newValue )
  {
    m_dataVector.push_back( newValue );
    setDataPtr();
    const_cast<INDEX_TYPE *>(m_dims)[0] = integer_conversion<INDEX_TYPE>( m_dataVector.size());
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type
  push_back( T && newValue )
  {
    m_dataVector.push_back( std::move( newValue ));
    setDataPtr();
    const_cast<INDEX_TYPE *>(m_dims)[0] = integer_conversion<INDEX_TYPE>( m_dataVector.size());
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type
  pop_back()
  {
    m_dataVector.pop_back();
    const_cast<INDEX_TYPE *>(m_dims)[0] = integer_conversion<INDEX_TYPE>( m_dataVector.size());
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type
  insert( INDEX_TYPE pos, T const& value )
  {
    m_dataVector.insert( pos, value );
    setDataPtr();
    const_cast<INDEX_TYPE *>(m_dims)[0] = integer_conversion<INDEX_TYPE>( m_dataVector.size());
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type
  insert( INDEX_TYPE pos, T && value )
  {
    m_dataVector.insert( pos, std::move( value ) );
    setDataPtr();
    const_cast<INDEX_TYPE *>(m_dims)[0] = integer_conversion<INDEX_TYPE>( m_dataVector.size());
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type
  insert( INDEX_TYPE pos, T const * const values, INDEX_TYPE n )
  {
    m_dataVector.insert( pos, values, n );
    setDataPtr();
    const_cast<INDEX_TYPE *>(m_dims)[0] = integer_conversion<INDEX_TYPE>( m_dataVector.size());
  }

  template<int N=NDIM>
  typename std::enable_if< N==1, void >::type erase( INDEX_TYPE pos )
  {
    m_dataVector.erase( pos );
    const_cast<INDEX_TYPE *>(m_dims)[0] = integer_conversion<INDEX_TYPE>( m_dataVector.size());
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


#ifdef USE_CHAI
  template< typename U = T >
  typename std::enable_if< !detail::is_array< U >::value, void >::type
  move( chai::ExecutionSpace space )
  {
    m_dataVector.move( space );
    setDataPtr();
  }

  template< typename U = T >
  typename std::enable_if< detail::is_array< U >::value, void >::type
  move( chai::ExecutionSpace space )
  {
    ChaiVector< typename T::asView > & dataVector =
      reinterpret_cast< ChaiVector< typename T::asView > & >( m_dataVector );
    dataVector.move( space );
    setDataPtr();
  }
#endif

#ifndef NDEBUG
  /**
   * @brief Static function that will be used by Totalview to display the array contents.
   * @param av A pointer to the array that is being displayed.
   * @return 0 if everything went OK
   */
  static int TV_ttf_display_type( Array const * av)
  {
    return ArrayView< T, NDIM, INDEX_TYPE, DATA_VECTOR_TYPE >::TV_ttf_display_type( av );
  }
#endif

private:

  void CalculateStrides()
  {
    const_cast<INDEX_TYPE *>(m_strides)[NDIM-1] = 1;
    for( int a=NDIM-2 ; a>=0 ; --a )
    {
      const_cast<INDEX_TYPE *>(m_strides)[a] = m_dims[a+1] * m_strides[a+1];
    }
  }

  template <class ...ARGS>
  void resizePrivate( INDEX_TYPE const oldLength, ARGS &&... args )
  {
    INDEX_TYPE const newLength = size_helper<NDIM, INDEX_TYPE>::f( m_dims );
    m_dataVector.resize( newLength, std::forward<ARGS>(args)... );
    this->setDataPtr();
  }

};

} /* namespace LvArray */

#endif /* ARRAY_HPP_ */
