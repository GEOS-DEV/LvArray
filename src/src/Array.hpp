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

// Source includes
#include "helpers.hpp"
#include "ArrayView.hpp"
#include "bufferManipulation.hpp"

// TPL includes
#include <RAJA/RAJA.hpp>

// System includes
#include <iostream>
#include <limits>


namespace LvArray
{

/**
 * @class Array
 * @brief This class provides a multi-dimensional array interface to a vector type object.
 * @tparam T type of data that is contained by the array
 * @tparam NDIM number of dimensions in array (e.g. NDIM=1->vector, NDIM=2->Matrix, etc. )
 * @tparam PERMUTATION a camp::idx_seq containing the values in [0, NDIM) which describes how the
 *         data is to be laid out in memory. See RAJA::PERM_IJK etc.
 * @tparam INDEX_TYPE the integer to use for indexing the components of the array
 */
template< typename T,
          int NDIM,
          typename PERMUTATION,
          typename INDEX_TYPE,
          template< typename... > class DATA_VECTOR_TYPE >
class Array : public ArrayView< T,
                                NDIM,
                                getStrideOneDimension( PERMUTATION {} ),
                                INDEX_TYPE,
                                DATA_VECTOR_TYPE >
{
public:

  // Check that the template arguments are valid.
  static_assert( NDIM >= 0, "The dimension of the Array must be positive." );
  static_assert( getDimension( PERMUTATION {} ) == NDIM, "The dimension of the permutation must match the dimension of the Array." );
  static_assert( isValidPermutation( PERMUTATION {} ), "The permutation must be valid." );
  static_assert( std::is_integral< INDEX_TYPE >::value, "INDEX_TYPE must be integral." );
  
  // The dimensionality of the array.
  static constexpr int ndim = NDIM;

  // The dimension with unit stride.
  static constexpr int UNIT_STRIDE_DIM = getStrideOneDimension( PERMUTATION {} );

  // Alias for the parent class.
  using ParentType = ArrayView< T, NDIM, UNIT_STRIDE_DIM, INDEX_TYPE, DATA_VECTOR_TYPE >;

  // Aliasing public methods of ArrayView so we don't have to use this->size etc.
  using ParentType::toSlice;
  using ParentType::size;
  using ParentType::empty;
  using ParentType::setDataPtr;
  using ParentType::data;
  using ParentType::operator[];
  using ParentType::operator();

  // Aliases to help with SFINAE
  using value_type = T;
  using typename ParentType::pointer;
  using typename ParentType::const_pointer;

  // Aliases to convert nested Arrays into ArrayViews
  using ViewType = typename AsView< Array< T, NDIM, PERMUTATION, INDEX_TYPE, DATA_VECTOR_TYPE > >::type;
  using ViewTypeConst = typename AsConstView< Array< T, NDIM, PERMUTATION, INDEX_TYPE, DATA_VECTOR_TYPE > >::type;

  /**
   * @brief default constructor
   */
  inline Array():
    ParentType()
  {
    CalculateStrides();
#if defined(USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__)
    Array::TV_ttf_display_type( nullptr );
#endif
  }

  /**
   * @brief Constructor that takes in the dimensions as a variadic parameter list
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
   * @brief Copy constructor
   * @param source object to copy
   *
   * Performs a deep copy of source
   */
  Array( Array const & source ):
    ParentType()
  {
    // This DOES NOT trigger Chai::ManagedArray CC
    *this = source;
  }

  /**
   * @brief Move constructor
   * @param source object to move
   *
   * Moves the source into *this, a shallow copy that invalidates the contents of source.
   */
  Array( Array && source ) = default;

  /**
   * @brief Destructor, free's the data.
   */
  ~Array()
  {
    bufferManipulation::free( m_dataVector, size() );
    setDataPtr();
  }

  /**
   * @brief Return a reference to *this after converting any nested Arrays to ArrayView const.
   */
  ViewType & toView() const
  {
    return reinterpret_cast< ViewType & >(*this);
  }

  /**
   * @brief Return a reference to *this after converting any nested Arrays to ArrayView const.
   *        In addition the data held by the innermost array will be const.
   */
  template< typename U = T >
  std::enable_if_t< !std::is_const< U >::value, ViewTypeConst & >
  toViewConst() const
  {
    return reinterpret_cast< ViewTypeConst & >( *this );
  }

  /**
   * @brief assignment operator, performs a deep copy of rhs.
   * @param rhs source for the assignment.
   * @return *this.
   */
  Array & operator=( Array const& rhs )
  {
    bufferManipulation::copyInto( m_dataVector, size(), rhs.m_dataVector, rhs.size() );
    setDataPtr();

    this->setDims( rhs.m_dims );
    this->setStrides( rhs.m_strides );

    setSingleParameterResizeIndex( rhs.getSingleParameterResizeIndex() );
    return *this;
  }

  /**
   * @brief Function to assign entire array to single value.
   * @param rhs value to set array.
   * @return *this.
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
   * @brief Resize the dimensions of the Array to match the given dimensions.
   * @param dims the new size of the dimensions, must be of length NDIM.
   * @param ndims must equal NDIMS.
   * @note This does not preserve the values in the Array.
   */
  template< typename DIMS_TYPE >
  void resize( int const numDims, DIMS_TYPE const * const dims )
  {
    static_assert( check_dim_type< INDEX_TYPE, DIMS_TYPE >::value, "DIMS_TYPE is incompatible with INDEX_TYPE." );

    GEOS_ERROR_IF_NE( numDims, NDIM );

    INDEX_TYPE const oldSize = size();
    this->setDims( dims );
    CalculateStrides();

    bufferManipulation::resize( m_dataVector, oldSize, size() );
    setDataPtr();
  }

  /**
   * @brief Resize the dimensions of the Array to match the given dimensions.
   * @param dims the new size of the dimensions, must be of length NDIM.
   * @param ndims must equal NDIMS.
   * @note This does not preserve the values in the Array.
   *
   * @note this is required due to an issue where some compilers may prefer the full variadic
   *       parameter pack template<typename ... DIMS> resize( DIMS... newDims)
   */
  template< typename DIMS_TYPE >
  void resize( int const numDims, DIMS_TYPE * const dims )
  { resize( numDims, const_cast<DIMS_TYPE const *>(dims) ); }

  /**
   * @brief function to resize the array.
   * @tparam DIMS variadic pack containing the dimension types
   * @param newDims the new dimensions
   * @note This does not preserve the values in the Array.
   */
  template< typename... DIMS >
  std::enable_if_t< sizeof ... (DIMS) == NDIM >
  resize( DIMS... newDims )
  {
    static_assert( check_dim_type< INDEX_TYPE, DIMS... >::value, "arguments to Array::resize(DIMS...newDims) are incompatible with INDEX_TYPE" );

    INDEX_TYPE const oldSize = size();
    dimUnpack( const_cast< INDEX_TYPE * >( m_dims ), newDims... );
    CalculateStrides();
    
    bufferManipulation::resize( m_dataVector, oldSize, size() );
    setDataPtr();
  }

  /**
   * @brief function to resize the array without initializing any new values or destroying any old values.
   *        Only safe on POD data, however it is much faster for large allocations.
   * @tparam DIMS variadic pack containing the dimension types
   * @param newDims the new dimensions
   * @note This does not preserve the values in the Array.
   */
  template< typename... DIMS >
  std::enable_if_t< sizeof ... (DIMS) == NDIM >
  resizeWithoutInitializationOrDestruction( DIMS... newDims )
  {
    static_assert( check_dim_type<INDEX_TYPE, DIMS...>::value, "arguments to Array::resizeWithoutInitializationOrDestruction(DIMS...newDims) are incompatible with INDEX_TYPE" );

    INDEX_TYPE const oldSize = size();
    dimUnpack( const_cast< INDEX_TYPE * >( m_dims ), newDims... );
    CalculateStrides();

    bufferManipulation::reserve( m_dataVector, oldSize, size() );
    setDataPtr();
  }

  template < INDEX_TYPE... INDICES, typename... DIMS>
  void resizeDimension( DIMS... newDims )
  {
    static_assert( sizeof ... (INDICES) <= NDIM, "Too many arguments provided." );
    static_assert( sizeof ... (INDICES) == sizeof ... (DIMS), "The number of indices must match the number of dimensions." );
    static_assert( check_dim_type<INDEX_TYPE, DIMS...>::value, "arguments to Array::resizeDimension(DIMS...newDims) are incompatible with INDEX_TYPE" );
    static_assert( check_dim_indices<INDEX_TYPE, NDIM, INDICES...>::value, "invalid dimension indices in Array::resizeDimension(DIMS...newDims)" );

    INDEX_TYPE const oldSize = size();
    dim_index_unpack<INDEX_TYPE, NDIM>( const_cast<INDEX_TYPE *>(m_dims),
                                        std::integer_sequence<INDEX_TYPE, INDICES...>(),
                                        newDims... );
    CalculateStrides();
    
    bufferManipulation::resize( m_dataVector, oldSize, size() );
    setDataPtr();
  }

  /**
   * @brief function to resize the array using a single dimension
   * @param newdim the new dimension
   *
   * This function will use set the the dimension specified by m_singleParameterResizeIndex
   * and reallocate based on that new size.
   */
  void resize( INDEX_TYPE const newdim )
  { resizeDefaultDimension( newdim ); }

  void resizeDefault( INDEX_TYPE const newdim, T const & defaultValue = T() )
  { resizeDefaultDimension( newdim, defaultValue ); }

  template <class ...ARGS>
  void resizeWithArgs( INDEX_TYPE const newdim, ARGS &&... args )
  { resizeDefaultDimension( newdim, std::forward<ARGS>(args)... ); }

  /**
   *
   * @param newCapacity
   */
  void reserve( INDEX_TYPE const newCapacity )
  {
    bufferManipulation::reserve( m_dataVector, size(), newCapacity );
    setDataPtr();
  }

  INDEX_TYPE capacity() const
  { return integer_conversion<INDEX_TYPE>( m_dataVector.capacity() ); }

  void clear()
  {
    bufferManipulation::resize( m_dataVector, size(), 0 );
    setDataPtr();

    for( int i = 0 ; i < NDIM ; ++i )
    {
      const_cast<INDEX_TYPE *>(m_dims)[i] = 1;
    }
    const_cast<INDEX_TYPE *>(m_dims)[getSingleParameterResizeIndex()] = 0;

    CalculateStrides();
  }

  template< int N=NDIM >
  std::enable_if_t< N == 1 >
  push_back( T const & newValue )
  {
    bufferManipulation::pushBack( m_dataVector, size(), newValue );
    setDataPtr();
    const_cast< INDEX_TYPE * >( m_dims )[ 0 ]++;
  }

  template< int N=NDIM >
  std::enable_if_t< N == 1 >
  push_back( T && newValue )
  {
    bufferManipulation::pushBack( m_dataVector, size(), std::move( newValue ) );
    setDataPtr();
    const_cast< INDEX_TYPE * >( m_dims )[ 0 ]++;
  }

  template< int N=NDIM >
  std::enable_if_t< N == 1 >
  pop_back()
  {
    bufferManipulation::popBack( m_dataVector, size() );
    const_cast< INDEX_TYPE * >( m_dims )[ 0 ]--;
  }

  template< int N=NDIM >
  std::enable_if_t< N == 1 >
  insert( INDEX_TYPE pos, T const& value )
  {
    bufferManipulation::insert( m_dataVector, size(), pos, value );
    setDataPtr();
    const_cast< INDEX_TYPE * >( m_dims )[ 0 ]++;
  }

  template< int N=NDIM >
  std::enable_if_t< N == 1 >
  insert( INDEX_TYPE pos, T && value )
  {
    bufferManipulation::insert( m_dataVector, size(), pos, std::move( value ) );
    setDataPtr();
    const_cast< INDEX_TYPE * >( m_dims )[ 0 ]++;
  }

  template< int N=NDIM >
  std::enable_if_t< N == 1 >
  insert( INDEX_TYPE pos, T const * const values, INDEX_TYPE n )
  {
    bufferManipulation::insert( m_dataVector, size(), pos, values, n );
    setDataPtr();
    const_cast< INDEX_TYPE * >( m_dims )[ 0 ] += n;
  }

  template< int N=NDIM >
  std::enable_if_t< N == 1 >
  erase( INDEX_TYPE pos )
  {
    bufferManipulation::erase( m_dataVector, size(), pos );
    const_cast< INDEX_TYPE * >( m_dims )[ 0 ]--;
  }

  inline int getSingleParameterResizeIndex() const
  {
    return m_singleParameterResizeIndex;
  }

  inline void setSingleParameterResizeIndex( int const index )
  {
    m_singleParameterResizeIndex = index;
  }


  void setUserCallBack( std::string const & name )
  {
    m_dataVector.template setUserCallBack<decltype(*this)>( name );
  }

  template< typename U = T >
  std::enable_if_t< !isArray< U > >
  move( chai::ExecutionSpace space, bool touch=true )
  {
    m_dataVector.move( space, touch );
    setDataPtr();
  }

  template< typename U = T >
  std::enable_if_t< isArray< U > >
  move( chai::ExecutionSpace space, bool touch=true )
  {
    reinterpret_cast< DATA_VECTOR_TYPE< typename T::ViewType > & >( m_dataVector ).move( space, touch );
    setDataPtr();
  }

#if defined(USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__)
  /**
   * @brief Static function that will be used by Totalview to display the array contents.
   * @param av A pointer to the array that is being displayed.
   * @return 0 if everything went OK
   */
  static int TV_ttf_display_type( Array const * av)
  {
    return ParentType::TV_ttf_display_type( av );
  }
#endif

  std::array< camp::idx_t, NDIM > getPermutation() const
  { return RAJA::as_array< PERMUTATION >::get(); }

private:

  using ParentType::m_dataVector;
  using ParentType::m_dims;
  using ParentType::m_strides;
  using ParentType::m_singleParameterResizeIndex;

  /**
   * @brief Calculate the strides given the dimensions and permutation.
   * @note Adapted from RAJA::make_permuted_layout.
   */
  void CalculateStrides()
  {
    constexpr std::array< camp::idx_t, NDIM > const permutation = RAJA::as_array< PERMUTATION >::get();
    std::array< INDEX_TYPE, NDIM > foldedStrides;
    
    for (int i = 0; i < NDIM; ++i)
    {
      foldedStrides[ i ] = 1;
      for (int j = i + 1; j < NDIM; ++j)
      {
        foldedStrides[ i ] *= m_dims[ permutation[ j ] ];
      }
    }

    INDEX_TYPE * const strides = const_cast< INDEX_TYPE * >( m_strides );
    for (int i = 0; i < NDIM; ++i)
    {
      strides[ permutation[ i ] ] = foldedStrides[ i ];
    }
  }

  template <class ...ARGS>
  void resizeDefaultDimension( INDEX_TYPE const newDimLength, ARGS &&... args )
  {
    INDEX_TYPE const curDimLength = m_dims[ m_singleParameterResizeIndex ];
    INDEX_TYPE const curDimStride = m_strides[ m_singleParameterResizeIndex ];

    INDEX_TYPE const oldSize = size();
    const_cast< INDEX_TYPE * >( m_dims )[ m_singleParameterResizeIndex ] = newDimLength;
    INDEX_TYPE const newSize = size();
    CalculateStrides();

    // If we aren't changing the total size then we can return early.
    if ( newSize == oldSize ) return;
    
    if ( newDimLength > curDimLength )
    {
      // Reserve space in the buffer but don't initialize the values.
      bufferManipulation::reserve( m_dataVector, oldSize, newSize );
      setDataPtr();
      T * const ptr = data();

      INDEX_TYPE const valuesToAddPerIteration = curDimStride * ( newDimLength - curDimLength );
      INDEX_TYPE const valuesToShiftPerIteration = curDimStride * curDimLength;
      INDEX_TYPE const numIterations = ( newSize - oldSize ) / valuesToAddPerIteration;

      // Iterate backwards over the iterations.
      for ( INDEX_TYPE i = numIterations - 1; i >= 0; --i )
      {
        INDEX_TYPE const valuesLeftToInsert = valuesToAddPerIteration * i;
        T * const startOfShift = ptr + valuesToShiftPerIteration * i;
        arrayManipulation::shiftUp( startOfShift, valuesToShiftPerIteration, INDEX_TYPE( 0 ), valuesLeftToInsert );

        // Initialize the new values.
        T * const startOfNewValues = startOfShift + valuesToShiftPerIteration + valuesLeftToInsert;
        for ( INDEX_TYPE j = 0; j < valuesToAddPerIteration; ++j )
        {
          new ( startOfNewValues + j ) T( std::forward< ARGS >( args )... );
        }
      }
    }
    else
    {
      T * const ptr = data();

      INDEX_TYPE const valuesToRemovePerIteration = curDimStride * ( curDimLength - newDimLength );
      INDEX_TYPE const valuesToShiftPerIteration = curDimStride * newDimLength;
      INDEX_TYPE const numIterations = ( oldSize - newSize ) / valuesToRemovePerIteration;

      // Iterate over the iterations, skipping the first.
      for ( INDEX_TYPE i = 1; i < numIterations; ++i )
      {
        INDEX_TYPE const amountToShift = valuesToRemovePerIteration * i;
        T * const startOfShift = ptr + valuesToShiftPerIteration * i;
        arrayManipulation::shiftDown( startOfShift, valuesToShiftPerIteration + amountToShift, amountToShift, amountToShift );
      }

      // Initialize the new values
      INDEX_TYPE const totalValuesToRemove = valuesToRemovePerIteration * numIterations;
      for ( INDEX_TYPE i = 0; i < totalValuesToRemove; ++i )
      {
        ptr[ newSize + i ].~T();
      }
    }
  }

};

} /* namespace LvArray */

#endif /* ARRAY_HPP_ */
