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
#include "arrayHelpers.hpp"
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
 * @brief This class provides a fixed dimensional resizeable array interface in addition to an
 *        interface similar to std::vector for a 1D array.
 * @tparam T type of data that is contained by the array.
 * @tparam NDIM number of dimensions in array (e.g. NDIM=1->vector, NDIM=2->Matrix, etc. ).
 *         Some methods such as push_back are only available for a 1D array.
 * @tparam PERMUTATION a camp::idx_seq containing the values in [0, NDIM) which describes how the
 *         data is to be laid out in memory. Given an 3-dimensional array A of dimension (L, M, N)
 *         the standard layout is A( i, j, k ) = A.data()[ i * M * N + j * N + k ], this layout is
 *         the given by the identity permutation (0, 1, 2) which says that the first dimension in
 *         memory corresponds to the first index (i), and similarly for the other dimensions. The
 *         same array with a layout of A( i, j, k ) = A.data()[ L * N * j + L * k + i ] would have
 *         a permutation of (1, 2, 0) because the first dimension in memory corresponds to the second
 *         index (j), the second dimension in memory corresponds to the third index (k) and the last
 *         dimension in memory goes with the first index (i).
 *         Note: RAJA provides aliases for every valid permutations up to 5D, the two permutations
 *               mentioned above can be used via RAJA::PERM_IJK and RAJA::PERM_JKI.
 *         Note: The dimension with unit stride is the last dimension in the permutation.
 * @tparam INDEX_TYPE the integer to use for indexing.
 * @tparam BUFFER_TYPE A class that defines how to actually allocate memory for the Array. Must take
 *         one template argument that describes the type of the data being stored (T).
 */
template< typename T,
          int NDIM,
          typename PERMUTATION,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class Array : public ArrayView< T,
                     NDIM,
  getStrideOneDimension( PERMUTATION {} ),
                     INDEX_TYPE,
                     BUFFER_TYPE >
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
  using ParentType = ArrayView< T, NDIM, UNIT_STRIDE_DIM, INDEX_TYPE, BUFFER_TYPE >;

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
  using ViewType = typename AsView< Array< T, NDIM, PERMUTATION, INDEX_TYPE, BUFFER_TYPE > >::type;
  using ViewTypeConst = typename AsConstView< Array< T, NDIM, PERMUTATION, INDEX_TYPE, BUFFER_TYPE > >::type;

  /**
   * @brief default constructor
   */
  inline Array():
    ParentType( true )
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
  template< typename ... DIMS >
  inline explicit Array( DIMS... dims ):
    Array()
  {
    static_assert( is_integer< INDEX_TYPE >::value, "Error: std::is_integral<INDEX_TYPE> is false" );
    static_assert( sizeof ... (DIMS) == NDIM, "Error: calling Array::Array with incorrect number of arguments." );
    static_assert( check_dim_type< INDEX_TYPE, DIMS... >::value, "arguments to constructor of geosx::Array( DIMS... dims ) are incompatible with INDEX_TYPE" );

    resize( dims ... );
  }

  /**
   * @brief Copy constructor.
   * @param source object to copy.
   *
   * @note Performs a deep copy of source
   */
  Array( Array const & source ):
    Array()
  {
    *this = source;
  }

  /**
   * @brief Move constructor.
   * @param source object to move.
   *
   * @note Moves the source into *this. This calls BUFFER_TYPE( BUFFER_TYPE && ) which usually is a
   *       shallow copy that invalidates the contents of source. However this depends on the
   *       implementation of BUFFER_TYPE since for instance it is impossible to create a shallow
   *       copy of a stack allocation.
   */
  Array( Array && source ) = default;

  /**
   * @brief Destructor, free's the data.
   */
  ~Array()
  {
    bufferManipulation::free( m_dataBuffer, size() );
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
  Array & operator=( Array const & rhs )
  {
    bufferManipulation::copyInto( m_dataBuffer, size(), rhs.m_dataBuffer, rhs.size() );
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

    LVARRAY_ERROR_IF_NE( numDims, NDIM );

    INDEX_TYPE const oldSize = size();
    this->setDims( dims );
    CalculateStrides();

    bufferManipulation::resize( m_dataBuffer, oldSize, size() );
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
  { resize( numDims, const_cast< DIMS_TYPE const * >(dims) ); }

  /**
   * @brief function to resize the array.
   * @tparam DIMS variadic pack containing the dimension types
   * @param newDims the new dimensions
   * @note This does not preserve the values in the Array.
   */
  template< typename ... DIMS >
  std::enable_if_t< sizeof ... (DIMS) == NDIM >
  resize( DIMS... newDims )
  {
    static_assert( check_dim_type< INDEX_TYPE, DIMS... >::value, "arguments to Array::resize(DIMS...newDims) are incompatible with INDEX_TYPE" );

    INDEX_TYPE const oldSize = size();
    dimUnpack( const_cast< INDEX_TYPE * >( m_dims ), newDims ... );
    CalculateStrides();

    bufferManipulation::resize( m_dataBuffer, oldSize, size() );
    setDataPtr();
  }

  /**
   * @brief Resize the array without initializing any new values or destroying any old values.
   *        Only safe on POD data, however it is much faster for large allocations.
   * @tparam DIMS variadic pack containing the dimension types
   * @param newDims the new dimensions
   * @note This does not preserve the values in the Array.
   */
  template< typename ... DIMS >
  std::enable_if_t< sizeof ... (DIMS) == NDIM >
  resizeWithoutInitializationOrDestruction( DIMS... newDims )
  {
    static_assert( check_dim_type< INDEX_TYPE, DIMS... >::value,
                   "arguments to Array::resizeWithoutInitializationOrDestruction(DIMS...newDims) are incompatible with INDEX_TYPE" );

    INDEX_TYPE const oldSize = size();
    dimUnpack( const_cast< INDEX_TYPE * >( m_dims ), newDims ... );
    CalculateStrides();

    bufferManipulation::reserve( m_dataBuffer, oldSize, size() );
    setDataPtr();
  }

  /**
   * @brief Resize specific dimensions of the array.
   * @tparam INDICES the indices of the dimensions to resize.
   * @tparam DIMS variadic pack containing the dimension types
   * @param newDims the new dimensions. newDims[ 0 ] will be the new size of
   *        dimensions INDICES[ 0 ].
   * @note This does not preserve the values in the Array.
   */
  template< INDEX_TYPE... INDICES, typename ... DIMS >
  void resizeDimension( DIMS... newDims )
  {
    static_assert( sizeof ... (INDICES) <= NDIM, "Too many arguments provided." );
    static_assert( sizeof ... (INDICES) == sizeof ... (DIMS), "The number of indices must match the number of dimensions." );
    static_assert( check_dim_type< INDEX_TYPE, DIMS... >::value, "arguments to Array::resizeDimension(DIMS...newDims) are incompatible with INDEX_TYPE" );
    static_assert( check_dim_indices< INDEX_TYPE, NDIM, INDICES... >::value, "invalid dimension indices in Array::resizeDimension(DIMS...newDims)" );

    INDEX_TYPE const oldSize = size();
    dim_index_unpack< INDEX_TYPE, NDIM >( const_cast< INDEX_TYPE * >(m_dims),
                                          std::integer_sequence< INDEX_TYPE, INDICES... >(),
                                          newDims ... );
    CalculateStrides();

    bufferManipulation::resize( m_dataBuffer, oldSize, size() );
    setDataPtr();
  }

  /**
   * @brief Resize the default dimension of the Array.
   * @param newdim the new size of the default dimension.
   * @note This preserves the values in the Array.
   * @note The default dimension is given by m_singleParameterResizeIndex.
   */
  void resize( INDEX_TYPE const newdim )
  { resizeDefaultDimension( newdim ); }

  /**
   * @brief Resize the default dimension of the Array.
   * @param newdim the new size of the default dimension.
   * @param defaultValue the value to initialize the new values with.
   * @note This preserves the values in the Array.
   * @note The default dimension is given by m_singleParameterResizeIndex.
   */
  void resizeDefault( INDEX_TYPE const newdim, T const & defaultValue = T() )
  { resizeDefaultDimension( newdim, defaultValue ); }

  /**
   * @brief Resize the default dimension of the Array.
   * @tparam ARGS variadic pack containing the types to initialize the new values with.
   * @param newdim the new size of the default dimension.
   * @param ARGS arguments to initialize the new values with.
   * @note This preserves the values in the Array.
   * @note The default dimension is given by m_singleParameterResizeIndex.
   */
  template< typename ... ARGS >
  void resizeWithArgs( INDEX_TYPE const newdim, ARGS && ... args )
  { resizeDefaultDimension( newdim, std::forward< ARGS >( args )... ); }

  /**
   * @brief Reserve space in the Array to hold at least the given number of values.
   * @param newCapacity the number of values to reserve space for. After this call
   *        capacity() >= newCapacity.
   */
  void reserve( INDEX_TYPE const newCapacity )
  {
    bufferManipulation::reserve( m_dataBuffer, size(), newCapacity );
    setDataPtr();
  }

  /**
   * @brief Return the maximum number of values the Array can hold without reallocation.
   */
  INDEX_TYPE capacity() const
  { return integer_conversion< INDEX_TYPE >( m_dataBuffer.capacity() ); }

  /**
   * @brief Sets the size of the Array to zero and destroys all the values.
   *        Sets the size of m_singleParameterResizeIndex to 0 but leaves the
   *        size of the other dimensions untouched. Equivalent to resize( 0 ).
   */
  void clear()
  {
    bufferManipulation::resize( m_dataBuffer, size(), 0 );
    setDataPtr();

    const_cast< INDEX_TYPE * >( m_dims )[ getSingleParameterResizeIndex() ] = 0;

    CalculateStrides();
  }

  /**
   * @brief Append a value to the end of the array. Equivalent to std::vector::push_back.
   * @param value the value to append via a copy.
   * @note This method is only available on 1D arrays.
   */
  template< int N=NDIM >
  std::enable_if_t< N == 1 >
  push_back( T const & value )
  {
    bufferManipulation::pushBack( m_dataBuffer, size(), value );
    setDataPtr();
    const_cast< INDEX_TYPE * >( m_dims )[ 0 ]++;
  }

  /**
   * @brief Append a value to the end of the array. Equivalent to std::vector::push_back.
   * @param value the value to append via a move.
   * @note This method is only available on 1D arrays.
   */
  template< int N=NDIM >
  std::enable_if_t< N == 1 >
  push_back( T && value )
  {
    bufferManipulation::pushBack( m_dataBuffer, size(), std::move( value ) );
    setDataPtr();
    const_cast< INDEX_TYPE * >( m_dims )[ 0 ]++;
  }

  /**
   * @brief Remove the last value in the array. Equivalent to std::vector::pop_back.
   * @note This method is only available on 1D arrays.
   */
  template< int N=NDIM >
  std::enable_if_t< N == 1 >
  pop_back()
  {
    bufferManipulation::popBack( m_dataBuffer, size() );
    const_cast< INDEX_TYPE * >( m_dims )[ 0 ]--;
  }

  /**
   * @brief Insert a value into the array at the given position. Similar to std::vector::insert
   *        but this method takes a position instead of an iterator.
   * @param pos the position to insert at.
   * @param value the value to insert via a copy.
   * @note This method is only available on 1D arrays.
   */
  template< int N=NDIM >
  std::enable_if_t< N == 1 >
  insert( INDEX_TYPE const pos, T const & value )
  {
    bufferManipulation::insert( m_dataBuffer, size(), pos, value );
    setDataPtr();
    const_cast< INDEX_TYPE * >( m_dims )[ 0 ]++;
  }

  /**
   * @brief Insert a value into the array at the given position. Similar to std::vector::insert
   *        but this method takes a position instead of an iterator.
   * @param pos the position to insert at.
   * @param value the value to insert via a move.
   * @note This method is only available on 1D arrays.
   */
  template< int N=NDIM >
  std::enable_if_t< N == 1 >
  insert( INDEX_TYPE const pos, T && value )
  {
    bufferManipulation::insert( m_dataBuffer, size(), pos, std::move( value ) );
    setDataPtr();
    const_cast< INDEX_TYPE * >( m_dims )[ 0 ]++;
  }

  /**
   * @brief Insert values into the array at the given position. Similar to std::vector::insert
   *        but this method takes a position, a pointer and a size instead of three iterators.
   * @param pos the position to insert at.
   * @param values a pointer to the values to insert via copies.
   * @param n the number of values to insert.
   * @note This method is only available on 1D arrays.
   */
  template< int N=NDIM >
  std::enable_if_t< N == 1 >
  insert( INDEX_TYPE const pos, T const * const values, INDEX_TYPE const n )
  {
    bufferManipulation::insert( m_dataBuffer, size(), pos, values, n );
    setDataPtr();
    const_cast< INDEX_TYPE * >( m_dims )[ 0 ] += n;
  }

  /**
   * @brief Remove the value at the given position. Similar to std::vector::erase
   *        but this method takes a position instead of an iterator.
   * @param pos the position of the value to remove.
   * @note This method is only available on 1D arrays.
   */
  template< int N=NDIM >
  std::enable_if_t< N == 1 >
  erase( INDEX_TYPE pos )
  {
    bufferManipulation::erase( m_dataBuffer, size(), pos );
    const_cast< INDEX_TYPE * >( m_dims )[ 0 ]--;
  }

  /**
   * @brief Get the default resize dimension.
   */
  inline int getSingleParameterResizeIndex() const
  {
    return m_singleParameterResizeIndex;
  }

  /**
   * @brief Set the default resize dimension.
   * @param index the new default dimension.
   */
  inline void setSingleParameterResizeIndex( int const index )
  {
    m_singleParameterResizeIndex = index;
  }

  /**
   * @brief Set the name of the Array to be displayed whenever
   *        the underlying Buffer's user call back is called.
   * @param name the name of the Array.
   */
  void setName( std::string const & name )
  {
    m_dataBuffer.template setName< decltype(*this) >( name );
  }

  /**
   * @brief Move the Array to the given execution space, optionally touching it.
   * @param space the space to move the Array to.
   * @param touch whether the Array should be touched in the new space or not.
   * @note This particular method is for when T is not itself an Array.
   * @note Not all Buffers support memory movement.
   */
  template< typename U = T >
  std::enable_if_t< !isArray< U > >
  move( chai::ExecutionSpace const space, bool const touch=true )
  {
    m_dataBuffer.move( space, touch );
    setDataPtr();
  }

  /**
   * @brief Move the Array to the given execution space, optionally touching it.
   * @param space the space to move the Array to.
   * @param touch whether the Array should be touched in the new space or not.
   * @note This particular method is for when T is itself an Array, and it also
   *       moves the inner Arrays. Not sure if the touch gets propagated correctly.
   * @note Not all Buffers support memory movement.
   */
  template< typename U = T >
  std::enable_if_t< isArray< U > >
  move( chai::ExecutionSpace const space, bool const touch=true )
  {
    reinterpret_cast< BUFFER_TYPE< typename std::remove_const< typename T::ViewType >::type > & >( m_dataBuffer ).move( space, touch );
    setDataPtr();
  }

#if defined(USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__)
  /**
   * @brief Static function that will be used by Totalview to display the array contents.
   * @param av A pointer to the array that is being displayed.
   * @return 0 if everything went OK
   */
  static int TV_ttf_display_type( Array const * av )
  {
    return ParentType::TV_ttf_display_type( av );
  }
#endif

private:

  // Aliasing the protected members of ArrayView.
  using ParentType::m_dataBuffer;
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

    for( int i = 0 ; i < NDIM ; ++i )
    {
      foldedStrides[ i ] = 1;
      for( int j = i + 1 ; j < NDIM ; ++j )
      {
        foldedStrides[ i ] *= m_dims[ permutation[ j ] ];
      }
    }

    INDEX_TYPE * const strides = const_cast< INDEX_TYPE * >( m_strides );
    for( int i = 0 ; i < NDIM ; ++i )
    {
      strides[ permutation[ i ] ] = foldedStrides[ i ];
    }
  }

  /**
   * @brief Resize the default dimension of the Array.
   * @tparam ARGS variadic pack containing the types to initialize the new values with.
   * @param newDimLength the new size of the default dimension.
   * @param ARGS arguments to initialize the new values with.
   * @note This preserves the values in the Array.
   * @note The default dimension is given by m_singleParameterResizeIndex.
   */
  template< typename ... ARGS >
  void resizeDefaultDimension( INDEX_TYPE const newDimLength, ARGS && ... args )
  {
    // Get the current length and stride of the dimension as well as the size of the whole Array.
    INDEX_TYPE const curDimLength = m_dims[ m_singleParameterResizeIndex ];
    INDEX_TYPE const curDimStride = m_strides[ m_singleParameterResizeIndex ];
    INDEX_TYPE const curSize = size();

    // Set the size of the dimension, recalculate the strides and get the new total size.
    const_cast< INDEX_TYPE * >( m_dims )[ m_singleParameterResizeIndex ] = newDimLength;
    CalculateStrides();
    INDEX_TYPE const newSize = size();

    // If we aren't changing the total size then we can return early.
    if( newSize == curSize ) return;

    // If the size is increasing do one thing, if it's decreasing do another.
    if( newDimLength > curDimLength )
    {
      // Reserve space in the buffer but don't initialize the values.
      bufferManipulation::reserve( m_dataBuffer, curSize, newSize );
      setDataPtr();
      T * const ptr = data();

      // The resizing consists of iterations where each iteration consists of the addition of a
      // contiguous segment of new values.
      INDEX_TYPE const valuesToAddPerIteration = curDimStride * ( newDimLength - curDimLength );
      INDEX_TYPE const valuesToShiftPerIteration = curDimStride * curDimLength;
      INDEX_TYPE const numIterations = ( newSize - curSize ) / valuesToAddPerIteration;

      // Iterate backwards over the iterations.
      for( INDEX_TYPE i = numIterations - 1 ; i >= 0 ; --i )
      {
        // First shift the values up to make remove for the values of subsequent iterations.
        // This step is a no-op on the last iteration (i=0).
        INDEX_TYPE const valuesLeftToInsert = valuesToAddPerIteration * i;
        T * const startOfShift = ptr + valuesToShiftPerIteration * i;
        arrayManipulation::shiftUp( startOfShift, valuesToShiftPerIteration, INDEX_TYPE( 0 ), valuesLeftToInsert );

        // Initialize the new values.
        T * const startOfNewValues = startOfShift + valuesToShiftPerIteration + valuesLeftToInsert;
        for( INDEX_TYPE j = 0 ; j < valuesToAddPerIteration ; ++j )
        {
          new ( startOfNewValues + j ) T( std::forward< ARGS >( args )... );
        }
      }
    }
    else
    {
      T * const ptr = data();

      // The resizing consists of iterations where each iteration consists of the removal of a
      // contiguous segment of new values.
      INDEX_TYPE const valuesToRemovePerIteration = curDimStride * ( curDimLength - newDimLength );
      INDEX_TYPE const valuesToShiftPerIteration = curDimStride * newDimLength;
      INDEX_TYPE const numIterations = ( curSize - newSize ) / valuesToRemovePerIteration;

      // Iterate over the iterations, skipping the first.
      for( INDEX_TYPE i = 1 ; i < numIterations ; ++i )
      {
        INDEX_TYPE const amountToShift = valuesToRemovePerIteration * i;
        T * const startOfShift = ptr + valuesToShiftPerIteration * i;
        arrayManipulation::shiftDown( startOfShift, valuesToShiftPerIteration + amountToShift, amountToShift, amountToShift );
      }

      // After the iterations are complete all the values to remove have been moved to the end of the array.
      // We destroy them here.
      INDEX_TYPE const totalValuesToRemove = valuesToRemovePerIteration * numIterations;
      for( INDEX_TYPE i = 0 ; i < totalValuesToRemove ; ++i )
      {
        ptr[ newSize + i ].~T();
      }
    }
  }

};

} /* namespace LvArray */

#endif /* ARRAY_HPP_ */
