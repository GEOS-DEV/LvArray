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
#include "StackBuffer.hpp"

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
  static_assert( isValidPermutation( PERMUTATION {} ), "The permutation must be valid." );
  static_assert( getDimension( PERMUTATION {} ) == NDIM, "The dimension of the permutation must match the dimension of the Array." );
  static_assert( std::is_integral< INDEX_TYPE >::value, "INDEX_TYPE must be integral." );

  /// The dimensionality of the array.
  static constexpr int ndim = NDIM;

  /// The permutation of the array.
  using permutation = PERMUTATION;

  /// The dimension with unit stride.
  static constexpr int USD = getStrideOneDimension( PERMUTATION {} );

  /// Alias for the parent class.
  using ParentClass = ArrayView< T, NDIM, USD, INDEX_TYPE, BUFFER_TYPE >;

  // Aliasing public methods of ArrayView so we don't have to use this->size etc.
  using ParentClass::toSlice;
  using ParentClass::size;
  using ParentClass::empty;
  using ParentClass::data;
  using ParentClass::operator[];
  using ParentClass::operator();

  /// The type when all nested arrays are converted to const views to mutable values.
  using ViewType = ArrayView< typename GetViewType< T >::type, NDIM, USD, INDEX_TYPE, BUFFER_TYPE > const;

  /// The type when all nested array arrays are converted to views to const values.
  using ViewTypeConst = ArrayView< typename GetViewTypeConst< T >::type const, NDIM, USD, INDEX_TYPE, BUFFER_TYPE > const;

  /**
   * @brief default constructor
   */
  LVARRAY_HOST_DEVICE
  inline Array():
    ParentClass( true )
  {
    CalculateStrides();
#if !defined(__CUDA_ARCH__)
    setName( "" );
#endif
#if defined(USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__)
    Array::TV_ttf_display_type( nullptr );
#endif
  }

  /**
   * @brief Constructor that takes in the dimensions as a variadic parameter list
   * @param dims the dimensions of the array in form ( n0, n1,..., n(NDIM-1) )
   */
  template< typename ... DIMS,
            typename=std::enable_if_t< sizeof ... ( DIMS ) == NDIM &&
                                       all_of_t< std::is_integral< DIMS > ... >::value > >
  LVARRAY_HOST_DEVICE
  inline explicit Array( DIMS const ... dims ):
    Array()
  { resize( dims ... ); }

  /**
   * @brief Copy constructor.
   * @param source object to copy.
   * @note Performs a deep copy of source
   * @return *this.
   */
  LVARRAY_HOST_DEVICE
  Array( Array const & source ):
    Array()
  { *this = source; }

  /**
   * @brief Move constructor.
   * @param source object to move.
   * @note Moves the source into *this. This calls BUFFER_TYPE( BUFFER_TYPE && ) which usually is a
   *   shallow copy that invalidates the contents of source. However this depends on the
   *   implementation of BUFFER_TYPE.
   */
  Array( Array && source ) = default;

  /**
   * @brief Destructor, free's the data.
   */
  LVARRAY_HOST_DEVICE
  ~Array()
  {
  #if !defined(__CUDA_ARCH__)
    if( !std::is_trivially_destructible< T >::value )
    { this->move( MemorySpace::CPU ); }
  #endif

    bufferManipulation::free( m_dataBuffer, size() );
  }

  /**
   * @brief @return Return a reference to *this after converting any nested arrays to const views to mutable values.
   */
  LVARRAY_HOST_DEVICE
  ViewType & toView() const
  { return reinterpret_cast< ViewType & >(*this); }

  /**
   * @brief @return Return a reference to *this after converting any nested arrays to const views to const values.
   */
  LVARRAY_HOST_DEVICE
  ViewTypeConst & toViewConst() const
  { return reinterpret_cast< ViewTypeConst & >( *this ); }

  /**
   * @brief Copy assignment operator, performs a deep copy of rhs.
   * @param rhs Source for the assignment.
   * @return *this.
   */
  LVARRAY_HOST_DEVICE
  Array & operator=( Array const & rhs )
  {
    bufferManipulation::copyInto( m_dataBuffer, size(), rhs.m_dataBuffer, rhs.size() );

    for( int i = 0; i < NDIM; ++i )
    {
      m_dims[ i ] = rhs.m_dims[ i ];
      m_strides[ i ] = rhs.m_strides[ i ];
    }

    setSingleParameterResizeIndex( rhs.getSingleParameterResizeIndex() );
    return *this;
  }

  /**
   * @brief Move assignment operator, performs a shallow copy of rhs.
   * @param rhs Source for the assignment.
   * @return *this.
   */
  LVARRAY_HOST_DEVICE
  Array & operator=( Array && rhs )
  {
    ParentClass::operator=( std::move( rhs ) );
    return *this;
  }

  /**
   * @brief Assignment operator to assign the entire array to single value.
   * @param rhs value to set array.
   * @return *this.
   */
  Array & operator=( T const & rhs )
  {
    ParentClass::operator=( rhs );
    return *this;
  }

  /**
   * @brief Resize the dimensions of the Array to match the given dimensions.
   * @param numDims must equal NDIMS.
   * @param dims the new size of the dimensions, must be of length NDIM.
   * @note This does not preserve the values in the Array NDIM == 1.
   */
  template< typename DIMS_TYPE >
  LVARRAY_HOST_DEVICE
  void resize( int const numDims, DIMS_TYPE const * const dims )
  {
    LVARRAY_ERROR_IF_NE( numDims, NDIM );

    INDEX_TYPE const oldSize = size();
    for( int i = 0; i < NDIM; ++i )
    {
      m_dims[ i ] = LvArray::integerConversion< INDEX_TYPE >( dims[ i ] );
      LVARRAY_ERROR_IF_LT( m_dims[ i ], 0 );
    }

    CalculateStrides();

    bufferManipulation::resize( m_dataBuffer, oldSize, size() );
  }

  /**
   * @brief function to resize the array.
   * @tparam DIMS Variadic list of integral types.
   * @param newDims The new dimensions, must be of length NDIM.
   * @note This does not preserve the values in the Array unless NDIM == 1.
   * @return void.
   */
  template< typename ... DIMS >
  LVARRAY_HOST_DEVICE
  std::enable_if_t< sizeof ... ( DIMS ) == NDIM && all_of_t< std::is_integral< DIMS > ... >::value >
  resize( DIMS... newDims )
  {
    static_assert( sizeof ... ( DIMS ) == NDIM, "The number of arguments provided does not equal NDIM!" );
    INDEX_TYPE const oldSize = size();

    int curDim = 0;
    forEachArg( [&curDim, dims=m_dims]( auto const newDim )
    {
      dims[ curDim++ ] = LvArray::integerConversion< INDEX_TYPE >( newDim );
      LVARRAY_ERROR_IF_LT( dims[ curDim ], 0 );
    }, newDims ... );

    CalculateStrides();

    bufferManipulation::resize( m_dataBuffer, oldSize, size() );
  }

  /**
   * @brief Resize the array without initializing any new values or destroying any old values.
   *   Only safe on POD data, however it is much faster for large allocations.
   * @tparam DIMS Variadic list of integral types.
   * @param newDims The new dimensions, must be of length NDIM.
   * @note This does not preserve the values in the Array unless NDIM == 1.
   */
  template< typename ... DIMS >
  LVARRAY_HOST_DEVICE
  void resizeWithoutInitializationOrDestruction( DIMS const ... newDims )
  {
    static_assert( sizeof ... ( DIMS ) == NDIM, "The number of arguments provided does not equal NDIM!" );
    static_assert( std::is_trivially_destructible< T >::value, "This function is only safe if T is trivially destructable." );

    INDEX_TYPE const oldSize = size();

    int i = 0;
    forEachArg( [&i, dims=m_dims]( auto const newDim )
    {
      dims[ i++ ] = LvArray::integerConversion< INDEX_TYPE >( newDim );
      LVARRAY_ERROR_IF_LT( dims[ i ], 0 );
    }, newDims ... );

    CalculateStrides();

    bufferManipulation::reserve( m_dataBuffer, oldSize, size() );
  }

  /**
   * @brief Resize specific dimensions of the array.
   * @tparam INDICES the indices of the dimensions to resize, should be sorted an unique.
   * @tparam DIMS variadic pack containing the dimension types
   * @param newDims the new dimensions. newDims[ 0 ] will be the new size of
   *        dimensions INDICES[ 0 ].
   * @note This does not preserve the values in the Array unless NDIM == 1.
   */
  template< INDEX_TYPE... INDICES, typename ... DIMS >
  LVARRAY_HOST_DEVICE
  void resizeDimension( DIMS const ... newDims )
  {
    static_assert( sizeof ... (INDICES) <= NDIM, "Too many arguments provided." );
    static_assert( sizeof ... (INDICES) == sizeof ... (DIMS), "The number of indices must match the number of dimensions." );
    static_assert( all_of< ( 0 <= INDICES ) ... >::value, "INDICES must all be positive." );
    static_assert( all_of< ( INDICES < NDIM ) ... >::value, "INDICES must all be less than NDIM." );

    INDEX_TYPE const oldSize = size();

    forEachArg( [&]( auto const & pair )
    {
      m_dims[ camp::get< 0 >( pair ) ] = LvArray::integerConversion< INDEX_TYPE >( camp::get< 1 >( pair ) );
      LVARRAY_ERROR_IF_LT( m_dims[ camp::get< 0 >( pair ) ], 0 );
    }, camp::make_tuple( INDICES, newDims )... );

    CalculateStrides();

    bufferManipulation::resize( m_dataBuffer, oldSize, size() );
  }

  /**
   * @brief Resize the default dimension of the Array.
   * @param newdim the new size of the default dimension.
   * @note This preserves the values in the Array.
   * @note The default dimension is given by m_singleParameterResizeIndex.
   */
  LVARRAY_HOST_DEVICE
  void resize( INDEX_TYPE const newdim )
  { resizeDefaultDimension( newdim ); }

  /**
   * @brief Resize the default dimension of the Array.
   * @param newdim the new size of the default dimension.
   * @param defaultValue the value to initialize the new values with.
   * @note This preserves the values in the Array.
   * @note The default dimension is given by m_singleParameterResizeIndex.
   */
  LVARRAY_HOST_DEVICE
  void resizeDefault( INDEX_TYPE const newdim, T const & defaultValue = T() )
  { resizeDefaultDimension( newdim, defaultValue ); }

  /**
   * @brief Reserve space in the Array to hold at least the given number of values.
   * @param newCapacity the number of values to reserve space for. After this call
   *        capacity() >= newCapacity.
   */
  void reserve( INDEX_TYPE const newCapacity )
  { bufferManipulation::reserve( m_dataBuffer, size(), newCapacity ); }

  /**
   * @brief @return Return the maximum number of values the Array can hold without reallocation.
   */
  LVARRAY_HOST_DEVICE
  INDEX_TYPE capacity() const
  { return LvArray::integerConversion< INDEX_TYPE >( m_dataBuffer.capacity() ); }

  /**
   * @brief Sets the size of the Array to zero and destroys all the values.
   * @details Sets the size of m_singleParameterResizeIndex to 0 but leaves the
   *  size of the other dimensions untouched. Equivalent to resize( 0 ).
   */
  void clear()
  {
    bufferManipulation::resize( m_dataBuffer, size(), 0 );

    m_dims[ getSingleParameterResizeIndex() ] = 0;

    CalculateStrides();
  }

  /**
   * @brief Construct a value in place at the end of the array.
   * @tparam ARGS A variadic pack of the types to construct the new value from.
   * @param args A variadic pack of values to construct the new value from.
   * @note This method is only available on 1D arrays.
   * @return void.
   */
  template< typename ... ARGS, int _NDIM=NDIM >
  std::enable_if_t< _NDIM == 1 >
  emplace_back( ARGS && ... args )
  {
    bufferManipulation::emplaceBack( m_dataBuffer, size(), std::forward< ARGS >( args ) ... );
    ++m_dims[ 0 ];
  }

  /**
   * @brief Insert a value into the array constructing it in place.
   * @tparam ARGS A variadic pack of the types to construct the new value from.
   * @param pos the position to insert at.
   * @param args A variadic pack of values to construct the new value from.
   * @note This method is only available on 1D arrays.
   * @return void.
   */
  template< typename ... ARGS, int _NDIM=NDIM >
  std::enable_if_t< _NDIM == 1 >
  emplace( INDEX_TYPE const pos, ARGS && ... args )
  {
    bufferManipulation::emplace( m_dataBuffer, size(), pos, std::forward< ARGS >( args ) ... );
    ++m_dims[ 0 ];
  }

  /**
   * @brief Insert values into the array at the given position.
   * @tparam ITER An iterator type, they type of @p first and @p last.
   * @param pos The position to insert at.
   * @param first An iterator to the first value to insert.
   * @param last An iterator to the last value to insert.
   * @note This method is only available on 1D arrays.
   * @return None.
   */
  template< typename ITER, int _NDIM=NDIM >
  std::enable_if_t< _NDIM == 1 >
  insert( INDEX_TYPE const pos, ITER const first, ITER const last )
  { m_dims[ 0 ] += bufferManipulation::insert( m_dataBuffer, size(), pos, first, last ); }

  /**
   * @brief Remove the last value in the array.
   * @note This method is only available on 1D arrays.
   * @return void.
   */
  template< int _NDIM=NDIM >
  std::enable_if_t< _NDIM == 1 >
  pop_back()
  {
    bufferManipulation::popBack( m_dataBuffer, size() );
    --m_dims[ 0 ];
  }

  /**
   * @brief Remove the value at the given position.
   * @param pos the position of the value to remove.
   * @note This method is only available on 1D arrays.
   * @return void.
   */
  template< int _NDIM=NDIM >
  std::enable_if_t< _NDIM == 1 >
  erase( INDEX_TYPE const pos )
  {
    bufferManipulation::erase( m_dataBuffer, size(), pos );
    --m_dims[ 0 ];
  }

  /**
   * @brief @return Return the default resize dimension.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  int getSingleParameterResizeIndex() const
  { return m_singleParameterResizeIndex; }

  /**
   * @brief Set the default resize dimension.
   * @param index The new default dimension.
   */
  LVARRAY_HOST_DEVICE
  inline void setSingleParameterResizeIndex( int const index )
  {
    LVARRAY_ERROR_IF_LT( index, 0 );
    LVARRAY_ERROR_IF_GE( index, NDIM );
    m_singleParameterResizeIndex = index;
  }

  /**
   * @brief Set the name to be displayed whenever the underlying Buffer's user call back is called.
   * @param name the name of the Array.
   */
  void setName( std::string const & name )
  { m_dataBuffer.template setName< decltype(*this) >( name ); }

#if defined(USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__)
  /**
   * @brief Static function that will be used by Totalview to display the array contents.
   * @param av A pointer to the array that is being displayed.
   * @return 0 if everything went OK
   */
  static int TV_ttf_display_type( Array const * av )
  {
    return ParentClass::TV_ttf_display_type( av );
  }
#endif

private:

  // Aliasing the protected members of ArrayView.
  using ParentClass::m_dataBuffer;
  using ParentClass::m_dims;
  using ParentClass::m_strides;
  using ParentClass::m_singleParameterResizeIndex;

  /**
   * @brief Calculate the strides given the dimensions and permutation.
   * @note Adapted from RAJA::make_permuted_layout.
   */
  LVARRAY_HOST_DEVICE
  void CalculateStrides()
  {
    constexpr CArray< camp::idx_t, NDIM > perm = asArray( PERMUTATION {} );
    INDEX_TYPE foldedStrides[ NDIM ];

    for( int i = 0; i < NDIM; ++i )
    {
      foldedStrides[ i ] = 1;
      for( int j = i + 1; j < NDIM; ++j )
      {
        foldedStrides[ i ] *= m_dims[ perm[ j ] ];
      }
    }

    for( int i = 0; i < NDIM; ++i )
    {
      m_strides[ perm[ i ] ] = foldedStrides[ i ];
    }
  }

  /**
   * @brief Resize the default dimension of the Array.
   * @tparam ARGS variadic pack containing the types to initialize the new values with.
   * @param newDimLength the new size of the default dimension.
   * @param args arguments to initialize the new values with.
   * @note This preserves the values in the Array.
   * @note The default dimension is given by m_singleParameterResizeIndex.
   */
  DISABLE_HD_WARNING
  template< typename ... ARGS >
  LVARRAY_HOST_DEVICE
  void resizeDefaultDimension( INDEX_TYPE const newDimLength, ARGS && ... args )
  {
    LVARRAY_ERROR_IF_LT( newDimLength, 0 );

    // If m_singleParameterResizeIndex is the first dimension in memory than a simple 1D resizing is sufficient. The
    // check if NDIM == 1 is to give the compiler compile time knowledge that this path is always taken for 1D arrays.
    if( NDIM == 1 || asArray( PERMUTATION {} )[ 0 ] == m_singleParameterResizeIndex )
    {
      INDEX_TYPE const oldSize = size();
      m_dims[ m_singleParameterResizeIndex ] = newDimLength;
      CalculateStrides();
      bufferManipulation::resize( m_dataBuffer, oldSize, size(), std::forward< ARGS >( args )... );
      return;
    }

    // Get the current length and stride of the dimension as well as the size of the whole Array.
    INDEX_TYPE const curDimLength = m_dims[ m_singleParameterResizeIndex ];
    INDEX_TYPE const curDimStride = m_strides[ m_singleParameterResizeIndex ];
    INDEX_TYPE const curSize = size();

    // Set the size of the dimension, recalculate the strides and get the new total size.
    m_dims[ m_singleParameterResizeIndex ] = newDimLength;
    CalculateStrides();
    INDEX_TYPE const newSize = size();

    // If we aren't changing the total size then we can return early.
    if( newSize == curSize ) return;

    // If the size is increasing do one thing, if it's decreasing do another.
    if( newDimLength > curDimLength )
    {
      // Reserve space in the buffer but don't initialize the values.
      bufferManipulation::reserve( m_dataBuffer, curSize, newSize );
      T * const ptr = data();

      // The resizing consists of iterations where each iteration consists of the addition of a
      // contiguous segment of new values.
      INDEX_TYPE const valuesToAddPerIteration = curDimStride * ( newDimLength - curDimLength );
      INDEX_TYPE const valuesToShiftPerIteration = curDimStride * curDimLength;
      INDEX_TYPE const numIterations = ( newSize - curSize ) / valuesToAddPerIteration;

      // Iterate backwards over the iterations.
      for( INDEX_TYPE i = numIterations - 1; i >= 0; --i )
      {
        // First shift the values up to make remove for the values of subsequent iterations.
        // This step is a no-op on the last iteration (i=0).
        INDEX_TYPE const valuesLeftToInsert = valuesToAddPerIteration * i;
        T * const startOfShift = ptr + valuesToShiftPerIteration * i;
        arrayManipulation::shiftUp( startOfShift, valuesToShiftPerIteration, INDEX_TYPE( 0 ), valuesLeftToInsert );

        // Initialize the new values.
        T * const startOfNewValues = startOfShift + valuesToShiftPerIteration + valuesLeftToInsert;
        for( INDEX_TYPE j = 0; j < valuesToAddPerIteration; ++j )
        {
          new ( startOfNewValues + j ) T( args ... );
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
      for( INDEX_TYPE i = 1; i < numIterations; ++i )
      {
        INDEX_TYPE const amountToShift = valuesToRemovePerIteration * i;
        T * const startOfShift = ptr + valuesToShiftPerIteration * i;
        arrayManipulation::shiftDown( startOfShift, valuesToShiftPerIteration + amountToShift, amountToShift, amountToShift );
      }

      // After the iterations are complete all the values to remove have been moved to the end of the array.
      // We destroy them here.
      INDEX_TYPE const totalValuesToRemove = valuesToRemovePerIteration * numIterations;
      for( INDEX_TYPE i = 0; i < totalValuesToRemove; ++i )
      {
        ptr[ newSize + i ].~T();
      }
    }
  }
};

/**
 * @brief True if the template type is an Array.
 */
template< typename >
constexpr bool isArray = false;

/**
 * @tparam T The type contained in the Array.
 * @tparam NDIM The number of dimensions in the Array.
 * @tparam PERMUTATION The way that the data is layed out in memory.
 * @tparam INDEX_TYPE The integral type used as an index.
 * @tparam BUFFER_TYPE The type used to manage the underlying allocation.
 * @brief Specialization of isArray for the Array class.
 */
template< typename T,
          int NDIM,
          typename PERMUTATION,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
constexpr bool isArray< Array< T, NDIM, PERMUTATION, INDEX_TYPE, BUFFER_TYPE > > = true;

namespace internal
{

// Since Array expects the BUFFER to only except a single template parameter we need to
// create an alias for a StackBuffer with a given length.
/**
 * @struct StackArrayHelper
 * @tparam T The type stored in the Array.
 * @tparam NDIM The number of dimensions in the Array.
 * @tparam PERMUTATION The dimension and permutation of the Array.
 * @tparam INDEX_TYPE The integer used to index the Array.
 * @tparam LENGTH The capacity of the underlying StackBuffer.
 */
template< typename T,
          int NDIM,
          typename PERMUTATION,
          typename INDEX_TYPE,
          int LENGTH >
struct StackArrayHelper
{
  /**
   * @brief An alias for a StackBuffer with the given length.
   * @tparam U The type contained in the StackBuffer.
   */
  template< typename U >
  using BufferType = StackBuffer< U, LENGTH >;

  /// An alias for the Array type.
  using type = Array< T, NDIM, PERMUTATION, INDEX_TYPE, BufferType >;
};

} // namespace internal

/**
 * @tparam T The type of the values stored in the Array.
 * @tparam NDIM The number of dimensions in the Array.
 * @tparam PERMUTATION The layout of the data in memory.
 * @tparam INDEX_TYPE The integer used for indexing.
 * @tparam LENGTH The capacity of the backing c-array.
 * @brief An alias for a Array backed by a StackBuffer.
 */
template< typename T,
          int NDIM,
          typename PERMUTATION,
          typename INDEX_TYPE,
          int LENGTH >
using StackArray = typename internal::StackArrayHelper< T, NDIM, PERMUTATION, INDEX_TYPE, LENGTH >::type;

} /* namespace LvArray */

#endif /* ARRAY_HPP_ */
