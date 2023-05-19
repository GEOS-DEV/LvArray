/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file Array.hpp
 * @brief Contains the implementation of LvArray::Array.
 */

#pragma once

// Source includes
#include "indexing.hpp"
#include "ArrayView.hpp"
#include "bufferManipulation.hpp"
#include "StackBuffer.hpp"

/**
 * @brief The top level namespace.
 */
namespace LvArray
{

/**
 * @class Array
 * @brief This class provides a fixed dimensional resizeable array interface in addition to an
 *   interface similar to std::vector for a 1D array.
 * @tparam T The type of data that is contained by the array.
 * @tparam NDIM The number of dimensions in array (e.g. NDIM=1->vector, NDIM=2->Matrix, etc. ).
 *   Some methods such as push_back are only available for a 1D array.
 * @tparam PERMUTATION A camp::idx_seq containing the values in [0, NDIM) which describes how the
 *   data is to be laid out in memory. Given an 3-dimensional array A of dimension (L, M, N)
 *   the standard layout is A( i, j, k ) = A.data()[ i * M * N + j * N + k ], this layout is
 *   the given by the identity permutation (0, 1, 2) which says that the first dimension in
 *   memory corresponds to the first index (i), and similarly for the other dimensions. The
 *   same array with a layout of A( i, j, k ) = A.data()[ L * N * j + L * k + i ] would have
 *   a permutation of (1, 2, 0) because the first dimension in memory corresponds to the second
 *   index (j), the second dimension in memory corresponds to the third index (k) and the last
 *   dimension in memory goes with the first index (i).
 * @note RAJA provides aliases for every valid permutations up to 5D, the two permutations
 *    mentioned above can be used via RAJA::PERM_IJK and RAJA::PERM_JKI.
 * @note The dimension with unit stride is the last dimension in the permutation.
 * @tparam INDEX_TYPE the integer to use for indexing.
 * @tparam BUFFER_TYPE A class that defines how to actually allocate memory for the Array. Must take
 *   one template argument that describes the type of the data being stored (T).
 */
template< typename T,
          typename EXTENT,
          typename PERMUTATION,
          template< typename > class BUFFER_TYPE >
class Array : public ArrayView< T, Layout< EXTENT, CompactStride< EXTENT, LayoutDefault, PERMUTATION > >, BUFFER_TYPE >
{
public:

  /// Alias for the permutation.
  using Permutation = PERMUTATION;

  /// Alias for the parent class.
  using ParentClass = ArrayView< T, Layout< EXTENT, CompactStride< EXTENT, LayoutDefault, PERMUTATION > >, BUFFER_TYPE >;

  using typename ParentClass::LayoutType;
  using typename ParentClass::ExtentType;
  using typename ParentClass::StrideType;
  using typename ParentClass::IndexType;
  using typename ParentClass::NestedViewType;
  using typename ParentClass::NestedViewTypeConst;

  using ParentClass::NDIM;
  using ParentClass::USD;

  /// Index of dimension for single-parameter resize. TODO: consider making this a template parameter?
  static constexpr int SingleResizeDim = 0;

  /// Whether array can be resized/cleared
  static constexpr bool IsResizable = !is_constant_v< camp::tuple_element_t< SingleResizeDim, ExtentType > >;

  /**
   * @name Constructors, destructor and assignment operators.
   */
  ///@{

  /**
   * @brief default constructor
   */
  LVARRAY_HOST_DEVICE
  Array():
    ParentClass( true )
  {
#if !defined(LVARRAY_DEVICE_COMPILE)
    setName( "" );
#endif
#if defined(LVARRAY_USE_TOTALVIEW_OUTPUT) && !defined(LVARRAY_DEVICE_COMPILE)
    Array::TV_ttf_display_type( nullptr );
#endif
  }

  /**
   * @brief Constructor that takes in the dimensions as a variadic parameter list
   * @param dims the dimensions of the array in form ( n0, n1,..., n(NDIM-1) )
   */
  template< typename ... DIMS,
            typename=std::enable_if_t< sizeof ... ( DIMS ) == NDIM &&
                                       typeManipulation::all_of_t< std::is_integral< DIMS > ... >::value > >
  LVARRAY_HOST_DEVICE
  inline explicit Array( DIMS const ... dims ):
    Array()
  { resize( dims ... ); }

  /**
   * @brief Construct an Array from @p buffer, taking ownership of its contents.
   * @param buffer The buffer to construct the Array from.
   * @note The Array is empty and @p buffer is expected to be empty as well.
   */
  Array( BUFFER_TYPE< T > && buffer ):
    ParentClass( std::move( buffer ) )
  {
#if !defined(LVARRAY_DEVICE_COMPILE)
    setName( "" );
#endif
#if defined(LVARRAY_USE_TOTALVIEW_OUTPUT) && !defined(LVARRAY_DEVICE_COMPILE)
    Array::TV_ttf_display_type( nullptr );
#endif
  }

  /**
   * @brief Copy constructor.
   * @param source object to copy.
   * @note Performs a deep copy of source
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
  LVARRAY_HOST_DEVICE
  Array( Array && source ):
    ParentClass( std::move( source ) )
  {
    // This resets to 0 all non-constant extents/strides
    source.m_layout.extent() = ExtentType{};
    source.m_layout.stride() = StrideType{};
  }

  /**
   * @brief Destructor, free's the data.
   */
  LVARRAY_HOST_DEVICE
  ~Array()
  { bufferManipulation::free( this->m_dataBuffer, this->size() ); }

  /**
   * @brief Copy assignment operator, performs a deep copy of rhs.
   * @param rhs Source for the assignment.
   * @return *this.
   */
  LVARRAY_HOST_DEVICE
  Array & operator=( Array const & rhs )
  {
    bufferManipulation::copyInto( this->m_dataBuffer, this->size(), rhs.m_dataBuffer, rhs.size() );
    this->m_layout = rhs.m_layout;
    return *this;
  }

  /**
   * @brief Copy assignment operator, performs a deep copy of rhs.
   * @param rhs Source for the assignment.
   * @return *this.
   */
  LVARRAY_HOST_DEVICE
  Array & operator=( typename ParentClass::ViewTypeConst const & rhs )
  {
    bufferManipulation::copyInto( this->m_dataBuffer, this->size(), rhs.dataBuffer(), rhs.size() );
    this->m_layout = rhs.m_layout;
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
    bufferManipulation::free( this->m_dataBuffer, this->size() );
    ParentClass::operator=( std::move( rhs ) );

    // Reset all dynamic extents/strides to 0
    rhs.m_layout.extent() = ExtentType{};
    rhs.m_layout.stride() = StrideType{};

    return *this;
  }

  ///@}

  /**
   * @name ArrayView and ArraySlice creation methods and user defined conversions.
   */
  ///@{

  using ParentClass::toView;

  /**
   * @brief Overload for rvalues that is deleted.
   * @return None.
   * @note This cannot be called on a rvalue since the @c ArrayView would
   *   contain the buffer of the current @c Array that is about to be destroyed.
   *   This overload prevents that from happening.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  ArrayView< T, LayoutType, BUFFER_TYPE > toView() const && = delete;

  using ParentClass::toViewConst;

  /**
   * @brief Overload for rvalues that is deleted.
   * @return None.
   * @note This cannot be called on a rvalue since the @c ArrayView would
   *   contain the buffer of the current @c Array that is about to be destroyed.
   *   This overload prevents that from happening.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  ArrayView< T const, LayoutType, BUFFER_TYPE > toViewConst() const && = delete;

  using ParentClass::toNestedView;

  /**
   * @brief Overload for rvalues that is deleted.
   * @return None.
   * @note This cannot be called on a rvalue since the @c ArrayView would
   *   contain the buffer of the current @c Array that is about to be destroyed.
   *   This overload prevents that from happening.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  NestedViewType toNestedView() const && = delete;

  using ParentClass::toNestedViewConst;

  /**
   * @brief Overload for rvalues that is deleted.
   * @return None.
   * @note This cannot be called on a rvalue since the @c ArrayView would
   *   contain the buffer of the current @c Array that is about to be destroyed.
   *   This overload prevents that from happening.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  NestedViewTypeConst toNestedViewConst() const && = delete;

  /**
   * @brief A user defined conversion operator (UDC) to an ArrayView< T const, ... >.
   * @return A new ArrayView where @c T is @c const.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  operator ArrayView< T const, LayoutType, BUFFER_TYPE >() const & noexcept
  { return toViewConst(); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return None.
   * @note This cannot be called on a rvalue since the @c ArrayView would
   *   contain the buffer of the current @c Array that is about to be destroyed.
   *   This overload prevents that from happening.
   */
  template< typename T_=T >
  inline LVARRAY_HOST_DEVICE constexpr
  operator std::enable_if_t< !std::is_const< T_ >::value,
                             ArrayView< T const, LayoutType, BUFFER_TYPE > >() const && noexcept = delete;

  ///@}

  /**
   * @name Resizing methods.
   */
  ///@{

public:

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
    IndexType const oldSize = this->size();
    this->m_layout.resize( dims, LayoutDefault{}, Permutation{} );
    bufferManipulation::resize( this->m_dataBuffer, oldSize, this->size() );
  }

  /**
   * @brief function to resize the array.
   * @tparam DIMS Variadic list of integral types.
   * @param newExtent The new extent
   * @note This does not preserve the values in the Array unless NDIM == 1.
   * @return void.
   */
  template< typename ... DIMS >
  LVARRAY_HOST_DEVICE
  void
  resize( Extent< DIMS ... > const & newExtent )
  {
    IndexType const oldSize = this->size();
    this->m_layout.resize( newExtent, LayoutDefault{}, Permutation{} );
    bufferManipulation::resize( this->m_dataBuffer, oldSize, this->size() );
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
  std::enable_if_t< sizeof...( DIMS ) == NDIM && typeManipulation::all_of< is_integral_v< DIMS > ... >::value >
  resize( DIMS const ... newDims )
  {
    resize( makeExtent( newDims... ) );
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
    return resizeWithoutInitializationOrDestruction( MemorySpace::host, newDims ... );
  }

  /**
   * @brief Resize the array without initializing any new values or destroying any old values.
   *   Only safe on POD data, however it is much faster for large allocations.
   * @tparam DIMS Variadic list of integral types.
   * @param space The space to perform the resize in.
   * @param newDims The new dimensions, must be of length NDIM.
   * @note This does not preserve the values in the Array unless NDIM == 1.
   */
  template< typename ... DIMS >
  LVARRAY_HOST_DEVICE
  void resizeWithoutInitializationOrDestruction( MemorySpace const space, DIMS const ... newDims )
  {
    static_assert( std::is_trivially_destructible< T >::value, "This function is only safe if T is trivially destructible." );

    IndexType const oldSize = this->size();
    this->m_layout.resize( makeExtent( newDims... ), LayoutDefault{}, Permutation{} );
    bufferManipulation::reserve( this->m_dataBuffer, oldSize, space, this->size() );
  }

  /**
   * @brief Resize specific dimensions of the array.
   * @tparam INDICES the indices of the dimensions to resize, should be sorted an unique.
   * @tparam DIMS variadic pack containing the dimension types
   * @param newDims the new dimensions. newDims[ 0 ] will be the new size of
   *        dimensions INDICES[ 0 ].
   * @note This does not preserve the values in the Array unless NDIM == 1.
   */
  template< int ... INDICES, typename ... DIMS >
  LVARRAY_HOST_DEVICE
  void resizeDimension( DIMS const ... newDims )
  {
    IndexType const oldSize = this->size();
    this->m_layout.template resize< INDICES... >( makeExtent( newDims... ), LayoutDefault{}, Permutation{} );
    bufferManipulation::resize( this->m_dataBuffer, oldSize, this->size() );
  }

  /**
   * @brief Resize the default dimension of the Array.
   * @param newdim the new size of the default dimension.
   * @note This preserves the values in the Array.
   * @note The default dimension is 0.
   */
  LVARRAY_HOST_DEVICE
  void resize( IndexType const newdim )
  { resizeDefaultDimension( newdim ); }

  /**
   * @brief Resize the default dimension of the Array.
   * @param newdim the new size of the default dimension.
   * @param defaultValue the value to initialize the new values with.
   * @note This preserves the values in the Array.
   * @note The default dimension is 0.
   */
  LVARRAY_HOST_DEVICE
  void resizeDefault( IndexType const newdim, T const & defaultValue )
  { resizeDefaultDimension( newdim, defaultValue ); }

  /**
   * @brief Sets the size of the Array to zero and destroys all the values.
   * @details Sets the size of 0th dimension to 0 but leaves the
   *  size of the other dimensions untouched. Equivalent to resize( 0 ).
   */
  void clear()
  {
    static_assert( IsResizable, "Array must have a dynamic leftmost extent to be resized." );
    bufferManipulation::resize( this->m_dataBuffer, this->size(), 0 );
    this->m_layout.template resize< SingleResizeDim >( makeExtent( _0{} ), LayoutDefault{}, Permutation{} );
  }

  ///@}

  /**
   * @name Methods to modify the capacity.
   */
  ///@{

  /**
   * @brief Reserve space in the Array to hold at least the given number of values.
   * @param newCapacity the number of values to reserve space for. After this call
   *        capacity() >= newCapacity.
   */
  void reserve( IndexType const newCapacity )
  { bufferManipulation::reserve( this->m_dataBuffer, this->size(), MemorySpace::host, newCapacity ); }

  ///@}

  /**
   * @name One dimensional interface.
   * @note These methods are only enabled for one dimensional arrays (NDIM == 1).
   */
  ///@{

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
    static_assert( IsResizable, "Array must have a dynamic leftmost extent to be resized." );
    bufferManipulation::emplaceBack( this->m_dataBuffer, this->size(), std::forward< ARGS >( args ) ... );
    ++camp::get< SingleResizeDim >( this->m_layout.extent() );
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
  emplace( IndexType const pos, ARGS && ... args )
  {
    static_assert( IsResizable, "Array must have a dynamic leftmost extent to be resized." );
    bufferManipulation::emplace( this->m_dataBuffer, this->size(), pos, std::forward< ARGS >( args ) ... );
    ++camp::get< SingleResizeDim >( this->m_layout.extent() );
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
  insert( IndexType const pos, ITER const first, ITER const last )
  {
    static_assert( IsResizable, "Array must have a dynamic leftmost extent to be resized." );
    camp::get< SingleResizeDim >( this->m_layout.extent() ) += bufferManipulation::insert( this->m_dataBuffer, this->size(), pos, first, last );
  }

  /**
   * @brief Remove the last value in the array.
   * @note This method is only available on 1D arrays.
   * @return void.
   */
  template< int _NDIM=NDIM >
  std::enable_if_t< _NDIM == 1 >
  pop_back()
  {
    static_assert( IsResizable, "Array must have a dynamic leftmost extent to be resized." );
    bufferManipulation::popBack( this->m_dataBuffer, this->size() );
    --camp::get< SingleResizeDim >( this->m_layout.extent() );
  }

  /**
   * @brief Remove the value at the given position.
   * @param pos the position of the value to remove.
   * @note This method is only available on 1D arrays.
   * @return void.
   */
  template< int _NDIM=NDIM >
  std::enable_if_t< _NDIM == 1 >
  erase( IndexType const pos )
  {
    static_assert( IsResizable, "Array must have a dynamic leftmost extent to be resized." );
    bufferManipulation::erase( this->m_dataBuffer, this->size(), pos );
    --camp::get< SingleResizeDim >( this->m_layout.extent() );
  }

  ///@}

  /**
   * @brief Set the name to be displayed whenever the underlying Buffer's user call back is called.
   * @param name the name of the Array.
   */
  void setName( std::string const & name )
  { this->m_dataBuffer.template setName< decltype(*this) >( name ); }

#if defined(LVARRAY_USE_TOTALVIEW_OUTPUT) && !defined(LVARRAY_DEVICE_COMPILE)
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

  /**
   * @brief Resize the default dimension of the Array.
   * @tparam ARGS variadic pack containing the types to initialize the new values with.
   * @param newDimLength the new size of the default dimension.
   * @param args arguments to initialize the new values with.
   * @note This preserves the values in the Array.
   * @note The default dimension is 0.
   */
  DISABLE_HD_WARNING
  template< typename ... ARGS >
  LVARRAY_HOST_DEVICE
  void resizeDefaultDimension( IndexType const newDimLength, ARGS && ... args )
  {
    static_assert( IsResizable, "Array must have a dynamic leftmost extent to be resized." );
    LVARRAY_ERROR_IF_LT( newDimLength, 0 );

    // If SingleResizeDim is the first dimension in memory than a simple 1D resizing is sufficient. The
    // check if NDIM == 1 is to give the compiler compile time knowledge that this path is always taken for 1D arrays.
    if constexpr( NDIM == 1 || typeManipulation::asArray( Permutation{} )[ 0 ] == SingleResizeDim )
    {
      IndexType const oldSize = this->size();
      this->m_layout.template resize< SingleResizeDim >( makeExtent( newDimLength ), LayoutDefault{}, Permutation{} );
      bufferManipulation::resize( this->m_dataBuffer, oldSize, this->size(), std::forward< ARGS >( args )... );
      return;
    }

    // Get the current length and stride of the dimension as well as the size of the whole Array.
    IndexType const curDimLength = camp::get< SingleResizeDim >( this->m_layout.extent() );
    IndexType const curDimStride = camp::get< SingleResizeDim >( this->m_layout.stride() );
    IndexType const curSize = this->size();

    // Set the size of the dimension, recalculate the strides and get the new total size.
    this->m_layout.template resize< SingleResizeDim >( makeExtent( newDimLength ), LayoutDefault{}, Permutation{} );
    IndexType const newSize = this->size();

    // If we aren't changing the total size then we can return early.
    if( newSize == curSize ) return;

    // If the size is increasing do one thing, if it's decreasing do another.
    if( newDimLength > curDimLength )
    {
      // Reserve space in the buffer but don't initialize the values.
      bufferManipulation::reserve( this->m_dataBuffer, curSize, MemorySpace::host, newSize );
      T * const ptr = this->data();

      // The resizing consists of iterations where each iteration consists of the addition of a
      // contiguous segment of new values.
      IndexType const valuesToAddPerIteration = curDimStride * ( newDimLength - curDimLength );
      IndexType const valuesToShiftPerIteration = curDimStride * curDimLength;
      IndexType const numIterations = ( newSize - curSize ) / valuesToAddPerIteration;

      // Iterate backwards over the iterations.
      for( IndexType i = numIterations - 1; i >= 0; --i )
      {
        // First shift the values up to make remove for the values of subsequent iterations.
        // This step is a no-op on the last iteration (i=0).
        IndexType const valuesLeftToInsert = valuesToAddPerIteration * i;
        T * const startOfShift = ptr + valuesToShiftPerIteration * i;
        arrayManipulation::shiftUp( startOfShift, valuesToShiftPerIteration, IndexType( 0 ), valuesLeftToInsert );

        // Initialize the new values.
        T * const startOfNewValues = startOfShift + valuesToShiftPerIteration + valuesLeftToInsert;
        for( IndexType j = 0; j < valuesToAddPerIteration; ++j )
        {
          new ( startOfNewValues + j ) T( args ... );
        }
      }
    }
    else
    {
      T * const ptr = this->data();

      // The resizing consists of iterations where each iteration consists of the removal of a
      // contiguous segment of new values.
      IndexType const valuesToRemovePerIteration = curDimStride * ( curDimLength - newDimLength );
      IndexType const valuesToShiftPerIteration = curDimStride * newDimLength;
      IndexType const numIterations = ( curSize - newSize ) / valuesToRemovePerIteration;

      // Iterate over the iterations, skipping the first.
      for( IndexType i = 1; i < numIterations; ++i )
      {
        IndexType const amountToShift = valuesToRemovePerIteration * i;
        T * const startOfShift = ptr + valuesToShiftPerIteration * i;
        arrayManipulation::shiftDown( startOfShift,
                                      valuesToShiftPerIteration + amountToShift,
                                      amountToShift, amountToShift );
      }

      // After the iterations are complete all the values to remove have been moved to the end of the array.
      // We destroy them here.
      IndexType const totalValuesToRemove = valuesToRemovePerIteration * numIterations;
      for( IndexType i = 0; i < totalValuesToRemove; ++i )
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
          typename EXTENT,
          typename PERMUTATION,
          template< typename > class BUFFER_TYPE >
constexpr bool isArray< Array< T, EXTENT, PERMUTATION, BUFFER_TYPE > > = true;

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
          typename EXTENT,
          typename PERMUTATION,
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
  using type = Array< T, EXTENT, PERMUTATION, BufferType >;
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
          typename EXTENT,
          typename PERMUTATION,
          int LENGTH >
using StackArray = typename internal::StackArrayHelper< T, EXTENT, PERMUTATION, LENGTH >::type;

} /* namespace LvArray */
