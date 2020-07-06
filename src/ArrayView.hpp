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
 * @file ArrayView.hpp
 */

#ifndef ARRAYVIEW_HPP_
#define ARRAYVIEW_HPP_

/// Source includes
#include "Permutation.hpp"
#include "ArraySlice.hpp"
#include "Macros.hpp"
#include "arrayHelpers.hpp"
#include "IntegerConversion.hpp"
#include "sliceHelpers.hpp"
#include "bufferManipulation.hpp"

/// System includes
#if defined(USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__)
#include "totalview/tv_helpers.hpp"
#include "totalview/tv_data_display.h"
#endif

namespace LvArray
{

/**
 * @class ArrayView
 * @brief This class serves to provide a "view" of a multidimensional array.
 * @tparam T type of data that is contained by the array
 * @tparam NDIM number of dimensions in array (e.g. NDIM=1->vector, NDIM=2->Matrix, etc. ).
 * @tparam USD the dimension with a unit stride, in an Array with a standard layout
 *         this is the last dimension.
 * @tparam INDEX_TYPE the integer to use for indexing.
 * @tparam BUFFER_TYPE A class that defines how to actually allocate memory for the Array. Must take
 *         one template argument that describes the type of the data being stored (T).
 *
 * When using CHAI the copy copy constructor of this class calls the copy constructor for CHAI
 * which will move the data to the location of the touch (host or device). In general, the
 * ArrayView should be what is passed by value into a lambda that is used to launch kernels as it
 * copy will trigger the desired data motion onto the appropriate memory space.
 *
 * Key features:
 * 1) When using a NewChaiBuffer as the BUFFER_TYPE the ArrayView copy constructor will move the data
 *    to the current execution space.
 * 2) Defines a slicing operator[].
 * 3) Defines operator() array accessor.
 * 3) operator[] and operator() are all const and may be called in non-mutable lambdas.
 * 4) Conversion operators to go from ArrayView<T> to ArrayView<T const>.
 * 5) Since the Array is derived from ArrayView, it may be upcasted:
 *      Array<T,NDIM> array;
 *      ArrayView<T,NDIM> const & arrView = array;
 *
 * A good guideline is to upcast to an ArrayView when you don't need allocation capabilities that
 * are only present in Array.
 */
template< typename T,
          int NDIM,
          int USD,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class ArrayView
{
public:

  static_assert( USD >= 0, "USD must be positive." );
  static_assert( USD < NDIM, "USD must be less than NDIM." );

  /// The number of dimensions.
  static constexpr int ndim = NDIM;

  /// The type when all inner array classes are converted to const views.
  using ViewType = ArrayView< typename GetViewType< T >::type, NDIM, USD, INDEX_TYPE, BUFFER_TYPE > const;

  /// The type when all inner array classes are converted to const views and the inner most view's values are also
  /// const.
  using ViewTypeConst = ArrayView< typename GetViewTypeConst< T >::type const, NDIM, USD, INDEX_TYPE, BUFFER_TYPE > const;

  /// The type of the ArrayView when converted to an ArraySlice.
  using SliceType = ArraySlice< T, NDIM, USD, INDEX_TYPE > const;

  /// The type of the ArrayView when converted to an immutable ArraySlice.
  using SliceTypeConst = ArraySlice< T const, NDIM, USD, INDEX_TYPE > const;

  /// The type of the value in the ArraySlice.
  using value_type = T;

  /// The type of the iterator used.
  using iterator = T *;

  /// The type of the const iterator used.
  using const_iterator = T const *;

  /**
   * @brief A constructor to create an uninitialized ArrayView.
   * @note An uninitialized ArrayView should not be used until it is assigned to.
   */
  ArrayView() = default;

  /**
   * @brief Copy Constructor.
   * @param source The object to copy.
   * @note The copy constructor will trigger the copy constructor for @tparam BUFFER_TYPE. When using the
   *   NewChaiBuffer this can move the underlying buffer to a new memory space if the execution context is set.
   * @return *this
   */
  DISABLE_HD_WARNING
  inline LVARRAY_HOST_DEVICE constexpr
  ArrayView( ArrayView const & source ) noexcept:
    m_dims{ 0 },
    m_strides{ 0 },
    m_dataBuffer{ source.m_dataBuffer, source.size() },
    m_singleParameterResizeIndex( source.m_singleParameterResizeIndex )
  {
    for( int i = 0; i < NDIM; ++i )
    {
      m_dims[ i ] = source.m_dims[ i ];
      m_strides[ i ] = source.m_strides[ i ];
    }
  }

  /**
   * @brief Move constructor, creates a shallow copy and invalidates the source.
   * @param source object to move.
   * @return *this.
   * @note Since this invalidates the source this should not be used when @p source is
   *   the parent of an Array. Do not do this:
   * @code
   * Array< int, RAJA::PERM_I, std::ptrdiff_t, MallocBuffer > array( 10 );
   * ArrayView< int, 1, 0, std::ptrdiff_t, MallocBuffer > view = std::move( array.toView() );
   * @endcode
   *   However this is ok:
   * @code
   * Array< int, RAJA::PERM_I, std::ptrdiff_t, MallocBuffer > array( 10 );
   * ArrayView< int, 1, 0, std::ptrdiff_t, MallocBuffer > view = array.toView();
   * ArrayView< int, 1, 0, std::ptrdiff_t, MallocBuffer > anotherView = std::move( view );
   * @endcode
   */
  inline LVARRAY_HOST_DEVICE constexpr
  ArrayView( ArrayView && source ):
    m_dims{ 0 },
    m_strides{ 0 },
    m_dataBuffer( std::move( source.m_dataBuffer ) ),
    m_singleParameterResizeIndex( source.m_singleParameterResizeIndex )
  {
    for( int i = 0; i < NDIM; ++i )
    {
      m_dims[ i ] = source.m_dims[ i ];
      m_strides[ i ] = source.m_strides[ i ];

      source.m_dims[ i ] = 0;
      source.m_strides[ i ] = 0;
    }
  }

  /**
   * @brief Move assignment operator, creates a shallow copy and invalidates the source.
   * @param rhs The object to copy.
   * @return *this.
   * @note Since this invalidates the source this should not be used when @p rhs is
   *   the parent of an Array. Do not do this:
   * @code
   * Array< int, RAJA::PERM_I, std::ptrdiff_t, MallocBuffer > array( 10 );
   * ArrayView< int, 1, 0, std::ptrdiff_t, MallocBuffer > view;
   * view = std::move( array.toView() );
   * @endcode
   *   However this is ok:
   * @code
   * Array< int, RAJA::PERM_I, std::ptrdiff_t, MallocBuffer > array( 10 );
   * ArrayView< int, 1, 0, std::ptrdiff_t, MallocBuffer > view = array.toView();
   * ArrayView< int, 1, 0, std::ptrdiff_t, MallocBuffer > anotherView;
   * anotherView = std::move( view );
   * @endcode
   */
  inline LVARRAY_HOST_DEVICE constexpr
  ArrayView & operator=( ArrayView && rhs )
  {
    m_dataBuffer = std::move( rhs.m_dataBuffer );
    m_singleParameterResizeIndex = rhs.m_singleParameterResizeIndex;
    for( int i = 0; i < NDIM; ++i )
    {
      m_dims[ i ] = rhs.m_dims[ i ];
      m_strides[ i ] = rhs.m_strides[ i ];

      rhs.m_dims[ i ] = 0;
      rhs.m_strides[ i ] = 0;
    }

    return *this;
  }

  /**
   * @brief Copy assignment operator, creates a shallow copy.
   * @param rhs object to copy.
   * @return *this
   */
  inline constexpr
  ArrayView & operator=( ArrayView const & rhs ) noexcept
  {
    m_dataBuffer = rhs.m_dataBuffer;
    m_singleParameterResizeIndex = rhs.m_singleParameterResizeIndex;
    for( int i = 0; i < NDIM; ++i )
    {
      m_dims[ i ] = rhs.m_dims[ i ];
      m_strides[ i ] = rhs.m_strides[ i ];
    }

    return *this;
  }

  /**
   * @brief Set all values of array to rhs.
   * @param rhs The value that array will be set to.
   * @return *this.
   */
  DISABLE_HD_WARNING
  inline LVARRAY_HOST_DEVICE constexpr
  ArrayView const & operator=( T const & rhs ) const noexcept
  {
    INDEX_TYPE const length = size();
    T * const data_ptr = data();
    for( INDEX_TYPE a = 0; a < length; ++a )
    {
      data_ptr[a] = rhs;
    }
    return *this;
  }

  /**
   * @brief @return Return *this after converting any nested arrays to const views.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  ViewType & toView() const
  { return reinterpret_cast< ViewType & >( *this ); }

  /**
   * @brief @return Return *this after converting any nested arrays to const views to const values.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  ViewTypeConst & toViewConst() const
  { return reinterpret_cast< ViewTypeConst & >( *this ); }

  /**
   * @brief @return Return *this interpret as ArrayView<T const> const &.
   */
  template< typename U = T >
  inline LVARRAY_HOST_DEVICE constexpr
  operator std::enable_if_t< !std::is_const< U >::value, ViewTypeConst & >() const noexcept
  { return toViewConst(); }

  /**
   * @brief @return Return an ArraySlice representing this ArrayView.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  ArraySlice< T, NDIM, USD, INDEX_TYPE >
  toSlice() const noexcept
  { return ArraySlice< T, NDIM, USD, INDEX_TYPE >( data(), m_dims, m_strides ); }

  /**
   * @brief @return Return an immutable ArraySlice representing this ArrayView.
   */
  template< typename U=T >
  inline LVARRAY_HOST_DEVICE constexpr
  ArraySlice< T const, NDIM, USD, INDEX_TYPE >
  toSliceConst() const noexcept
  { return ArraySlice< T const, NDIM, USD, INDEX_TYPE >( data(), m_dims, m_strides ); }

  /**
   * @brief @return Return an ArraySlice representing this ArrayView.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  operator ArraySlice< T, NDIM, USD, INDEX_TYPE >() const noexcept
  { return toSlice(); }

  /**
   * @brief @return Return an immutable ArraySlice representing this ArrayView.
   */
  template< typename U=T >
  inline LVARRAY_HOST_DEVICE constexpr
  operator std::enable_if_t< !std::is_const< U >::value,
                             ArraySlice< T const, NDIM, USD, INDEX_TYPE > const >
    () const noexcept
  { return toSliceConst(); }

  /**
   * @brief @return Return the allocated size.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE size() const noexcept
  { return multiplyAll< NDIM >( m_dims ); }

  /**
   * @brief @return Return the length of the given dimension.
   * @param dim The dimension to get the length of.
   */
  LVARRAY_HOST_DEVICE inline
  INDEX_TYPE size( int const dim ) const noexcept
  {
#ifdef USE_ARRAY_BOUNDS_CHECK
    LVARRAY_ASSERT_GE( dim, 0 );
    LVARRAY_ASSERT_GT( NDIM, dim );
#endif
    return m_dims[dim];
  }

  /**
   * @brief @return Return true if the array is empty.
   */
  inline bool empty() const
  { return size() == 0; }

  /**
   * @brief @return Return an iterator to the begining of the data.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  T * begin() const
  { return data(); }

  /**
   * @brief @return Return an iterator to the end of the data.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  T * end() const
  { return data() + size(); }

  /**
   * @brief @return Return a reference to the first value.
   */
  T & front() const
  { return data()[ 0 ]; }

  /**
   * @brief @return Return a reference to the last value.
   */
  T & back() const
  { return data()[size() - 1]; }

  ///***********************************************************************************************
  ///**** DO NOT EDIT!!! THIS CODE IS COPIED FROM ArraySlice *****
  ///***********************************************************************************************

  /**
   * @brief @return Return a lower dimensionsal slice of this ArrayView.
   * @param index The index of the slice to create.
   * @note This method is only active when NDIM > 1.
   */
  template< int U=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
  std::enable_if_t< (U > 1), ArraySlice< T, NDIM - 1, USD - 1, INDEX_TYPE > >
  operator[]( INDEX_TYPE const index ) const noexcept LVARRAY_RESTRICT_THIS
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return ArraySlice< T, NDIM-1, USD-1, INDEX_TYPE >( data() + ConditionalMultiply< USD == 0 >::multiply( index, m_strides[ 0 ] ),
                                                       m_dims + 1,
                                                       m_strides + 1 );
  }

  /**
   * @brief @return Return a reference to the value at the given index.
   * @param index The index of the value to access.
   * @note This method is only active when NDIM == 1.
   */
  template< int U=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
  std::enable_if_t< U == 1, T & >
  operator[]( INDEX_TYPE const index ) const noexcept LVARRAY_RESTRICT_THIS
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return data()[ ConditionalMultiply< USD == 0 >::multiply( index, m_strides[ 0 ] ) ];
  }

  /**
   * @tparam INDICES A variadic pack of integral types.
   * @brief @return Return a reference to the value at the given multidimensional index.
   * @param indices The indices of the value to access.
   */
  template< typename ... INDICES >
  LVARRAY_HOST_DEVICE inline constexpr
  T & operator()( INDICES... indices ) const
  {
    static_assert( sizeof ... (INDICES) == NDIM, "number of indices does not match NDIM" );
    return data()[ linearIndex( indices ... ) ];
  }

  /**
   * @tparam INDICES A variadic pack of integral types.
   * @brief @return Return the linear index from a multidimensional index.
   * @param indices The indices of the value to get the linear index of.
   */
  template< typename ... INDICES >
  LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
  INDEX_TYPE linearIndex( INDICES const ... indices ) const
  {
    static_assert( sizeof ... (INDICES) == NDIM, "number of indices does not match NDIM" );
#ifdef USE_ARRAY_BOUNDS_CHECK
    checkIndices( m_dims, indices ... );
#endif
    return getLinearIndex< USD >( m_strides, indices ... );
  }

  ///***********************************************************************************************
  ///**** END DO NOT EDIT!!! THIS CODE IS COPIED FROM ArraySlice *****
  ///***********************************************************************************************

  /**
   * @brief @return Return a pointer to the values.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  T * data() const
  { return m_dataBuffer.data(); }

  /**
   * @brief @return A pointer to the array containing the size of each dimension.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  INDEX_TYPE const * dims() const noexcept
  { return m_dims; }

  /**
   * @brief @return A pointer to the array containing the stride of each dimension.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  INDEX_TYPE const * strides() const noexcept
  { return m_strides; }

  /**
   * @brief Move the Array to the given execution space, optionally touching it.
   * @param space the space to move the Array to.
   * @param touch whether the Array should be touched in the new space or not.
   * @note Not all Buffers support memory movement.
   */
  void move( MemorySpace const space, bool const touch=true ) const
  { m_dataBuffer.move( space, size(), touch ); }

#if defined(USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__)
  /**
   * @brief Static function that will be used by Totalview to display the array contents.
   * @param av A pointer to the array that is being displayed.
   * @return 0 if everything went OK
   */
  static int TV_ttf_display_type( ArrayView const * av )
  {
    if( av!=nullptr )
    {
      int constexpr ndim = NDIM;
      //std::cout<<"Totalview using ("<<totalview::format<T,INDEX_TYPE>(NDIM, av->m_dims )<<") for display of
      // m_data;"<<std::endl;
      // TV_ttf_add_row( "tv(m_data)", totalview::format< T, INDEX_TYPE >( NDIM, av->m_dims ).c_str(), (av->m_data) );
      // TV_ttf_add_row( "m_data", totalview::format< T, INDEX_TYPE >( 0, av->m_dims ).c_str(), (av->m_data) );
      TV_ttf_add_row( "m_dims", totalview::format< INDEX_TYPE, int >( 1, &ndim ).c_str(), (av->m_dims) );
      TV_ttf_add_row( "m_strides", totalview::format< INDEX_TYPE, int >( 1, &ndim ).c_str(), (av->m_strides) );
      TV_ttf_add_row( "m_dataBuffer", LvArray::demangle< BUFFER_TYPE< T > >().c_str(), &(av->m_dataBuffer) );
      TV_ttf_add_row( "m_singleParameterResizeIndex", "int", &(av->m_singleParameterResizeIndex) );
    }
    return 0;
  }
#endif

protected:

  /**
   * @brief Protected constructor to be used by the Array class.
   * @note The unused boolean parameter is to distinguish this from the default constructor.
   */
  DISABLE_HD_WARNING
  LVARRAY_HOST_DEVICE inline explicit CONSTEXPR_WITHOUT_BOUNDS_CHECK
  ArrayView( bool ) noexcept:
    m_dims{ 0 },
    m_strides{ 0 },
    m_dataBuffer( true )
  {
#if defined(USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__)
    ArrayView::TV_ttf_display_type( nullptr );
#endif
  }

  /// the dimensions of the array.
  INDEX_TYPE m_dims[ NDIM ] = { 0 };

  /// the strides of the array.
  INDEX_TYPE m_strides[ NDIM ] = { 0 };

  /// this data member contains the actual data for the array.
  BUFFER_TYPE< T > m_dataBuffer;

  /// this data member specifies the dimension that will be resized as a result of a call to the
  /// single dimension resize method.
  int m_singleParameterResizeIndex = 0;
};

/**
 * @brief True if the template type is a ArrayView.
 */
template< class >
constexpr bool isArrayView = false;

/**
 * @tparam T The type contained in the ArrayView.
 * @tparam NDIM The number of dimensions in the ArrayView.
 * @tparam USD The unit stride dimension.
 * @tparam INDEX_TYPE The integral type used as an index.
 * @tparam BUFFER_TYPE The type used to manager the underlying allocation.
 * @brief Specialization of isArrayView for the ArrayView class.
 */
template< typename T,
          int NDIM,
          int USD,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
constexpr bool isArrayView< ArrayView< T, NDIM, USD, INDEX_TYPE, BUFFER_TYPE > > = true;

} // namespace LvArray

#endif // ARRAYVIEW_HPP_
