/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file ArrayView.hpp
 * @brief Contains the implementation of LvArray::ArrayView.
 */

#pragma once

// Source includes
#include "ArraySlice.hpp"
#include "Macros.hpp"
#include "indexing.hpp"
#include "limits.hpp"
#include "sliceHelpers.hpp"
#include "bufferManipulation.hpp"
#include "umpireInterface.hpp"

// System includes
#if defined(LVARRAY_USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__)
#include "totalview/tv_helpers.hpp"
#include "totalview/tv_data_display.h"
#endif

namespace LvArray
{

/**
 * @class ArrayView
 * @brief This class serves to provide a "view" of a multidimensional array.
 * @tparam T type of data that is contained by the array
 * @tparam NDIM_TPARAM number of dimensions in array (e.g. NDIM=1->vector, NDIM=2->Matrix, etc. ).
 * @tparam USD the dimension with a unit stride, in an Array with a standard layout
 *   this is the last dimension.
 * @tparam INDEX_TYPE the integer to use for indexing.
 * @tparam BUFFER_TYPE A class that defines how to actually allocate memory for the Array. Must take
 *   one template argument that describes the type of the data being stored (T).
 *
 * @details When using the ChaiBuffer the copy copy constructor of this class calls the copy constructor for the
 *   ChaiBuffer which will move the data to the location of the touch (host or device). In general, the
 *   ArrayView should be what is passed by value into a lambda that is used to launch kernels as it
 *   copy will trigger the desired data motion onto the appropriate memory space.
 *
 *   Key features:
 *   1) When using a ChaiBuffer as the BUFFER_TYPE the ArrayView copy constructor will move the data
 *      to the current execution space.
 *   2) Defines a slicing operator[].
 *   3) Defines operator() array accessor.
 *   3) operator[] and operator() are all const and may be called in non-mutable lambdas.
 *   4) Conversion operators to go from ArrayView<T> to ArrayView<T const>.
 *   5) Since the Array is derived from ArrayView, it may be upcasted:
 *        Array<T,NDIM> array;
 *        ArrayView<T,NDIM> const & arrView = array;
 *
 *   A good guideline is to upcast to an ArrayView when you don't need allocation capabilities that
 *   are only present in Array.
 */
template< typename T,
          int NDIM_TPARAM,
          int USD_TPARAM,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class ArrayView
{
public:

  static_assert( NDIM_TPARAM > 0, "Number of dimensions must be greater than zero." );
  static_assert( USD_TPARAM >= 0, "USD must be positive." );
  static_assert( USD_TPARAM < NDIM_TPARAM, "USD must be less than NDIM." );

  /// The type of the values in the ArrayView.
  using ValueType = T;

  /// The number of dimensions.
  static constexpr int NDIM = NDIM_TPARAM;

  /// The unit stride dimension.
  static constexpr int USD = USD_TPARAM;

  /// The integer type used for indexing.
  using IndexType = INDEX_TYPE;

  /// The type when the data type is const.
  using ViewTypeConst = ArrayView< std::remove_const_t< T > const, NDIM, USD, INDEX_TYPE, BUFFER_TYPE >;


  /// The type when all inner array classes are converted to const views.
  using NestedViewType = ArrayView< std::remove_reference_t< typeManipulation::NestedViewType< T > >,
                                    NDIM, USD, INDEX_TYPE, BUFFER_TYPE >;

  /// The type when all inner array classes are converted to const views and the inner most view's values are also const.
  using NestedViewTypeConst = ArrayView< std::remove_reference_t< typeManipulation::NestedViewTypeConst< T > >,
                                         NDIM, USD, INDEX_TYPE, BUFFER_TYPE >;

  /// The type of the ArrayView when converted to an ArraySlice.
  using SliceType = ArraySlice< T, NDIM, USD, INDEX_TYPE > const;

  /// The type of the ArrayView when converted to an immutable ArraySlice.
  using SliceTypeConst = ArraySlice< T const, NDIM, USD, INDEX_TYPE > const;

  /// The type of the values in the ArrayView, here for stl compatability.
  using value_type = T;

  /// The integer type used for indexing, here for stl compatability.
  using size_type = INDEX_TYPE;

  /**
   * @name Constructors, destructor and assignment operators.
   */
  ///@{

  /**
   * @brief A constructor to create an uninitialized ArrayView.
   * @note An uninitialized ArrayView should not be used until it is assigned to.
   */
  ArrayView() = default;

  /**
   * @brief Copy Constructor.
   * @param source The object to copy.
   * @note Triggers the copy constructor for @tparam BUFFER_TYPE. When using the
   *   ChaiBuffer this can move the underlying buffer to a new memory space if the execution context is set.
   */
  DISABLE_HD_WARNING
  inline LVARRAY_HOST_DEVICE constexpr
  ArrayView( ArrayView const & source ) noexcept:
    m_dims{ source.m_dims },
    m_strides{ source.m_strides },
    m_dataBuffer{ source.m_dataBuffer, source.size() },
    m_singleParameterResizeIndex( source.m_singleParameterResizeIndex )
  {}

  /**
   * @brief Construct a new ArrayView from an ArrayView with a different type.
   * @tparam U The type to convert from.
   * @param source The ArrayView to convert.
   * @details If the size of @c T and @c U are different then either the size of @c T must be a
   *   multiple of the size of @c U or vice versa. If the types have different size then size of the unit stride
   *   dimension is changed accordingly.
   * @note This is useful for converting between single values and SIMD types such as CUDA's @c __half and @c __half2.
   * @code
   *   Array< int, 2, RAJA::PERM_IJ, std::ptrdiff_t, MallocBuffer > x( 5, 10 );
   *   ArrayView< int[ 2 ], 2, 1, std::ptrdiff_t, MallocBuffer > y( x.toView() );
   *   assert( y.size( 1 ) == x.size( 1 ) / 2 );
   *   assert( y( 3, 4 )[ 0 ] == x( 3, 8 ) );
   * @endcode
   */
  template< typename U, typename=std::enable_if_t< !std::is_same< T, U >::value > >
  inline LVARRAY_HOST_DEVICE constexpr
  explicit ArrayView( ArrayView< U, NDIM, USD, INDEX_TYPE, BUFFER_TYPE > const & source ):
    m_dims{ source.dimsArray() },
    m_strides{ source.stridesArray() },
    m_dataBuffer{ source.dataBuffer() },
    m_singleParameterResizeIndex( source.getSingleParameterResizeIndex() )
  {
    m_dims[ USD ] = typeManipulation::convertSize< T, U >( m_dims[ USD ] );

    for( int i = 0; i < NDIM; ++i )
    {
      if( i != USD )
      {
        m_strides[ i ] = typeManipulation::convertSize< T, U >( m_strides[ i ] );
      }
    }
  }

  /**
   * @brief Move constructor, creates a shallow copy and invalidates the source.
   * @param source object to move.
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
  ArrayView( ArrayView && source ) = default;

  /**
   * @brief Construct a new ArrayView from existing components.
   * @param dims The array of dimensions.
   * @param strides The array of strides.
   * @param singleParameterResizeIndex The single parameter resize index.
   * @param buffer The buffer to copy construct.
   */
  inline LVARRAY_HOST_DEVICE constexpr explicit
  ArrayView( typeManipulation::CArray< INDEX_TYPE, NDIM > const & dims,
             typeManipulation::CArray< INDEX_TYPE, NDIM > const & strides,
             int const singleParameterResizeIndex,
             BUFFER_TYPE< T > const & buffer ):
    m_dims( dims ),
    m_strides( strides ),
    m_dataBuffer( buffer ),
    m_singleParameterResizeIndex( singleParameterResizeIndex )
  {}

  /// The default destructor.
  ~ArrayView() = default;

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
  inline LVARRAY_HOST_DEVICE LVARRAY_INTEL_CONSTEXPR
  ArrayView & operator=( ArrayView && rhs )
  {
    m_dataBuffer = std::move( rhs.m_dataBuffer );
    m_singleParameterResizeIndex = rhs.m_singleParameterResizeIndex;
    for( int i = 0; i < NDIM; ++i )
    {
      m_dims[ i ] = rhs.m_dims[ i ];
      m_strides[ i ] = rhs.m_strides[ i ];
    }

    return *this;
  }

  /**
   * @brief Copy assignment operator, creates a shallow copy.
   * @param rhs object to copy.
   * @return *this
   */
  inline LVARRAY_INTEL_CONSTEXPR
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

  ///@}

  /**
   * @name ArrayView and ArraySlice creation methods and user defined conversions.
   */
  ///@{

  /**
   * @return Return a new ArrayView.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  ArrayView toView() const &
  { return ArrayView( m_dims, m_strides, m_singleParameterResizeIndex, m_dataBuffer ); }

  /**
   * @return Return a new ArrayView where @c T is @c const.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  ViewTypeConst toViewConst() const &
  {
    return ViewTypeConst( m_dims,
                          m_strides,
                          m_singleParameterResizeIndex,
                          m_dataBuffer );
  }

  /**
   * @brief @return Return *this after converting any nested arrays to const views.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  NestedViewType toNestedView() const &
  { return reinterpret_cast< NestedViewType const & >( *this ); }

  /**
   * @brief @return Return *this after converting any nested arrays to const views to const values.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  NestedViewTypeConst toNestedViewConst() const &
  { return reinterpret_cast< NestedViewTypeConst const & >( *this ); }

  /**
   * @return Return an ArraySlice representing this ArrayView.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  ArraySlice< T, NDIM, USD, INDEX_TYPE >
  toSlice() const & noexcept
  { return ArraySlice< T, NDIM, USD, INDEX_TYPE >( data(), m_dims.data, m_strides.data ); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null ArraySlice.
   * @note This cannot be called on a rvalue since the @c ArraySlice would
   *   contain pointers to the dims and strides of the current @c ArrayView that is
   *   about to be destroyed. This overload prevents that from happening.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  ArraySlice< T, NDIM, USD, INDEX_TYPE >
  toSlice() const && noexcept = delete;

  /**
   * @return Return an immutable ArraySlice representing this ArrayView.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  ArraySlice< T const, NDIM, USD, INDEX_TYPE >
  toSliceConst() const & noexcept
  { return ArraySlice< T const, NDIM, USD, INDEX_TYPE >( data(), m_dims.data, m_strides.data ); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null ArraySlice.
   * @brief This cannot be called on a rvalue since the @c ArraySlice would
   *   contain pointers to the dims and strides of the current @c ArrayView that is
   *   about to be destroyed. This overload prevents that from happening.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  ArraySlice< T const, NDIM, USD, INDEX_TYPE >
  toSliceConst() const && noexcept = delete;

  /**
   * @brief A user defined conversion operator (UDC) to an ArrayView< T const, ... >.
   * @return A new ArrayView where @c T is @c const.
   */
  template< typename _T=T >
  inline LVARRAY_HOST_DEVICE constexpr
  operator std::enable_if_t< !std::is_const< _T >::value,
                             ArrayView< T const, NDIM, USD, INDEX_TYPE, BUFFER_TYPE > >() const noexcept
  { return toViewConst(); }

  /**
   * @return Return an ArraySlice representing this ArrayView.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  operator ArraySlice< T, NDIM, USD, INDEX_TYPE >() const & noexcept
  { return toSlice(); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null ArraySlice.
   * @brief This conversion cannot be called on a rvalue since the @c ArraySlice would
   *   contain pointers to the dims and strides of the current @c ArrayView that is
   *   about to be destroyed. This overload prevents that from happening.
   */
  inline LVARRAY_HOST_DEVICE constexpr
  operator ArraySlice< T, NDIM, USD, INDEX_TYPE >() const && noexcept = delete;

  /**
   * @return Return an immutable ArraySlice representing this ArrayView.
   */
  template< typename _T=T >
  inline LVARRAY_HOST_DEVICE constexpr
  operator std::enable_if_t< !std::is_const< _T >::value,
                             ArraySlice< T const, NDIM, USD, INDEX_TYPE > const >() const & noexcept
  { return toSliceConst(); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null ArraySlice.
   * @brief This conversion cannot be called on a rvalue since the @c ArraySlice would
   *   contain pointers to the dims and strides of the current @c ArrayView that is
   *   about to be destroyed. This overload prevents that from happening.
   */
  template< typename _T=T >
  inline LVARRAY_HOST_DEVICE constexpr
  operator std::enable_if_t< !std::is_const< _T >::value,
                             ArraySlice< T const, NDIM, USD, INDEX_TYPE > const >() const && noexcept = delete;

  ///@}

  /**
   * @name Attribute querying methods
   */
  ///@{

  /**
   * @return Return the allocated size.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  INDEX_TYPE size() const noexcept
  {
  #if defined( __ibmxl__ )
    // Note: This used to be done with a recursive template but XL-release would produce incorrect results.
    // Specifically in exampleArray it would return an "old" size even after being updated, strange.
    INDEX_TYPE val = m_dims[ 0 ];
    for( int i = 1; i < NDIM; ++i )
    { val *= m_dims[ i ]; }

    return val;
  #else
    return indexing::multiplyAll< NDIM >( m_dims.data );
  #endif
  }

  /**
   * @return Return the length of the given dimension.
   * @param dim The dimension to get the length of.
   */
  LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
  INDEX_TYPE size( int const dim ) const noexcept
  {
#ifdef LVARRAY_BOUNDS_CHECK
    LVARRAY_ASSERT_GE( dim, 0 );
    LVARRAY_ASSERT_GT( NDIM, dim );
#endif
    return m_dims[ dim ];
  }

  /**
   * @return Return true if the array is empty.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  bool empty() const
  { return size() == 0; }

  /**
   * @return Return the maximum number of values the Array can hold without reallocation.
   */
  LVARRAY_HOST_DEVICE constexpr
  INDEX_TYPE capacity() const
  { return LvArray::integerConversion< INDEX_TYPE >( m_dataBuffer.capacity() ); }

  /**
   * @return Return the default resize dimension.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  int getSingleParameterResizeIndex() const
  { return m_singleParameterResizeIndex; }

  /**
   * @tparam INDICES A variadic pack of integral types.
   * @return Return the linear index from a multidimensional index.
   * @param indices The indices of the value to get the linear index of.
   */
  template< typename ... INDICES >
  LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
  INDEX_TYPE linearIndex( INDICES const ... indices ) const
  {
    static_assert( sizeof ... (INDICES) == NDIM, "number of indices does not match NDIM" );
#ifdef LVARRAY_BOUNDS_CHECK
    indexing::checkIndices( m_dims.data, indices ... );
#endif
    return indexing::getLinearIndex< USD >( m_strides.data, indices ... );
  }

  /**
   * @return A pointer to the array containing the size of each dimension.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  INDEX_TYPE const * dims() const noexcept
  { return m_dims.data; }

  /**
   * @return The CArray containing the size of each dimension.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  typeManipulation::CArray< INDEX_TYPE, NDIM > const & dimsArray() const
  { return m_dims; }

  /**
   * @return A pointer to the array containing the stride of each dimension.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  INDEX_TYPE const * strides() const noexcept
  { return m_strides.data; }

  /**
   * @return The CArray containing the stride of each dimension.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  typeManipulation::CArray< INDEX_TYPE, NDIM > const & stridesArray() const
  { return m_strides; }

  /**
   * @return A reference to the underlying buffer.
   * @note Use with caution.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  BUFFER_TYPE< T > const & dataBuffer() const
  { return m_dataBuffer; }

  ///@}

  /**
   * @name Methods that provide access to the data.
   */
  ///@{

  /**
   * @return Return a lower dimensional slice of this ArrayView.
   * @param index The index of the slice to create.
   * @note This method is only active when NDIM > 1.
   */
  template< int _NDIM=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
  std::enable_if_t< (_NDIM > 1), ArraySlice< T, NDIM - 1, USD - 1, INDEX_TYPE > >
  operator[]( INDEX_TYPE const index ) const & noexcept
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return ArraySlice< T, NDIM-1, USD-1, INDEX_TYPE >( data() + indexing::ConditionalMultiply< USD == 0 >::multiply( index, m_strides[ 0 ] ),
                                                       m_dims.data + 1,
                                                       m_strides.data + 1 );
  }

  /**
   * @brief Overload for rvalues that is deleted.
   * @param index Not used.
   * @return A null ArraySlice.
   * @brief The multidimensional operator[] cannot be called on a rvalue since the @c ArraySlice
   *   would contain pointers to the object that is about to be destroyed. This overload
   *   prevents that from happening.
   */
  template< int _NDIM=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
  std::enable_if_t< (_NDIM > 1), ArraySlice< T, NDIM - 1, USD - 1, INDEX_TYPE > >
  operator[]( INDEX_TYPE const index ) const && noexcept = delete;

  /**
   * @return Return a reference to the value at the given index.
   * @param index The index of the value to access.
   * @note This method is only active when NDIM == 1.
   */
  template< int _NDIM=NDIM >
  LVARRAY_HOST_DEVICE inline CONSTEXPR_WITHOUT_BOUNDS_CHECK
  std::enable_if_t< _NDIM == 1, T & >
  operator[]( INDEX_TYPE const index ) const & noexcept
  {
    ARRAY_SLICE_CHECK_BOUNDS( index );
    return data()[ indexing::ConditionalMultiply< USD == 0 >::multiply( index, m_strides[ 0 ] ) ];
  }

  /**
   * @tparam INDICES A variadic pack of integral types.
   * @return Return a reference to the value at the given multidimensional index.
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
   * @return Return a pointer to the values.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  T * data() const
  { return m_dataBuffer.data(); }

  /**
   * @return Return an iterator to the begining of the data.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  T * begin() const
  { return data(); }

  /**
   * @return Return an iterator to the end of the data.
   */
  LVARRAY_HOST_DEVICE inline constexpr
  T * end() const
  { return data() + size(); }

  /**
   * @return Return a reference to the first value.
   */
  T & front() const
  { return data()[ 0 ]; }

  /**
   * @return Return a reference to the last value.
   */
  T & back() const
  { return data()[size() - 1]; }

  ///@}

  /**
   * @name Methods that set all the values
   */
  ///@{

  /**
   * @brief Set all entries in the array to @p value.
   * @tparam POLICY The RAJA policy to use.
   * @param value The value to set entries to.
   * @note The view is moved to and touched in the appropriate space.
   */
  DISABLE_HD_WARNING
  template< typename POLICY >
  void setValues( T const & value ) const
  {
    auto const view = toView();
    RAJA::forall< POLICY >( RAJA::TypedRangeSegment< INDEX_TYPE >( 0, size() ),
                            [value, view] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
      {
        view.data()[ i ] = value;
      } );
  }

  /**
   * @brief Use memset to set all the values in the array to 0.
   * @details This is preferred over setValues< POLICY >( 0 ) for numeric types since it is much faster in most cases.
   *   If the buffer is allocated using Umpire then the Umpire ResouceManager is used, otherwise std::memset is used.
   * @note The memset occurs in the last space the array was used in and the view is moved and touched in that space.
   */
  inline void zero() const
  {
  #if !defined( LVARRAY_USE_UMPIRE )
    LVARRAY_ERROR_IF_NE_MSG( getPreviousSpace(), MemorySpace::host, "Without Umpire only host memory is supported." );
  #endif

    if( size() > 0 )
    {
      move( getPreviousSpace(), true );
      umpireInterface::memset( data(), 0, size() * sizeof( T ) );
    }
  }

  /**
   * @brief Set entries to values from another compatible ArrayView.
   * @tparam POLICY The RAJA policy to use.
   * @param rhs The source array view, must have the same dimensions and strides as *this.
   */
  template< typename POLICY >
  void setValues( ArrayView< T const, NDIM, USD, INDEX_TYPE, BUFFER_TYPE > const & rhs ) const
  {
    for( int dim = 0; dim < NDIM; ++dim )
    {
      LVARRAY_ERROR_IF_NE( size( dim ), rhs.size( dim ) );
      LVARRAY_ERROR_IF_NE_MSG( strides()[ dim ], rhs.strides()[ dim ],
                               "This method only works with Arrays with the same data layout." );
    }

    auto const view = toView();
    RAJA::forall< POLICY >( RAJA::TypedRangeSegment< INDEX_TYPE >( 0, size() ), [view, rhs] LVARRAY_HOST_DEVICE ( INDEX_TYPE const i )
      {
        view.data()[ i ] = rhs.data()[ i ];
      } );
  }

  ///@}

  /**
   * @name Methods dealing with memory spaces
   */
  ///@{

  /**
   * @return The last space the Array was moved to.
   */
  MemorySpace getPreviousSpace() const
  { return m_dataBuffer.getPreviousSpace(); }

  /**
   * @brief Touch the memory in @p space.
   * @param space The memory space in which a touch will be recorded.
   */
  void registerTouch( MemorySpace const space ) const
  { m_dataBuffer.registerTouch( space ); }

  /**
   * @brief Move the Array to the given execution space, optionally touching it.
   * @param space the space to move the Array to.
   * @param touch whether the Array should be touched in the new space or not.
   * @note Not all Buffers support memory movement.
   */
  void move( MemorySpace const space, bool const touch=true ) const
  { m_dataBuffer.moveNested( space, size(), touch ); }

  ///@}

#if defined(LVARRAY_USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__)
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
      TV_ttf_add_row( "m_dataBuffer", LvArray::system::demangle< BUFFER_TYPE< T > >().c_str(), &(av->m_dataBuffer) );
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
#if defined(LVARRAY_USE_TOTALVIEW_OUTPUT) && !defined(__CUDA_ARCH__) && defined(LVARRAY_BOUNDS_CHECK)
    ArrayView::TV_ttf_display_type( nullptr );
#endif
  }

  /**
   * @brief Protected constructor to be used by the Array class.
   * @details Construct an empty ArrayView from @p buffer.
   * @param buffer The buffer use.
   */
  DISABLE_HD_WARNING
  inline LVARRAY_HOST_DEVICE constexpr
  ArrayView( BUFFER_TYPE< T > && buffer ) noexcept:
    m_dims{ 0 },
    m_strides{ 0 },
    m_dataBuffer{ std::move( buffer ) },
    m_singleParameterResizeIndex{ 0 }
  {}

  /// the dimensions of the array.
  typeManipulation::CArray< INDEX_TYPE, NDIM > m_dims = { 0 };

  /// the strides of the array.
  typeManipulation::CArray< INDEX_TYPE, NDIM > m_strides = { 0 };

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
