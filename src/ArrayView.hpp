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
#include "Layout.hpp"
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
 * @tparam LAYOUT type of layout function that defines the mapping from logical coordinates to memory offsets
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
          typename LAYOUT,
          template< typename > class BUFFER_TYPE >
class ArrayView
{
public:

  static_assert( is_layout_v< LAYOUT >, "LAYOUT template parameter must be a Layout instantiation" );

  /// The type of layout.
  using LayoutType = LAYOUT;

  /// The type of extent.
  using ExtentType = typename LayoutType::ExtentType;

  /// The type of stride.
  using StrideType = typename LayoutType::StrideType;

  /// The number of dimensions.
  static constexpr int NDIM = LayoutType::NDIM;

  /// The unit-stride dimension index.
  static constexpr int USD = LayoutType::USD;

  static_assert( NDIM > 0, "Number of dimensions must be greater than zero." );

  /// The type of the values in the ArrayView.
  using ValueType = T;

  /// The integer type used for indexing.
  using IndexType = typename LayoutType::IndexType;

  /// The view type
  using ViewType = ArrayView;

  /// The type when the data type is const.
  using ViewTypeConst = ArrayView< std::remove_const_t< T > const, LAYOUT, BUFFER_TYPE >;

  /// The type when all inner array classes are converted to const views.
  using NestedViewType = ArrayView< std::remove_reference_t< typeManipulation::NestedViewType< T > >, LAYOUT, BUFFER_TYPE >;

  /// The type when all inner array classes are converted to const views and the inner most view's values are also const.
  using NestedViewTypeConst = ArrayView< std::remove_reference_t< typeManipulation::NestedViewTypeConst< T > >, LAYOUT, BUFFER_TYPE >;

  /// The type of the ArrayView when converted to an ArraySlice.
  using SliceType = ArraySlice< T, LayoutType > const;

  /// The type of the ArrayView when converted to an immutable ArraySlice.
  using SliceTypeConst = ArraySlice< T const, LayoutType > const;

  /// The type of the values in the ArrayView, here for stl compatability.
  using value_type = T;

  /// The integer type used for indexing, here for stl compatability.
  using size_type = IndexType;

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
  LVARRAY_HOST_DEVICE constexpr
  ArrayView( ArrayView const & source ) noexcept:
    m_layout( source.m_layout ),
    m_dataBuffer{ source.m_dataBuffer, source.size() }
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
  template< typename U,
            LVARRAY_REQUIRES( !std::is_same< T, U >::value && ( sizeof( T ) % sizeof( U ) == 0 ) ) >
  LVARRAY_HOST_DEVICE constexpr
  explicit ArrayView( ArrayView< U, decltype( LayoutType{}.template downcast< sizeof( T ) / ( sizeof( U ) ) >() ), BUFFER_TYPE > const & source ):
    m_layout( source.layout().template upcast< sizeof( T ) / ( sizeof( U ) ) >() ),
    m_dataBuffer{ source.dataBuffer() }
  {}

  template< typename U,
            LVARRAY_REQUIRES( ( sizeof( U ) > sizeof( T ) ) && ( sizeof( U ) % sizeof( T ) == 0 ) ) >
  LVARRAY_HOST_DEVICE constexpr
  explicit ArrayView( ArrayView< U, decltype( LayoutType{}.template upcast< sizeof( U ) / ( sizeof( T ) ) >() ), BUFFER_TYPE > const & source ):
    m_layout( source.layout().template downcast< sizeof( U ) / ( sizeof( T ) ) >() ),
    m_dataBuffer{ source.dataBuffer() }
  {}

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
   * @param layout The array view layout.
   * @param buffer The buffer to copy construct.
   */
  LVARRAY_HOST_DEVICE constexpr explicit
  ArrayView( LayoutType const & layout,
              BUFFER_TYPE< T > const & buffer ):
    m_layout( layout ),
    m_dataBuffer( buffer )
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
  LVARRAY_HOST_DEVICE LVARRAY_INTEL_CONSTEXPR
  ArrayView & operator=( ArrayView && rhs ) noexcept = default;

  /**
   * @brief Copy assignment operator, creates a shallow copy.
   * @param rhs object to copy.
   * @return *this
   */
  LVARRAY_INTEL_CONSTEXPR
  //ArrayView & operator=( ArrayView const & rhs ) noexcept = default;
  ArrayView & operator=( ArrayView const & rhs ) noexcept
  {
    m_dataBuffer = rhs.m_dataBuffer;
    m_layout = rhs.m_layout;
  }

  ///@}

  /**
   * @name ArrayView and ArraySlice creation methods and user defined conversions.
   */
  ///@{

  /**
   * @return Return a new ArrayView.
   */
  LVARRAY_HOST_DEVICE constexpr
  ArrayView toView() const &
  { return ArrayView( layout(), dataBuffer() ); }

  /**
   * @return Return a new ArrayView.
   */
  template< typename NEW_LAYOUT >
  LVARRAY_HOST_DEVICE constexpr
  ArrayView< T, NEW_LAYOUT, BUFFER_TYPE > toView() const &
  {
    NEW_LAYOUT newLayout( layout() );
    LVARRAY_ERROR_IF_MSG( newLayout != layout(), "Target layout does not match the source." );
    return ArrayView< T, NEW_LAYOUT, BUFFER_TYPE >( newLayout, dataBuffer() );
  }

  /**
   * @return Return a new ArrayView where @c T is @c const.
   */
  LVARRAY_HOST_DEVICE constexpr
  ViewTypeConst toViewConst() const &
  {
    return ViewTypeConst( layout(), dataBuffer() );
  }

  /**
   * @brief @return Return *this after converting any nested arrays to const views.
   */
  LVARRAY_HOST_DEVICE constexpr
  NestedViewType toNestedView() const &
  { return reinterpret_cast< NestedViewType const & >( *this ); }

  /**
   * @brief @return Return *this after converting any nested arrays to const views to const values.
   */
  LVARRAY_HOST_DEVICE constexpr
  NestedViewTypeConst toNestedViewConst() const &
  { return reinterpret_cast< NestedViewTypeConst const & >( *this ); }

  /**
   * @return Return an ArraySlice representing this ArrayView.
   */
  LVARRAY_HOST_DEVICE constexpr
  ArraySlice< T, LayoutType >
  toSlice() const & noexcept
  { return ArraySlice< T, LayoutType >( data(), layout() ); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null ArraySlice.
   * @note This cannot be called on a rvalue since the @c ArraySlice would
   *   contain pointers to the dims and strides of the current @c ArrayView that is
   *   about to be destroyed. This overload prevents that from happening.
   */
  LVARRAY_HOST_DEVICE constexpr
  ArraySlice< T, LayoutType >
  toSlice() const && noexcept = delete;

  /**
   * @return Return an immutable ArraySlice representing this ArrayView.
   */
  LVARRAY_HOST_DEVICE constexpr
  ArraySlice< T const, LayoutType >
  toSliceConst() const & noexcept
  { return ArraySlice< T const, LayoutType >( data(), layout() ); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null ArraySlice.
   * @brief This cannot be called on a rvalue since the @c ArraySlice would
   *   contain pointers to the dims and strides of the current @c ArrayView that is
   *   about to be destroyed. This overload prevents that from happening.
   */
  LVARRAY_HOST_DEVICE constexpr
  ArraySlice< T const, LayoutType >
  toSliceConst() const && noexcept = delete;

  /**
   * @brief A user defined conversion operator (UDC) to an ArrayView< T const, ... >.
   * @return A new ArrayView where @c T is @c const.
   */
  template< typename T_=T,
            LVARRAY_REQUIRES( !std::is_const< T_ >::value )>
  LVARRAY_HOST_DEVICE constexpr
  operator ArrayView< T const, LayoutType, BUFFER_TYPE >() const noexcept
  { return toViewConst(); }

  /**
   * @return Return an ArraySlice representing this ArrayView.
   */
  LVARRAY_HOST_DEVICE constexpr
  operator ArraySlice< T, LayoutType >() const & noexcept
  { return toSlice(); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null ArraySlice.
   * @brief This conversion cannot be called on a rvalue since the @c ArraySlice would
   *   contain pointers to the dims and strides of the current @c ArrayView that is
   *   about to be destroyed. This overload prevents that from happening.
   */
  LVARRAY_HOST_DEVICE constexpr
  operator ArraySlice< T, LayoutType >() const && noexcept = delete;

  /**
   * @return Return an immutable ArraySlice representing this ArrayView.
   */
  template< typename T_=T,
            LVARRAY_REQUIRES( !std::is_const< T_ >::value ) >
  LVARRAY_HOST_DEVICE constexpr
  operator ArraySlice< T const, LayoutType >() const & noexcept
  { return toSliceConst(); }

  /**
   * @brief Overload for rvalues that is deleted.
   * @return A null ArraySlice.
   * @brief This conversion cannot be called on a rvalue since the @c ArraySlice would
   *   contain pointers to the dims and strides of the current @c ArrayView that is
   *   about to be destroyed. This overload prevents that from happening.
   */
  template< typename T_=T,
            LVARRAY_REQUIRES( !std::is_const< T_ >::value ) >
  LVARRAY_HOST_DEVICE constexpr
  operator ArraySlice< T const, LayoutType >() const && noexcept = delete;

  ///@}

  /**
   * @name Attribute querying methods
   */
  ///@{

  /**
   * @return A reference to the view's layout.
   */
  LVARRAY_HOST_DEVICE constexpr
  LayoutType const & layout() const noexcept
  { return m_layout; }

  /**
   * @return Return the allocated size.
   */
  LVARRAY_HOST_DEVICE constexpr
  IndexType size() const noexcept
  {
    return layout().size();
  }

  /**
   * @return Return the length of the given dimension.
   * @param dim The dimension to get the length of.
   */
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK
  IndexType size( int const dim ) const noexcept
  {
#ifdef LVARRAY_BOUNDS_CHECK
    LVARRAY_ASSERT_GE( dim, 0 );
    LVARRAY_ASSERT_GT( NDIM, dim );
#endif
    return tupleManipulation::getValue( layout().extent(), dim );
  }

  /**
   * @return Return the length of the given dimension.
   * @param N the dimension to get the length of.
   */
  template< int N >
  LVARRAY_HOST_DEVICE constexpr
  auto size() const noexcept
  {
    static_assert( N < NDIM, "Dimension index must be less than number of dimensions." );
    return camp::get< N >( m_layout.extent() );
  }

  /**
   * @return Return the stride of the given dimension.
   * @param dim the dimension to get the stride of.
   */
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK
  IndexType stride( int const dim ) const noexcept
  {
#ifdef LVARRAY_BOUNDS_CHECK
    LVARRAY_ERROR_IF_GE( dim, NDIM );
#endif
    return tupleManipulation::getValue( m_layout.stride(), dim );
  }

  /**
   * @return Return the stride of the given dimension.
   * @param N the dimension to get the stride of.
   */
  template< int N >
  LVARRAY_HOST_DEVICE constexpr
  auto stride() const noexcept
  {
    static_assert( N < NDIM, "Dimension index must be less than number of dimensions." );
    return camp::get< N >( m_layout.stride() );
  }

  /**
   * @return Return true if the array is empty.
   */
  LVARRAY_HOST_DEVICE constexpr
  bool empty() const
  { return size() == 0; }

  /**
   * @return Return the maximum number of values the Array can hold without reallocation.
   */
  LVARRAY_HOST_DEVICE constexpr
  IndexType capacity() const
  { return LvArray::integerConversion< IndexType >( m_dataBuffer.capacity() ); }

  /**
   * @tparam INDICES A variadic pack of integral types.
   * @return Return the linear index from a multidimensional index.
   * @param indices The indices of the value to get the linear index of.
   */
  template< typename ... INDICES >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK
  IndexType linearIndex( INDICES const ... indices ) const
  {
    static_assert( sizeof ... (INDICES) == NDIM, "number of indices does not match NDIM" );
    return layout()( indices... );
  }

  /**
   * @return A reference to the underlying buffer.
   * @note Use with caution.
   */
  LVARRAY_HOST_DEVICE constexpr
  BUFFER_TYPE< T > const & dataBuffer() const
  { return m_dataBuffer; }

  ///@}

  /**
   * @name Methods that provide access to the data.
   */
  ///@{

  /**
   * @return Return a value reference or a lower dimensional slice of this ArrayView.
   * @param index The index along the left-most dimension
   */
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK
  decltype(auto)
  operator[]( IndexType const index ) const & noexcept
  {
    return operator()( index );
  }

  /**
   * @return Return a value reference or a lower dimensional slice of this ArrayView.
   * @param index The index along the left-most dimension
   * @note Deleted rvalue overload
   */
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK
  decltype(auto)
  operator[]( IndexType const index ) && = delete;

  /**
   * @tparam INDICES A variadic pack of integral types.
   * @return Return a reference to the value at the given multidimensional index.
   * @param indices The indices of the value to access.
   */
  template< typename ... INDICES >
  LVARRAY_HOST_DEVICE constexpr
  decltype(auto)
  operator()( INDICES... indices ) const
  {
    auto const sliceLayout = layout().slice( indices... );
    auto const sliceOffset = layout()( indices... );
    if constexpr ( decltype( sliceLayout )::NDIM == 0 )
    {
      return data()[ sliceOffset ];
    }
    else
    {
      return ArraySlice< T, std::remove_cv_t< decltype( sliceLayout ) > >( data() + sliceOffset, sliceLayout );
    }
  }

  /**
   * @return Return a pointer to the values.
   */
  LVARRAY_HOST_DEVICE constexpr
  T * data() const
  { return m_dataBuffer.data(); }

  /**
   * @return Return an iterator to the begining of the data.
   */
  LVARRAY_HOST_DEVICE constexpr
  T * begin() const
  { return data(); }

  /**
   * @return Return an iterator to the end of the data.
   */
  LVARRAY_HOST_DEVICE constexpr
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
    RAJA::forall< POLICY >( RAJA::TypedRangeSegment< IndexType >( 0, size() ),
                            [value, view] LVARRAY_HOST_DEVICE ( IndexType const i )
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
  void zero() const
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
  void setValues( ArrayView< T const, LayoutType, BUFFER_TYPE > const & rhs ) const
  {
    LVARRAY_ERROR_IF_NE_MSG( layout(), rhs.layout(), "This method only works with Arrays with the same data layout." );
    auto const view = toView();
    RAJA::forall< POLICY >( RAJA::TypedRangeSegment< IndexType >( 0, size() ), [view, rhs] LVARRAY_HOST_DEVICE ( IndexType const i )
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
  LVARRAY_HOST_DEVICE explicit CONSTEXPR_WITHOUT_BOUNDS_CHECK
  ArrayView( bool ) noexcept:
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
  LVARRAY_HOST_DEVICE constexpr
  ArrayView( BUFFER_TYPE< T > && buffer ) noexcept:
    m_dataBuffer{ std::move( buffer ) }
  {}

  /// layout describing the mapping from coordinates to memory offsets
  LayoutType m_layout;

  /// this data member contains the actual data for the array.
  BUFFER_TYPE< T > m_dataBuffer;
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
          typename LAYOUT,
          template< typename > class BUFFER_TYPE >
constexpr bool isArrayView< ArrayView< T, LAYOUT, BUFFER_TYPE > > = true;

} // namespace LvArray
