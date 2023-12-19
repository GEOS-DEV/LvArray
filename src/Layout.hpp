/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file Layout.hpp
 * @brief Contains the implementation of Layout.
 */

#pragma once

#include "Macros.hpp"
#include "Constant.hpp"
#include "tupleManipulation.hpp"
#include "typeManipulation.hpp"

#include "RAJA/util/types.hpp"
#include "RAJA/util/Permutations.hpp"

#include <ostream>

namespace LvArray
{

/**
 * @brief Type used to represent multidimensional array extents
 */
template< typename ... Ts >
using Extent = camp::tuple< Ts... >;

/**
 * @brief Make function for extent
 * @tparam Ts argument type list
 * @param args variadic arguments
 * @return the constructed extent object
 */
template< typename ... Ts >
LVARRAY_HOST_DEVICE
constexpr auto makeExtent( Ts && ... args )
{
  return makeTuple( LVARRAY_FWD( args )... );
}

/**
 * @brief Make function for extent from a pointer
 * @tparam T type of values
 * @param vals extent values
 * @return the constructed extent object
 */
template< camp::idx_t N, typename T >
LVARRAY_HOST_DEVICE
constexpr auto makeExtentFromPtr( T const * const vals )
{
  return makeTupleFromPtr< N >( vals );
}

/**
 * @brief Type used to represent multidimensional array strides
 */
template< typename ... Ts >
using Stride = camp::tuple< Ts... >;

/**
 * @brief Make function for stride
 * @tparam Ts argument type list
 * @param args variadic arguments
 * @return the constructed stride object
 */
template< typename ... Ts >
LVARRAY_HOST_DEVICE
constexpr auto makeStride( Ts && ... args )
{
  return makeTuple( LVARRAY_FWD( args )... );
}

/**
 * @brief Type used to represent multidimensional array coordinate
 */
template< typename ... Ts >
using Coord = camp::tuple< Ts... >;

/**
 * @brief Make function for coordinate
 * @tparam Ts argument type list
 * @param args variadic arguments
 * @return the constructed coordinate object
 */
template< typename ... Ts >
LVARRAY_HOST_DEVICE
constexpr auto makeCoord( Ts && ... args )
{
  return makeTuple( LVARRAY_FWD( args )... );
}

/**
 * @brief Special type to indicate slicing a layout dimension.
 * @note Inheriting from _0 is convenient because:
 *       - Automatically identified as a compile-time constant int
 *       - Automatically correctly treated in offset computation
 */
struct _ : _0 {};

template< typename T >
inline constexpr bool is_slicer = std::is_same_v< camp::decay< T >, _ >;

/**
 * @brief Shortcut to define an all-dynamic extent type (default for arrays)
 */
template< int NDIM, typename INDEX >
using DynamicExtent = decltype( tupleManipulation::repeat< NDIM >( INDEX{} ) );

struct LayoutLeft{};
struct LayoutRight{};

namespace internal
{

/**
 * @brief Compute the product of all tuple elements.
 * @tparam T type of tuple
 * @param t the tuple
 * @return the product
 */
template< typename T >
LVARRAY_HOST_DEVICE constexpr
auto
product( T && t )
{
  return tupleManipulation::apply( LVARRAY_FWD( t ), []( auto && ... vs ){ return ( _1{} * ... * vs ); } );
}

/**
 * @brief Compute the inner product of two tuples
 * @tparam T1 type of first tuple
 * @tparam T2 type of second tuple
 * @param t1 the first tuple
 * @param t2 the second tuple
 * @return the inner product
 */
template< typename T1, typename T2 >
LVARRAY_HOST_DEVICE constexpr
auto
innerProduct( T1 && t1, T2 && t2 )
{
  return tupleManipulation::transformApply2( LVARRAY_FWD( t1 ), LVARRAY_FWD( t2 ),
                                             []( auto && v1, auto && v2 ){ return v1 * v2; },
                                             []( auto && ... vs ){ return ( _0{} + ... + vs ); } );
}

/**
 * @brief Return only elements of tuple that correspond to _ in the mask
 * @tparam T input tuple type
 * @tparam U mask type (also a tuple)
 * @param t the input tuple to slice
 * @param mask the slicing mask
 * @return the sliced (filtered) tuple
 */
template< typename T, typename U >
LVARRAY_HOST_DEVICE constexpr
auto
slice( T && t, U && mask )
{
  return tupleManipulation::filter( LVARRAY_FWD( t ), LVARRAY_FWD( mask ), []( auto && e, auto && m )
  {
    if constexpr( is_slicer< decltype( m ) > ) return makeTuple( LVARRAY_FWD( e ) );
    else return camp::tuple<>{};
  } );
}

template< typename EXTENT >
LVARRAY_HOST_DEVICE constexpr
auto
foldExtent( LayoutLeft, EXTENT const & extent )
{
  auto scanner = []( auto curr, auto e )
  {
    return makeTuple( tupleManipulation::append( camp::get< 0 >( curr ), camp::get< 1 >( curr ) ), camp::get< 1 >( curr ) * e );
  };
  auto init = Constant< value_type_t< camp::tuple_element_t< 0, EXTENT > >, 1 >{};
  return tupleManipulation::foldLeft( extent, makeTuple( makeStride(), init ), scanner );
}

template< typename EXTENT >
LVARRAY_HOST_DEVICE constexpr
auto
foldExtent( LayoutRight, EXTENT const & extent )
{
  auto scanner = []( auto e, auto curr )
  {
    return makeTuple( tupleManipulation::prepend( camp::get< 0 >( curr ), camp::get< 1 >( curr ) ), camp::get< 1 >( curr ) * e );
  };
  auto init = Constant< value_type_t< camp::tuple_element_t< camp::tsize_v< EXTENT > - 1, EXTENT > >, 1 >{};
  return tupleManipulation::foldRight( extent, makeTuple( makeStride(), init ), scanner );
}

} // namespace internal

/**
 * @brief Computes a compact stride given an extent and permutation
 * @tparam EXTENT type of extent
 * @tparam PERM type of permutation sequence
 * @param extent the extent
 * @param perm the permutation
 * @return the computed stride object
 *
 * Packed stride is defined assuming a compact right layout:
 * rightmost dimension is assigned stride _1, each next stride in
 * right-to-left order is a product of previous stride and extent.
 */
template< typename EXTENT, typename LAYOUT_TAG, typename PERMUTATION >
LVARRAY_HOST_DEVICE
auto
makeCompactStride( EXTENT const & extent, LAYOUT_TAG tag, PERMUTATION perm = camp::idx_seq_from_t< EXTENT >{} )
{
  static_assert( typeManipulation::getDimension< PERMUTATION > == camp::tsize_v< EXTENT >, "Permutation size must match the extent." );
  static_assert( typeManipulation::isValidPermutation( PERMUTATION{} ), "The permutation must be valid." );
  auto folded = internal::foldExtent( tag, tupleManipulation::select( extent, perm ) );
  return tupleManipulation::select( camp::get< 0 >( folded ), RAJA::invert_permutation< PERMUTATION >{} );
}

template< typename EXTENT, typename LAYOUT_TAG, typename PERM = camp::idx_seq_from_t< EXTENT > >
using CompactStride = decltype( makeCompactStride( EXTENT{}, LAYOUT_TAG{}, PERM{} ) );

template< typename EXTENT, typename STRIDE >
struct Layout;

template< typename EXTENT, typename STRIDE >
LVARRAY_HOST_DEVICE
auto
makeLayout( EXTENT && extent, STRIDE && stride )
{
  return Layout< camp::decay< EXTENT >, camp::decay< STRIDE > >( LVARRAY_FWD( extent ), LVARRAY_FWD( stride ) );
}

/**
 * @brief Layout defines a mapping from a multidimensional coordinate to an offset in memory
 * @tparam EXTENT type of extent
 * @tparam STRIDE type of stride
 */
template< typename EXTENT, typename STRIDE >
struct Layout : private camp::tuple< EXTENT, STRIDE > // EBO for static layouts
{
public:

  using ExtentType = EXTENT;
  using StrideType = STRIDE;
  using Base = camp::tuple< ExtentType, StrideType >;

  // Convenience type to denote indices
  using IndexType = tupleManipulation::common_type_t< ExtentType >;

  static_assert( camp::tsize_v< ExtentType > == camp::tsize_v< StrideType >, "Extent and Stride must have matching number of dims" );

  /// Number of dimensions (rank) of the layout
  static constexpr int NDIM = camp::tsize_v< ExtentType >;

  struct USDHelper
  {
    static constexpr int value = tupleManipulation::find( StrideType{}, []( auto s ){ return is_constant_value< 1, decltype( s ) >{}; } );
  };

  /// Unit-stride dimension index.
  static constexpr int USD = ( USDHelper::value < NDIM ) ? USDHelper::value : -1;

  LVARRAY_HOST_DEVICE constexpr
  explicit Layout( ExtentType const & extent = ExtentType{}, StrideType const & stride = StrideType{} ) noexcept : Base( extent, stride ) {}

  template< typename RExtent, typename RStride,
            LVARRAY_REQUIRES( Layout< RExtent, RStride >::NDIM == NDIM ) >
  LVARRAY_HOST_DEVICE constexpr
  explicit Layout( Layout< RExtent, RStride > const & rhs ) noexcept : Base( rhs.extent(), rhs.stride() ) {}

  template< typename RExtent, typename RStride,
            LVARRAY_REQUIRES( Layout< RExtent, RStride >::NDIM == NDIM )>
  LVARRAY_HOST_DEVICE constexpr
  Layout &
  operator=( Layout< RExtent, RStride > const & rhs ) noexcept
  {
    this->extent() = rhs.extent();
    this->stride() = rhs.stride();
    return *this;
  }

  LVARRAY_HOST_DEVICE constexpr
  ExtentType const &
  extent() const { return camp::get< 0 >( static_cast< Base const & >( *this ) ); }

  LVARRAY_HOST_DEVICE constexpr
  ExtentType &
  extent() { return camp::get< 0 >( static_cast< Base & >( *this ) ); }

  LVARRAY_HOST_DEVICE constexpr
  StrideType const &
  stride() const { return camp::get< 1 >( static_cast< Base const & >( *this ) ); }

  LVARRAY_HOST_DEVICE constexpr
  StrideType &
  stride() { return camp::get< 1 >( static_cast< Base & >( *this ) ); }

private:

  struct CoordCheck
  {
    template< typename C, typename E >
    auto operator()( C const & c, E const & e )
    {
      if constexpr( is_slicer< C > ) return true;
      else return c < e;
    }
  };

public:

  /**
   * @brief Compute an offset at given coordinates.
   * @tparam Ts types of coordinates
   * @param coords the target coordinates, may include slicing operators (_)c
   * @return the offset in the codomain
   */
  template< typename ... Ts >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK
  auto
  operator()( Coord< Ts... > const & coords ) const noexcept
  {
    // We allow specifying only leading coordinates and right-pad to NDIM with _{}
    static_assert( sizeof...( Ts ) <= NDIM, "The number of coordinates must not exceed the number of dimensions." );
    auto coordFull = tupleManipulation::append< NDIM - sizeof...( Ts ) >( coords, _{} );
#ifdef LVARRAY_BOUNDS_CHECK
    LVARRAY_ERROR_IF( !tupleManipulation::allOf2( coordFull, extent(), CoordCheck{} ), "Invalid coordinates." );
#endif
    return internal::innerProduct( stride(), coordFull );
  }

  /**
   * @brief Compute an offset at given coordinates.
   * @tparam Ts types of coordinates
   * @param coords the target coordinates, may include slicing operators (_)
   * @return the offset in the codomain
   */
  template< typename ... Ts,
            LVARRAY_REQUIRES( typeManipulation::all_of< ( is_integral_v< Ts > )... >::value ) >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK
  auto
  operator()( Ts ... coords ) const noexcept
  {
    return operator()( makeCoord( coords... ) );
  }

  /**
   * @brief Produce a sliced layout at given coordinates.
   * @tparam Ts types of coordinates
   * @param coord the target coordinates, may include slicing operators (_)
   * @return the sliced layout
   * @note if number of coordinates is less than layout rank (NDIM),
   *       they are right-padded with slicing operators (_)
   *
   * If none of the coordinates are a slicing operator (_), the result
   * is a rank-0 layout that describes a single value. Otherwise, the
   * resulting layout describes a (possibly non-contiguous) slice of
   * the original layout's (co)domain.
   */
  template< typename ... Ts >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK
  auto
  slice( Coord< Ts... > const & coords ) const noexcept
  {
    static_assert( sizeof...( Ts ) <= NDIM, "The number of coordinates must not exceed the number of dimensions." );
    auto coordFull = tupleManipulation::append< NDIM - sizeof...( Ts ) >( coords, _{} );
#ifdef LVARRAY_BOUNDS_CHECK
    LVARRAY_ERROR_IF( !tupleManipulation::allOf2( coordFull, extent(), CoordCheck{} ), "Invalid coordinates." );
#endif
    return makeLayout( internal::slice( extent(), coordFull ), internal::slice( stride(), coordFull ) );
  }

  /**
   * @brief Produce a sliced layout at given coordinates.
   * @tparam Ts types of coordinates
   * @param coord the target coordinates, may include slicing operators (_)
   * @return the sliced layout
   * @note if number of coordinates is less than layout rank (NDIM),
   *       they are right-padded with slicing operators (_)
   *
   * If none of the coordinates are a slicing operator (_), the result
   * is a rank-0 layout that describes a single value. Otherwise, the
   * resulting layout describes a (possibly non-contiguous) slice of
   * the original layout's (co)domain.
   */
  template< typename ... Ts,
            LVARRAY_REQUIRES( typeManipulation::all_of< ( is_integral_v< Ts > )... >::value ) >
  LVARRAY_HOST_DEVICE CONSTEXPR_WITHOUT_BOUNDS_CHECK
  auto
  slice( Ts ... coords ) const noexcept
  {
    return slice( makeCoord( coords... ) );
  }

  /**
   * @brief @return Size of the layout's domain, i.e. the product of its extents.
   */
  LVARRAY_HOST_DEVICE constexpr
  auto
  size() const noexcept
  {
    return internal::product( extent() );
  }

  /**
   * @brief Convert the layout to one representing N times more values.
   * @tparam N the downcast factor
   * @return the new layout
   * @note This is useful when casting arrays to a smaller data type.
   */
  template< int N >
  LVARRAY_HOST_DEVICE constexpr
  auto
  downcast() const noexcept
  {
    auto [newExtent, newStride ] = tupleManipulation::unzip2( tupleManipulation::transform2( extent(), stride(), []( auto e, auto s )
    {
      if constexpr( is_constant_value_v< 1, decltype( s ) > ) return makeTuple( e * Constant< int, N >{}, s );
      else                                                    return makeTuple( e, s * Constant< int, N >{} );
    } ) );
    return makeLayout( newExtent, newStride );
  }

  /**
   * @brief Convert the layout to one representing N times fewer values.
   * @tparam N the downcast factor
   * @return the new layout
   * @note This is useful when casting arrays to a larger data type.
   */
  template< int N >
  LVARRAY_HOST_DEVICE constexpr
  auto
  upcast() const noexcept
  {
    auto [newExtent, newStride] = tupleManipulation::unzip2( tupleManipulation::transform2( extent(), stride(), []( auto e, auto s )
    {
      if constexpr( is_constant_value_v< 1, decltype( s ) > ) return makeTuple( e / Constant< int, N >{}, s );
      else                                                    return makeTuple( e, s / Constant< int, N >{} );
    } ) );
    return makeLayout( newExtent, newStride );
  }

  /**
   * @brief Equality comparison operator for layouts.
   * @tparam RExtent right-hand side layout's extent type
   * @tparam RStride right-hand side layout's stride type
   * @param lhs left-hand side layout
   * @param rhs right-hand side layout
   * @return whether the layout are equal.
   *
   * Layouts are considered equal when their extents and strides consist of equal values,
   * regardless of data types (i.e. they represent equivalent linear functions of coordinates).
   */
  template< typename RExtent, typename RStride >
  LVARRAY_HOST_DEVICE constexpr
  friend bool operator==( Layout const & lhs, Layout< RExtent, RStride > const & rhs )
  {
    return tupleManipulation::allOf2( lhs.extent(), rhs.extent(), tupleManipulation::cmp_eq{} )
        && tupleManipulation::allOf2( lhs.stride(), rhs.stride(), tupleManipulation::cmp_eq{} );
  }

  /**
   * @brief Non-equality comparison operator for layouts.
   * @tparam RExtent right-hand side layout's extent type
   * @tparam RStride right-hand side layout's stride type
   * @param lhs left-hand side layout
   * @param rhs right-hand side layout
   * @return whether the layout are not equal.
   */
  template< typename RExtent, typename RStride >
  LVARRAY_HOST_DEVICE constexpr
  friend bool operator!=( Layout const & lhs, Layout< RExtent, RStride > const & rhs )
  {
    return !( lhs == rhs );
  }

  /**
   * @brief Stream output operator for layouts.
   * @param os the output stream
   * @param layout the layout
   * @return @p os
   */
  friend std::ostream & operator<<( std::ostream & os, Layout const & layout )
  {
    // Unqualified lookup via ADL does not work for camp::tuple because its
    // operator<< is defined outside its namespace. This trips up compilers.
    // return os << layout.extent() << ':' << layout.stride();
    ::operator<<(os, layout.extent());
    os << ':';
    ::operator<<(os, layout.stride());
    return os;
  }

  /**
   * @name Resizing API.
   *
   * Generally a layout is just a container of an arbitrary extent and stride
   * and does not enforce any particular relationship between the two, nor any
   * specific class invariant (hence it's an struct with write access to both).
   * Often however it is convenient to assume that layout is compact and strides
   * represent a product fold of extents in some order, and recompute the strides
   * correspondingly as extent is updated. The resizing methods below aid in this.
   */
  ///@{

  template< typename ... DIMS, typename LAYOUT_TAG, typename PERMUTATION >
  LVARRAY_HOST_DEVICE
  void resize( Extent< DIMS ... > const & newExtent, LAYOUT_TAG layoutTag, PERMUTATION permutation )
  {
    static_assert( sizeof ... ( DIMS ) == NDIM, "The rank of new extents must match the layout rank." );
    tupleManipulation::forEach( newExtent, []( auto const newDim ) { LVARRAY_ERROR_IF_LT( newDim, 0 ); } );

    extent() = newExtent;
    stride() = makeCompactStride( extent(), layoutTag, permutation );
  }

  template< typename DIMS_TYPE, typename LAYOUT_TAG, typename PERMUTATION >
  LVARRAY_HOST_DEVICE
  void resize( DIMS_TYPE const * const dims, LAYOUT_TAG layoutTag, PERMUTATION permutation  )
  {
    resize( makeExtentFromPtr< NDIM >( dims ), layoutTag, permutation );
  }

  template< int INDEX, int ... INDICES, typename ... DIMS, typename LAYOUT_TAG, typename PERMUTATION >
  LVARRAY_HOST_DEVICE
  void resize( Extent< DIMS ... > const & newExtent, LAYOUT_TAG layoutTag, PERMUTATION permutation )
  {
    static_assert( sizeof...( INDICES ) + 1 <= NDIM, "The number of dimension indices must match the layout rank." );
    static_assert( sizeof...( INDICES ) + 1 == sizeof...( DIMS ), "The number of dimension indices must match the number of arguments." );
    static_assert( typeManipulation::all_of< ( 0 <= INDEX ), ( 0 <= INDICES ) ... >::value, "Dimension indices must all be positive." );
    static_assert( typeManipulation::all_of< ( INDEX < NDIM), ( INDICES < NDIM ) ... >::value, "Dimension indices must all be less than layout rank." );
    tupleManipulation::forEach( newExtent, []( auto const newDim ) { LVARRAY_ERROR_IF_LT( newDim, 0 ); } );

    tupleManipulation::selectRefs( extent(), camp::idx_seq< INDEX, INDICES... >{} ) = newExtent;
    stride() = makeCompactStride( extent(), layoutTag, permutation );
  }

  ///@}
};

/**
 * @brief Detection trait for Layout.
 * @tparam T the type to test
 */
template< typename T >
struct is_layout : Constant< bool, false > {};

/**
 * @brief Specialization of detection trait for Layout.
 * @tparam EXTENT extent type
 * @tparam STRIDE stride type
 */
template< typename EXTENT, typename STRIDE >
struct is_layout< Layout< EXTENT, STRIDE > > : Constant< bool, true > {};

/**
 * @brief Variable version of detection trait for Layout.
 * @tparam T the type to test
 */
template< typename T >
inline constexpr bool is_layout_v = is_layout< camp::decay< T > >::value;

// Useful default types

/// LvArray defaults to right layouts (last index is contiguous by default)
using LayoutDefault = LayoutRight;

/// Default dynamic layout type
template< int NDIM, typename INDEX_TYPE, typename PERM = camp::make_idx_seq_t< NDIM > >
using DynamicLayout = Layout< DynamicExtent< NDIM, INDEX_TYPE >, CompactStride< DynamicExtent< NDIM, INDEX_TYPE >, LayoutDefault, PERM > >;

/// Default 1D dynamic layout type
template< typename INDEX_TYPE >
using DynamicLayout1D = DynamicLayout< 1, INDEX_TYPE >;

/// Dynamic layout type without a contiguous dimension
template< int NDIM, typename INDEX_TYPE >
using NonContiguousLayout = Layout< DynamicExtent< NDIM, INDEX_TYPE >, DynamicExtent< NDIM, INDEX_TYPE > >;

/// Default static layout type
template< typename INDEX_TYPE, INDEX_TYPE ... DIMS >
using StaticLayout = Layout< Extent< Constant< INDEX_TYPE, DIMS >... >, CompactStride< Extent< Constant< INDEX_TYPE, DIMS >... >, LayoutDefault > >;

template< typename EXTENT >
LVARRAY_HOST_DEVICE
auto
makeLayout( EXTENT && extent )
{
  auto stride = makeCompactStride( extent, LayoutDefault{}, camp::idx_seq_from_t< EXTENT >{} );
  return makeLayout( LVARRAY_FWD( extent ), LVARRAY_FWD( stride ) );
}

} // namespace LvArray
