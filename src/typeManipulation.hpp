/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file typeManipulation.hpp
 * @brief Contains templates useful for type manipulation.
 */

#pragma once

// Source includes
#include "LvArrayConfig.hpp"
#include "Macros.hpp"

// TPL includes
#include <camp/camp.hpp>

// System includes
#include <initializer_list>
#include <utility>
#include <type_traits>

#if defined(__GLIBCXX__) && __GLIBCXX__ <= 20150626
namespace std
{
/**
 * @brief This is a fix for LC's old standard library that doesn't conform to C++14.
 */
template< typename T >
using is_trivially_default_constructible = has_trivial_default_constructor< T >;
}
#endif

/**
 * @brief Macro that expands to a static constexpr bool with one template argument
 *   which is true only if the expression is valid.
 * @param NAME The name to give the boolean variable.
 * @param T Name of the template argument in the expression.
 * @note The remaining macro arguments consist of the expresion to check for validity.
 */
#define IS_VALID_EXPRESSION( NAME, T, ... ) \
  template< typename _T > \
  struct NAME ## _impl \
  { \
private: \
    template< typename T > static constexpr auto test( int )->decltype( __VA_ARGS__, bool() ) \
    { return true; } \
    template< typename T > static constexpr auto test( ... )->bool \
    { return false; } \
public: \
    static constexpr bool value = test< _T >( 0 ); \
  }; \
  template< typename T > \
  static constexpr bool NAME = NAME ## _impl< T >::value

/**
 * @brief Macro that expands to a static constexpr bool with two template arguments
 *   which is true only if the expression is valid.
 * @param NAME The name to give the boolean variable.
 * @param T Name of the first template argument in the expression.
 * @param U Name of the second template argument in the expression.
 * @note The remaining macro arguments consist of the expresion to check for validity.
 */
#define IS_VALID_EXPRESSION_2( NAME, T, U, ... ) \
  template< typename _T, typename _U > \
  struct NAME ## _impl \
  { \
private: \
    template< typename T, typename U > static constexpr auto test( int )->decltype( __VA_ARGS__, bool() ) \
    { return true; } \
    template< typename T, typename U > static constexpr auto test( ... )->bool \
    { return false; } \
public: \
    static constexpr bool value = test< _T, _U >( 0 ); \
  }; \
  template< typename T, typename U > \
  static constexpr bool NAME = NAME ## _impl< T, U >::value

/**
 * @brief Macro that expands to a static constexpr bool templated on a type that is only true when
 *   the type has a method @p NAME which takes the given arguments. The name of the boolean variable
 *   is HasMemberFunction_ ## @p NAME.
 * @param NAME The name of the method to look for.
 * @note The remaining macro arguments is the argument list to call the method with.
 * @note The class type is available through the name CLASS.
 */
#define HAS_MEMBER_FUNCTION_NO_RTYPE( NAME, ... ) \
  IS_VALID_EXPRESSION( HasMemberFunction_ ## NAME, CLASS, std::declval< CLASS & >().NAME( __VA_ARGS__ ) )

/**
 * @brief Macro that expands to a static constexpr bool templated on a type that is only true when
 *        the type has a nested type (or type alias) called @p NAME.
 *        The name of the boolean variable is HasMemberType_ ## @p NAME.
 * @param NAME The name of the type to look for.
 */
#define HAS_MEMBER_TYPE( NAME ) \
  IS_VALID_EXPRESSION( HasMemberType_ ## NAME, CLASS, std::declval< typename CLASS::NAME >() )

/**
 * @brief Macro that expands to a static constexpr bool templated on a type that is only true when
 *        the type has a static member called @p NAME.
 *        The name of the boolean variable is HasStaticMember_ ## @p NAME.
 * @param NAME The name of the variable to look for.
 */
#define HAS_STATIC_MEMBER( NAME ) \
  IS_VALID_EXPRESSION( HasStaticMember_ ## NAME, CLASS, std::enable_if_t< !std::is_member_pointer< decltype( &CLASS::NAME ) >::value, bool >{} )

namespace LvArray
{

/**
 * @brief Contains templates for manipulating types.
 */
namespace typeManipulation
{

/**
 * @tparam F The type of the function to call.
 * @brief The recursive base case where no argument is provided.
 * @param f The function to not call.
 */
DISABLE_HD_WARNING
template< typename F >
inline constexpr LVARRAY_HOST_DEVICE
void forEachArg( F && f )
{ LVARRAY_UNUSED_VARIABLE( f ); }

/**
 * @tparam F The type of the function to call.
 * @tparam ARG The type of the first argument to pass to the function.
 * @tparam ARGS The types of the remaining arguments.
 * @brief Call the @p f with @p arg and then again with each argument in @p args .
 * @param f The function to call.
 * @param arg The first argument to call the function with.
 * @param args The rest of the arguments to call the function with.
 */
DISABLE_HD_WARNING
template< typename F, typename ARG, typename ... ARGS >
inline constexpr LVARRAY_HOST_DEVICE
void forEachArg( F && f, ARG && arg, ARGS && ... args )
{
  f( arg );
  forEachArg( std::forward< F >( f ), std::forward< ARGS >( args ) ... );
}

/**
 * @brief A struct that contains a static constexpr bool value that is true if all of BOOLS are true.
 * @tparam BOOLS A variadic pack of bool.
 * @note Not a static constexpr bool so it can be used on device.
 */
template< bool ... BOOLS >
using all_of = camp::concepts::metalib::all_of< BOOLS ... >;

/**
 * @brief A struct that contains a static constexpr bool value that is true if all of TYPES::value are true.
 * @tparam TYPES A variadic pack of types all of which define a static constexpr bool value.
 * @note Not a static constexpr bool so it can be used on device.
 */
template< typename ... TYPES >
using all_of_t = camp::concepts::metalib::all_of_t< TYPES ... >;

/**
 * @tparam TEMPLATE The template to check if @p TYPE is an instantiation of.
 * @tparam TYPE The type to check.
 * @brief Trait to detect if @tparam TYPE is an instantiation of @tparam Tempate.
 * @details Usage: @code is_instantiaton_of< std::vector, std::vector< int > > @endcode.
 */
template< template< typename ... > class TEMPLATE,
          typename TYPE >
constexpr bool is_instantiation_of = false;

/**
 * @tparam TEMPLATE The template to check.
 * @tparam ARGS The variadic pack of argument types used to instantiate TEMPLATE.
 * @brief Trait to detect if @tparam TYPE is an instantiation of @tparam Tempate.
 * @details Usage: @code is_instantiaton_of< std::vector, std::vector< int > > @endcode.
 * @note Specialization for the true case.
 */
template< template< typename ... > class TEMPLATE,
          typename ... ARGS >
constexpr bool is_instantiation_of< TEMPLATE, TEMPLATE< ARGS... > > = true;

/**
 * @brief Defines a template static constexpr bool @c HasMemberFunction_toView
 *   that is true if @c CLASS has a method @c toView().
 * @tparam CLASS The type to test.
 */
HAS_MEMBER_FUNCTION_NO_RTYPE( toView, );

/**
 * @brief Defines a template static constexpr bool @c HasMemberFunction_toViewConstSizes
 *   that is true if @c CLASS has a method @c toView().
 * @tparam CLASS The type to test.
 */
HAS_MEMBER_FUNCTION_NO_RTYPE( toViewConstSizes, );

/**
 * @brief Defines a template static constexpr bool @c HasMemberFunction_toViewConst
 *   that is true if @c CLASS has a method @c toViewConst().
 * @tparam CLASS The type to test.
 */
HAS_MEMBER_FUNCTION_NO_RTYPE( toViewConst, );

/**
 * @brief Defines a template static constexpr bool @c HasMemberFunction_toNestedView
 *   that is true if @c CLASS has a method @c toNestedView().
 * @tparam CLASS The type to test.
 */
HAS_MEMBER_FUNCTION_NO_RTYPE( toNestedView, );

/**
 * @brief Defines a template static constexpr bool @c HasMemberFunction_toNestedViewConst
 *   that is true if @c CLASS has a method @c toNestedViewConst().
 * @tparam CLASS The type to test.
 */
HAS_MEMBER_FUNCTION_NO_RTYPE( toNestedViewConst, );

namespace internal
{

/**
 * @struct GetViewType
 * @brief A helper struct used to get the view type of an object.
 * @tparam T The type to get the view type of.
 * @note By default this is T &.
 */
template< typename T, bool=HasMemberFunction_toView< T > >
struct GetViewType
{
  /// An alias for the view type.
  using type = T &;
};

/**
 * @struct GetViewType< T, true >
 * @brief A helper struct used to get the view type of an object.
 * @tparam T The type to get the view type of.
 * @note This is a specialization for objects with a toView method.
 */
template< typename T >
struct GetViewType< T, true >
{
  /// An alias for the view type.
  using type = decltype( std::declval< T & >().toView() );
};

/**
 * @struct GetViewTypeConstSizes
 * @tparam T The type to get the un-resizable view type of.
 * @brief A helper struct used to get the un-resizable view type of an object.
 * @note By default this is GetViewType< T >.
 */
template< typename T, bool=HasMemberFunction_toViewConstSizes< T > >
struct GetViewTypeConstSizes
{
  /// An alias for the view type.
  using type = typename GetViewType< T >::type;
};

/**
 * @struct GetViewTypeConstSizes< T, true >
 * @brief A helper struct used to get the un-resizable view type of an object.
 * @tparam T The type to get the un-resizable view type of T.
 * @note This is a specialization for objects with a toView method.
 */
template< typename T >
struct GetViewTypeConstSizes< T, true >
{
  /// An alias for the view type.
  using type = decltype( std::declval< T & >().toViewConstSizes() );
};

/**
 * @struct GetViewTypeConst
 * @brief A helper struct used to get the const view type of an object.
 * @tparam T The type to get the const view type of.
 * @note By default this is T const &.
 */
template< typename T, bool=HasMemberFunction_toViewConst< T > >
struct GetViewTypeConst
{
  /// An alias for the const view type.
  using type = std::remove_reference_t< T > const &;
};

/**
 * @struct GetViewTypeConst< T, true >
 * @brief A helper struct used to get the const view type of an object.
 * @tparam T The type to get the const view type of.
 * @note This is a specialization for objects with a toViewConst method.
 */
template< typename T >
struct GetViewTypeConst< T, true >
{
  /// An alias for the const view type.
  using type = decltype( std::declval< T & >().toViewConst() );
};

/**
 * @struct GetNestedViewType
 * @brief A helper struct used to get the nested view type of an object.
 * @tparam T The type to get the nested view type of.
 */
template< typename T, bool=HasMemberFunction_toNestedView< T > >
struct GetNestedViewType
{
  /// An alias for the view type.
  using type = typename GetViewType< T >::type;
};

/**
 * @struct GetNestedViewType< T, true >
 * @brief A helper struct used to get the nested view type of an object.
 * @tparam T The type to get the nested view type of.
 * @note This is a specialization for objects with a toNestedView method.
 */
template< typename T >
struct GetNestedViewType< T, true >
{
  /// An alias for the view type.
  using type = decltype( std::declval< T & >().toNestedView() ) const;
};

/**
 * @struct GetNestedViewTypeConst
 * @brief A helper struct used to get the nested view const type of an object.
 * @tparam T The type to get the nested view const type of.
 */
template< typename T, bool=HasMemberFunction_toNestedViewConst< T > >
struct GetNestedViewTypeConst
{
  /// An alias for the view type.
  using type = typename GetViewTypeConst< T >::type;
};

/**
 * @struct GetNestedViewTypeConst< T, true >
 * @brief A helper struct used to get the nested view const type of an object.
 * @tparam T The type to get the nested view const type of.
 * @note This is a specialization for objects with a toNestedViewConst method.
 */
template< typename T >
struct GetNestedViewTypeConst< T, true >
{
  /// An alias for the view type.
  using type = decltype( std::declval< T & >().toNestedViewConst() ) const;
};

} // namespace internal

/**
 * @brief An alias for the view type of T.
 * @tparam T The type to get the view type of.
 */
template< typename T >
using ViewType = typename internal::GetViewType< T >::type;

/**
 * @brief An alias for the un-resizable view type of T.
 * @tparam T The type to get the view type of.
 */
template< typename T >
using ViewTypeConstSizes = typename internal::GetViewTypeConstSizes< T >::type;

/**
 * @brief An alias for the const view type of T.
 * @tparam T The type to get the view type of.
 */
template< typename T >
using ViewTypeConst = typename internal::GetViewTypeConst< T >::type;

/**
 * @brief An alias for the nested view type of T.
 * @tparam T The type to get the nested view type of.
 */
template< typename T >
using NestedViewType = typename internal::GetNestedViewType< T >::type;

/**
 * @brief An alias for the nested const view type of T.
 * @tparam T The type to get the nested const view type of.
 */
template< typename T >
using NestedViewTypeConst = typename internal::GetNestedViewTypeConst< T >::type;


namespace internal
{

/**
 * @return Return true iff INDEX_TO_FIND == INDEX.
 * @tparam INDEX_TO_FIND The value of the index to find.
 * @tparam INDEX The index.
 */
template< camp::idx_t INDEX_TO_FIND, camp::idx_t INDEX >
constexpr bool contains( camp::idx_seq< INDEX > )
{ return INDEX_TO_FIND == INDEX; }

/**
 * @return Return true iff INDEX_TO_FIND is in { INDEX0, INDEX1, INDICES ... }.
 * @tparam INDEX_TO_FIND The value of the index to find.
 * @tparam INDEX0 The first index.
 * @tparam INDEX1 The second index.
 * @tparam INDICES The remaining indices.
 */
template< camp::idx_t INDEX_TO_FIND, camp::idx_t INDEX0, camp::idx_t INDEX1, camp::idx_t... INDICES >
constexpr bool contains( camp::idx_seq< INDEX0, INDEX1, INDICES... > )
{ return ( INDEX_TO_FIND == INDEX0 ) || contains< INDEX_TO_FIND >( camp::idx_seq< INDEX1, INDICES... > {} ); }

/**
 * @return Return true iff PERMUTATION is a valid permutation of INDICES.
 * @tparam PERMUTATION The permutation to check.
 * @tparam INDICES The indices.
 */
template< typename PERMUTATION, camp::idx_t... INDICES >
constexpr bool isValidPermutation( PERMUTATION, camp::idx_seq< INDICES... > )
{ return all_of< contains< INDICES >( PERMUTATION {} )... >::value; }

/**
 * @tparam INDICES A variadic list of indices.
 * @return The number of indices.
 */
template< camp::idx_t... INDICES >
constexpr camp::idx_t getDimension( camp::idx_seq< INDICES... > )
{ return sizeof...( INDICES ); }

} // namespace internal

/**
 * @tparam T The type for which the number of indices is requested.
 * @return the number of indices for type T
 */
template< typename T >
static constexpr camp::idx_t getDimension = internal::getDimension( T {} );

/**
 * @tparam INDICES A variadic list of indices.
 * @return The unit stride dimension, the last index in the sequence.
 */
template< camp::idx_t... INDICES >
LVARRAY_HOST_DEVICE constexpr camp::idx_t getStrideOneDimension( camp::idx_seq< INDICES... > )
{
  constexpr camp::idx_t dimension = camp::seq_at< sizeof...( INDICES ) - 1, camp::idx_seq< INDICES... > >::value;
  static_assert( dimension >= 0, "The dimension must be greater than zero." );
  static_assert( dimension < sizeof...( INDICES ), "The dimension must be less than NDIM." );
  return dimension;
}

/**
 * @tparam PERMUTATION A camp::idx_seq.
 * @return True iff @tparam PERMUTATION is a permutation of [0, N] for some N.
 */
template< typename PERMUTATION >
constexpr bool isValidPermutation( PERMUTATION )
{
  constexpr int NDIM = getDimension< PERMUTATION >;
  return internal::isValidPermutation( PERMUTATION {}, camp::make_idx_seq_t< NDIM > {} );
}

/**
 * @brief Convert a number of values of type @c U to a number of values of type @c T.
 * @tparam T The type to convert to.
 * @tparam U The type to convert from.
 * @tparam INDEX_TYPE The type of @p numU.
 * @param numU The number of @c U to convert to number of @c T.
 * @return @p numU converted to a number of types @p T.
 * @details This is a specialization for when @code sizeof( T ) <= sizeof( U ) @endcode.
 */
template< typename T, typename U, typename INDEX_TYPE >
constexpr inline LVARRAY_HOST_DEVICE
std::enable_if_t< ( sizeof( T ) <= sizeof( U ) ), INDEX_TYPE >
convertSize( INDEX_TYPE const numU )
{
  static_assert( sizeof( U ) % sizeof( T ) == 0, "T and U need to have compatable sizes." );

  return numU * sizeof( U ) / sizeof( T );
}

/**
 * @brief Convert a number of values of type @c U to a number of values of type @c T.
 * @tparam T The type to convert to.
 * @tparam U The type to convert from.
 * @tparam INDEX_TYPE The type of @p numU.
 * @param numU The number of @c U to convert to number of @c T.
 * @return @p numU converted to a number of types @p T.
 * @details This is a specialization for when @code sizeof( T ) > sizeof( U ) @endcode.
 */
template< typename T, typename U, typename INDEX_TYPE >
inline LVARRAY_HOST_DEVICE
std::enable_if_t< ( sizeof( T ) > sizeof( U ) ), INDEX_TYPE >
convertSize( INDEX_TYPE const numU )
{
  static_assert( sizeof( T ) % sizeof( U ) == 0, "T and U need to have compatable sizes." );

  INDEX_TYPE const numUPerT = sizeof( T ) / sizeof( U );
  INDEX_TYPE const remainder = numU % numUPerT;
  LVARRAY_ERROR_IF_NE( remainder, 0 );

  return numU / numUPerT;
}

/**
 * @tparam T The type of values stored in the array.
 * @tparam N The number of values in the array.
 * @struct CArray
 * @brief A wrapper around a compile time c array.
 */
template< typename T, camp::idx_t N >
struct CArray
{
  /**
   * @return Return a reference to the value at position @p i.
   * @param i The position to access.
   */
  LVARRAY_INTEL_CONSTEXPR inline LVARRAY_HOST_DEVICE
  T & operator[]( camp::idx_t const i )
  { return data[ i ]; }

  /**
   * @return Return a const reference to the value at position @p i.
   * @param i The position to access.
   */
  constexpr inline LVARRAY_HOST_DEVICE
  T const & operator[]( camp::idx_t const i ) const
  { return data[ i ]; }

  /**
   * @return Return the size of the array.
   */
  constexpr inline LVARRAY_HOST_DEVICE
  camp::idx_t size()
  { return N; }

  /// The backing c array, public so that aggregate initialization works.
  // TODO(corbett5) remove {}
  T data[ N ] = {};
};

/**
 * @tparam INDICES A variadic pack of numbers.
 * @return A CArray containing @tparam INDICES.
 */
template< camp::idx_t... INDICES >
LVARRAY_HOST_DEVICE inline constexpr
CArray< camp::idx_t, sizeof...( INDICES ) > asArray( camp::idx_seq< INDICES... > )
{ return { INDICES ... }; }

} // namespace typeManipulation
} // namespace LvArray
