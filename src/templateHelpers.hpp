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
#ifndef TEMPLATEHELPERS_HPP_
#define TEMPLATEHELPERS_HPP_

#include <initializer_list>
#include <utility>

namespace LvArray
{

/**
 * @brief Macro that expands to a static constexpr bool with one template argument
 *        which is true only if the expression @p __VA_ARGS__ is valid.
 * @param NAME The name to give the boolean variable.
 * @param T Name of the template argument in the expression.
 * @param __VA_ARGS__ The expresion to check for validity.
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
 *        which is true only if the expression @p __VA_ARGS__ is valid.
 * @param NAME The name to give the boolean variable.
 * @param T Name of the first template argument in the expression.
 * @param U Name of the second template argument in the expression.
 * @param __VA_ARGS__ The expresion to check for validity.
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
 *        the type has a method @p NAME which takes arguments @p __VA_ARGS__. The name of the boolean variable
 *        is HasMemberFunction_ ## @p NAME.
 * @param NAME The name of the method to look for.
 * @param __VA_ARGS__ The argument list to call the method with.
 * @note The class type is available through the name CLASS.
 */
#define HAS_MEMBER_FUNCTION_NO_RTYPE( NAME, ... ) \
  IS_VALID_EXPRESSION( HasMemberFunction_ ## NAME, CLASS, std::declval< CLASS >().NAME( __VA_ARGS__ ) )

/**
 * @brief @return Return the max of the two values.
 * @param a The first value.
 * @param b The second value.
 */
template< typename U >
constexpr U const & max( const U & a, const U & b )
{ return (a > b) ? a : b; }

/**
 * @brief @return Return the minimum of the two values.
 * @param a The first value.
 * @param b The second value.
 */
template< typename U >
constexpr U const & min( const U & a, const U & b )
{ return (a < b) ? a : b; }

/**
 * @tparam F The type of the function to call.
 * @tparam ARG The type of argument to pass to the function.
 * @brief Call the function @p f with @p arg.
 * @param f The function to call.
 * @param arg The argument to call the function with.
 */
DISABLE_HD_WARNING
template< typename F, typename ARG >
inline constexpr LVARRAY_HOST_DEVICE
void forEachArg( F && f, ARG && arg )
{ f( arg ); }

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
 * @tparam FIRST The first argument.
 * @tparam REST A variadic pack of the remaining arguments.
 * @brief Evaluates to true iff all of the arguments are true.
 */
template< bool FIRST, bool ... REST >
constexpr bool conjunction = FIRST && conjunction< REST ... >;

/**
 * @tparam FIRST The first argument.
 * @brief Evaluates to true iff the argument is true.
 */
template< bool FIRST >
constexpr bool conjunction< FIRST > = FIRST;

/**
 * @tparam ARGS A variadic pack of bools.
 * @struct Conjunction.
 * @brief A helper struct since host variables can't be accessed in device code.
 */
template< bool ... ARGS >
struct Conjunction
{
  /// Evaluates to true of all of ARGS are true.
  static constexpr bool value = conjunction< ARGS ... >;
};

/**
 * @tparam TEMPLATE The template to check if @p TYPE is an instantiation of.
 * @tparam TYPE The type to check.
 * @brief Trait to detect if @tparam TYPE is an instantiation of @tparam Tempate.
 * @details Usage: @code is_instantiaton_of< std::vector, std::vector< int > > @endcode.
 * @note Taken from https://cukic.co/2019/03/15/template-meta-functions-for-detecting-template-instantiation/
 */
template< template< typename ... > class TEMPLATE,
          typename TYPE >
constexpr bool is_instantiation_of = false;

/**
 * @tparam TEMPLATE The template to check.
 * @tparam ARGS The variadic pack of argument types used to instantiate TEMPLATE.
 * @brief Trait to detect if @tparam TYPE is an instantiation of @tparam Tempate.
 * @details Usage: @code is_instantiaton_of< std::vector, std::vector< int > > @endcode.
 * @note Taken from https://cukic.co/2019/03/15/template-meta-functions-for-detecting-template-instantiation/
 * @note Specialization for the true case.
 */
template< template< typename ... > class TEMPLATE,
          typename ... ARGS >
constexpr bool is_instantiation_of< TEMPLATE, TEMPLATE< ARGS... > > = true;

/**
 * @tparam ITER An iterator type.
 * @brief @return The distance between two non-random access iterators.
 * @param first The iterator to the beginning of the range.
 * @param last The iterator to the end of the range.
 */
DISABLE_HD_WARNING
template< typename ITER >
inline constexpr LVARRAY_HOST_DEVICE
typename std::iterator_traits< ITER >::difference_type
iterDistance( ITER first, ITER const last, std::input_iterator_tag )
{
  typename std::iterator_traits< ITER >::difference_type n = 0;
  while( first != last )
  {
    ++first;
    ++n;
  }

  return n;
}

/**
 * @tparam ITER An iterator type.
 * @brief @return The distance between two random access iterators.
 * @param first The iterator to the beginning of the range.
 * @param last The iterator to the end of the range.
 */
DISABLE_HD_WARNING
template< typename RandomAccessIterator >
inline constexpr LVARRAY_HOST_DEVICE
typename std::iterator_traits< RandomAccessIterator >::difference_type
iterDistance( RandomAccessIterator first, RandomAccessIterator last, std::random_access_iterator_tag )
{ return last - first; }

/**
 * @tparam ITER An iterator type.
 * @brief @return The distance between two iterators.
 * @param first The iterator to the beginning of the range.
 * @param last The iterator to the end of the range.
 */
DISABLE_HD_WARNING
template< typename ITER >
inline constexpr LVARRAY_HOST_DEVICE
typename std::iterator_traits< ITER >::difference_type
iterDistance( ITER const first, ITER const last )
{ return iterDistance( first, last, typename std::iterator_traits< ITER >::iterator_category() ); }

/**
 * @tparam CLASS The type to test.
 * @brief Defines a static constexpr bool HasMemberFunction_toView is true if @tparam CLASS has a method toView().
 */
HAS_MEMBER_FUNCTION_NO_RTYPE( toView, );

/**
 * @tparam T The type to get the view type of.
 * @struct GetViewType
 * @brief A helper struct used to get the view type of an object.
 */
template< typename T, bool=HasMemberFunction_toView< T > >
struct GetViewType
{
  /// An alias for the view type.
  using type = T;
};

/**
 * @tparam T The type to get the view type of.
 * @struct GetViewType< T, true >
 * @brief A helper struct used to get the view type of an object.
 * @note This is a specialization for objects with a toView method.
 */
template< typename T >
struct GetViewType< T, true >
{
  /// An alias for the view type.
  using type = std::remove_reference_t< decltype( std::declval< T >().toView() ) > const;
};

/**
 * @tparam CLASS The type to test.
 * @brief Defines a static constexpr bool HasMemberFunction_toViewConst is true if @tparam CLASS has a method
 * toViewConst().
 */
HAS_MEMBER_FUNCTION_NO_RTYPE( toViewConst, );

/**
 * @tparam T The type to get the view type of.
 * @struct GetViewTypeConst
 * @brief A helper struct used to get the const view type of an object.
 */
template< typename T, bool=HasMemberFunction_toViewConst< T > >
struct GetViewTypeConst
{
  /// An alias for the const view type.
  using type = T const;
};

/**
 * @tparam T The type to get the view type of.
 * @struct GetViewTypeConst< T, true >
 * @brief A helper struct used to get the const view type of an object.
 * @note This is a specialization for objects with a toViewConst method.
 */
template< typename T >
struct GetViewTypeConst< T, true >
{
  /// An alias for the const view type.
  using type = std::remove_reference_t< decltype( std::declval< T >().toViewConst() ) > const;
};

} // namespace LvArray

namespace std
{

/// Implementation of std::is_same_v, replace when upgrading to C++17.
template< typename T, typename U >
constexpr bool is_same_v = std::is_same< T, U >::value;

/// Implementation of std::is_base_of_v, replace when upgrading to C++17.
template< class T, class U >
constexpr bool is_base_of_v = std::is_base_of< T, U >::value;

} // namespace std

#endif // TEMPLATEHELPERS_HPP_
