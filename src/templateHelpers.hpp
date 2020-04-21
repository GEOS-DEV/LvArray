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
 * @brief return the max of the two values.
 */
template< typename U >
constexpr U const & max( const U & a, const U & b )
{ return (a > b) ? a : b; }

/**
 * @brief return the minimum of the two values.
 */
template< typename U >
constexpr U const & min( const U & a, const U & b )
{ return (a < b) ? a : b; }

/**
 * @tparam F The type of the function to apply.
 * @tparam ARGS the type of the arguments to pass to the function.
 * @brief Apply the given function to each argument.
 * @note Taken from
 * https://www.fluentcpp.com/2019/03/05/for_each_arg-applying-a-function-to-each-argument-of-a-function-in-cpp/
 */
template< typename F, typename ... ARGS >
void for_each_arg( F && f, ARGS && ... args )
{
  std::initializer_list< int > unused{ ( ( void ) f( std::forward< ARGS >( args ) ), 0 )... };
  (void) unused;
}

template< bool... B >
struct conjunction {};

template< bool Head, bool... Tail >
struct conjunction< Head, Tail... >
  : std::integral_constant< bool, Head && conjunction< Tail... >::value > {};

template< bool B >
struct conjunction< B > : std::integral_constant< bool, B > {};

/**
 * @brief usage : static_assert(is_instantiation_of_v<LvArray::Array, LvArray::Array<int, 1, int>>)
 * @note Taken from https://cukic.co/2019/03/15/template-meta-functions-for-detecting-template-instantiation/
 */
template< template< typename ... > class Template,
          typename Type >
constexpr bool is_instantiation_of = false;

template< template< typename ... > class Template,
          typename ... Args >
constexpr bool is_instantiation_of< Template, Template< Args... > > = true;

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
