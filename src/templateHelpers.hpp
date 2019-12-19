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
 * @class is_instantiation_of
 * @brief usage :  static_assert(is_instance_of_v<LvArray::Array, LvArray::Array<int, 1, int>>)
 * @note Taken from https://cukic.co/2019/03/15/template-meta-functions-for-detecting-template-instantiation/
 */
template< template< typename ... > class Template,
          typename Type >
constexpr bool is_instantiation_of = false;

template< template< typename ... > class Template,
          typename ... Args >
constexpr bool is_instantiation_of< Template, Template< Args... > > = true;

/**
 * @class is_instance_of of
 * @brief usage :  static_assert(is_instance_of_v<LvArray::Array<int 1, int>, LvArray::Array<int, 1, int>>)
 */
template< typename T, typename U >
struct is_instance_of : std::is_same< T, U > {};

template< typename T, typename U >
constexpr bool is_instance_of_v = is_instance_of< T, U >::value;

template< class T, class U >
constexpr bool is_base_of_v = std::is_base_of< T, U >::value;

#endif // TEMPLATEHELPERS_HPP_
