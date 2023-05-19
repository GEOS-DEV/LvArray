/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file Constant.hpp
 * @brief Contains the implementation of Constant.
 */

#pragma once

#include "Macros.hpp"

#include <ostream>

namespace LvArray
{

/**
 * @brief A simple compile-time number with arithmetic operations that preserve compile-timeness when possible.
 * @tparam T type of value
 * @tparam v the value
 * @note this cannot be declared as an alias, as we don't want to define arithmetic operations for all integral constants
 */
template< typename T, T v = T{} >
struct Constant
{
  static constexpr T value = v;
  using value_type = T;
  using type = Constant< T, v >;
  LVARRAY_HOST_DEVICE constexpr operator   value_type() const noexcept { return value; }
  LVARRAY_HOST_DEVICE constexpr value_type operator()() const noexcept { return value; }

  template< typename U, U u >
  Constant & operator=( Constant< U, u > const & ) noexcept
  {
    static_assert( u == v, "Unable to change a compile-time constant value" );
    return *this;
  }

  template< typename U >
  Constant & operator=( U const & u ) noexcept
  {
    LVARRAY_ERROR_IF_NE_MSG( u, v, "Unable to change a compile-time constant value" );
    return *this;
  }
};

namespace internal
{

template <typename T, T v>
constexpr std::true_type is_constant( Constant< T, v > && ) { return {}; }

constexpr std::false_type is_constant( ... ) { return {}; }

}

/**
 * @brief Detection trait for Constant. Detects both exact match and derived types.
 * @tparam T the type to test
 */
template< typename T >
struct is_constant : Constant< bool, decltype( internal::is_constant( std::declval< T >() ) )::value > {};

/**
 * @brief Variable version of the detection trait for Constant.
 * @tparam T the type to test
 */
template< typename T >
inline constexpr bool is_constant_v = is_constant< T >::value;

/**
 * @brief Trait for detecting integral types, which considers all Constant specializations to be integral.
 * @tparam T type to test
 */
template< typename T >
inline constexpr bool is_integral_v = std::is_integral_v< T > || is_constant_v< T >;

/**
 * @brief Detection trait for Constant holding a particular value.
 * @tparam val the value to test against
 * @tparam T the type to test
 */
template< auto val, typename T >
struct is_constant_value : Constant< bool, false > {};

/**
 * @brief Detection trait specialization for Constant holding a particular value.
 * @tparam val the value to test against
 * @tparam T type of value
 * @tparam v the value in constant
 */
template< auto val, typename T, T v >
struct is_constant_value< val, Constant< T, v > > : Constant< bool, ( v == val ) ? true : false > {};

/**
 * @brief Variable version of the detection trait for Constant holding a particular value.
 * @tparam val the value to test against
 * @tparam T the type to test
 */
template< auto val, typename T >
inline constexpr bool is_constant_value_v = is_constant_value< val, std::decay_t< T > >::value;

template< typename T >
struct value_type
{
  using type = T;
};

template< typename T, T v >
struct value_type< Constant< T, v > >
{
  using type = T;
};

template< typename T >
using value_type_t = typename value_type< T >::type;

// Useful special compile-time constants
using _0  = Constant< int,  0 >; ///< Alias for compile-time 0 type
using _1  = Constant< int,  1 >; ///< Alias for compile-time 1 type
using _2  = Constant< int,  2 >; ///< Alias for compile-time 2 type
using _3  = Constant< int,  3 >; ///< Alias for compile-time 3 type
using _4  = Constant< int,  4 >; ///< Alias for compile-time 4 type
using _5  = Constant< int,  5 >; ///< Alias for compile-time 5 type
using _6  = Constant< int,  6 >; ///< Alias for compile-time 6 type
using _7  = Constant< int,  7 >; ///< Alias for compile-time 7 type
using _8  = Constant< int,  8 >; ///< Alias for compile-time 8 type
using _9  = Constant< int,  9 >; ///< Alias for compile-time 9 type
using _10 = Constant< int, 10 >; ///< Alias for compile-time 10 type
using _11 = Constant< int, 11 >; ///< Alias for compile-time 11 type
using _12 = Constant< int, 12 >; ///< Alias for compile-time 12 type
using _t  = Constant< bool, true >;  ///< Alias for compile-time true
using _f  = Constant< bool, false >; ///< Alias for compile-time false

template< typename T, T v >
std::ostream & operator<<( std::ostream & os, Constant< T, v > )
{
  os << '_' << v;
  return os;
}

namespace literals
{

namespace internal
{

// Parser adapted from:
// https://stackoverflow.com/questions/13869193/compile-time-type-conversion-check-constexpr-and-user-defined-literals/13869688#13869688
template< typename T, T S, char ... Chars >
struct IntegralLiteralParser;

template< typename T, T S, char C, char... Cs >
struct IntegralLiteralParser< T, S, C, Cs... >
{
  static_assert( '0' <= C && C <= '9', "Invalid constant literal" );
  static constexpr T value = IntegralLiteralParser< T, S * 10 + ( C - '0' ), Cs... >::value;
};

template< typename T, T S >
struct IntegralLiteralParser< T, S >
{
  static constexpr int value = S;
};

} // namespace internal

template< char... Chars >
auto
operator""_Ci()
{
  return Constant< int, internal::IntegralLiteralParser< int, 0, Chars... >::value >{};
}

template< char... Chars >
auto
operator""_Cl()
{
  return Constant< long, internal::IntegralLiteralParser< long, 0l, Chars... >::value >{};
}

template< char... Chars >
auto
operator""_Cll()
{
  return Constant< long long, internal::IntegralLiteralParser< long long, 0ll, Chars... >::value >{};
}

template< char... Chars >
auto
operator""_Cui()
{
  return Constant< unsigned int, internal::IntegralLiteralParser< unsigned int, 0u, Chars... >::value >{};
}

template< char... Chars >
auto
operator""_Cul()
{
  return Constant< unsigned long, internal::IntegralLiteralParser< unsigned long, 0ul, Chars... >::value >{};
}

template< char... Chars >
auto
operator""_Cull()
{
  return Constant< unsigned long long, internal::IntegralLiteralParser< unsigned long long, 0ull, Chars... >::value >{};
}

} // namespace literals

#define LVARRAY_CONSTANT_BINARY_OP( OP )      \
template< typename T, T v, typename U, U s >  \
LVARRAY_HOST_DEVICE constexpr                 \
Constant< decltype( v OP s ), ( v OP s ) >    \
operator OP( Constant< T, v >,                \
             Constant< U, s > )               \
{                                             \
  return {};                                  \
}                                             \

LVARRAY_CONSTANT_BINARY_OP( + )
LVARRAY_CONSTANT_BINARY_OP( - )
LVARRAY_CONSTANT_BINARY_OP( * )
LVARRAY_CONSTANT_BINARY_OP( / )
LVARRAY_CONSTANT_BINARY_OP( % )
LVARRAY_CONSTANT_BINARY_OP( == )
LVARRAY_CONSTANT_BINARY_OP( != )
LVARRAY_CONSTANT_BINARY_OP( > )
LVARRAY_CONSTANT_BINARY_OP( >= )
LVARRAY_CONSTANT_BINARY_OP( < )
LVARRAY_CONSTANT_BINARY_OP( <= )
LVARRAY_CONSTANT_BINARY_OP( && )
LVARRAY_CONSTANT_BINARY_OP( || )

#undef LVARRAY_CONSTANT_BINARY_OP

// Shortcuts to preserve static type where we can

template< typename T, T v, typename U,
          LVARRAY_REQUIRES( v == T{} ) >
LVARRAY_HOST_DEVICE constexpr
Constant< decltype( v * U{} ) >
operator*( Constant< T, v >, U )
{
  return {};
}

template< typename T, T v, typename U,
          LVARRAY_REQUIRES( v == T{} ) >
LVARRAY_HOST_DEVICE constexpr
Constant< decltype( U{} * v ) >
operator*( U, Constant< T, v > )
{
  return {};
}

template< typename T, T v, typename U,
          LVARRAY_REQUIRES( v == T{} ) >
LVARRAY_HOST_DEVICE constexpr
Constant< decltype( v / U{} ) >
operator/( Constant< T, v >, U )
{
  return {};
}

template< typename T, T v, typename U,
          LVARRAY_REQUIRES( v == T{} ) >
LVARRAY_HOST_DEVICE constexpr
Constant< decltype( U{} / v ) >
operator/( U, Constant< T, v > )
{
  static_assert( sizeof( U ) == 0, "Division by zero!");
  return {};
}

template< typename T, T v, typename U,
          LVARRAY_REQUIRES( v == T{} ) >
LVARRAY_HOST_DEVICE constexpr
Constant< decltype( v % U{} ) >
operator%( Constant< T, v >, U )
{
  return {};
}

template< typename T, T v, typename U,
          LVARRAY_REQUIRES( v == T{} ) >
LVARRAY_HOST_DEVICE constexpr
Constant< decltype( U{} % v ) >
operator%( U, Constant< T, v > )
{
  static_assert( sizeof( U ) == 0, "Division by zero!");
  return {};
}

} // namespace LvArray
