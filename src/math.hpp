/*
 * Copyright (c) 2020, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file math.hpp
 * @brief Contains some portable math functions.
 */

#pragma once

// Source includes
#include "LvArrayConfig.hpp"
#include "Macros.hpp"

// System includes
#include <cmath>
#include <type_traits>

namespace LvArray
{

/**
 * @brief Contains protable wrappers around cmath functions and some cuda specific functions.
 */
namespace math
{

/**
 * @name General purpose functions
 */
///@{

/**
 * @return Return the maximum of the numbers @p a and @p b.
 * @tparam T A numeric type.
 * @param a The first number.
 * @param b The second number.
 */
template< typename T >
LVARRAY_HOST_DEVICE inline constexpr
std::enable_if_t< std::is_arithmetic< T >::value, T >
max( T const a, T const b )
{
#if defined(__CUDA_ARCH__)
  return ::max( a, b );
#else
  return std::max( a, b );
#endif
}

/**
 * @return Return the minimum of the numbers @p a and @p b.
 * @tparam T A numeric type.
 * @param a The first number.
 * @param b The second number.
 */
template< typename T >
LVARRAY_HOST_DEVICE inline constexpr
std::enable_if_t< std::is_arithmetic< T >::value, T >
min( T const a, T const b )
{
#if defined(__CUDA_ARCH__)
  return ::min( a, b );
#else
  return std::min( a, b );
#endif
}

/**
 * @return The absolute value of @p x.
 * @param x The number to get the absolute value of.
 * @note This set of overloads is valid for any numeric type.
 */
template< typename T >
LVARRAY_HOST_DEVICE inline
T abs( T const x )
{
#if defined(__CUDA_ARCH__)
  return ::abs( x );
#else
  return std::abs( x );
#endif
}

///@}

template< typename T >
LVARRAY_HOST_DEVICE inline constexpr
T square( T const x )
{ return x * x; }

/**
 * @name Square root and inverse square root.
 */
///@{

/**
 * @return The square root of @p x.
 * @param x The number to get the square root of.
 * @note This set of overloads is valid for any numeric type. If @p x is not a float
 *   it is converted to a double and the return type is double.
 */
LVARRAY_HOST_DEVICE inline
float sqrt( float const x )
{ return ::sqrtf( x ); }

/**
 * @copydoc sqrt( float )
 */
template< typename T >
LVARRAY_HOST_DEVICE inline
double sqrt( T const x )
{
#if defined(__CUDA_ARCH__)
  return ::sqrt( double( x ) );
#else
  return std::sqrt( x );
#endif
}

/**
 * @return One over the square root of @p x.
 * @param x The number to get the inverse square root of.
 * @note This set of overloads is valid for any numeric type. If @p x is not a float
 *   it is converted to a double and the return type is double.
 */
LVARRAY_HOST_DEVICE inline
float invSqrt( float const x )
{
#if defined(__CUDA_ARCH__)
  return ::rsqrtf( x );
#else
  return 1 / std::sqrt( x );
#endif
}

/**
 * @copydoc invSqrt( float )
 */
template< typename T >
LVARRAY_HOST_DEVICE inline
double invSqrt( T const x )
{
#if defined(__CUDACC__)
  return ::rsqrt( double( x ) );
#else
  return 1 / std::sqrt( x );
#endif
}

///@}

/**
 * @name Trigonometric functions
 */
///@{

/**
 * @return The sine of @p theta.
 * @param theta The angle in radians.
 * @note This set of overloads is valid for any numeric type. If @p x is not a float
 *   it is converted to a double and the return type is double.
 */
LVARRAY_HOST_DEVICE inline
float sin( float const x )
{ return ::sinf( x ); }

/**
 * @copydoc sin( float )
 */
template< typename T >
LVARRAY_HOST_DEVICE inline
double sin( T const x )
{
#if defined(__CUDA_ARCH__)
  return ::sin( double( x ) );
#else
  return std::sin( x );
#endif
}

/**
 * @return The cosine of @p theta.
 * @param theta The angle in radians.
 * @note This set of overloads is valid for any numeric type. If @p x is not a float
 *   it is converted to a double and the return type is double.
 */
LVARRAY_HOST_DEVICE inline
float cos( float const x )
{ return ::cosf( x ); }

/**
 * @copydoc cos( float )
 */
template< typename T >
LVARRAY_HOST_DEVICE inline
double cos( T const x )
{
#if defined(__CUDA_ARCH__)
  return ::cos( double( x ) );
#else
  return std::cos( x );
#endif
}

/**
 * @return The tangent of @p theta.
 * @param theta The angle in radians.
 * @note This set of overloads is valid for any numeric type. If @p x is not a float
 *   it is converted to a double and the return type is double.
 */
LVARRAY_HOST_DEVICE inline
float tan( float const x )
{ return ::tanf( x ); }

/**
 * @copydoc tan( float )
 */
template< typename T >
LVARRAY_HOST_DEVICE inline
double tan( T const x )
{
#if defined(__CUDA_ARCH__)
  return ::tan( double( x ) );
#else
  return std::tan( x );
#endif
}

/**
 * @brief Compute the sine and cosine of @p theta.
 * @param theta The angle in radians.
 * @param sinTheta The sine of @p theta.
 * @param cosTheta The cosine of @p theta.
 */
template< typename T >
LVARRAY_HOST_DEVICE inline
void sincos( T const theta, T & sinTheta, T & cosTheta )
{
#if defined(__CUDA_ARCH__)
  ::sincos( theta, &sinTheta, &cosTheta );
#else
  sinTheta = std::sin( theta );
  cosTheta = std::cos( theta );
#endif
}

///@}

/**
 * @name Inverse trigonometric functions
 */
///@{

/**
 * @return The arcsine of the value @p x.
 * @param x The value to get the arcsine of, must be in [-1, 1].
 * @note This set of overloads is valid for any numeric type. If @p x is not a float
 *   it is converted to a double and the return type is double.
 */
LVARRAY_HOST_DEVICE inline
float asin( float const x )
{ return ::asinf( x ); }

/**
 * @copydoc asin( float )
 */
template< typename T >
LVARRAY_HOST_DEVICE inline
double asin( T const x )
{
#if defined(__CUDA_ARCH__)
  return ::asin( double( x ) );
#else
  return std::asin( x );
#endif
}

/**
 * @return The arccosine of the value @p x.
 * @param x The value to get the arccosine of, must be in [-1, 1].
 * @note This set of overloads is valid for any numeric type. If @p x is not a float
 *   it is converted to a double and the return type is double.
 */
LVARRAY_HOST_DEVICE inline
float acos( float const x )
{ return ::acosf( x ); }

/**
 * @copydoc acos( float )
 */
template< typename T >
LVARRAY_HOST_DEVICE inline
double acos( T const x )
{
#if defined(__CUDA_ARCH__)
  return ::acos( double( x ) );
#else
  return std::acos( x );
#endif
}

/**
 * @return The angle corresponding to the point (x, y).
 * @param y The y coordinate.
 * @param x The x coordinate.
 * @note This set of overloads is valid for any numeric type. If @p x is not a float
 *   it is converted to a double and the return type is double.
 */
LVARRAY_HOST_DEVICE inline
float atan2( float const y, float const x )
{ return ::atan2f( y, x ); }

/// @copydoc atan2( float, float )
template< typename T >
LVARRAY_HOST_DEVICE inline
double atan2( T const y, T const x )
{
#if defined(__CUDA_ARCH__)
  return ::atan2( y, x );
#else
  return std::atan2( y, x );
#endif
}

///@}

/**
 * @name Exponential functions
 */
///@{

/**
 * @return e raised to the @p x.
 * @param x The power to raise e to.
 * @note This set of overloads is valid for any numeric type. If @p x is not a float
 *   it is converted to a double and the return type is double.
 */
LVARRAY_HOST_DEVICE inline
float exp( float const x )
{ return ::expf( x ); }

/**
 * @copydoc exp( float )
 */
template< typename T >
LVARRAY_HOST_DEVICE inline
double exp( T const x )
{
#if defined(__CUDA_ARCH__)
  return ::exp( double( x ) );
#else
  return std::exp( x );
#endif
}

/**
 * @return The natural logarithm of @p x.
 * @param x The number to get the natural logarithm of.
 * @note This set of overloads is valid for any numeric type. If @p x is not a float
 *   it is converted to a double and the return type is double.
 */
LVARRAY_HOST_DEVICE inline
float log( float const x )
{ return ::logf( x ); }

/**
 * @copydoc log( float )
 */
template< typename T >
LVARRAY_HOST_DEVICE inline
double log( T const x )
{
#if defined(__CUDA_ARCH__)
  return ::log( double( x ) );
#else
  return std::log( x );
#endif
}

///@}

} // namespace math
} // namespace LvArray
