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
  static_assert( std::is_same< decltype( ::max( a, b ) ), T >::value, "Should return a T." );
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
  static_assert( std::is_same< decltype( ::min( a, b ) ), T >::value, "Should return a T." );
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
LVARRAY_HOST_DEVICE inline
float abs( float const x )
{ return ::fabsf( x ); }

/// @copydoc abs( float )
LVARRAY_HOST_DEVICE inline
double abs( double const x )
{ return ::fabs( x ); }

/**
 * @copydoc abs( float )
 * @tparam T An integral type.
 */
template< typename T >
LVARRAY_HOST_DEVICE inline
std::enable_if_t< std::is_integral< T >::value, T >
abs( T const x )
{
#if defined(__CUDA_ARCH__)
  static_assert( std::is_same< decltype( ::abs( x ) ), T >::value, "Should return a T." );
  return ::abs( x );
#else
  static_assert( std::is_same< decltype( std::abs( x ) ), T >::value, "Should return a T." );
  return std::abs( x );
#endif
}

///@}

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
 * @tparam T An integral type.
 */
template< typename T >
LVARRAY_HOST_DEVICE inline
std::enable_if_t< std::is_arithmetic< T >::value, double >
sqrt( T const x )
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
 * @tparam T An integral type.
 */
template< typename T >
LVARRAY_HOST_DEVICE inline
std::enable_if_t< std::is_arithmetic< T >::value, double >
invSqrt( T const x )
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
 */
LVARRAY_HOST_DEVICE inline
float sin( float const theta )
{ return ::sinf( theta ); }

/// @copydoc sin( float )
LVARRAY_HOST_DEVICE inline
double sin( double const theta )
{ return ::sin( theta ); }

/**
 * @return The cosine of @p theta.
 * @param theta The angle in radians.
 */
LVARRAY_HOST_DEVICE inline
float cos( float const theta )
{ return ::cosf( theta ); }

/// @copydoc cos( float )
LVARRAY_HOST_DEVICE inline
double cos( double const theta )
{ return ::cos( theta ); }

/**
 * @return The tangent of @p theta.
 * @param theta The angle in radians.
 */
LVARRAY_HOST_DEVICE inline
float tan( float const theta )
{ return ::tanf( theta ); }

/// @copydoc tan( float )
LVARRAY_HOST_DEVICE inline
double tan( double const theta )
{ return ::tan( theta ); }

/**
 * @brief Compute the sine and cosine of @p theta.
 * @param theta The angle in radians.
 * @param sinTheta The sine of @p theta.
 * @param cosTheta The cosine of @p theta.
 */
LVARRAY_HOST_DEVICE inline
void sincos( float const theta, float & sinTheta, float & cosTheta )
{
#if defined(__CUDACC__)
  ::sincosf( theta, &sinTheta, &cosTheta );
#else
  sinTheta = ::sin( theta );
  cosTheta = ::cos( theta );
#endif
}

/// @copydoc sincos( float, float &, float & )
LVARRAY_HOST_DEVICE inline
void sincos( double const theta, double & sinTheta, double & cosTheta )
{
#if defined(__CUDACC__)
  ::sincos( theta, &sinTheta, &cosTheta );
#else
  sinTheta = ::sin( theta );
  cosTheta = ::cos( theta );
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
 */
LVARRAY_HOST_DEVICE inline
float asin( float const x )
{ return ::asinf( x ); }

/// @copydoc asin( float )
LVARRAY_HOST_DEVICE inline
double asin( double const x )
{ return ::asin( x ); }

/**
 * @return The arccosine of the value @p x.
 * @param x The value to get the arccosine of, must be in [-1, 1].
 */
LVARRAY_HOST_DEVICE inline
float acos( float const x )
{ return ::acosf( x ); }

/// @copydoc acos( float )
LVARRAY_HOST_DEVICE inline
double acos( double const x )
{ return ::acos( x ); }

/**
 * @return The angle corresponding to the point (x, y).
 * @param y The y coordinate.
 * @param x The x coordinate.
 */
LVARRAY_HOST_DEVICE inline
float atan2( float const y, float const x )
{ return ::atan2f( y, x ); }

/// @copydoc atan2( float, float )
LVARRAY_HOST_DEVICE inline
double atan2( double const y, double const x )
{ return ::atan2( y, x ); }

///@}

} // namespace math
} // namespace LvArray
