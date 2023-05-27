/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
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

#if defined( LVARRAY_USE_CUDA )
  #include <cuda_fp16.h>
#endif

namespace LvArray
{

/**
 * @brief Contains protable wrappers around cmath functions and some cuda specific functions.
 */
namespace math
{

namespace internal
{

/**
 * @brief Convert @p u to type @tparam T.
 * @tparam T The type to convert to.
 * @tparam U The type to convert from.
 * @param u The value to convert.
 * @note No checking is done to ensure the conversion is safe. Ie you could convert -1 to uint.
 * @return @p u converted to @tparam T.
 */
template< typename T, typename U >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE constexpr
T convert( T const, U const u )
{ return u; }

/**
 * @brief Return the number of values stored in @tparam T, by default this is 1.
 * @tparam T The type to query.
 * @return The number of values stored in @tparam T, by default this is 1.
 */
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE constexpr
int numValues( T const )
{ return 1; }

/**
 * @brief The type of a single value of type @c T.
 * @tparam T The type to query.
 */
template< typename T >
struct SingleType
{
  /// An alias for T.
  using type = T;
};

/**
 * @return @p x.
 * @tparam T The type of @p x.
 * @param x The value to return.
 */
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE constexpr
SingleType< T > getFirst( T const x )
{ return x; }

/**
 * @return @p x.
 * @tparam T The type of @p x.
 * @param x The value to return.
 */
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE constexpr
SingleType< T > getSecond( T const x )
{ return x; }

/**
 * @return 1 if @p x is less than @p y, else 0.
 * @tparam T The type of @p x and @p y.
 * @param x The first value.
 * @param y The second value.
 */
template< typename T >
LVARRAY_HOST_DEVICE inline constexpr
T lessThan( T const x, T const y )
{ return __hlt( x, y ); }

#if defined( LVARRAY_USE_CUDA )
/**
 * @brief Convert @p u to @c __half.
 * @tparam U The type to convert from.
 * @param u The value to convert.
 * @note No checking is done to ensure the conversion is safe.
 * @return @p u converted to @c __half.
 */
template< typename U >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE constexpr
__half convert( __half const, U const u )
{ return __float2half_rn( u ); }

/**
 * @brief Convert @p u to @c __half2, filling both halves of the returned value with @p u converted to @c __half.
 * @tparam U The type to convert from.
 * @param u The value to convert.
 * @note No checking is done to ensure the conversion is safe.
 * @return A @c __half2 with both halves having value @p u converted to @c __half.
 */
template< typename U >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE constexpr
__half2 convert( __half2 const, U const u )
{ return __float2half2_rn( u ); }

/**
 * @brief Convert @p u to @c __half2, filling both halves of the returned value with @p u.
 * @param u The value to convert.
 * @return A @c __half2 with both halves having value @p u.
 */
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
__half2 convert( __half2 const, __half const u )
{
#if defined( LVARRAY_DEVICE_COMPILE )
  return __half2half2( u );
#else
  return __float2half2_rn( u );
#endif
}

/**
 * @brief Convert @p u, @p v to @c __half2.
 * @tparam U The first type to convert from.
 * @tparam V The second type to convert from.
 * @param u The first value to convert.
 * @param v The second value to convert.
 * @note No checking is done to ensure the conversion is safe.
 * @return A @c __half2 containing @p u as the first value and @p v as the second.
 */
template< typename U, typename V >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE constexpr
__half2 convert( __half2 const, U const u, V const v )
{ return __floats2half2_rn( u, v ); }

/**
 * @brief Convert @p u, @p v to @c __half2.
 * @param u The first value to convert.
 * @param v The second value to convert.
 * @return A @c __half2 containing @p u as the first value and @p v as the second.
 */
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
__half2 convert( __half2 const, __half const u, __half const v )
{
#if defined( LVARRAY_DEVICE_COMPILE )
  return __halves2half2( u, v );
#else
  return __floats2half2_rn( u, v );
#endif
}

/**
 * @brief Return the number of values stored in a @c __half2, which is 2.
 * @return The number of values stored in a @c __half2, which is 2.
 */
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE constexpr
int numValues( __half2 const & )
{ return 2; }

/**
 * @brief The type of a single value of type @c __half2.
 */
template<>
struct SingleType< __half2 >
{
  /// An alias for __half.
  using type = __half;
};

/**
 * @return The fist @c __half in @p x.
 * @param x The value to query.
 */
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half getFirst( __half2 const x )
{ return __low2half( x ); }

/**
 * @return The second @c __half in @p x.
 * @param x The value to query.
 */
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half getSecond( __half2 const x )
{ return __high2half( x ); }

#endif

#if defined( LVARRAY_USE_DEVICE )
/**
 * @return 1 if @p x is less than @p y, else 0.
 * @param x The first value.
 * @param y The second value.
 */
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half lessThan( __half const x, __half const y )
{ return __hlt( x, y ); }

/**
 * @return 1 if @p x is less than @p y, else 0, performed elementwise accross the two values.
 * @param x The first value.
 * @param y The second value.
 */
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half2 lessThan( __half2 const x, __half2 const y )
{ return __hlt2( x, y ); }
#endif

} // namespace internal

/**
 * @name General purpose functions
 */
///@{

/**
 * @brief Return the number of values stored in type @tparam T.
 * @param T The type to query.
 * @return The number of values stored in type @tparam T.
 */
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE constexpr
int numValues()
{ return internal::numValues( T() ); }

/**
 * @brief The type of a single value of type @c T.
 * @tparam T The type to query.
 */
template< typename T >
using SingleType = typename internal::SingleType< T >::type;

/**
 * @brief Convert @p u to type @tparam T.
 * @tparam T The type to convert to.
 * @tparam U The type to convert from.
 * @param u The value to convert.
 * @note No checking is done to ensure the conversion is safe. Ie you could convert -1 to uint.
 * @return @p u converted to @tparam T.
 */
template< typename T, typename U >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE constexpr
T convert( U const u )
{ return internal::convert( T(), u ); }

/**
 * @brief Convert @p u and @p v to a dual type @tparam T.
 * @tparam T The type to convert to, must hold two values such as __half2.
 * @tparam U The first type to convert from.
 * @tparam U The second type to convert from.
 * @param u The first value to convert.
 * @param v The second value to convert.
 * @note No checking is done to ensure the conversion is safe. Ie you could convert -1 to uint.
 * @return @p u, @p v converted to @tparam T.
 */
template< typename T, typename U, typename V >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE constexpr
T convert( U const u, V const v )
{ return internal::convert( T(), u, v ); }

/**
 * @return The fist value in @p x.
 * @tparam T The type of @p x.
 * @param x The value to query.
 * @note If @code numValues< T >() == 1 @endcode then @p x is returned.
 */
template< typename T >
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
SingleType< T > getFirst( T const x )
{ return internal::getFirst( x ); }

/**
 * @return The second value in @p x.
 * @tparam T The type of @p x.
 * @param x The value to query.
 * @note If @code numValues< T >() == 1 @endcode then @p x is returned.
 */
template< typename T >
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
SingleType< T > getSecond( T const x )
{ return internal::getSecond( x ); }

/**
 * @return Return the maximum of the numbers @p a and @p b.
 * @tparam T A numeric type.
 * @param a The first number.
 * @param b The second number.
 */
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE constexpr
std::enable_if_t< std::is_arithmetic< T >::value, T >
max( T const a, T const b )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::max( a, b );
#else
  return std::max( a, b );
#endif
}

#if defined( LVARRAY_USE_DEVICE )

/// @copydoc max( T, T )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half max( __half const a, __half const b )
{
#if defined(LVARRAY_USE_CUDA) && CUDART_VERSION > 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
  return __hmax( a, b );
#elif defined(LVARRAY_USE_HIP)
  return __hgt( a, b ) ? a : b;
#else
  return a > b ? a : b;
#endif
}

/// @copydoc max( T, T )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half2 max( __half2 const a, __half2 const b )
{
#if defined(LVARRAY_USE_CUDA) && CUDART_VERSION > 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
  return __hmax2( a, b );
#else
  __half2 const aFactor = __hge2( a, b );
  __half2 const bFactor = convert< __half2 >( 1 ) - aFactor;
  return a * aFactor + bFactor * b;
#endif
}

#endif


/**
 * @return Return the minimum of the numbers @p a and @p b.
 * @tparam T A numeric type.
 * @param a The first number.
 * @param b The second number.
 */
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE constexpr
std::enable_if_t< std::is_arithmetic< T >::value, T >
min( T const a, T const b )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::min( a, b );
#else
  return std::min( a, b );
#endif
}

#if defined( LVARRAY_USE_CUDA )

/// @copydoc min( T, T )
LVARRAY_DEVICE
LVARRAY_FORCE_INLINE
__half min( __half const a, __half const b )
{
#if CUDART_VERSION > 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
  return __hmin( a, b );
#else
  return a < b ? a : b;
#endif
}

/// @copydoc min( T, T )
LVARRAY_DEVICE
LVARRAY_FORCE_INLINE
__half2 min( __half2 const a, __half2 const b )
{
#if CUDART_VERSION > 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
  return __hmin2( a, b );
#else
  __half2 const aFactor = __hle2( a, b );
  __half2 const bFactor = convert< __half2 >( 1 ) - aFactor;
  return a * aFactor + bFactor * b;
#endif
}

#endif

/**
 * @return The absolute value of @p x.
 * @param x The number to get the absolute value of.
 * @note This set of overloads is valid for any numeric type.
 */
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE constexpr
T abs( T const x )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::abs( x );
#else
  return std::abs( x );
#endif
}

#if defined( LVARRAY_USE_DEVICE )

/// @copydoc abs( T )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half abs( __half const x )
{
#if CUDART_VERSION > 11000
  return __habs( x );
#else
  return x > __half( 0 ) ? x : -x;
#endif
}

/// @copydoc abs( T )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half2 abs( __half2 const x )
{
#if CUDART_VERSION > 11000
  return __habs2( x );
#else
  return x - __hle2( x, convert< __half2 >( 0 ) ) * ( x + x );
#endif
}

#endif

/**
 * @return @code x * x @endcode.
 * @tparam T The typeof @p x.
 * @param x The value to square.
 */
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE constexpr
T square( T const x )
{ return x * x; }

///@}

/**
 * @name Square root and inverse square root.
 */
///@{

/**
 * @return The square root of @p x.
 * @param x The number to get the square root of.
 * @note This set of overloads is valid for any numeric type. If @p x is integral it is converted to @c double
 *   and the return type is @c double.
 */
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
float sqrt( float const x )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::sqrtf( x );
#else
  return std::sqrt( x );
#endif
}

/// @copydoc sqrt( float )
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
double sqrt( T const x )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::sqrt( double( x ) );
#else
  return std::sqrt( x );
#endif
}

#if defined( LVARRAY_USE_DEVICE )

/// @copydoc sqrt( float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half sqrt( __half const x )
{ return ::hsqrt( x ); }

/// @copydoc sqrt( float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half2 sqrt( __half2 const x )
{ return ::h2sqrt( x ); }

#endif

/**
 * @return One over the square root of @p x.
 * @param x The number to get the inverse square root of.
 * @note This set of overloads is valid for any numeric type. If @p x is integral it is converted to @c double
 *   and the return type is double.
 */
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
float invSqrt( float const x )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::rsqrtf( x );
#else
  return 1 / std::sqrt( x );
#endif
}

/// @copydoc invSqrt( float )
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
double invSqrt( T const x )
{
#if defined( LVARRAY_DEVICE_COMPILE )
  return ::rsqrt( double( x ) );
#else
  return 1 / std::sqrt( x );
#endif
}

#if defined( LVARRAY_USE_DEVICE )

/// @copydoc invSqrt( float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half invSqrt( __half const x )
{ return ::hrsqrt( x ); }

/// @copydoc invSqrt( float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half2 invSqrt( __half2 const x )
{ return ::h2rsqrt( x ); }

#endif

///@}

/**
 * @name Trigonometric functions
 */
///@{

/**
 * @return The sine of @p theta.
 * @param theta The angle in radians.
 * @note This set of overloads is valid for any numeric type. If @p x is integral it is converted to @c double
 *   and the return type is double.
 */
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
float sin( float const theta )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::sinf( theta );
#else
  return std::sin( theta );
#endif
}

/// @copydoc sin( float )
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
double sin( T const theta )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::sin( double( theta ) );
#else
  return std::sin( theta );
#endif
}

#if defined( LVARRAY_USE_DEVICE )

/// @copydoc sin( float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half sin( __half const theta )
{ return ::hsin( theta ); }

/// @copydoc sin( float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half2 sin( __half2 const theta )
{ return ::h2sin( theta ); }

#endif

/**
 * @return The cosine of @p theta.
 * @param theta The angle in radians.
 * @note This set of overloads is valid for any numeric type. If @p theta is not a float
 *   it is converted to a double and the return type is double.
 */
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
float cos( float const theta )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::cosf( theta );
#else
  return std::cos( theta );
#endif
}

/// @copydoc cos( float )
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
double cos( T const theta )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::cos( double( theta ) );
#else
  return std::cos( theta );
#endif
}

#if defined( LVARRAY_USE_DEVICE )

/// @copydoc cos( float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half cos( __half const theta )
{ return ::hcos( theta ); }

/// @copydoc cos( float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half2 cos( __half2 const theta )
{ return ::h2cos( theta ); }

#endif

/**
 * @brief Compute the sine and cosine of @p theta.
 * @param theta The angle in radians.
 * @param sinTheta The sine of @p theta.
 * @param cosTheta The cosine of @p theta.
 */
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
void sincos( float const theta, float & sinTheta, float & cosTheta )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  #if defined(LVARRAY_USE_CUDA)
  ::sincos( theta, &sinTheta, &cosTheta );
  #elif defined(LVARRAY_USE_HIP)
  ::sincosf( theta, &sinTheta, &cosTheta );
  #endif
#else
  sinTheta = std::sin( theta );
  cosTheta = std::cos( theta );
#endif
}

/// @copydoc sincos( float, float &, float & )
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
void sincos( double const theta, double & sinTheta, double & cosTheta )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  ::sincos( theta, &sinTheta, &cosTheta ); // hip and cuda versions both use double
#else
  sinTheta = std::sin( theta );
  cosTheta = std::cos( theta );
#endif
}

/// @copydoc sincos( float, float &, float & )
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
void sincos( T const theta, double & sinTheta, double & cosTheta )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  double s, c;
  ::sincos( theta, &s, &c );
  sinTheta = s;
  cosTheta = c;
#else
  sinTheta = std::sin( theta );
  cosTheta = std::cos( theta );
#endif
}

#if defined( LVARRAY_USE_DEVICE )

/// @copydoc sincos( float, float &, float & )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
void sincos( __half const theta, __half & sinTheta, __half & cosTheta )
{
  sinTheta = ::hsin( theta );
  cosTheta = ::hcos( theta );
}

/// @copydoc sincos( float, float &, float & )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
void sincos( __half2 const theta, __half2 & sinTheta, __half2 & cosTheta )
{
  sinTheta = ::h2sin( theta );
  cosTheta = ::h2cos( theta );
}

#endif

/**
 * @return The tangent of @p theta.
 * @param theta The angle in radians.
 * @note This set of overloads is valid for any numeric type. If @p theta is not a float
 *   it is converted to a double and the return type is double.
 */
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
float tan( float const theta )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::tanf( theta );
#else
  return std::tan( theta );
#endif
}

/// @copydoc tan( float )
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
double tan( T const theta )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::tan( double( theta ) );
#else
  return std::tan( theta );
#endif
}

#if defined( LVARRAY_USE_DEVICE )

/// @copydoc tan( float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half tan( __half const theta )
{
  __half s, c;
  sincos( theta, s, c );
  return s / c;
}

/// @copydoc tan( float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half2 tan( __half2 const theta )
{
  __half2 s, c;
  sincos( theta, s, c );
  return s / c;
}

#endif

///@}

/**
 * @name Inverse trigonometric functions
 */
///@{

namespace internal
{

/**
 * @return The arcsine of @p x.
 * @param x The value to get the arcsine of, must be in [-1, 1].
 * @details Has a max error of 1.36e-3 and max relative error of 6.95e-3 for half precision.
 *   The largeish relative error occurs where the algorithm transitions away from the small value approximation.
 *   The original algorithm didn't the small value approximation and it had huge (~3.0x) relative error,
 *   I believe due to the poor cancellation when subtracting from pi/2. @c acos doesn't have this problem
 *   and calculating @c asin as @code pi/2 - acos @endcode led to the the same error. To improve precision
 *   you could always add in another term in the taylor series (x^3 / 6) but I don't think it's worth it.
 * @note Modified from https://developer.download.nvidia.com/cg/asin.html
 */
template< typename T >
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
T asinImpl( T const x )
{
  T const negate = lessThan( x, math::convert< T >( 0 ) );
  T const absX = abs( x );

  T ret = math::convert< T >( -0.0187293 ) * absX + math::convert< T >( 0.0742610 );
  ret = ret * absX - math::convert< T >( 0.2121144 );
  ret = ret * absX + math::convert< T >( 1.5707288 );
  ret = math::convert< T >( 3.14159265358979 * 0.5 ) - ret * sqrt( math::convert< T >( 1 ) - absX );
  ret = ret - negate * ( ret + ret );
  T const smallAngle = lessThan( absX, math::convert< T >( 1.7e-1 ) );
  return smallAngle * x + ( math::convert< T >( 1 ) - smallAngle ) * ret;
}

/**
 * @return The arcsine of @p x.
 * @param x The value to get the arcsine of, must be in [-1, 1].
 * @details Has a max error of 2.64e-3 and max relative error of 1.37e-3 for half precision.
 * @note Modified from https://developer.download.nvidia.com/cg/acos.html
 */
template< typename T >
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
T acosImpl( T const x )
{
  T const negate = lessThan( x, math::convert< T >( 0 ) );
  T const absX = abs( x );

  T ret = math::convert< T >( -0.0187293 ) * absX + math::convert< T >( 0.0742610 );
  ret = ret * absX - math::convert< T >( 0.2121144 );
  ret = ret * absX + math::convert< T >( 1.5707288 );
  ret = ret * sqrt( math::convert< T >( 1 ) - absX );
  ret = ret - negate * ( ret + ret );
  return negate * math::convert< T >( 3.14159265358979 ) + ret;
}

/**
 * @return The arctangent of the point @p x, @p y.
 * @param y The y coordinate.
 * @param x The x coordinate.
 * @details Has a max error of 2.29e-3 and max relative error of 1.51e-3 for half precision.
 * @note Modified from https://developer.download.nvidia.com/cg/atan2.html
 */
template< typename T >
LVARRAY_DEVICE
LVARRAY_FORCE_INLINE
T atan2Impl( T const y, T const x )
{
  T const absX = abs( x );
  T const absY = abs( y );
  T const ratio = min( absX, absY ) / max( absX, absY );
  T const ratio2 = ratio * ratio;

  T ret = math::convert< T >( -0.013480470 ) * ratio2 + math::convert< T >( 0.057477314 );
  ret = ret * ratio2 - math::convert< T >( 0.121239071 );
  ret = ret * ratio2 + math::convert< T >( 0.195635925 );
  ret = ret * ratio2 - math::convert< T >( 0.332994597 );
  ret = ret * ratio2 + math::convert< T >( 0.999995630 );
  ret = ret * ratio;

  // For single values the following works out to:
  // ret = absX < absY ? 1.570796327 - ret : ret;
  // ret = x < 0 ? 3.141592654 - ret : ret;
  // ret = y < 0 ? -ret : ret;
  ret = internal::lessThan( absX, absY ) * ( math::convert< T >( 1.570796327 ) - ret - ret ) + ret;
  ret = internal::lessThan( x, math::convert< T >( 0 ) ) * ( math::convert< T >( 3.141592654 ) - ret - ret ) + ret;
  ret = ret - internal::lessThan( y, math::convert< T >( 0 ) ) * ( ret + ret );

  return ret;
}

} // namespace internal

/**
 * @return The arcsine of the value @p x.
 * @param x The value to get the arcsine of, must be in [-1, 1].
 * @note This set of overloads is valid for any numeric type. If @p x is integral it is converted to @c double
 *   and the return type is double.
 */
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
float asin( float const x )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::asinf( x );
#else
  return std::asin( x );
#endif
}

/// @copydoc asin( float )
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
double asin( T const x )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::asin( double( x ) );
#else
  return std::asin( x );
#endif
}

#if defined( LVARRAY_USE_DEVICE )

/// @copydoc asin( float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half asin( __half const x )
{ return internal::asinImpl( x ); }

/// @copydoc asin( float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half2 asin( __half2 const x )
{ return internal::asinImpl( x ); }

#endif

/**
 * @return The arccosine of the value @p x.
 * @param x The value to get the arccosine of, must be in [-1, 1].
 * @note This set of overloads is valid for any numeric type. If @p x is integral it is converted to @c double
 *   and the return type is double.
 */
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
float acos( float const x )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::acosf( x );
#else
  return std::acos( x );
#endif
}

/// @copydoc acos( float )
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
double acos( T const x )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::acos( double( x ) );
#else
  return std::acos( x );
#endif
}

#if defined( LVARRAY_USE_DEVICE )

/// @copydoc acos( float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half acos( __half const x )
{ return internal::acosImpl( x ); }

/// @copydoc acos( float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half2 acos( __half2 const x )
{ return internal::acosImpl( x ); }

#endif

/**
 * @return The angle corresponding to the point (x, y).
 * @param y The y coordinate.
 * @param x The x coordinate.
 * @note This set of overloads is valid for any numeric type. If @p x is integral it is converted to @c double
 *   and the return type is double.
 */
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
float atan2( float const y, float const x )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::atan2f( y, x );
#else
  return std::atan2( y, x );
#endif
}

/// @copydoc atan2( float, float )
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
double atan2( T const y, T const x )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::atan2( double( y ), double( x ) );
#else
  return std::atan2( y, x );
#endif
}

#if defined( LVARRAY_USE_CUDA )

/// @copydoc atan2( float, float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half atan2( __half const y, __half const x )
{ return internal::atan2Impl( y, x ); }

/// @copydoc atan2( float, float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half2 atan2( __half2 const y, __half2 const x )
{ return internal::atan2Impl( y, x ); }

#endif

///@}

/**
 * @name Exponential functions
 */
///@{

/**
 * @return e raised to the @p x.
 * @param x The power to raise e to.
 * @note This set of overloads is valid for any numeric type. If @p x is integral it is converted to @c double
 *   and the return type is double.
 */
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
float exp( float const x )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::expf( x );
#else
  return std::exp( x );
#endif
}

/// @copydoc exp( float )
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
double exp( T const x )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::exp( double( x ) );
#else
  return std::exp( x );
#endif
}

#if defined( LVARRAY_USE_DEVICE )

/// @copydoc exp( float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half exp( __half const x )
{ return ::hexp( x ); }

/// @copydoc exp( float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half2 exp( __half2 const x )
{ return ::h2exp( x ); }

#endif

/**
 * @return The natural logarithm of @p x.
 * @param x The number to get the natural logarithm of.
 * @note This set of overloads is valid for any numeric type. If @p x is integral it is converted to @c double
 *   and the return type is double.
 */
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
float log( float const x )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::logf( x );
#else
  return std::log( x );
#endif
}

/// @copydoc log( float )
template< typename T >
LVARRAY_HOST_DEVICE LVARRAY_FORCE_INLINE
double log( T const x )
{
#if defined(LVARRAY_DEVICE_COMPILE)
  return ::log( double( x ) );
#else
  return std::log( x );
#endif
}

#if defined( LVARRAY_USE_DEVICE )

/// @copydoc log( float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half log( __half const x )
{ return ::hlog( x ); }

/// @copydoc log( float )
LVARRAY_DEVICE LVARRAY_FORCE_INLINE
__half2 log( __half2 const x )
{ return ::h2log( x ); }

#endif

///@}

} // namespace math
} // namespace LvArray
