/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file limits.hpp
 * @brief Contains portable access to std::numeric_limits and functions for
 *   converting between integral types.
 */

#pragma once

// Source includes
#include "Macros.hpp"

// System includes
#include <limits>

#if defined( LVARRAY_USE_CUDA )
  #include <cuda_fp16.h>
#endif

namespace LvArray
{

/**
 * @struct NumericLimits
 * @brief A wrapper for the std::numeric_limits< T > member functions,
 *  this allows their values to be used on device.
 * @tparam T The numeric type to query.
 */
template< typename T >
struct NumericLimits : public std::numeric_limits< T >
{
  /// The smallest finite value T can hold.
  static constexpr T min = std::numeric_limits< T >::min();
  /// The lowest finite value T can hold.
  static constexpr T lowest = std::numeric_limits< T >::lowest();
  /// The largest finite value T can hold.
  static constexpr T max = std::numeric_limits< T >::max();
  /// The difference between 1.0 and the next representable value (if T is floating point).
  static constexpr T epsilon = std::numeric_limits< T >::epsilon();
  /// The maximum rounding error (if T is a floating point).
  static constexpr T round_error = std::numeric_limits< T >::round_error();
  /// A positive infinity value (if T is a floating point).
  static constexpr T infinity = std::numeric_limits< T >::infinity();
  /// A quiet NaN (if T is a floating point).
  static constexpr T quiet_NaN = std::numeric_limits< T >::quiet_NaN();
  /// A signaling NaN (if T is a floating point).
  static constexpr T signaling_NaN = std::numeric_limits< T >::signaling_NaN();
  /// The smallest positive subnormal value (if T is a floating point).
  static constexpr T denorm_min = std::numeric_limits< T >::denorm_min();
};

#if defined( LVARRAY_USE_CUDA )
/**
 * @brief An overload for half precision.
 * @note The values here are stored as float so that they can be constexpr.
 */
template<>
struct NumericLimits< __half >
{
  /// The smallest finite value T can hold.
  static constexpr float min = 1.0 / 16384;
  /// The lowest finite value T can hold.
  static constexpr float lowest = -65504;
  /// The largest finite value T can hold.
  static constexpr float max = 65504;
  /// The difference between 1.0 and the next representable value (if T is floating point).
  static constexpr float epsilon = 1.0 / 1024;
  /// The smallest positive subnormal value (if T is a floating point).
  static constexpr float denorm_min = 1.0 / 16777216;
};

template<>
struct NumericLimits< __half2 > : public NumericLimits< __half >
{};

#endif

/**
 * @struct NumericLimitsNC
 * @brief The same as @c NumericLimits except the entries are not static or constexpr.
 * @details This is useful for solving "undefined reference" errors that pop up often in lambdas.
 * @tparam T the numeric type to query.
 */
template< typename T >
struct NumericLimitsNC
{
  /// The smallest finite value T can hold.
  T const min = NumericLimits< T >::min;
  /// The lowest finite value T can hold.
  T const lowest = NumericLimits< T >::lowest;
  /// The largest finite value T can hold.
  T const max = NumericLimits< T >::max;
  /// The difference between 1.0 and the next representable value (if T is floating point).
  T const epsilon = NumericLimits< T >::epsilon;
  /// The smallest positive subnormal value (if T is a floating point).
  T const denorm_min = NumericLimits< T >::denorm_min;
};

namespace internal
{

/**
 * @tparam T The first type to check.
 * @tparam U The second type to check.
 * @brief True iff @tparam T and @tparam U are both signed or both unsigned.
 */
template< typename T, typename U >
constexpr bool sameSignedness = ( std::is_signed< T >::value && std::is_signed< U >::value ) ||
                                ( std::is_unsigned< T >::value && std::is_unsigned< U >::value );

/**
 * @tparam INPUT The input type.
 * @tparam OUTPUT The output type.
 * @brief True iff @tparam OUTPUT can hold every possible value of @tparam INPUT.
 */
template< typename INPUT, typename OUTPUT >
constexpr bool canEasilyConvert = sizeof( INPUT ) < sizeof( OUTPUT ) ||
                                  ( sizeof( INPUT ) == sizeof( OUTPUT ) && sameSignedness< INPUT, OUTPUT > );

} // namespace internal

/**
 * @tparam INPUT The input integer type.
 * @tparam OUTPUT The output integer type.
 * @brief @return Return @p input to type @tparam OUTPUT, aborting execution if the conversion can't be performed.
 * @param input The value to convert.
 */
template< typename OUTPUT, typename INPUT >
std::enable_if_t< internal::canEasilyConvert< INPUT, OUTPUT >, OUTPUT >
inline constexpr LVARRAY_HOST_DEVICE
integerConversion( INPUT input )
{
  static_assert( std::is_integral< INPUT >::value, "INPUT must be an integral type." );
  static_assert( std::is_integral< OUTPUT >::value, "OUTPUT must be an integral type." );

//  return OUTPUT{ input };
  return static_cast< OUTPUT >(input);
}

/**
 * @tparam INPUT The input integer type.
 * @tparam OUTPUT The output integer type.
 * @brief @return Return @p input to type @tparam OUTPUT, aborting execution if the conversion can't be performed.
 * @param input The value to convert.
 */
template< typename OUTPUT, typename INPUT >
std::enable_if_t< !internal::canEasilyConvert< INPUT, OUTPUT > &&
                  std::is_unsigned< INPUT >::value,
                  OUTPUT >
inline LVARRAY_HOST_DEVICE
integerConversion( INPUT input )
{
  static_assert( std::is_integral< INPUT >::value, "INPUT must be an integral type." );
  static_assert( std::is_integral< OUTPUT >::value, "OUTPUT must be an integral type." );

  LVARRAY_ERROR_IF_GT( input, NumericLimits< OUTPUT >::max );

  return static_cast< OUTPUT >( input );
}

/**
 * @tparam INPUT The input integer type.
 * @tparam OUTPUT The output integer type.
 * @brief @return Return @p input to type @tparam OUTPUT, aborting execution if the conversion can't be performed.
 * @param input The value to convert.
 */
template< typename OUTPUT, typename INPUT >
std::enable_if_t< !internal::canEasilyConvert< INPUT, OUTPUT > &&
                  !std::is_unsigned< INPUT >::value,
                  OUTPUT >
inline LVARRAY_HOST_DEVICE
integerConversion( INPUT input )
{
  static_assert( std::is_integral< INPUT >::value, "INPUT must be an integral type." );
  static_assert( std::is_integral< OUTPUT >::value, "OUTPUT must be an integral type." );

  LVARRAY_ERROR_IF_LT( input, std::make_signed_t< OUTPUT >{ NumericLimits< OUTPUT >::min } );

  // If OUTPUT is unsigned we convert input to an unsigned type. This is safe because it must be
  // positive due to the check above.
  using ConditionallyUnsigned = std::conditional_t< std::is_unsigned< OUTPUT >::value, std::make_unsigned_t< INPUT >, INPUT >;
  LVARRAY_ERROR_IF_GT( ConditionallyUnsigned( input ), NumericLimits< OUTPUT >::max );

  return static_cast< OUTPUT >( input );
}

}
