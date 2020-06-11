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
 * @file IntegerConversion.hpp
 */

#ifndef INTEGERCONVERSION_HPP_
#define INTEGERCONVERSION_HPP_

// Source includes
#include "Macros.hpp"

// System includes
#include <limits>

/**
 * @file IntegerConversion.hpp
 */

namespace LvArray
{

namespace internal
{

template< typename INTEGER >
struct IntegerTraits
{
  static constexpr INTEGER min = std::numeric_limits< INTEGER >::min();
  static constexpr INTEGER max = std::numeric_limits< INTEGER >::max();
};

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

  return OUTPUT{ input };
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

  LVARRAY_ERROR_IF_GT( input, internal::IntegerTraits< OUTPUT >::max );

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

  LVARRAY_ERROR_IF_LT( input, std::make_signed_t< OUTPUT >{ internal::IntegerTraits< OUTPUT >::min } );

  // If OUTPUT is unsigned we convert input to an unsigned type. This is safe because it must be
  // positive due to the check above.
  using ConditionallyUnsigned = std::conditional_t< std::is_unsigned< OUTPUT >::value, std::make_unsigned_t< INPUT >, INPUT >;
  LVARRAY_ERROR_IF_GT( ConditionallyUnsigned( input ), internal::IntegerTraits< OUTPUT >::max );

  return static_cast< OUTPUT >( input );
}

} // namespace LvArray

#endif /* SRC_SRC_INTEGERCONVERSION_HPP_ */
