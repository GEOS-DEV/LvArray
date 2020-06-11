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

#pragma once

// Source includes
#include "templateHelpers.hpp"

// TPL includes
#include <camp/camp.hpp>

namespace LvArray
{

namespace internal
{

template< camp::idx_t INDEX_TO_FIND, camp::idx_t INDEX >
constexpr bool contains( camp::idx_seq< INDEX > )
{ return INDEX_TO_FIND == INDEX; }

template< camp::idx_t INDEX_TO_FIND, camp::idx_t INDEX0, camp::idx_t INDEX1, camp::idx_t... INDICES >
constexpr bool contains( camp::idx_seq< INDEX0, INDEX1, INDICES... > )
{ return ( INDEX_TO_FIND == INDEX0 ) || contains< INDEX_TO_FIND >( camp::idx_seq< INDEX1, INDICES... > {} ); }

template< typename PERMUTATION, camp::idx_t... INDICES >
constexpr bool isValidPermutation( PERMUTATION, camp::idx_seq< INDICES... > )
{ return conjunction< contains< INDICES >( PERMUTATION {} )... >; }

} // namespace internal

/**
 * @tparam INDICES A variadic list of indices.
 * @brief @return The number of indices.
 */
template< camp::idx_t... INDICES >
constexpr camp::idx_t getDimension( camp::idx_seq< INDICES... > )
{ return sizeof...( INDICES ); }

/**
 * @tparam INDICES A variadic list of indices.
 * @brief @return The unit stride dimension, the last index in the sequence.
 */
template< camp::idx_t... INDICES >
constexpr camp::idx_t getStrideOneDimension( camp::idx_seq< INDICES... > )
{
  constexpr camp::idx_t dimension = camp::seq_at< sizeof...( INDICES ) - 1, camp::idx_seq< INDICES... > >::value;
  static_assert( dimension >= 0, "The dimension must be greater than zero." );
  static_assert( dimension < sizeof...( INDICES ), "The dimension must be less than NDIM." );
  return dimension;
}

/**
 * @tparam PERMUTATION A camp::idx_seq.
 * @brief @return True iff @tparam PERMUTATION is a permutation of [0, N] for some N.
 */
template< typename PERMUTATION >
constexpr bool isValidPermutation( PERMUTATION )
{
  constexpr int NDIM = getDimension( PERMUTATION {} );
  return internal::isValidPermutation( PERMUTATION {}, camp::make_idx_seq_t< NDIM > {} );
}

/**
 * @tparam T The type of values stored in the array.
 * @tparam N The number of values in the array.
 * @struct CArray
 * @brief A wrapper around a compile time c array.
 */
template< typename T, std::ptrdiff_t N >
struct CArray
{
  /**
   * @brief @return Return a reference to the value at position @p i.
   * @param i The position to access.
   */
  constexpr inline LVARRAY_HOST_DEVICE
  T & operator[]( std::ptrdiff_t const i )
  { return m_d_a_t_a[ i ]; }

  /**
   * @brief @return Return a const reference to the value at position @p i.
   * @param i The position to access.
   */
  constexpr inline LVARRAY_HOST_DEVICE
  T const & operator[]( std::ptrdiff_t const i ) const
  { return m_d_a_t_a[ i ]; }

  /**
   * @brief @return Return the size of the array.
   */
  constexpr inline LVARRAY_HOST_DEVICE
  std::ptrdiff_t size()
  { return N; }

  /// The backing c array, public so that aggregate initialization works.
  /// The funny name is to dissuade the user from accessing it directly.
  T m_d_a_t_a[ N ];
};

/**
 * @tparam INDICES A variadic pack of numbers.
 * @brief @return A CArray containing @tparam INDICES.
 */
template< camp::idx_t... INDICES >
LVARRAY_HOST_DEVICE inline constexpr
CArray< camp::idx_t, sizeof...( INDICES ) > asArray( camp::idx_seq< INDICES... > )
{ return { INDICES ... }; }

} // namespace LvArray
