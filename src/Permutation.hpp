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
{ return conjunction< contains< INDICES >( PERMUTATION {} )... >::value; }

} // namespace internal

template< camp::idx_t... INDICES >
constexpr camp::idx_t getDimension( camp::idx_seq< INDICES... > )
{ return sizeof...( INDICES ); }

template< camp::idx_t... INDICES >
constexpr camp::idx_t getStrideOneDimension( camp::idx_seq< INDICES... > )
{
  constexpr camp::idx_t dimension = camp::seq_at< sizeof...( INDICES ) - 1, camp::idx_seq< INDICES... > >::value;
  static_assert( dimension >= 0, "The dimension must be greater than zero." );
  static_assert( dimension < sizeof...( INDICES ), "The dimension must be less than NDIM." );
  return dimension;
}

template< typename PERMUTATION >
constexpr bool isValidPermutation( PERMUTATION )
{
  constexpr int NDIM = getDimension( PERMUTATION {} );
  return internal::isValidPermutation( PERMUTATION {}, camp::make_idx_seq_t< NDIM > {} );
}

} // namespace LvArray
