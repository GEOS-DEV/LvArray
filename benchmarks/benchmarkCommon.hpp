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

#include "Array.hpp"

namespace LvArray
{

namespace benchmarking
{

#define ACCESS_IJ( N, M, i, j ) M * i + j
#define ACCESS_JI( N, M, i, j ) N * j + i

#define ACCESS_IJK( N, M, P, i, j, k ) M * P * i + P * j + k
#define ACCESS_KJI( N, M, P, i, j, k ) M * N * k + N * j + i

using INDEX_TYPE = std::ptrdiff_t;

template< typename T, typename PERMUTATION >
using Array = LvArray::Array< T, getDimension( PERMUTATION {} ), PERMUTATION, INDEX_TYPE >;

template< typename T, typename PERMUTATION, int LENGTH >
using StackArray = LvArray::StackArray< T, getDimension( PERMUTATION {} ), PERMUTATION, INDEX_TYPE, LENGTH >;

template< typename T, typename PERMUTATION >
using ArrayView = LvArray::ArrayView< T, getDimension( PERMUTATION {} ), getStrideOneDimension( PERMUTATION {} ), INDEX_TYPE >;

template< typename T, typename PERMUTATION >
using ArraySlice = LvArray::ArraySlice< T, getDimension( PERMUTATION {} ), getStrideOneDimension( PERMUTATION {} ), INDEX_TYPE >;

template< typename T, typename PERMUTATION >
using RajaView = RAJA::View< T, RAJA::Layout< getDimension( PERMUTATION {} ), INDEX_TYPE, getStrideOneDimension( PERMUTATION {} ) >>;

} // namespace benchmarking
} // namespace LvArray
