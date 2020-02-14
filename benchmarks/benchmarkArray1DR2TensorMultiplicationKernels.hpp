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
#include "benchmarkCommon.hpp"

namespace LvArray
{
namespace benchmarking
{

constexpr unsigned long THREADS_PER_BLOCK = 256;

template< typename PERMUTATION >
struct Array1DR2TensorMultiplicationNative
{
  static void fortran( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
                       ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
                       ArrayView< VALUE_TYPE, PERMUTATION > const & c,
                       INDEX_TYPE const N );

  static void subscript( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
                         ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
                         ArrayView< VALUE_TYPE, PERMUTATION > const & c,
                         INDEX_TYPE const N );

  static void tensorAbstraction( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
                                 ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
                                 ArrayView< VALUE_TYPE, PERMUTATION > const & c,
                                 INDEX_TYPE const N );

  static void rajaView( RajaView< VALUE_TYPE const, PERMUTATION > const & a,
                        RajaView< VALUE_TYPE const, PERMUTATION > const & b,
                        RajaView< VALUE_TYPE, PERMUTATION > const & c,
                        INDEX_TYPE const N );

  static void pointer( VALUE_TYPE const * const LVARRAY_RESTRICT a,
                       VALUE_TYPE const * const LVARRAY_RESTRICT b,
                       VALUE_TYPE * const LVARRAY_RESTRICT c,
                       INDEX_TYPE const N );
};

template< typename PERMUTATION, typename POLICY >
struct Array1DR2TensorMultiplicationRaja
{
  static void fortran( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
                       ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
                       ArrayView< VALUE_TYPE, PERMUTATION > const & c,
                       INDEX_TYPE const N );

  static void subscript( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
                         ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
                         ArrayView< VALUE_TYPE, PERMUTATION > const & c,
                         INDEX_TYPE const N );

  static void tensorAbstraction( ArrayView< VALUE_TYPE const, PERMUTATION > const & a,
                                 ArrayView< VALUE_TYPE const, PERMUTATION > const & b,
                                 ArrayView< VALUE_TYPE, PERMUTATION > const & c,
                                 INDEX_TYPE const N );

  static void rajaView( RajaView< VALUE_TYPE const, PERMUTATION > const & a,
                        RajaView< VALUE_TYPE const, PERMUTATION > const & b,
                        RajaView< VALUE_TYPE, PERMUTATION > const & c,
                        INDEX_TYPE const N );

  static void pointer( VALUE_TYPE const * const LVARRAY_RESTRICT a,
                       VALUE_TYPE const * const LVARRAY_RESTRICT b,
                       VALUE_TYPE * const LVARRAY_RESTRICT c,
                       INDEX_TYPE const N );
};

} // namespace benchmarking
} // namespace LvArray
