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
#include "benchmarkHelpers.hpp"

namespace LvArray
{
namespace benchmarking
{

constexpr unsigned long THREADS_PER_BLOCK = 256;

struct InnerProductNative
{
  static VALUE_TYPE fortran( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                             ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                             INDEX_TYPE const N );

  static VALUE_TYPE subscript( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                               ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                               INDEX_TYPE const N );

  static VALUE_TYPE rajaView( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                              RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                              INDEX_TYPE const N );

  static VALUE_TYPE pointer( VALUE_TYPE const * const LVARRAY_RESTRICT a,
                             VALUE_TYPE const * const LVARRAY_RESTRICT b,
                             INDEX_TYPE const N );
};

template< typename POLICY >
struct InnerProductRAJA
{
  static VALUE_TYPE fortran( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                             ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                             INDEX_TYPE const N );

  static VALUE_TYPE subscript( ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                               ArrayView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                               INDEX_TYPE const N );

  static VALUE_TYPE rajaView( RajaView< VALUE_TYPE const, RAJA::PERM_I > const & a,
                              RajaView< VALUE_TYPE const, RAJA::PERM_I > const & b,
                              INDEX_TYPE const N );

  static VALUE_TYPE pointer( VALUE_TYPE const * const LVARRAY_RESTRICT a,
                             VALUE_TYPE const * const LVARRAY_RESTRICT b,
                             INDEX_TYPE const N );
};

} // namespace benchmarking
} // namespace LvArray
