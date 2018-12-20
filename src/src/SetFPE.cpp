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
 * @file SetFPE.cpp
 */

#include "SetFPE.hpp"
// #include <xmmintrin.h>
// #include <pmmintrin.h>



namespace cxx_utilities
{


/**
 * function to set the floating point environment. We typically want to throw floating point exceptions on
 * FE_DIVBYZERO, FE_OVERFLOW | FE_INVALID. We would also like to flush denormal numbers to zero.
 */
void SetFPE()
{
// #if defined(__APPLE__) && defined(__MACH__)
//   fesetenv( FE_DFL_DISABLE_SSE_DENORMS_ENV );
// #else
//   _MM_SET_FLUSH_ZERO_MODE( _MM_FLUSH_ZERO_ON );
//   _MM_SET_DENORMALS_ZERO_MODE( _MM_DENORMALS_ZERO_ON );
// #endif
//   feenableexcept( FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID );
}

}
