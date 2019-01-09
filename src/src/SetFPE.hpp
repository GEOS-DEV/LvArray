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

/*
 * SetFPE.hpp
 *
 *  Created on: Dec 4, 2018
 *      Author: settgast
 */

#ifndef SRC_CORECOMPONENTS_CXX_UTILITIES_SRC_SRC_SETFPE_HPP_
#define SRC_CORECOMPONENTS_CXX_UTILITIES_SRC_SRC_SETFPE_HPP_
#include <fenv.h>

#if defined(__APPLE__) && defined(__MACH__)

// Public domain polyfill for feenableexcept on OS X
// http://www-personal.umich.edu/~williams/archive/computation/fe-handling-example.c

inline int feenableexcept( unsigned int excepts )
{
  static fenv_t fenv;
  unsigned int new_excepts = excepts & FE_ALL_EXCEPT;
  // previous masks
  unsigned int old_excepts;

  if( fegetenv( &fenv ))
  {
    return -1;
  }
  old_excepts = fenv.__control & FE_ALL_EXCEPT;

  // unmask
  fenv.__control &= ~new_excepts;
  fenv.__mxcsr   &= ~(new_excepts << 7);

  return fesetenv( &fenv ) ? -1 : old_excepts;
}

inline int fedisableexcept( unsigned int excepts )
{
  static fenv_t fenv;
  unsigned int new_excepts = excepts & FE_ALL_EXCEPT;
  // all previous masks
  unsigned int old_excepts;

  if( fegetenv( &fenv ))
  {
    return -1;
  }
  old_excepts = fenv.__control & FE_ALL_EXCEPT;

  // mask
  fenv.__control |= new_excepts;
  fenv.__mxcsr   |= new_excepts << 7;

  return fesetenv( &fenv ) ? -1 : old_excepts;
}
#endif

namespace cxx_utilities
{

void SetFPE();

}

#endif /* SRC_CORECOMPONENTS_CXX_UTILITIES_SRC_SRC_SETFPE_HPP_ */
