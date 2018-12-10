/**
 * @file SetFPE.cpp
 */

#include "SetFPE.hpp"
#include <fenv.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
//#include <bitset>
//#include <iostream>

namespace cxx_utilities
{

/**
 * function to set the floating point environment. We typically want to throw floating point exceptions on
 * FE_DIVBYZERO, FE_OVERFLOW | FE_INVALID. We would also like to flush denormal numbers to zero. Flushing doesn't
 * seem to work in conjunction with FPE on mac, so we turn it off.
 */
void SetFPE()
{
#ifdef __APPLE__

  fenv_t fenv_local ;//= _FE_DFL_DISABLE_SSE_DENORMS_ENV;
  fenv_local.__mxcsr =  FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID  ;
  fesetenv( &fenv_local );

#else
  feenableexcept( FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID  );
  SetUnderflowFlush();
#endif

}

void SetUnderflowFlush()
{
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

void UnsetUnderflowFlush()
{
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
}

}
