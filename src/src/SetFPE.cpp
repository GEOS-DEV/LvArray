/**
 * @file SetFPE.cpp
 */

#include "SetFPE.hpp"
#include <xmmintrin.h>
#include <pmmintrin.h>
//#include <bitset>
//#include <iostream>



namespace cxx_utilities
{


/**
 * function to set the floating point environment. We typically want to throw floating point exceptions on
 * FE_DIVBYZERO, FE_OVERFLOW | FE_INVALID. We would also like to flush denormal numbers to zero.
 */
void SetFPE()
{
#if defined(__APPLE__) && defined(__MACH__)
  fesetenv(FE_DFL_DISABLE_SSE_DENORMS_ENV);
#else
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
  feenableexcept( FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID  );
}

}
