

#include "SetFPE.hpp"
#include <fenv.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

namespace cxx_utilities
{
void SetFPE()
{
#ifdef __APPLE__
#if __clang__

#elif __GNUC__
//  _MM_SET_EXCEPTION_MASK( ( _MM_EXCEPT_INVALID |
//          _MM_EXCEPT_DENORM |
//          _MM_EXCEPT_DIV_ZERO |
//          _MM_EXCEPT_OVERFLOW |
//          _MM_EXCEPT_UNDERFLOW |
//          _MM_EXCEPT_INEXACT ) );
//  _MM_SET_EXCEPTION_MASK( _MM_GET_EXCEPTION_MASK()
//           & ~( _MM_EXCEPT_INVALID |
//                _MM_EXCEPT_DENORM |
//                _MM_EXCEPT_DIV_ZERO |
//                _MM_EXCEPT_OVERFLOW |
//                _MM_EXCEPT_UNDERFLOW |
//                _MM_EXCEPT_INEXACT ) );
//  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_EXCEPT_DIV_ZERO);
#endif
#else
  feenableexcept( FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID  );
//  fesetenv(FE_DFL_DISABLE_SSE_DENORMS_ENV);

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
