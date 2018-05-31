// Copyright (c) 2018, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-746361. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the GEOSX Simulation Framework.

//
// GEOSX is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
/*
 * FloatingPointExceptions.cpp
 *
 *  Created on: Jun 14, 2016
 *      Author: settgast1
 */

#include "SetSignalHandling.hpp"
#include "stackTrace.hpp"
#include <fenv.h>
#include <xmmintrin.h>

namespace cxx_utilities
{

void setSignalHandling( void (*handler)( int ) )
{


  signal(SIGHUP, handler);
  signal(SIGINT, handler);
  signal(SIGQUIT, handler);
  signal(SIGILL, handler);
  signal(SIGTRAP, handler);
  signal(SIGABRT, handler);
#if  (defined(_POSIX_C_SOURCE) && !defined(_DARWIN_C_SOURCE))
  signal(SIGPOLL, handler);
#else
  signal(SIGIOT, handler);
  signal(SIGEMT, handler);
#endif
  signal(SIGFPE, handler);
  signal(SIGKILL, handler);
  signal(SIGBUS, handler);
  signal(SIGSEGV, handler);
  signal(SIGSYS, handler);
  signal(SIGPIPE, handler);
  signal(SIGALRM, handler);
  signal(SIGTERM, handler);
  signal(SIGURG, handler);
  signal(SIGSTOP, handler);
  signal(SIGTSTP, handler);
  signal(SIGCONT, handler);
  signal(SIGCHLD, handler);

#if __APPLE__
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
  feenableexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID);
#endif

  return;
}



} /* namespace geosx */
