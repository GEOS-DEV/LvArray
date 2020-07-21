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

#include "system.hpp"

// System includes
#include <map>
#include <sstream>
#include <iostream>
#include <signal.h>
#include <fenv.h>
#include <cxxabi.h>
#include <execinfo.h>
#include <string.h>
#include <cstdio>

#if defined(__x86_64__)
  #include <xmmintrin.h>
  #include <pmmintrin.h>
#endif

std::string demangleBacktrace( char * backtraceString, int frame )
{
  char * mangledName = nullptr;
  char * functionOffset = nullptr;
  char * returnOffset = nullptr;

#ifdef __APPLE__
  /* On apple machines the mangled function name always starts at the 58th
   * character */
  constexpr int APPLE_OFFSET = 58;
  mangledName = backtraceString + APPLE_OFFSET;
  for( char * p = backtraceString; *p; ++p )
  {
    if( *p == '+' )
    {
      functionOffset = p;
    }
    returnOffset = p;
  }
#else
  for( char * p = backtraceString; *p; ++p )
  {
    if( *p == '(' )
    {
      mangledName = p;
    }
    else if( *p == '+' )
    {
      functionOffset = p;
    }
    else if( *p == ')' )
    {
      returnOffset = p;
      break;
    }
  }
#endif

  std::ostringstream oss;

  if( mangledName && functionOffset && returnOffset &&
      mangledName < functionOffset )
  {
    // if the line could be processed, attempt to demangle the symbol
    *mangledName = 0;
    mangledName++;
#ifdef __APPLE__
    *(functionOffset - 1) = 0;
#endif
    *functionOffset = 0;
    ++functionOffset;
    *returnOffset = 0;
    ++returnOffset;

    oss << "Frame " << frame << ": " << LvArray::system::demangle( mangledName ) << "\n";
  }
  else
  {
    // otherwise, print the whole line
    oss << "Frame " << frame << ": " << backtraceString << "\n";
  }

  return ( oss.str() );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string getFpeDetails()
{
  std::ostringstream oss;
  int const fpe = fetestexcept( FE_ALL_EXCEPT );

  oss << "Floating point exception:";

  if( fpe & FE_DIVBYZERO )
  {
    oss << " Division by zero;";
  }
  if( fpe & FE_INEXACT )
  {
    oss << " Inexact result;";
  }
  if( fpe & FE_INVALID )
  {
    oss << " Invalid argument;";
  }
  if( fpe & FE_OVERFLOW )
  {
    oss << " Overflow;";
  }
  if( fpe & FE_UNDERFLOW )
  {
    oss << " Underflow;";
  }

  return oss.str();
}

namespace LvArray
{
namespace system
{

/// An alias for a function that takes an int and returns nothing.
using handle_type = void ( * )( int );

/// A map containing the initial signal handlers.
static std::map< int, handle_type > initialHandler;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string stackTrace()
{
  constexpr int MAX_FRAMES = 25;
  void * array[ MAX_FRAMES ];

  const int size = backtrace( array, MAX_FRAMES );
  char * * strings = backtrace_symbols( array, size );

  // skip first stack frame (points here)
  std::ostringstream oss;
  oss << "\n** StackTrace of " << size - 1 << " frames **\n";
  for( int i = 1; i < size && strings != nullptr; ++i )
  {
    oss << demangleBacktrace( strings[ i ], i );
  }

  oss << "=====\n";

  free( strings );

  return ( oss.str() );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string demangle( char const * const name )
{
  if( name == nullptr )
  {
    return "";
  }

  int status = -4; // some arbitrary value to eliminate the compiler warning
  char * const demangledName = abi::__cxa_demangle( name, nullptr, nullptr, &status );

  std::string const result = (status == 0) ? demangledName : name;

  std::free( demangledName );

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string calculateSize( size_t const bytes )
{
  char const * suffix;
  uint shift;
  if( bytes >> 30 != 0 )
  {
    suffix = "GB";
    shift = 30;
  }
  else if( bytes >> 20 != 0 )
  {
    suffix = "MB";
    shift = 20;
  }
  else
  {
    suffix = "KB";
    shift = 10;
  }

  double const units = double( bytes ) / ( 1 << shift );

  char result[10];
  std::snprintf( result, 10, "%.1f %s", units, suffix );
  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void abort()
{
#ifdef USE_MPI
  int mpi = 0;
  MPI_Initialized( &mpi );
  if( mpi )
  {
    MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
  }
#endif
  std::abort();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void stackTraceHandler( int const sig, bool const exit )
{
  std::ostringstream oss;

  if( sig >= 0 && sig < NSIG )
  {
    // sys_signame not available on linux, so just print the code; strsignal is POSIX
    oss << "Received signal " << sig << ": " << strsignal( sig ) << "\n";

    if( sig == SIGFPE )
    {
      oss << getFpeDetails() << "\n";
    }
  }

  oss << stackTrace() << std::endl;
  std::cout << oss.str();

  if( exit )
  {
    // An infinite loop was encountered when an FPE was received. Resetting the handlers didn't
    // fix it because they would just recurse. This does.
    setSignalHandling( nullptr );
    abort();
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void setSignalHandling( void (* handler)( int ) )
{
  initialHandler[SIGHUP] = signal( SIGHUP, handler );
  initialHandler[SIGINT] = signal( SIGINT, handler );
  initialHandler[SIGQUIT] = signal( SIGQUIT, handler );
  initialHandler[SIGILL] = signal( SIGILL, handler );
  initialHandler[SIGTRAP] = signal( SIGTRAP, handler );
  initialHandler[SIGABRT] = signal( SIGABRT, handler );
#if  (defined(_POSIX_C_SOURCE) && !defined(_DARWIN_C_SOURCE))
  initialHandler[SIGPOLL] = signal( SIGPOLL, handler );
#else
  initialHandler[SIGIOT] = signal( SIGIOT, handler );
  initialHandler[SIGEMT] = signal( SIGEMT, handler );
#endif
  initialHandler[SIGFPE] = signal( SIGFPE, handler );
  initialHandler[SIGKILL] = signal( SIGKILL, handler );
  initialHandler[SIGBUS] = signal( SIGBUS, handler );
  initialHandler[SIGSEGV] = signal( SIGSEGV, handler );
  initialHandler[SIGSYS] = signal( SIGSYS, handler );
  initialHandler[SIGPIPE] = signal( SIGPIPE, handler );
  initialHandler[SIGTERM] = signal( SIGTERM, handler );

  return;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void resetSignalHandling()
{
  for( auto a : initialHandler )
  {
    signal( a.first, a.second );
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int enableFloatingPointExceptions( int const exceptions )
{
#if defined(__APPLE__) && defined(__MACH__)
  // Public domain polyfill for feenableexcept on OS X
  // http://www-personal.umich.edu/~williams/archive/computation/fe-handling-example.c
  static fenv_t fenv;
  int const newExcepts = exceptions & FE_ALL_EXCEPT;
  // previous masks
  int oldExcepts;

  if( fegetenv( &fenv ))
  {
    return -1;
  }
  oldExcepts = fenv.__control & FE_ALL_EXCEPT;

  // unmask
  fenv.__control &= ~newExcepts;
  fenv.__mxcsr   &= ~(newExcepts << 7);

  return fesetenv( &fenv ) ? -1 : oldExcepts;
#else
  return feenableexcept( exceptions );
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int disableFloatingPointExceptions( int const exceptions )
{
#if defined(__APPLE__) && defined(__MACH__)
  // Public domain polyfill for feenableexcept on OS X
  // http://www-personal.umich.edu/~williams/archive/computation/fe-handling-example.c
  static fenv_t fenv;
  int const newExcepts = exceptions & FE_ALL_EXCEPT;
  // all previous masks
  int oldExcepts;

  if( fegetenv( &fenv ))
  {
    return -1;
  }
  oldExcepts = fenv.__control & FE_ALL_EXCEPT;

  // mask
  fenv.__control |= newExcepts;
  fenv.__mxcsr   |= newExcepts << 7;

  return fesetenv( &fenv ) ? -1 : oldExcepts;
#else
  return fedisableexcept( exceptions );
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void setFPE()
{
#if defined(__APPLE__) && defined(__MACH__)
  fesetenv( FE_DFL_DISABLE_SSE_DENORMS_ENV );
#elif defined(__x86_64__)
  _MM_SET_FLUSH_ZERO_MODE( _MM_FLUSH_ZERO_ON );
  _MM_SET_DENORMALS_ZERO_MODE( _MM_DENORMALS_ZERO_ON );
#endif
#if defined(__x86_64__)
  enableFloatingPointExceptions( FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID );
#endif
}

} // namespace system
} // namespace LvArray
