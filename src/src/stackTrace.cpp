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

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <cxxabi.h>
#include <sys/ucontext.h>
#include <sstream>
#include <cfenv>
#include <string.h>

#include "Macros.hpp"
#include "stackTrace.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif


constexpr int MAX_FRAMES = 25;

namespace cxx_utilities
{

namespace internal
{

std::string demangle( char* backtraceString, int frame )
{
  char* mangledName = nullptr;
  char* functionOffset = nullptr;
  char* returnOffset = nullptr;

#ifdef __APPLE__
  /* On apple machines the mangled function name always starts at the 58th
   * character */
  constexpr int APPLE_OFFSET = 58;
  mangledName = backtraceString + APPLE_OFFSET;
  for( char* p = backtraceString ; *p ; ++p )
  {
    if( *p == '+' )
    {
      functionOffset = p;
    }
    returnOffset = p;
  }
#else
  for( char* p = backtraceString ; *p ; ++p )
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

  // if the line could be processed, attempt to demangle the symbol
  if( mangledName && functionOffset && returnOffset &&
      mangledName < functionOffset )
  {
    *mangledName = 0;
    mangledName++;
#ifdef __APPLE__
    *(functionOffset - 1) = 0;
#endif
    *functionOffset = 0;
    ++functionOffset;
    *returnOffset = 0;
    ++returnOffset;

    int status;
    char* realName = abi::__cxa_demangle( mangledName, nullptr, nullptr,
                                          &status );

    // if demangling is successful, output the demangled function name
    if( status == 0 )
    {
      oss << "Frame " << frame << ": " << realName << std::endl;
    }
    // otherwise, output the mangled function name
    else
    {
      oss << "Frame " << frame << ": " << mangledName << std::endl;
    }

    free( realName );
  }
  // otherwise, print the whole line
  else
  {
    oss << "Frame " << frame << ": " << backtraceString << std::endl;
  }

  return ( oss.str() );
}

} // namespace internal

std::string stackTrace( )
{
  void *array[ MAX_FRAMES ];

  const int size = backtrace( array, MAX_FRAMES );
  char** strings = backtrace_symbols( array, size );

  // skip first stack frame (points here)
  std::ostringstream oss;
  oss << "\n** StackTrace of " << size - 1 << " frames **\n";
  for( int i = 1 ; i < size && strings != nullptr ; ++i )
  {
    oss << internal::demangle( strings[ i ], i );
  }

  oss << "=====\n";

  free( strings );

  return ( oss.str() );
}

std::string getFpeDetails()
{
  std::ostringstream oss;
  int const fpe = std::fetestexcept( FE_ALL_EXCEPT );

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

void handler( int sig, int exitFlag, int CXX_UTILS_UNUSED_ARG( exitCode ) )
{
  std::ostringstream oss;

  if( sig >= 0 && sig < NSIG )
  {
    // sys_signame not available on linux, so just print the code; strsignal is POSIX
    oss << "Received signal " << sig << ": " << strsignal( sig ) << std::endl;

    if( sig == SIGFPE )
    {
      oss << getFpeDetails() << std::endl;
    }
  }
  oss << stackTrace() << std::endl;
  std::cout << oss.str();

  if( exitFlag == 1 )
  {
#ifdef USE_MPI
    int mpi = 0;
    MPI_Initialized( &mpi );
    if( mpi )
    {
      MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
    }
    else
#endif
    {
      abort();
    }

  }
}

} // namespace cxx_utilities
