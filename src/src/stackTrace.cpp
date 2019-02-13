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
#include <unistd.h>
#include <iostream>
#include <cxxabi.h>
#include <sys/ucontext.h>

#include "stackTrace.hpp"

#ifdef GEOSX_USE_MPI
#include <mpi.h>
#endif

namespace cxx_utilities
{

void handler( int sig, int exitFlag, int exitCode )
{
  fprintf( stdout, "executing stackTrace.cpp::handler(%d,%d,%d)\n", sig, exitFlag, exitCode );
  void *array[100];
  int size;

  // get void*'s for all entries on the stack
  size = backtrace( array, 100 );
  char ** messages    = backtrace_symbols( array, size );
  fprintf( stdout, "attempting unmangled trace: \n" );
  fprintf( stdout, "0         1         2         3         4         5         6         7         8         9         : \n" );
  fprintf( stdout, "0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789: \n" );

  // skip first stack frame (points here)
  for( int i = 1 ; i < size && messages != nullptr ; ++i )
  {
    char *mangled_name = nullptr, *offset_begin = nullptr, *offset_end = nullptr;

#ifdef __APPLE__
    mangled_name = &(messages[i][58]);
    for( char *p = messages[i] ; *p ; ++p )
    {
      if( *p == '+' )
      {
        offset_begin = p;
      }
      offset_end = p;
    }

#else
    // find parentheses and +address offset surrounding mangled name
    for( char *p = messages[i] ; *p ; ++p )
    {
      if( *p == '(' )
      {
        mangled_name = p;
      }
      else if( *p == '+' )
      {
        offset_begin = p;
      }
      else if( *p == ')' )
      {
        offset_end = p;
        break;
      }
    }
#endif

    // if the line could be processed, attempt to demangle the symbol
    if( mangled_name && offset_begin && offset_end &&
        mangled_name < offset_begin )
    {
      *mangled_name++ = '\0';
#ifdef __APPLE__
#ifdef __MACH__
      *(offset_begin-1) = '\0';
#endif
#endif
      *offset_begin++ = '\0';
      *offset_end++ = '\0';

      int status;
      char * real_name = abi::__cxa_demangle( mangled_name, nullptr, nullptr, &status );

      // if demangling is successful, output the demangled function name
      if( status == 0 )
      {
        std::cout <<messages[i]<<"("<<i<<") "<< " : "
                  << real_name << "+" << offset_begin << offset_end
                  << std::endl;

      }
      // otherwise, output the mangled function name
      else
      {
        std::cout << messages[i]<<"("<<i<<") "<< " : "
                  << mangled_name << "+" << offset_begin << offset_end
                  << std::endl;
      }
      free( real_name );
    }
    // otherwise, print the whole line
    else
    {
      std::cout << messages[i] << std::endl;
    }
  }
  std::cout << std::endl;

  free( messages );
  if( exitFlag == 1 )
  {
#ifdef GEOSX_USE_MPI
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
}
