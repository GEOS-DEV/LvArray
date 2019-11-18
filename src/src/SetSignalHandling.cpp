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
 * @file FloatingPointExceptions.cpp
 */

#include "SetSignalHandling.hpp"
#include "stackTrace.hpp"


namespace cxx_utilities
{

void setSignalHandling( void (*handler)( int ) )
{
  signal( SIGHUP, handler );
  signal( SIGINT, handler );
  signal( SIGQUIT, handler );
  signal( SIGILL, handler );
  signal( SIGTRAP, handler );
  signal( SIGABRT, handler );
#if  (defined(_POSIX_C_SOURCE) && !defined(_DARWIN_C_SOURCE))
  signal( SIGPOLL, handler );
#else
  signal( SIGIOT, handler );
  signal( SIGEMT, handler );
#endif
  signal( SIGFPE, handler );
  signal( SIGKILL, handler );
  signal( SIGBUS, handler );
  signal( SIGSEGV, handler );
  signal( SIGSYS, handler );
  signal( SIGPIPE, handler );
//  signal( SIGALRM, handler );
  signal( SIGTERM, handler );
//  signal( SIGURG, handler );
//  signal( SIGSTOP, handler );
//  signal( SIGTSTP, handler );
//  signal( SIGCONT, handler );
//  signal( SIGCHLD, handler );


  return;
}



} /* namespace geosx */
