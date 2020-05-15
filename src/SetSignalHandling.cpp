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

#include <map>


namespace LvArray
{

typedef void ( *handle_type )( int );
static std::map< int, handle_type > initialHandler;

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

void resetSignalHandling()
{
  for( auto a : initialHandler )
  {
    signal( a.first, a.second );
  }
}


} /* namespace geosx */
