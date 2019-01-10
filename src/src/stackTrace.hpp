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
 * stackTrace.hpp
 *
 *  Created on: May 31, 2016
 *      Author: settgast
 */

#ifndef SRC_CODINGUTILITIES_STACKTRACE_HPP_
#define SRC_CODINGUTILITIES_STACKTRACE_HPP_

#define GEOSX_USE_MPI

#include <signal.h>

namespace cxx_utilities
{

void handler( int sig, int exitFlag=1, int exitCode=1 );


inline void handler0( int sig )
{
  handler( sig, 0, 1 );
}

inline void handler1( int sig )
{
  handler( sig, 1, 1 );
}


}


#endif /* SRC_CODINGUTILITIES_STACKTRACE_HPP_ */
