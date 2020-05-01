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
 * @file stackTrace.hpp
 */

#ifndef SRC_CODINGUTILITIES_STACKTRACE_HPP_
#define SRC_CODINGUTILITIES_STACKTRACE_HPP_

#include "CXX_UtilsConfig.hpp"

#include <signal.h>
#include <string>

namespace cxx_utilities
{

/// @brief @return Return a demangled stack trace of the last 25 frames.
std::string stackTrace();

/**
 * @brief Print signal information and a stack trace to standard out, optionally aborting.
 * @param sig The signal received.
 * @param exit If true abort execution.
 */
void stackTraceHandler( int const sig, bool const exit );

}


#endif /* SRC_CODINGUTILITIES_STACKTRACE_HPP_ */
