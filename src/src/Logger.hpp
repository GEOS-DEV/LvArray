/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
 * Logger.hpp
 *
 *  Created on: Jul 17, 2017
 *      Author: settgast1
 */

#ifndef CXX_UTILITIES_SRC_SRC_LOGGER_HPP_
#define CXX_UTILITIES_SRC_SRC_LOGGER_HPP_

#include <fstream>
#include "common/GeosxConfig.hpp"
#include "axom/slic/interface/slic.hpp"

#ifdef GEOSX_USE_MPI
#include <mpi.h>
#endif

#define GEOS_LOG(msg)                                                          \
  do {                                                                         \
    std::ostringstream oss;                                                    \
    oss << msg;                                                                \
    std::cout << oss.str() << std::endl;                                       \
  } while (false);

#define GEOS_LOG_RANK_0(msg)                                                   \
  do {                                                                         \
    if (geosx::logger::internal::rank == 0)                                    \
    {                                                                          \
      std::ostringstream oss;                                                  \
      oss << msg;                                                              \
      std::cout << oss.str() << std::endl;                                     \
    }                                                                          \
  } while (false)

#define GEOS_LOG_RANK(msg)                                                     \
  do {                                                                         \
    std::ostringstream oss;                                                    \
    if (geosx::logger::internal::using_cout_for_rank_stream)                   \
    {                                                                          \
      if (geosx::logger::internal::n_ranks > 1)                                \
      {                                                                        \
        oss << "Rank " << geosx::logger::internal::rank << ": ";               \
      }                                                                        \
                                                                               \
      oss << msg;                                                              \
      std::cout << oss.str() << std::endl;                                     \
    }                                                                          \
    else                                                                       \
    {                                                                          \
      oss << msg;                                                              \
      geosx::logger::internal::rank_stream << oss.str() << std::endl;          \
    }                                                                          \
  } while (false)

/* Always active */
#define GEOS_ERROR(msg) SLIC_ERROR(msg)
#define GEOS_ERROR_IF(EXP, msg) SLIC_ERROR_IF(EXP, msg)
#define GEOS_WARNING(msg) SLIC_WARNING(msg)
#define GEOS_WARNING_IF(EXP, msg) SLIC_WARNING_IF(EXP, msg)
#define GEOS_INFO(EXP, msg) SLIC_INFO(EXP, msg)
#define GEOS_INFO_IF(EXP, msg) SLIC_INFO_IF(EXP, msg)

/* Active with AXOM_DEBUG */
#define GEOS_ASSERT(EXP) SLIC_ASSERT(EXP)
#define GEOS_ASSERT_MSG(EXP, msg) SLIC_ASSERT_MSG(EXP, msg)
#define GEOS_CHECK(EXP, msg) SLIC_CHECK_MSG(EXP, msg)

namespace geosx
{

namespace logger
{

namespace internal
{

extern int rank;

extern int n_ranks;

extern bool using_cout_for_rank_stream;

extern std::ofstream rank_stream;

} /* namespace internal */

#ifdef GEOSX_USE_MPI
void InitializeLogger(MPI_Comm comm, const std::string& rank_output_dir="");
#else
void InitializeLogger(const std::string& rank_output_dir="");
#endif

void FinalizeLogger();

} /* namespace logger */

} /* namespace geosx */

#endif /* CXX_UTILITIES_SRC_SRC_LOGGER_HPP_ */
