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
#include <sstream>
#include <iostream>

#include "CXX_UtilsConfig.hpp"

#ifdef USE_AXOM
#include "axom/slic/interface/slic.hpp"
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#define GEOS_LOG( msg )                                                          \
  do {                                                                         \
    std::ostringstream oss;                                                    \
    oss << msg;                                                                \
    std::cout << oss.str() << std::endl;                                       \
  } while( false )

#define GEOS_LOG_RANK_0( msg )                                                   \
  do {                                                                         \
    if( logger::internal::rank == 0 )                                           \
    {                                                                          \
      std::ostringstream oss;                                                  \
      oss << msg;                                                              \
      std::cout << oss.str() << std::endl;                                     \
    }                                                                          \
  } while( false )

#define GEOS_LOG_RANK( msg )                                                     \
  do {                                                                         \
    std::ostringstream oss;                                                    \
    if( logger::internal::using_cout_for_rank_stream )                          \
    {                                                                          \
      if( logger::internal::n_ranks > 1 )                                       \
      {                                                                        \
        oss << "Rank " << logger::internal::rank << ": ";                      \
      }                                                                        \
                                                                               \
      oss << msg;                                                              \
      std::cout << oss.str() << std::endl;                                     \
    }                                                                          \
    else                                                                       \
    {                                                                          \
      oss << msg;                                                              \
      logger::internal::rank_stream << oss.str() << std::endl;                 \
    }                                                                          \
  } while( false )

#ifdef USE_AXOM

/* Always active */
#define GEOS_ERROR( msg ) SLIC_ERROR( msg )
#define GEOS_ERROR_IF( EXP, msg ) SLIC_ERROR_IF( EXP, msg )
#define GEOS_WARNING( msg ) SLIC_WARNING( msg )
#define GEOS_WARNING_IF( EXP, msg ) SLIC_WARNING_IF( EXP, msg )
#define GEOS_INFO( msg ) SLIC_INFO_IF( true, msg )
#define GEOS_INFO_IF( EXP, msg ) SLIC_INFO_IF( EXP, msg )

/* Active with AXOM_DEBUG */
#define GEOS_ASSERT( EXP ) SLIC_ASSERT( EXP )
#define GEOS_ASSERT_MSG( EXP, msg ) SLIC_ASSERT_MSG( EXP, msg )
#define GEOS_CHECK( EXP, msg ) SLIC_CHECK_MSG( EXP, msg )


#else

#define GEOS_ERROR_IF( EXP, msg )                               \
  do {                                                        \
    if( EXP )                                                  \
    {                                                         \
      std::cout << "***** GEOS_ERROR "<<std::endl;            \
      std::cout << "***** FILE: " << __FILE__ << std::endl;   \
      std::cout << "***** LINE: " << __LINE__ << std::endl;   \
      std::cout << msg << std::endl;                          \
      logger::abort();                                        \
    }                                                         \
  } while( false )

#define GEOS_WARNING_IF( EXP, msg )                             \
  do {                                                        \
    if( EXP )                                                  \
    {                                                         \
      std::cout << "***** GEOS_WARNING "<<std::endl;          \
      std::cout << "***** FILE: " << __FILE__ << std::endl;   \
      std::cout << "***** LINE: " << __LINE__ << std::endl;   \
      std::cout << msg << std::endl;                          \
    }                                                         \
  } while( false )

#define GEOS_INFO_IF( EXP, msg )                                \
  do {                                                        \
    if( EXP )                                                  \
    {                                                         \
      std::cout << "***** GEOS_INFO "<<std::endl;             \
      std::cout << "***** FILE: " << __FILE__ << std::endl;   \
      std::cout << "***** LINE: " << __LINE__ << std::endl;   \
      std::cout << msg << std::endl;                          \
    }                                                         \
  } while( false )

#ifdef GEOSX_DEBUG

#define GEOS_ASSERT_MSG( EXP, msg ) GEOS_ERROR_IF( !(EXP), msg )

#define GEOS_CHECK( EXP, msg ) GEOS_WARNING_IF( !(EXP), msg )

#else /* #ifdef GEOSX_DEBUG */

#define GEOS_ASSERT_MSG( EXP, msg ) ((void) 0)

#define GEOS_CHECK( EXP, msg ) ((void) 0)

#endif /* #ifdef GEOSX_DEBUG */

#define GEOS_ERROR( msg ) GEOS_ERROR_IF( true, msg )

#define GEOS_WARNING( msg ) GEOS_WARNING_IF( true, msg )

#define GEOS_INFO( msg ) GEOS_INFO_IF( true, msg )

#define GEOS_ASSERT( EXP ) GEOS_ASSERT_MSG( EXP, "" )

#endif /* #ifdef GEOSX_USE_ATK */

#ifndef NDEBUG
#define GEOS_ERROR_IF_DEBUG( EXP, msg ) GEOS_ERROR_IF( EXP, msg )
#else
#define GEOS_ERROR_IF_DEBUG( EXP, msg ) ((void) 0)
#endif


namespace logger
{

#ifndef USE_MPI
[[noreturn]]
#endif
void abort();


namespace internal
{

extern int rank;

extern int n_ranks;

extern bool using_cout_for_rank_stream;

extern std::ofstream rank_stream;

#ifdef USE_MPI
extern MPI_Comm comm;
#endif
} /* namespace internal */

#ifdef USE_MPI
void InitializeLogger( MPI_Comm comm, const std::string& rank_output_dir="" );
#else
void InitializeLogger( const std::string& rank_output_dir="" );
#endif

void FinalizeLogger();

} /* namespace logger */

#endif /* CXX_UTILITIES_SRC_SRC_LOGGER_HPP_ */
