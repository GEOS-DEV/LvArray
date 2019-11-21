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
 * Logger.hpp
 *
 *  Created on: Jul 17, 2017
 */

#ifndef CXX_UTILITIES_SRC_SRC_LOGGER_HPP_
#define CXX_UTILITIES_SRC_SRC_LOGGER_HPP_



// Source incldes
#include "CXX_UtilsConfig.hpp"

// TPL includes
#ifdef USE_ATK
  #if !defined(NDEBUG) && !defined(AXOM_DEBUG)
    #define AXOM_DEBUG
  #endif

  #include <axom/slic.hpp>
#endif

// System includes
#include <fstream>
#include <sstream>
#include <iostream>

#if defined(USE_CUDA)
  #include <cassert>
#endif

#if defined(USE_MPI)
  #include <mpi.h>
#endif


/**
 * @brief Macro used to turn on/off a function based on the log level.
 * @param[in] minLevel Minimum log level
 * @param[in] fn Function to filter
 */
#define LOG_LEVEL_FN( minLevel, fn )                                           \
  do {                                                                         \
    if( this->getLogLevel() >= minLevel )                                      \
    {                                                                          \
      fn;                                                                      \
    }                                                                          \
  } while( false )

/**
 * @brief Macro used to output messages based on the log level.
 * @param[in] minLevel Minimum log level
 * @param[in] msg Log message
 */
#define LOG_LEVEL( minLevel, msg )                                             \
  do {                                                                         \
    if( this->getLogLevel() >= minLevel )                                      \
    {                                                                          \
      std::ostringstream oss;                                                  \
      oss << msg;                                                              \
      std::cout << oss.str() << std::endl;                                     \
    }                                                                          \
  } while( false )

/**
 * @brief Macro used to output messages (only on rank 0) based on the log level.
 * @param[in] minLevel Minimum log level
 * @param[in] msg Log message
 */
#define LOG_LEVEL_RANK_0( minLevel, msg )                                      \
  do {                                                                         \
    if( this->getLogLevel() >= minLevel )                                      \
    {                                                                          \
      if( logger::internal::rank == 0 )                                        \
      {                                                                        \
        std::ostringstream oss;                                                \
        oss << msg;                                                            \
        std::cout << oss.str() << std::endl;                                   \
      }                                                                        \
    }                                                                          \
  } while( false )

/**
 * @brief Macro used to output messages (with one line per rank) based on the log level.
 * @param[in] minLevel Minimum log level
 * @param[in] msg Log message
 */
#define LOG_LEVEL_BY_RANK( minLevel, msg )                                     \
  do {                                                                         \
    if( this->getLogLevel() >= minLevel )                                      \
    {                                                                          \
      std::ostringstream oss;                                                  \
      if( logger::internal::using_cout_for_rank_stream )                       \
      {                                                                        \
        if( logger::internal::n_ranks > 1 )                                    \
        {                                                                      \
          oss << "Rank " << logger::internal::rank << ": ";                    \
        }                                                                      \
                                                                               \
        oss << msg;                                                            \
        std::cout << oss.str() << std::endl;                                   \
      }                                                                        \
      else                                                                     \
      {                                                                        \
        oss << msg;                                                            \
        logger::internal::rank_stream << oss.str() << std::endl;               \
      }                                                                        \
    }                                                                          \
  } while( false )


#define GEOS_LOG( msg )                                                        \
  do {                                                                         \
    std::ostringstream oss;                                                    \
    oss << msg;                                                                \
    std::cout << oss.str() << std::endl;                                       \
  } while( false )

#define GEOS_LOG_VAR( var ) GEOS_LOG( #var << " = " << var )

#define GEOS_LOG_RANK_0( msg )                                                 \
  do {                                                                         \
    if( logger::internal::rank == 0 )                                          \
    {                                                                          \
      std::ostringstream oss;                                                  \
      oss << msg;                                                              \
      std::cout << oss.str() << std::endl;                                     \
    }                                                                          \
  } while( false )

#define GEOS_LOG_RANK( msg )                                                   \
  do {                                                                         \
    std::ostringstream oss;                                                    \
    if( logger::internal::using_cout_for_rank_stream )                         \
    {                                                                          \
      if( logger::internal::n_ranks > 1 )                                      \
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

#define GEOS_LOG_RANK_VAR( var ) GEOS_LOG_RANK( #var " = " << var )

#if defined(__CUDA_ARCH__) && !defined(NDEBUG)
    #define GEOS_ERROR_IF( EXP, msg ) assert( !(EXP) )
    #define GEOS_ERROR( msg ) assert( false )
    #define GEOS_ASSERT_MSG( EXP, msg ) assert( EXP )
    #define GEOS_ASSERT( EXP ) assert( EXP );
#endif

#if defined(__CUDA_ARCH__) && defined(NDEBUG)
  #define GEOS_ERROR_IF( EXP, msg ) if( EXP ) asm( "trap;" )
  #define GEOS_ERROR( msg ) GEOS_ERROR_IF( true, msg )
  #define GEOS_ASSERT_MSG( EXP, msg ) ((void) 0)
  #define GEOS_ASSERT( EXP ) ((void) 0)
#endif

#if defined(USE_ATK)

  #if !defined(__CUDA_ARCH__)
    #define GEOS_ERROR_IF( EXP, msg ) SLIC_ERROR_IF( EXP, msg )
    #define GEOS_ERROR( msg ) SLIC_ERROR( msg )
    #define GEOS_ASSERT_MSG( EXP, msg ) SLIC_ASSERT_MSG( EXP, msg )
    #define GEOS_ASSERT( EXP ) SLIC_ASSERT( EXP )
  #endif

  #define GEOS_WARNING( msg ) SLIC_WARNING( msg )
  #define GEOS_WARNING_IF( EXP, msg ) SLIC_WARNING_IF( EXP, msg )
  #define GEOS_INFO( msg ) SLIC_INFO_IF( true, msg )
  #define GEOS_INFO_IF( EXP, msg ) SLIC_INFO_IF( EXP, msg )
  #define GEOS_CHECK( EXP, msg ) SLIC_CHECK_MSG( EXP, msg )

#else // #if defined(USE_ATK)

  #if !defined(__CUDA_ARCH__)
    #define GEOS_ERROR_IF( EXP, msg )                                      \
  do {                                                                     \
    if( EXP )                                                              \
    {                                                                      \
      std::cout << "***** GEOS_ERROR " <<std::endl;                        \
      std::cout << "***** FILE: " << __FILE__ << std::endl;                \
      std::cout << "***** LINE: " << __LINE__ << std::endl;                \
      std::cout << msg << std::endl;                                       \
      logger::abort();                                                     \
    }                                                                      \
  } while( false )

    #define GEOS_ERROR( msg ) GEOS_ERROR_IF( true, msg )

    #if !defined(NDEBUG)
      #define GEOS_ASSERT_MSG( EXP, msg ) GEOS_ERROR_IF( !(EXP), msg )
      #define GEOS_ASSERT( EXP ) GEOS_ASSERT_MSG( EXP, "" )
    #else
      #define GEOS_ASSERT_MSG( EXP, msg ) ((void) 0)
      #define GEOS_ASSERT( EXP ) ((void) 0)
    #endif

  #endif

  #define GEOS_WARNING_IF( EXP, msg )                                        \
  do {                                                                       \
    if( EXP )                                                                \
    {                                                                        \
      std::cout << "***** GEOS_WARNING "<<std::endl;                         \
      std::cout << "***** FILE: " << __FILE__ << std::endl;                  \
      std::cout << "***** LINE: " << __LINE__ << std::endl;                  \
      std::cout << msg << std::endl;                                         \
    }                                                                        \
  } while( false )

  #define GEOS_WARNING( msg ) GEOS_WARNING_IF( true, msg )

  #define GEOS_INFO_IF( EXP, msg )                                           \
  do {                                                                       \
    if( EXP )                                                                \
    {                                                                        \
      std::cout << "***** GEOS_INFO "<<std::endl;                            \
      std::cout << "***** FILE: " << __FILE__ << std::endl;                  \
      std::cout << "***** LINE: " << __LINE__ << std::endl;                  \
      std::cout << msg << std::endl;                                         \
    }                                                                        \
  } while( false )

  #define GEOS_INFO( msg ) GEOS_INFO_IF( true, msg )

  #if !defined(NDEBUG)
    #define GEOS_CHECK( EXP, msg ) GEOS_WARNING_IF( !(EXP), msg )
  #else
    #define GEOS_CHECK( EXP, msg ) ((void) 0)
  #endif

#endif // #if defined(USE_ATK)

#define GEOS_ERROR_IF_OP_MSG( lhs, OP, NOP, rhs, msg ) GEOS_ERROR_IF( lhs OP rhs,                                                  \
                                                                      "Expected " << #lhs << " " << #NOP << " " << #rhs << "\n" << \
                                                                      "  " << #lhs << " = " << lhs << "\n" <<                      \
                                                                      "  " << #rhs << " = " << rhs << "\n" <<                      \
                                                                      msg )

#define GEOS_ERROR_IF_EQ_MSG( lhs, rhs, msg ) GEOS_ERROR_IF_OP_MSG( lhs, ==, !=, rhs, msg )
#define GEOS_ERROR_IF_EQ( lhs, rhs ) GEOS_ERROR_IF_NE_MSG( lhs, rhs, "" )

#define GEOS_ERROR_IF_NE_MSG( lhs, rhs, msg ) GEOS_ERROR_IF_OP_MSG( lhs, !=, ==, rhs, msg )
#define GEOS_ERROR_IF_NE( lhs, rhs ) GEOS_ERROR_IF_NE_MSG( lhs, rhs, "" )

#define GEOS_ERROR_IF_GT_MSG( lhs, rhs, msg ) GEOS_ERROR_IF_OP_MSG( lhs, >, <=, rhs, msg )
#define GEOS_ERROR_IF_GT( lhs, rhs ) GEOS_ERROR_IF_GT_MSG( lhs, rhs, "" )

#define GEOS_ERROR_IF_GE_MSG( lhs, rhs, msg ) GEOS_ERROR_IF_OP_MSG( lhs, >=, <, rhs, msg )
#define GEOS_ERROR_IF_GE( lhs, rhs ) GEOS_ERROR_IF_GE_MSG( lhs, rhs, "" )

#define GEOS_ERROR_IF_LT_MSG( lhs, rhs, msg ) GEOS_ERROR_IF_OP_MSG( lhs, <, >=, rhs, msg )
#define GEOS_ERROR_IF_LT( lhs, rhs ) GEOS_ERROR_IF_GT_MSG( lhs, rhs, "" )

#define GEOS_ERROR_IF_LE_MSG( lhs, rhs, msg ) GEOS_ERROR_IF_OP_MSG( lhs, <=, >, rhs, msg )
#define GEOS_ERROR_IF_LE( lhs, rhs ) GEOS_ERROR_IF_GE_MSG( lhs, rhs, "" )


#define GEOS_ASSERT_OP_MSG( lhs, OP, rhs, msg ) GEOS_ASSERT_MSG( lhs OP rhs,                                                 \
                                                                 "  " << #lhs << " = " << lhs << "\n" <<                     \
                                                                 "  " << #rhs << " = " << rhs << "\n" <<                     \
                                                                 msg )

#define GEOS_ASSERT_EQ_MSG( lhs, rhs, msg ) GEOS_ASSERT_OP_MSG( lhs, ==, rhs, msg )
#define GEOS_ASSERT_EQ( lhs, rhs ) GEOS_ASSERT_EQ_MSG( lhs, rhs, "" )

#define GEOS_ASSERT_GT_MSG( lhs, rhs, msg ) GEOS_ASSERT_OP_MSG( lhs, >, rhs, msg )
#define GEOS_ASSERT_GT( lhs, rhs ) GEOS_ASSERT_GT_MSG( lhs, rhs, "" )

#define GEOS_ASSERT_GE_MSG( lhs, rhs, msg ) GEOS_ASSERT_OP_MSG( lhs, >=, rhs, msg )
#define GEOS_ASSERT_GE( lhs, rhs ) GEOS_ASSERT_GE_MSG( lhs, rhs, "" )


namespace logger
{

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
void InitializeLogger( MPI_Comm comm, const std::string & rank_output_dir="" );
#endif

void InitializeLogger( const std::string & rank_output_dir="" );

void FinalizeLogger();

} /* namespace logger */

#endif /* CXX_UTILITIES_SRC_SRC_LOGGER_HPP_ */
