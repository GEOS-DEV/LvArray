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

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif

/*
 * Logger.cpp
 *
 *  Created on: Aug 31, 2017
 *      Author: settgast
 */

#include "Logger.hpp"

#include <stdlib.h>
#include <fstream>
#include <string>

#ifdef USE_AXOM

#ifdef USE_MPI
#include "axom/slic/streams/LumberjackStream.hpp"
#endif

#include "axom/slic/streams/GenericOutputStream.hpp"

using namespace axom;

#endif

#include "stackTrace.hpp"

namespace logger
{
namespace internal
{

int rank = 0;

int n_ranks = 1;

bool using_cout_for_rank_stream = true;

std::ofstream rank_stream;

#ifdef USE_MPI
MPI_Comm comm;
#endif

#ifdef USE_AXOM
slic::GenericOutputStream* createGenericStream()
{
  std::string format =  std::string( 100, '*' ) + std::string( "\n" ) +
                       std::string( "[<LEVEL> in line <LINE> of file <FILE>]\n" ) +
                       std::string( "MESSAGE=<MESSAGE>\n" ) +
                       std::string( "Rank " ) + std::to_string( internal::rank ) + std::string( "\n" ) +
                       std::string( "<TIMESTAMP>\n" ) +
                       std::string( 100, '*' ) + std::string( "\n" );

  return new slic::GenericOutputStream( &std::cout, format );
}
#endif

} /* namespace internal */

#ifdef USE_MPI

void InitializeLogger( MPI_Comm mpi_comm, const std::string& rank_output_dir )
{
  internal::comm = mpi_comm;
  MPI_Comm_rank( mpi_comm, &internal::rank );
  MPI_Comm_size( mpi_comm, &internal::n_ranks );

#ifdef USE_AXOM
  slic::initialize();
  slic::setLoggingMsgLevel( slic::message::Debug );
  slic::GenericOutputStream* stream = internal::createGenericStream();
  slic::addStreamToAllMsgLevels( stream );
#endif

  if( rank_output_dir != "" )
  {
    internal::using_cout_for_rank_stream = false;

    if( internal::rank != 0 )
    {
      MPI_Barrier( mpi_comm );
    }
    else
    {
      std::string cmd = "mkdir -p " + rank_output_dir;
      int ret = std::system( cmd.c_str());
      if( ret != 0 )
      {
        GEOS_LOG( "Failed to initialize Logger: command '" << cmd << "' exited with code " << std::to_string( ret ));
        abort();
      }
      MPI_Barrier( mpi_comm );
    }

    std::string output_file_path = rank_output_dir + "/rank_" + std::to_string( internal::rank ) + ".out";
    internal::rank_stream.rdbuf()->open( output_file_path, std::ios_base::out );
  }
}

#else

void InitializeLogger( const std::string& rank_output_dir )
{
#ifdef USE_AXOM
  slic::initialize();
  slic::setLoggingMsgLevel( slic::message::Debug );
  slic::GenericOutputStream* stream = internal::createGenericStream();
  slic::addStreamToAllMsgLevels( stream );
#endif

  if( rank_output_dir != "" )
  {
    internal::using_cout_for_rank_stream = false;

    std::string cmd = "mkdir -p " + rank_output_dir;
    std::system( cmd.c_str());

    std::string output_file_path = rank_output_dir + "/rank_" + std::to_string( internal::rank ) + ".out";
    internal::rank_stream.rdbuf()->open( output_file_path, std::ios_base::out );
  }
}

#endif

void FinalizeLogger()
{
#ifdef USE_AXOM
  slic::flushStreams();
  slic::finalize();
#endif
}

#ifndef USE_MPI
[[noreturn]]
#endif
void abort()
{
  cxx_utilities::handler1( EXIT_FAILURE );
}

} /* namespace logger */

#ifdef __clang__
#pragma clang diagnostic pop
#endif
