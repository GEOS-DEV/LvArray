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

#ifdef GEOSX_USE_ATK

#ifdef GEOSX_USE_MPI
#include "axom/slic/streams/LumberjackStream.hpp"
#endif

#include "axom/slic/streams/GenericOutputStream.hpp"

#endif

#include "stackTrace.hpp"

namespace geosx
{

namespace logger
{
namespace internal
{

int rank = 0;

int n_ranks = 1;

bool using_cout_for_rank_stream = true;

std::ofstream rank_stream;

} /* namespace internal */

#ifdef GEOSX_USE_MPI

void InitializeLogger(MPI_Comm mpi_comm, const std::string& rank_output_dir)
{
  MPI_Comm_rank(mpi_comm, &internal::rank);
  MPI_Comm_size(mpi_comm, &internal::n_ranks);

#ifdef GEOSX_USE_ATK
  axom::slic::initialize();
  axom::slic::setLoggingMsgLevel( axom::slic::message::Debug );

  std::string format =  std::string( 100, '*' ) + std::string( "\n" ) +
                        std::string( "[LEVEL in line <LINE> of file <FILE>]\n" ) +
                        std::string( "RANKS=<RANK>\n") +
                        std::string( "MESSAGE=<MESSAGE>\n" ) +
                        std::string( "<TIMESTAMP>\n" ) +
                        std::string( 100, '*' ) + std::string("\n");

  const int ranks_limit = 5;
  axom::slic::LumberjackStream * const stream = new axom::slic::LumberjackStream(&std::cout, mpi_comm, ranks_limit, format);
  axom::slic::addStreamToAllMsgLevels( stream );
#endif

  if ( rank_output_dir != "" )
  {
    internal::using_cout_for_rank_stream = false;

    if ( internal::rank != 0 )
    {
      MPI_Barrier(mpi_comm);
    }
    else
    {
      std::string cmd = "mkdir -p " + rank_output_dir;
      std::system(cmd.c_str());
      MPI_Barrier(mpi_comm);
    }

    std::string output_file_path = rank_output_dir + "/rank_" + std::to_string(internal::rank) + ".out";
    internal::rank_stream.rdbuf()->open(output_file_path, std::ios_base::out);
  }
}

#else

void InitializeLogger(const std::string& rank_output_dir)
{
#ifdef GEOSX_USE_ATK
  axom::slic::initialize();
  axom::slic::setLoggingMsgLevel( axom::slic::message::Debug );

  std::string format =  std::string( 100, '*' ) + std::string( "\n" ) +
                        std::string( "[LEVEL in line <LINE> of file <FILE>]\n" ) +
                        std::string( "MESSAGE=<MESSAGE>\n" ) +
                        std::string( "<TIMESTAMP>\n" ) +
                        std::string( 100, '*' ) + std::string("\n");

  axom::slic::GenericOutputStream * const stream = new axom::slic::GenericOutputStream(&std::cout, format );
  axom::slic::addStreamToAllMsgLevels( stream );
#endif

  if ( rank_output_dir != "" )
  {
    internal::using_cout_for_rank_stream = false;

    std::string cmd = "mkdir -p " + rank_output_dir;
    std::system(cmd.c_str());

    std::string output_file_path = rank_output_dir + "/rank_" + std::to_string(internal::rank) + ".out";
    internal::rank_stream.rdbuf()->open(output_file_path, std::ios_base::out);
  }
}

#endif

void FinalizeLogger()
{
#ifdef GEOSX_USE_ATK
  axom::slic::flushStreams();
  axom::slic::finalize();
#endif
}

[[noreturn]] void geos_abort()
{
  cxx_utilities::handler1(EXIT_FAILURE);
#ifdef GEOSX_USE_MPI
  int mpi = 0;
  MPI_Initialized( &mpi );
  if ( mpi )
  {
    MPI_Abort( MPI_COMM_GEOSX, EXIT_FAILURE );
  }
  else
#endif
  {
    exit( EXIT_FAILURE );
  }
}

} /* namespace logger */

} /* namespace geosx */

#ifdef __clang__
#pragma clang diagnostic pop
#endif
