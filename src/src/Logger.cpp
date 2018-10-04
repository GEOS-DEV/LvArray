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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wglobal-constructors"

/*
 * Logger.cpp
 *
 *  Created on: Aug 31, 2017
 *      Author: settgast
 */

#include "Logger.hpp"
#include <mpi.h>
#include <stdlib.h>
#include <fstream>
#include <string>

#include "slic/GenericOutputStream.hpp"
#include "slic/LumberjackStream.hpp"

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

void InitializeLogger(int mpi_rank, int mpi_n_ranks, int mpi_comm, const std::string& rank_output_dir)
{
  internal::rank = mpi_rank;
  internal::n_ranks = mpi_n_ranks;
  axom::slic::initialize();
  axom::slic::setLoggingMsgLevel( axom::slic::message::Debug );

  if ( internal::n_ranks > 1 )
  {
    std::string format =  std::string( 100, '*' ) + std::string( "\n" ) +
                          std::string( "MESSAGE=<MESSAGE>\n" ) +
                          std::string( "<TIMESTAMP>\n" ) +
                          std::string( "LEVEL=<LEVEL>\n" ) +
                          std::string( "RANKS=<RANK>\n") +
                          std::string( "LOCATION=<FILE> : <LINE>\n" ) +
                          std::string( 100, '*' ) + std::string("\n");

    const int ranks_limit = 5;
    axom::slic::LumberjackStream * const stream = new axom::slic::LumberjackStream(&std::cout, mpi_comm, ranks_limit, format);
    axom::slic::addStreamToAllMsgLevels( stream );
  }
  else
  {
    std::string format =  std::string( 100, '*' ) + std::string( "\n" ) +
                          std::string( "MESSAGE=<MESSAGE>\n" ) +
                          std::string( "<TIMESTAMP>\n" ) +
                          std::string( "LEVEL=<LEVEL>\n" ) +
                          std::string( "RANKS=<RANK>\n") +
                          std::string( "LOCATION=<FILE> : <LINE>\n" ) +
                          std::string( 100, '*' ) + std::string("\n");

    axom::slic::GenericOutputStream * const stream = new axom::slic::GenericOutputStream(&std::cout, format );
    axom::slic::addStreamToAllMsgLevels( stream );
  }

  if ( rank_output_dir != "" )
  {
    internal::using_cout_for_rank_stream = false;

#ifdef GEOSX_USE_MPI
    if ( rank != 0 )
    {
      MPI_Barrier(MPI_COMM_GEOSX);
    }
    else
    {
#endif
      std::string cmd = "mkdir -p " + rank_output_dir;
      std::system(cmd.c_str());
#ifdef GEOSX_USE_MPI
      MPI_Barrier(MPI_COMM_GEOSX);
    }
#endif

    std::string output_file_path = rank_output_dir + "/rank_" + std::to_string(internal::rank) + ".out";
    internal::rank_stream.rdbuf()->open(output_file_path, std::ios_base::out);
  }
}

void FinalizeLogger()
{
  axom::slic::flushStreams();
  axom::slic::finalize();
}

} /* namespace logger */

} /* namespace geosx */

#pragma clang diagnostic pop
