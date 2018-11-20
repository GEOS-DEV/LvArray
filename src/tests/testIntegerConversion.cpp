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
 * GEOSX is a free software; you can redistrubute it and/or modify it under
 * the terms of the GNU Lesser General Public Liscense (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

#include "gtest/gtest.h"
#include "IntegerConversion.hpp"


typedef  int32_t  int32;
typedef uint32_t uint32;
typedef  int64_t  int64;
typedef uint64_t uint64;

TEST( IntegerConversion, unsignedToSigned )
{
  uint32 source0 = std::numeric_limits<int32>::max();
  uint32 source1 = std::numeric_limits<uint32>::max();
  uint64 source2 = std::numeric_limits<int64>::max();
  uint64 source3 = std::numeric_limits<uint64>::max();

  int32 compare0 = std::numeric_limits<int32>::max();
  int64 compare2 = std::numeric_limits<int64>::max();

  ASSERT_TRUE( compare0 == integer_conversion<int32>(source0) );
  ASSERT_DEATH_IF_SUPPORTED( integer_conversion<int32>(source1), "" );

  ASSERT_TRUE( compare2 == integer_conversion<int32>(source2) );
  ASSERT_DEATH_IF_SUPPORTED( integer_conversion<int64>(source3), "" );
}


int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  logger::InitializeLogger( MPI_COMM_WORLD );

  int result = 0;
  testing::InitGoogleTest( &argc, argv );
  result = RUN_ALL_TESTS();

  logger::FinalizeLogger();
  MPI_Finalize();
  return result;
}

