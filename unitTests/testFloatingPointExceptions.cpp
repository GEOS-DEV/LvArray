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

// Source includes
#include "testFloatingPointExceptionsHelpers.hpp"
#include "system.hpp"
#include "system.hpp"

// TPL includes
#include <gtest/gtest.h>

// System includes
#include <fenv.h>
#include <cmath>
#include <float.h>


#if defined(__x86_64__)

using namespace testFloatingPointExceptionsHelpers;

const char IGNORE_OUTPUT[] = ".*";

TEST( TestFloatingPointEnvironment, test_FE_UNDERFLOW_flush )
{
  LvArray::system::enableFloatingPointExceptions( FE_UNDERFLOW );
  EXPECT_DEATH_IF_SUPPORTED( uf_test( DBL_MIN, 2 ), IGNORE_OUTPUT );
  LvArray::system::disableFloatingPointExceptions( FE_UNDERFLOW );

  LvArray::system::setFPE();
  double fpnum = uf_test( DBL_MIN, 2 );
  int fpclassification = std::fpclassify( fpnum );
  EXPECT_NE( fpclassification, FP_SUBNORMAL );
}

TEST( TestFloatingPointEnvironment, test_FE_DIVBYZERO )
{
  LvArray::system::setFPE();
  EXPECT_DEATH_IF_SUPPORTED( func3( 0.0 ), IGNORE_OUTPUT );
}


TEST( TestFloatingPointEnvironment, test_FE_OVERFLOW )
{
  LvArray::system::setFPE();
  EXPECT_DEATH_IF_SUPPORTED( of_test( 2, DBL_MAX ), IGNORE_OUTPUT );
}

TEST( TestFloatingPointEnvironment, test_FE_INVALID )
{
  LvArray::system::setFPE();
  EXPECT_DEATH_IF_SUPPORTED( invalid_test( 0.0 ), IGNORE_OUTPUT );
}

#endif
