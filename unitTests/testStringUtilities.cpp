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

// SRC includes
#include "StringUtilities.hpp"

// TPL includes
#include <gtest/gtest.h>

TEST( testStringUtilities, calculateSize )
{
  EXPECT_EQ( "0.0 KB", LvArray::calculateSize( 0 ) );
  EXPECT_EQ( "0.0 KB", LvArray::calculateSize( 10 ) );
  EXPECT_EQ( "0.5 KB", LvArray::calculateSize( 1 << 9 ) );
  EXPECT_EQ( "1.3 KB", LvArray::calculateSize( 1.34 * (1 << 10) ) );
  EXPECT_EQ( "1.4 KB", LvArray::calculateSize( 1.39 * (1 << 10) ) );

  EXPECT_EQ( "1.0 MB", LvArray::calculateSize( 1 << 20 ) );
  EXPECT_EQ( "105.4 MB", LvArray::calculateSize( 105.4 * (1 << 20) ) );
  EXPECT_EQ( "1000.3 MB", LvArray::calculateSize( 1000.3 * (1 << 20) ) );

  EXPECT_EQ( "53.5 GB", LvArray::calculateSize( 53.5 * (1 << 30) ) );
  EXPECT_EQ( "53.5 GB", LvArray::calculateSize( 53.5 * (1 << 30) ) );
  EXPECT_EQ( "789.7 GB", LvArray::calculateSize( 789.74563 * (1 << 30) ) );
}
