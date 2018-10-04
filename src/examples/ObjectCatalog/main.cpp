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
 * Copyright (c) 2015, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */


#include <string>
#include <iostream>

#include "Logger.hpp"
#include "Base.hpp"
#include "Derived1.hpp"
using namespace cxx_utilities;

int main( int argc, char *argv[] )
{
  GEOS_LOG("EXECUTING MAIN");
  int junk = 1;
  double junk2 = 3.14;
  double junk3 = 2*3.14;
  Parameter param;


  GEOS_LOG("Attempting to create a Derived1 object");
  std::unique_ptr<Base> derived1 = Base::CatalogInterface::Factory( "derived1", junk, junk2, param);
  GEOS_LOG("Attempting to create a Derived2 object");
  std::unique_ptr<Base> derived2 = Base::CatalogInterface::Factory( "derived2", junk, junk3, param);

  Base::CatalogInterface::catalog_cast<Derived1>(*(derived2.get()));
  GEOS_LOG("EXITING MAIN");
}
