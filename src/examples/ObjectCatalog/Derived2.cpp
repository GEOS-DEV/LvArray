// Copyright (c) 2018, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-746361. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the GEOSX Simulation Framework.

//
// GEOSX is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
/*
 * Copyright (c) 2015, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

#include "Derived2.hpp"


Derived2::Derived2( int junk, double const & junk2, Parameter& param ):
  Base(junk,junk2,param)
{
  std::cout<<"calling Derived2 constructor with arguments ("<<junk<<" "<<junk2<<")"<<std::endl;
}

Derived2::~Derived2()
{
  std::cout<<"calling Derived2 destructor"<<std::endl;
}

REGISTER_CATALOG_ENTRY( Base, Derived2, int, double const &, Parameter& )
