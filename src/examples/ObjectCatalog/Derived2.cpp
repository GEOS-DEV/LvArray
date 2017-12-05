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
