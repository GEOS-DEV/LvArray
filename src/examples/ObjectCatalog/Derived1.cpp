/*
 * Copyright (c) 2015, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

#include "Derived1.hpp"


Derived1::Derived1( int junk, double const & junk2):
  Base(junk,junk2)
{
  std::cout<<"calling Derived1 constructor with arguments ("<<junk<<" "<<junk2<<")"<<std::endl;
}

Derived1::~Derived1()
{
  std::cout<<"calling Derived1 destructor"<<std::endl;
}

REGISTER_CATALOG_ENTRY( Base, Derived1, int, double const & )


