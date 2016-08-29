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

#include "Base.hpp"
#include "Derived1.hpp"
using namespace cxx_utilities;


int main( int argc, char *argv[] )
{
  std::cout<<"EXECUTING MAIN"<<std::endl;
  int junk = 1;
  double junk2 = 3.14;
  double junk3 = 2*3.14;
  Parameter param;

  std::unique_ptr<Base> derived1 = Base::CatalogInterface::Factory( "derived1", junk, junk2, param);
  std::unique_ptr<Base> derived2 = Base::CatalogInterface::Factory( "derived2", junk, junk3, param);

  Base::CatalogInterface::catalog_cast<Derived1>(*(derived2.get()));


  std::cout<<"EXITING MAIN"<<std::endl;
}

