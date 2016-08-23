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

template< typename T, typename... ARGS >
std::unique_ptr<T> wrapper( ARGS&&... args )
{
  return std::make_unique<T>(std::forward<ARGS>(args)...);
}

int main( int argc, char *argv[] )
{
  std::cout<<"EXECUTING MAIN"<<std::endl;
  int junk = 1;
  double junk2 = 3.14;
  double junk3 = 2*3.14;
//  std::unique_ptr<Base> dptr1a = std::make_unique<Derived1>( junk,junk2 );

//  std::unique_ptr<Base> derived1 = Base::CatalogInterface::Factory("derived1",junk,junk2);
//  std::unique_ptr<Base> derived2 = Base::CatalogInterface::Factory("derived2",junk,junk3);

  std::cout<<"EXITING MAIN"<<std::endl;
}
