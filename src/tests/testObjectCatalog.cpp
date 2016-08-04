/*
 * Copyright (c) 2015, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */


#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif

#include <gtest/gtest.h>

#ifdef __clang__
#pragma clang diagnostic push
#endif

#include "ObjectCatalog.hpp"
#include <sys/time.h>
#include <stdint.h>
#include <string>

using namespace cxx_utilities;

class Base
{
public:
  Base( int& junk, double const & junk2)
  {
    std::cout<<"calling Base constructor with arguments ("<<junk<<" "<<junk2<<")"<<std::endl;
  }

  ~Base()
  {
    std::cout<<"calling Base destructor"<<std::endl;
  }

  using CatalogInterface = objectcatalog::CatalogInterface< Base, int&, double const &  >;
  static CatalogInterface::CatalogType& GetCatalog()
  {
    static CatalogInterface::CatalogType catalog;
    return catalog;
  }
};

class Derived1 : public Base
{
public:
  Derived1( int& junk, double const & junk2):
    Base(junk,junk2)
  {
    std::cout<<"calling Derived1 constructor with arguments ("<<junk<<" "<<junk2<<")"<<std::endl;
  }

  ~Derived1()
  {
    std::cout<<"calling Derived1 destructor"<<std::endl;
  }
  static std::string CatalogName() { return "derived1"; }
};
REGISTER_CATALOG_ENTRY( Base, Derived1, int&, double const & )


TEST(testObjectCatalog,testRegistration)
{
  int junk = 1;
  double junk2 = 3.14;
  std::unique_ptr<Base> derived1 = Base::CatalogInterface::Factory("derived1",junk,junk2);
}
