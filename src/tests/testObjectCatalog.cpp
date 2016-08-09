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

#define OBJECTCATALOGVERBOSE 2
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

  using CatalogInterface = CatalogInterface< Base, int&, double const &  >;
  static CatalogInterface::CatalogType& GetCatalog()
  {
    static CatalogInterface::CatalogType catalog;
    return catalog;
  }

  virtual std::string getName() = 0;
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
  std::string getName() { return CatalogName(); }

};
REGISTER_CATALOG_ENTRY( Base, Derived1, int&, double const & )

class Derived2 : public Base
{
public:
  Derived2( int& junk, double const & junk2):
    Base(junk,junk2)
  {
    std::cout<<"calling Derived2 constructor with arguments ("<<junk<<" "<<junk2<<")"<<std::endl;
  }

  ~Derived2()
  {
    std::cout<<"calling Derived2 destructor"<<std::endl;
  }
  static std::string CatalogName() { return "derived2"; }
  std::string getName() { return CatalogName(); }

};
REGISTER_CATALOG_ENTRY( Base, Derived2, int&, double const & )


TEST(testObjectCatalog,testRegistration)
{
  std::cout<<"EXECUTING MAIN"<<std::endl;
  int junk = 1;
  double junk2 = 3.14;
  std::unique_ptr<Base> derived1 = Base::CatalogInterface::Factory("derived1",junk,junk2);
  std::unique_ptr<Base> derived2 = Base::CatalogInterface::Factory("derived2",junk,junk2);

  assert( derived1->getName() == Derived1::CatalogName() );
  assert( derived2->getName() == Derived2::CatalogName() );
  std::cout<<"EXITING MAIN"<<std::endl;
}
