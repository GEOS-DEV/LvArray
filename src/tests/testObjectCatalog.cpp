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

#include <gtest/gtest.h>

#define OBJECTCATALOGVERBOSE 2
#include "dataRepository/ObjectCatalog.hpp"
#include "Logger.hpp"
#include <sys/time.h>
#include <stdint.h>
#include <string>

using namespace cxx_utilities;

class Base
{
public:
  Base( int& junk, double const & junk2 )
  {
    GEOS_LOG( "calling Base constructor with arguments ("<<junk<<" "<<junk2<<")" );
  }

  virtual ~Base()
  {
    GEOS_LOG( "calling Base destructor" );
  }

  using CatalogInterface = dataRepository::CatalogInterface< Base, int&, double const &  >;
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
  Derived1( int& junk, double const & junk2 ):
    Base( junk, junk2 )
  {
    GEOS_LOG( "calling Derived1 constructor with arguments ("<<junk<<" "<<junk2<<")" );
  }

  ~Derived1()
  {
    GEOS_LOG( "calling Derived1 destructor" );
  }
  static std::string CatalogName() { return "derived1"; }
  std::string getName() { return CatalogName(); }

};
REGISTER_CATALOG_ENTRY( Base, Derived1, int&, double const & )

class Derived2 : public Base
{
public:
  Derived2( int& junk, double const & junk2 ):
    Base( junk, junk2 )
  {
    GEOS_LOG( "calling Derived2 constructor with arguments ("<<junk<<" "<<junk2<<")" );
  }

  ~Derived2()
  {
    GEOS_LOG( "calling Derived2 destructor" );
  }
  static std::string CatalogName() { return "derived2"; }
  std::string getName() { return CatalogName(); }

};
REGISTER_CATALOG_ENTRY( Base, Derived2, int&, double const & )


TEST( testObjectCatalog, testRegistration )
{
  GEOS_LOG( "EXECUTING MAIN" );
  int junk = 1;
  double junk2 = 3.14;
  std::unique_ptr<Base> derived1 = Base::CatalogInterface::Factory( "derived1", junk, junk2 );
  std::unique_ptr<Base> derived2 = Base::CatalogInterface::Factory( "derived2", junk, junk2 );

  assert( derived1->getName() == Derived1::CatalogName() );
  assert( derived2->getName() == Derived2::CatalogName() );
  GEOS_LOG( "EXITING MAIN" );
}
