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

/*
 * Copyright (c) 2015, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

#ifndef BASE_HPP
#define BASE_HPP
#define OBJECTCATALOGVERBOSE 2
#include "ObjectCatalog.hpp"
#include <string>

class Parameter
{
public:
  Parameter(){}
  ~Parameter(){}
  Parameter( Parameter const & source ):
    member( source.member )
  {
    GEOS_LOG( "called copy constructor for Parameter" );
  }

#if ( __cplusplus >= 201103L )
  Parameter( Parameter && source ):
    member( std::move( source.member ))
  {
    GEOS_LOG( "called move constructor for Parameter" );
  }
#endif

  double member;


};

class Base
{
public:
  Base( int junk, double const & junk2, Parameter& pbv )
  {
    GEOS_LOG( "calling Base constructor with arguments (" << junk << " " << junk2 << ")" );
  }

  ~Base()
  {
    GEOS_LOG( "calling Base destructor" );
  }

  using CatalogInterface = cxx_utilities::CatalogInterface< Base, int, double const &, Parameter& >;
  static CatalogInterface::CatalogType& GetCatalog()
  {
    static CatalogInterface::CatalogType catalog;
    return catalog;
  }

  virtual std::string const getName() const = 0;
};

#endif
