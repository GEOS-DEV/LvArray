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
  Parameter(Parameter const & source ):
    member(source.member)
  {
    std::cout<<"called copy constructor for Parameter"<<std::endl;
  }

#if( __cplusplus >= 201103L )
  Parameter(Parameter && source ):
    member(std::move(source.member))
  {
    std::cout<<"called move constructor for Parameter"<<std::endl;
  }
#endif

  double member;


};

class Base
{
public:
  Base( int junk, double const & junk2, Parameter& pbv )
  {
    std::cout<<"calling Base constructor with arguments ("<<junk<<" "<<junk2<<")"<<std::endl;
  }

  ~Base()
  {
    std::cout<<"calling Base destructor"<<std::endl;
  }

  using CatalogInterface = cxx_utilities::CatalogInterface< Base, int, double const & , Parameter& >;
  static CatalogInterface::CatalogType& GetCatalog()
  {
    static CatalogInterface::CatalogType catalog;
    return catalog;
  }

  virtual std::string const getName() const = 0;
};

#endif
