/*
 * Copyright (c) 2015, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

#include "Base.hpp"


class Derived1 : public Base
{
public:
  Derived1( int junk, double const & junk2, Parameter& param );

  ~Derived1();

  static std::string CatalogName() { return "derived1"; }
  std::string const getName() const override final { return CatalogName(); }

};


