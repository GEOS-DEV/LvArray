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

#pragma once

// Sphinx start after Foo
class Foo
{
public:
  Foo( int const x ):
    m_x( x )
  {}

  int get() const
  { return m_x; }

  void set( int const x )
  { m_x = x; }

private:
  int m_x;
};
// Sphinx end before Foo

// Sphinx start after FooTemplate
template< typename T >
class FooTemplate
{
public:
  FooTemplate( T const & x ):
    m_x( x )
  {}

  T const & get() const
  { return m_x; }

  void set( T const & x )
  { m_x = x; }

private:
  T m_x;
};
// Sphinx end before FooTemplate
