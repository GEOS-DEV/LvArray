/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
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
