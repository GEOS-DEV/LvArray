/*
 * Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

// Source includes
#include "Foo.hpp"

// TPL includes
#include <gtest/gtest.h>

// Sphinx start after Foo
TEST( Foo, get )
{
  Foo foo( 5 );
  EXPECT_EQ( foo.get(), 5 );
}

TEST( Foo, set )
{
  Foo foo( 3 );
  EXPECT_EQ( foo.get(), 3 );
  foo.set( 8 );
  EXPECT_EQ( foo.get(), 8 );
}
// Sphinx end before Foo

// Sphinx start after FooTemplate
TEST( FooTemplate, get )
{
  {
    FooTemplate< int > foo( 5 );
    EXPECT_EQ( foo.get(), 5 );
  }

  {
    FooTemplate< double > foo( 5 );
    EXPECT_EQ( foo.get(), 5 );
  }
}

TEST( FooTemplate, set )
{
  {
    FooTemplate< int > foo( 3 );
    EXPECT_EQ( foo.get(), 3 );
    foo.set( 8 );
    EXPECT_EQ( foo.get(), 8 );
  }

  {
    FooTemplate< double > foo( 3 );
    EXPECT_EQ( foo.get(), 3 );
    foo.set( 8 );
    EXPECT_EQ( foo.get(), 8 );
  }
}
// Sphinx end before FooTemplate

// Sphinx start after FooTemplateTest
template< typename T >
class FooTemplateTest : public ::testing::Test
{
public:
  void get()
  {
    FooTemplate< T > foo( 5 );
    EXPECT_EQ( foo.get(), 5 );
  }

  void set()
  {
    FooTemplate< T > foo( 3 );
    EXPECT_EQ( foo.get(), 3 );
    foo.set( 8 );
    EXPECT_EQ( foo.get(), 8 );
  }
};

using FooTemplateTestTypes = ::testing::Types<
  int
  , double
  >;

TYPED_TEST_SUITE( FooTemplateTest, FooTemplateTestTypes, );

TYPED_TEST( FooTemplateTest, get )
{
  this->get();
}

TYPED_TEST( FooTemplateTest, set )
{
  this->set();
}
// Sphinx end before FooTemplateTest


// This is the default gtest main method. It is included for ease of debugging.
int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
