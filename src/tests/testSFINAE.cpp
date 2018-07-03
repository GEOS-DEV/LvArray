/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-746361
 *
 * All rights reserved. See COPYRIGHT for details.
 *
 * This file is part of the GEOSX Simulation Framework.
 *
 * GEOSX is a free software; you can redistrubute it and/or modify it under
 * the terms of the GNU Lesser General Public Liscense (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */


//#ifdef __clang__
//#pragma clang diagnostic push
//#pragma clang diagnostic ignored "-Wglobal-constructors"
//#pragma clang diagnostic ignored "-Wexit-time-destructors"
//#endif

#include <iostream>
#include "gtest/gtest.h"

//#ifdef __clang__
//#pragma clang diagnostic push
//#endif

#include "SFINAE_Macros.hpp"


struct Foo_MemberData
{
  int memberName = 1;
};
struct Bar_MemberData : Foo_MemberData {};

struct Foo_StaticMemberData
{
  static int memberName;
};
int Foo_StaticMemberData::memberName = 1;
struct Bar_StaticMemberData : Foo_StaticMemberData {};


struct Foo_MemberFunction_1Arg
{
  int memberName(int a) { return a; }
};
struct Bar_MemberFunction_1Arg : Foo_MemberFunction_1Arg {};

struct Foo_MemberFunction_2Arg
{
  double memberName2(int a,double b) { return a+b; }
};
struct Bar_MemberFunction_2Arg : Foo_MemberFunction_2Arg {};

struct Foo_StaticMemberFunction_1Arg
{
  static int memberName( int a) { return a; }
};
struct Bar_StaticMemberFunction_1Arg : Foo_StaticMemberFunction_1Arg {};


struct Foo_Using
{
  using memberName = int;
};
struct Bar_Using : Foo_Using {};

struct Foo_Typedef
{
  typedef int memberName;
};
struct Bar_Typedef : Foo_Typedef {};

struct Foo_EnumClass
{
  enum class memberName
  {
    enum1,
    enum2
  };
};
struct Bar_EnumClass : Foo_EnumClass {};



HAS_MEMBER_DATA(memberName)
TEST(test_sfinae,test_has_datamember)
{
//  std::cout<<has_datamember_memberName<Foo_MemberData>::value<<std::endl;
//  bool test = has_datamember_memberName<Foo_MemberData>::value;
//  EXPECT_TRUE( test );
//  EXPECT_TRUE( has_datamember_memberName<Foo_MemberData>::value );
  EXPECT_FALSE( !has_datamember_memberName<Foo_MemberData>::value );
  EXPECT_FALSE( has_datamember_memberName<Foo_StaticMemberData>::value );
  EXPECT_FALSE( has_datamember_memberName<Foo_MemberFunction_1Arg>::value );
  EXPECT_FALSE( has_datamember_memberName<Foo_StaticMemberFunction_1Arg>::value );
  EXPECT_FALSE( has_datamember_memberName<Foo_Using>::value );
  EXPECT_FALSE( has_datamember_memberName<Foo_Typedef>::value );
  EXPECT_FALSE( has_datamember_memberName<Foo_EnumClass>::value );
}

HAS_STATIC_MEMBER_DATA(memberName)
TEST(test_sfinae,test_has_staticdatamember)
{
  EXPECT_FALSE( has_staticdatamember_memberName<Foo_MemberData>::value );
//  ASSERT_TRUE(  has_staticdatamember_memberName<Foo_StaticMemberData>::value);
  EXPECT_FALSE( has_staticdatamember_memberName<Foo_MemberFunction_1Arg>::value );
  EXPECT_FALSE( has_staticdatamember_memberName<Foo_StaticMemberFunction_1Arg>::value );
  EXPECT_FALSE( has_staticdatamember_memberName<Foo_Using>::value );
  EXPECT_FALSE( has_staticdatamember_memberName<Foo_Typedef>::value );
  EXPECT_FALSE( has_staticdatamember_memberName<Foo_EnumClass>::value );
}

HAS_MEMBER_FUNCTION_NAME( memberName )
HAS_MEMBER_FUNCTION_NAME( memberName2 )
TEST(test_sfinae,test_has_memberfunction_name)
{
  EXPECT_FALSE( has_memberfunction_name_memberName<Foo_MemberData>::value );
  EXPECT_FALSE( has_memberfunction_name_memberName<Foo_StaticMemberData>::value );
//  EXPECT_TRUE(
//  has_memberfunction_name_memberName<Foo_MemberFunction_1Arg>::value );
  EXPECT_FALSE( has_memberfunction_name_memberName<Foo_StaticMemberFunction_1Arg>::value );
  EXPECT_FALSE( has_memberfunction_name_memberName<Foo_Using>::value );
  EXPECT_FALSE( has_memberfunction_name_memberName<Foo_Typedef>::value );
  EXPECT_FALSE( has_memberfunction_name_memberName<Foo_EnumClass>::value );
//  EXPECT_TRUE(
//  has_memberfunction_name_memberName2<Foo_MemberFunction_2Arg>::value );

}

HAS_MEMBER_FUNCTION_VARIANT( memberName, 0, int,          , VA_LIST(int), VA_LIST(int(1)) )
HAS_MEMBER_FUNCTION_VARIANT( memberName, 1, unsigned int, , VA_LIST(int), VA_LIST(int(1)) )
HAS_MEMBER_FUNCTION_VARIANT( memberName, 2, int,          , VA_LIST( unsigned int), VA_LIST(int(1)) )
HAS_MEMBER_FUNCTION_VARIANT( memberName2, 0, double, , VA_LIST(int,double), VA_LIST(int(1),double(1)) )
TEST(test_sfinae,test_has_memberfunction_variant)
{
  EXPECT_FALSE( has_memberfunction_v0_memberName<Foo_MemberData>::value );
  EXPECT_FALSE( has_memberfunction_v0_memberName<Foo_StaticMemberData>::value );

//  EXPECT_TRUE(
//   has_memberfunction_v0_memberName<Foo_MemberFunction_1Arg>::value );
  EXPECT_FALSE(  has_memberfunction_v1_memberName<Foo_MemberFunction_1Arg>::value );
  EXPECT_FALSE(  has_memberfunction_v2_memberName<Foo_MemberFunction_1Arg>::value );

  EXPECT_FALSE( has_memberfunction_v0_memberName<Foo_StaticMemberFunction_1Arg>::value );
  EXPECT_FALSE( has_memberfunction_v0_memberName<Foo_Using>::value );
  EXPECT_FALSE( has_memberfunction_v0_memberName<Foo_Typedef>::value );
  EXPECT_FALSE( has_memberfunction_v0_memberName<Foo_EnumClass>::value );

//  EXPECT_TRUE(
//  has_memberfunction_v0_memberName2<Foo_MemberFunction_2Arg>::value );

}

HAS_MEMBER_FUNCTION( memberName, int, , VA_LIST(int), VA_LIST(int(1)) )
HAS_MEMBER_FUNCTION( memberName2, double, , VA_LIST(int,double), VA_LIST(int(1),double(1)) )
TEST(test_sfinae,test_has_memberfunction)
{
  EXPECT_FALSE( has_memberfunction_memberName<Foo_MemberData>::value );
  EXPECT_FALSE( has_memberfunction_memberName<Foo_StaticMemberData>::value );
//  EXPECT_TRUE(  has_memberfunction_memberName<Foo_MemberFunction_1Arg>::value
// );
  EXPECT_FALSE( has_memberfunction_memberName<Foo_StaticMemberFunction_1Arg>::value );
  EXPECT_FALSE( has_memberfunction_memberName<Foo_Using>::value );
  EXPECT_FALSE( has_memberfunction_memberName<Foo_Typedef>::value );
  EXPECT_FALSE( has_memberfunction_memberName<Foo_EnumClass>::value );
//  EXPECT_TRUE(  has_memberfunction_memberName2<Foo_MemberFunction_2Arg>::value
// );
}


HAS_STATIC_MEMBER_FUNCTION(memberName,int,int(1))
TEST(test_sfinae,test_has_staticmemberfunction)
{
  EXPECT_FALSE( has_staticmemberfunction_memberName<Foo_MemberData>::value );
  EXPECT_FALSE( has_staticmemberfunction_memberName<Foo_StaticMemberData>::value );
  EXPECT_FALSE(  has_staticmemberfunction_memberName<Foo_MemberFunction_1Arg>::value );
//  EXPECT_TRUE(
// has_staticmemberfunction_memberName<Foo_StaticMemberFunction_1Arg>::value );
  EXPECT_FALSE( has_staticmemberfunction_memberName<Foo_Using>::value );
  EXPECT_FALSE( has_staticmemberfunction_memberName<Foo_Typedef>::value );
  EXPECT_FALSE( has_staticmemberfunction_memberName<Foo_EnumClass>::value );
}

HAS_ENUM(memberName)
TEST(test_sfinae,test_has_enum)
{
  EXPECT_FALSE( has_enum_memberName<Foo_MemberData>::value );
  EXPECT_FALSE( has_enum_memberName<Foo_StaticMemberData>::value );
  EXPECT_FALSE( has_enum_memberName<Foo_MemberFunction_1Arg>::value );
  EXPECT_FALSE( has_enum_memberName<Foo_StaticMemberFunction_1Arg>::value );
  EXPECT_FALSE( has_enum_memberName<Foo_Using>::value );
  EXPECT_FALSE( has_enum_memberName<Foo_Typedef>::value );
//  EXPECT_TRUE(  has_enum_memberName<Foo_EnumClass>::value );
}


HAS_ALIAS(memberName)
TEST(test_sfinae,test_has_alias)
{
  EXPECT_FALSE( has_alias_memberName<Foo_MemberData>::value );
  EXPECT_FALSE( has_alias_memberName<Foo_StaticMemberData>::value );
  EXPECT_FALSE(  has_alias_memberName<Foo_MemberFunction_1Arg>::value );
  EXPECT_FALSE( has_alias_memberName<Foo_StaticMemberFunction_1Arg>::value );
//  EXPECT_TRUE(  has_alias_memberName<Foo_Using>::value );
//  EXPECT_TRUE(  has_alias_memberName<Foo_Typedef>::value );
  EXPECT_FALSE( has_alias_memberName<Foo_EnumClass>::value );
}
