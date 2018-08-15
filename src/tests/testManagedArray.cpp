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
 * GEOSX is a free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */


//#ifdef __clang__
//#pragma clang diagnostic push
//#pragma clang diagnostic ignored "-Wglobal-constructors"
//#pragma clang diagnostic ignored "-Wexit-time-destructors"
//#endif


#if 1
#include <iostream>
#include "gtest/gtest.h"

//#ifdef __clang__
//#pragma clang diagnostic push
//#endif

#include "SFINAE_Macros.hpp"
#include "ManagedArray.hpp"




using namespace multidimensionalArray;

HAS_ALIAS(pointer)
HAS_ALIAS(iterator)
HAS_MEMBER_FUNCTION(begin,typename U::iterator,,,)
HAS_MEMBER_FUNCTION(end,typename U::iterator,,,)


TEST(test_managedarray,test_has_sfinae)
{
  using array = ManagedArray<int, 2, int>;

  std::cout<<has_alias_pointer<array>::value<<std::endl;
  std::cout<<has_alias_iterator<array>::value<<std::endl;
  std::cout<<has_memberfunction_begin<array>::value<<std::endl;
  std::cout<<has_memberfunction_end<array>::value<<std::endl;

//  std::cout<<std::is_member_function_pointer<decltype(static_cast<array::iterator (array::*)()>(&array::begin) )>::value<<std::endl;
//  std::cout<<std::is_member_function_pointer<decltype(static_cast<array::iterator (array::*)()>(&array::begin) )>::value<<std::endl;
////  EXPECT_TRUE( has_datamember_memberName<Foo_MemberData>::value );
//  EXPECT_FALSE( has_datamember_memberName<Foo_StaticMemberData>::value );
//  EXPECT_FALSE( has_datamember_memberName<Foo_MemberFunction_1Arg>::value );
//  EXPECT_FALSE( has_datamember_memberName<Foo_StaticMemberFunction_1Arg>::value );
//  EXPECT_FALSE( has_datamember_memberName<Foo_Using>::value );
//  EXPECT_FALSE( has_datamember_memberName<Foo_Typedef>::value );
//  EXPECT_FALSE( has_datamember_memberName<Foo_EnumClass>::value );
}


TEST(test_managedarray,test_const)
{
  int junk[10] = {1,2,3,4,5,6,7,8,9,10};
  int dim[1]={10};
  int stride[1]={1};


  ArrayView<int, 1, int> array(junk,dim,stride);

  ArrayView<const int, 1, int> reducedArray = array;

  EXPECT_TRUE( array[9]==array[9] );

  ManagedArray<int, 2 , int> managedArray;
  managedArray.resize(10,1);

  ArrayView<int, 1, int> array2 = managedArray;

}


TEST(test_managedarray,test_udc_dimreduce)
{
  int junk[10];
  int dim[2]={10,1};
  int stride[2]={1,1};


  ArrayView<int, 2, int> array(junk,dim,stride);

  ArrayView<int, 1, int> arrayC = array;

  EXPECT_TRUE( arrayC[9]==array[9][0] );

}
#else

#include <iostream>


class Foo
{
public:
  double       * Bar()       { return &m_data; }
private:
  double m_data;
};

class Foo2
{
public:
  double       * Bar()       { return &m_data; }
  double const * Bar() const { return &m_data; }
private:
  double m_data;
};


template<typename TYPE >
  struct has_memberfunction_name_Bar
  {
private:
    template<typename U>
    static constexpr auto test(int)->
//    decltype( std::is_member_function_pointer<decltype(&U::Bar)>::value, bool() )
//    decltype( U::Bar, bool() )
    decltype( static_cast<double const * (U::*)() const>(&U::Bar), bool() )
    {
      return true;
    }
    template<typename U>
    static constexpr auto test(...)->
    bool
    {
      return false;
    }
public:
    static constexpr bool value = test<TYPE>(0);
  };

int main()
{
  std::cout<<has_memberfunction_name_Bar<Foo>::value<<std::endl;
  std::cout<<has_memberfunction_name_Bar<Foo2>::value<<std::endl;
}

#endif
