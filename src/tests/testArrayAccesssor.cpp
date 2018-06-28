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

#include "../src/MultidimensionalArray.hpp"
#include <sys/time.h>
#include <stdint.h>
#include <string>

using namespace multidimensionalArray;


TEST(testArrayAccessor,ArrayInterface1)
{
  integer_t const lengths[1] = {8};
  integer_t const n = lengths[0];
  double memblock[n];

  for( integer_t a=0 ; a<n ; ++a )
  {
    memblock[a] = a;
  }

  ArrayAccessor<double,1> array(memblock,lengths);

  for( integer_t a=0 ; a<n ; ++a )
  {
    std::cout<<array[a]<<std::endl;
  }
}

TEST(testArrayAccessor,ArrayInterface2)
{
  integer_t const lengths[2] = {2,8};
  integer_t const n = lengths[0]*lengths[1];
  double memblock[n];

  for( integer_t a=0 ; a<lengths[0] ; ++a )
  {
    for( integer_t b=0 ; b<lengths[1] ; ++b )
    {
      memblock[a*lengths[1]+b] = a*lengths[1]+b;
    }
  }

  ArrayAccessor<double,2> array(memblock,lengths);


  for( integer_t a=0 ; a<lengths[0] ; ++a )
  {
    for( integer_t b=0 ; b<lengths[1] ; ++b )
    {
      std::cout<<memblock[a*lengths[1]+b]<<" ?= ";
      std::cout<<array[a][b]<<std::endl;
      assert(&(memblock[a*lengths[1]+b])==&(array[a][b]));
    }
  }
}

TEST(testArrayAccessor,ArrayInterface3)
{
  constexpr integer_t lengths[3] = {2,2,3};
  double memblock[lengths[0]][lengths[1]][lengths[2]];

  for( integer_t a=0 ; a<lengths[0] ; ++a )
  {
    for( integer_t b=0 ; b<lengths[1] ; ++b )
    {
      for( integer_t c=0 ; c<lengths[2] ; ++c )
      {
        memblock[a][b][c] = a*lengths[1]*lengths[2]+b*lengths[2]+c;
      }
    }
  }

  ArrayAccessor<double,3> array3(&(memblock[0][0][0]),lengths);


  for( integer_t a=0 ; a<lengths[0] ; ++a )
  {
    ArrayAccessor<double,2> array2 = array3[a];

    for( integer_t b=0 ; b<lengths[1] ; ++b )
    {
      ArrayAccessor<double,1> array1 = array2[b];

      for( integer_t c=0 ; c<lengths[2] ; ++c )
      {
        std::cout<<"("<<a<<","<<b<<","<<c<<") -> ";
        std::cout<<memblock[a][b][c]<<" ?= ";
        std::cout<<array3[a][b][c]<<" ?= "<<array1[c]<<std::endl;

        assert(&(memblock[a][b][c])==&(array1[c]));
        assert(&(memblock[a][b][c])==&(array2[b][c]));
        assert(&(memblock[a][b][c])==&(array3[a][b][c]));
      }
      std::cout<<std::endl;
    }
  }
}
