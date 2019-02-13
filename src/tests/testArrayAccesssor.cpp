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
#include "../src/Logger.hpp"
#include <sys/time.h>
#include <stdint.h>

using namespace multidimensionalArray;


TEST( testArrayAccessor, ArrayInterface1 )
{
  integer_t const lengths[1] = {8};
  integer_t const n = lengths[0];
  double memblock[n];

  for( integer_t a=0 ; a<n ; ++a )
  {
    memblock[a] = a;
  }

  ArrayAccessor<double, 1> array( memblock, lengths );

  for( integer_t a=0 ; a<n ; ++a )
  {
    GEOS_LOG( array[a] );
  }
}

TEST( testArrayAccessor, ArrayInterface2 )
{
  integer_t const lengths[2] = {2, 8};
  integer_t const n = lengths[0]*lengths[1];
  double memblock[n];

  for( integer_t a=0 ; a<lengths[0] ; ++a )
  {
    for( integer_t b=0 ; b<lengths[1] ; ++b )
    {
      memblock[a*lengths[1]+b] = a*lengths[1]+b;
    }
  }

  ArrayAccessor<double, 2> array( memblock, lengths );


  for( integer_t a=0 ; a<lengths[0] ; ++a )
  {
    for( integer_t b=0 ; b<lengths[1] ; ++b )
    {
      GEOS_LOG( memblock[a*lengths[1]+b] << " ?= " << array[a][b] );
      assert( &(memblock[a*lengths[1]+b])==&(array[a][b]));
    }
  }
}

TEST( testArrayAccessor, ArrayInterface3 )
{
  constexpr integer_t lengths[3] = {2, 2, 3};
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

  ArrayAccessor<double, 3> array3( &(memblock[0][0][0]), lengths );


  for( integer_t a=0 ; a<lengths[0] ; ++a )
  {
    ArrayAccessor<double, 2> array2 = array3[a];

    for( integer_t b=0 ; b<lengths[1] ; ++b )
    {
      ArrayAccessor<double, 1> array1 = array2[b];

      for( integer_t c=0 ; c<lengths[2] ; ++c )
      {
        GEOS_LOG( "(" << a << "," << b << "," << c << ") -> " << memblock[a][b][c] <<" ?= "
                      << array3[a][b][c] << " ?= " << array1[c] );

        assert( &(memblock[a][b][c])==&(array1[c]));
        assert( &(memblock[a][b][c])==&(array2[b][c]));
        assert( &(memblock[a][b][c])==&(array3[a][b][c]));
      }
      GEOS_LOG( "" );
    }
  }
}
