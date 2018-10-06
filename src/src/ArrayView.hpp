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

#ifndef ARRAYVIEW_HPP_
#define ARRAYVIEW_HPP_


#include "ArraySlice.hpp"


namespace LvArray
{
template< typename T, int NDIM, typename INDEX_TYPE >
class ArrayView : public ArraySlice<T, NDIM, INDEX_TYPE >
{
public:


  ArrayView( ArrayView const & source )
  {
    //This triggers Chai::ManagedArray CC
  }

  ArrayView( ArrayView && source )
  {

  }

  ArrayView & operator=( ArrayView const & source )
  {

  }

  ArrayView & operator=( ArrayView && source )
  {

  }

  using ArrayType = ChaiVector<T>;
  using iterator = typename ArrayType::iterator;
  using const_iterator = typename ArrayType::const_iterator;

protected:
  ArrayType m_dataVector;

  INDEX_TYPE m_dimsMem[NDIM];

  INDEX_TYPE m_stridesMem[NDIM];


  ArrayView():
    ArraySlice( nullptr, m_dimsMem, m_stridesMem ),
    m_dataVector()
  {}

};
}

#endif /* ARRAYVIEW_HPP_ */
