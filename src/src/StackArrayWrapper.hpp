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

#ifndef STACKARRAYWRAPPER_HPP_
#define STACKARRAYWRAPPER_HPP_

#include <stddef.h>

#include "CXX_UtilsConfig.hpp"
#include "Logger.hpp"

namespace LvArray
{

template< typename T, int LENGTH >
struct StackArrayWrapper
{
  typedef T * iterator;
  typedef T const * const_iterator;

  void free() {}

  void resize( ptrdiff_t length, T const & = T())
  {
    GEOS_ERROR_IF( length > LENGTH, "C_Array::resize("<<length<<") is larger than template argument LENGTH=" << LENGTH );
  }

  LVARRAY_HOST_DEVICE T * data()             { return m_data; }
  LVARRAY_HOST_DEVICE T const * data() const { return m_data; }

  T m_data[LENGTH];
};
}

#endif
