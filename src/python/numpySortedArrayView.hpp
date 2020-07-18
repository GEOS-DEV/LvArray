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

/**
 * @file numpySortedArrayView.hpp
 */

#pragma once
#include "numpyConversion.hpp"
#include "../SortedArrayView.hpp"

namespace LvArray
{

namespace python
{

/**
 * Return a Numpy view of a SortedArray
 * @param arr the SortedArrayView to convert to numpy.
 */
template< typename T, typename INDEX_TYPE, template<typename> class BUFFER_TYPE >
PyObject * create( SortedArrayView< T const, INDEX_TYPE, BUFFER_TYPE > const & arr ){
    arr.move( MemorySpace::CPU );
    INDEX_TYPE dims = arr.size();
    INDEX_TYPE strides = 1;
    return internal::create( arr.data(), 1, &dims, &strides);
}

} // namespace python

} // namespace LvArray
