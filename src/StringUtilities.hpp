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
 * @file StringUtilities.hpp
 */

#ifndef CXX_UTILITIES_STRINGUTILITIES_HPP_
#define CXX_UTILITIES_STRINGUTILITIES_HPP_


#include <string>
#include <vector>
#include <iostream>
#include <typeinfo>

namespace cxx_utilities
{

std::string demangle( std::string const & name );

template< class T >
inline std::string demangleType()
{ return demangle( typeid( T ).name() ); }

template< class T >
inline std::string demangleType( T const & )
{ return demangle( typeid( T ).name() ); }

}

#endif /* STRINGUTILITIES_HPP_ */
