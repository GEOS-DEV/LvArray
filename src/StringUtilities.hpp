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

namespace LvArray
{

/**
 * @brief @return The demangled name corresponding to the given
 *        mangled name @p name.
 * @param name The mangled name.
 */
std::string demangle( std::string const & name );

/**
 * @brief @return A demangled type name corresponding to the type @tparam T.
 * @tparam T The type to demangle.
 */
template< class T >
inline std::string demangleType()
{ return demangle( typeid( T ).name() ); }

/**
 * @brief @return A demangled type name corresponding to the type @tparam T.
 * @tparam T The type to demangle.
 */
template< class T >
inline std::string demangleType( T const & )
{ return demangle( typeid( T ).name() ); }

/**
 * @brief @return A string representing @p bytes converted to either
 *        KB, MB, or GB.
 * @param bytes The number of bytes.
 */
std::string calculateSize( size_t const bytes );

} // namespace LvArray

#endif // CXX_UTILITIES_STRINGUTILITIES_HPP_
