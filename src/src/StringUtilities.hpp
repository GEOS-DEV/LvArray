/*
 * StringUtilities.hpp
 *
 *  Created on: Nov 12, 2014
 *      Author: rrsettgast
 */

#ifndef STRINGUTILITIES_HPP_
#define STRINGUTILITIES_HPP_


#include <cxxabi.h>
#include <string>
#include <memory>

namespace cxx_utilities
{
std::string demangle( const std::string& name );

}

#endif /* STRINGUTILITIES_HPP_ */
