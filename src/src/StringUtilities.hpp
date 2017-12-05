/*
 * StringUtilities.hpp
 *
 *  Created on: Nov 12, 2014
 *      Author: rrsettgast
 */

#ifndef CXX_UTILITIES_STRINGUTILITIES_HPP_
#define CXX_UTILITIES_STRINGUTILITIES_HPP_


//#include <cxxabi.h>
#include <string>
#include <vector>
//#include <memory>
#include <iostream>

namespace cxx_utilities
{
std::string demangle( const std::string& name );


template< typename T_RHS, typename VECTORTYPE >
void equateStlVector( T_RHS & lhs, std::vector<VECTORTYPE> const & rhs )
{
  for( unsigned int a=0 ; a<rhs.size() ; ++a )
  {
    lhs[a] = rhs[a];
  }
}

template<>
void equateStlVector<std::string,std::string>( std::string & lhs, std::vector<std::string> const & rhs );



}

#endif /* STRINGUTILITIES_HPP_ */
