
#include "StringUtilities.hpp"
#include <cxxabi.h>
#include <cstring>
#include <memory>
#include <sstream>
#include <algorithm>
#include <map>

namespace cxx_utilities
{
std::string demangle( const std::string& name )
{

  int status = -4; // some arbitrary value to eliminate the compiler warning

  // enable c++11 by passing the flag -std=c++11 to g++
  std::unique_ptr<char, void (*)( void* )> res
  {
    abi::__cxa_demangle( name.c_str(), nullptr, nullptr, &status ),
    std::free
  };

  return ( status == 0 ) ? res.get() : name;
}


template<>
void equateStlVector<std::string,std::string>( std::string & lhs, std::vector<std::string> const & rhs )
{
  // need error checking for size mismatch
  lhs = rhs[0];
}

}
