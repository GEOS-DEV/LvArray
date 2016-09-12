/*
 * InputDocumentation.hpp
 *
 *  Created on: Sep 7, 2016
 *      Author: rrsettgast
 */

#ifndef SRC_COMPONENTS_CXX_UTILITIES_SRC_SRC_INPUTDOCUMENTATION_HPP_
#define SRC_COMPONENTS_CXX_UTILITIES_SRC_SRC_INPUTDOCUMENTATION_HPP_

#include <memory>
#include <iostream>
namespace cxx_utilities
{
struct InputDocumentation
{
  std::string m_varName = "";
  std::string m_varType = "";
  std::string m_varDescription = "";
  int m_level = 0;
  std::unordered_map<std::string,InputDocumentation> m_child;

  // TODO THIS ISN'T CORRECT. FIX IT.
  void Print(  ) const
  {
    std::string indent( m_level*2, ' ');
    std::cout<<indent<<"m_varName        = "<<m_varName<<"\n";
    std::cout<<indent<<"m_varType        = "<<m_varType<<"\n";
    std::cout<<indent<<"m_varDescription = "<<m_varDescription<<"\n\n";
    for( auto const & child : m_child )
    {
      std::cout<<indent<<"node "<<child.first<<"\n";
      child.second.Print();
    }
  }
};
}

#endif /* SRC_COMPONENTS_CXX_UTILITIES_SRC_SRC_INPUTDOCUMENTATION_HPP_ */
