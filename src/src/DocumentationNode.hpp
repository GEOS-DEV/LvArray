/*
 * InputDocumentation.hpp
 *
 *  Created on: Sep 7, 2016
 *      Author: rrsettgast
 */

#ifndef SRC_COMPONENTS_CXX_UTILITIES_SRC_SRC_DOCUMENTATIONNODE_HPP_
#define SRC_COMPONENTS_CXX_UTILITIES_SRC_SRC_DOCUMENTATIONNODE_HPP_

#include <memory>
#include <iostream>
#include <map>

namespace cxx_utilities
{
struct DocumentationNode
{
  std::string m_name              = "";
  std::string m_stringKey         = "";
  int m_intKey                    = -1;
  std::string m_type              = "";
  std::string m_shortDescription  = "";
  std::string m_longDescription   = "";
  int m_level                     = -1;
  std::string m_default           = "";
  unsigned int m_isInput          =  0;
  std::map<std::string,DocumentationNode> m_child = {};

  // TODO THIS ISN'T CORRECT. FIX IT.
  void Print(  ) const
  {
    std::string indent( m_level*2, ' ');
    std::cout<<indent<<"m_varName        = "<<m_name<<"\n";
    std::cout<<indent<<"m_varType        = "<<m_type<<"\n";
    std::cout<<indent<<"m_varDescription = "<<m_shortDescription<<"\n\n";
    for( auto const & child : m_child )
    {
      std::cout<<indent<<"node "<<child.first<<"\n";
      child.second.Print();
    }
  }
};
}

#endif /* SRC_COMPONENTS_CXX_UTILITIES_SRC_SRC_DOCUMENTATIONNODE_HPP_ */
