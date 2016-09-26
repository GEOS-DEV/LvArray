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

  /*
  void Write(std::string const & fname) const
  {
    FILE * outputFile = fopen(fname.c_str(), "w");
    fprintf(outputFile, "<?xml version=\"1.0\" ?>\n\n");
    WriteXMLStructure(outputFile);
    fclose(outputFile);
  }

  void WriteXMLStructure(FILE * outputFile) const
  {
    std::string indent( m_level*2, ' ');
    std::string indent_b( m_level*2 + strlen(m_varName.c_str()), ' ');
    
    fprintf(outputFile, "%s<%s m_varType=\"%s\"\n%s  m_varDescription=\"%s\">\n", indent.c_str(),
                                                                  m_varName.c_str(),
                                                                  m_varType.c_str(),
                                                                  indent_b.c_str(),
                                                                  m_varDescription.c_str());
    
    for( auto const & child : m_child )
    {
      child.second.WriteXMLStructure(outputFile);
    }

    fprintf(outputFile, "%s</%s>\n", indent.c_str(), m_varName.c_str());
  }
  */
};
}

#endif /* SRC_COMPONENTS_CXX_UTILITIES_SRC_SRC_DOCUMENTATIONNODE_HPP_ */
