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
#include <map>
namespace cxx_utilities
{
struct InputDocumentation
{
  std::string m_varName = "";
  std::string m_varType = "";
  std::string m_varDescription = "";
  unsigned int m_level = 0;
  std::map<std::string,InputDocumentation> m_child;

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

#endif /* SRC_COMPONENTS_CXX_UTILITIES_SRC_SRC_INPUTDOCUMENTATION_HPP_ */
