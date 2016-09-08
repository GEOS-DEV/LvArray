/*
 * InputDocumentation.hpp
 *
 *  Created on: Sep 7, 2016
 *      Author: rrsettgast
 */

#ifndef SRC_COMPONENTS_CXX_UTILITIES_SRC_SRC_INPUTDOCUMENTATION_HPP_
#define SRC_COMPONENTS_CXX_UTILITIES_SRC_SRC_INPUTDOCUMENTATION_HPP_

#include <memory>

struct InputDocumentation
{
  std::string m_varName;
  std::string m_varType;
  std::string m_varDescription;
  std::unordered_map<std::string,InputDocumentation> m_child;
};


#endif /* SRC_COMPONENTS_CXX_UTILITIES_SRC_SRC_INPUTDOCUMENTATION_HPP_ */
