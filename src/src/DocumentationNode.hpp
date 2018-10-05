/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
#include "../../../../coreComponents/dataRepository/RestartFlags.hpp"

namespace cxx_utilities
{
class DocumentationNode
{
public:

  DocumentationNode() = default;
  DocumentationNode( std::string const & name,
                     std::string const & stringKey,
                     int const & intKey,
                     std::string const & dataType,
                     std::string const & schemaType,
                     std::string const & shortDescription,
                     std::string const & longDescription,
                     std::string const & defaultVal,
                     std::string const & groups,
                     unsigned int const & parentManaged,
                     unsigned int const & level,
                     unsigned int const & isInput,
                     unsigned int const & verbosity,
                     DocumentationNode * m_parentNode,
                     geosx::dataRepository::RestartFlags const & restart_flag=geosx::dataRepository::RestartFlags::WRITE_AND_READ );


  bool operator==(DocumentationNode const & rhs);
  bool operator!=(DocumentationNode const & rhs) { return !(*this==rhs); }
//  friend bool operator==( cxx_utilities::DocumentationNode const & lhs,
//                          cxx_utilities::DocumentationNode const & rhs );
//  friend bool operator!=( DocumentationNode const & lhs, DocumentationNode
// const & rhs );


  DocumentationNode * AllocateChildNode( std::string const & name,
                                         std::string const & stringKey,
                                         int const & intKey,
                                         std::string const & dataType,
                                         std::string const & schemaType,
                                         std::string const & shortDescription,
                                         std::string const & longDescription,
                                         std::string const & default0,
                                         std::string const & groups,
                                         unsigned int const & parentManaged,
                                         unsigned int const & isInput,
                                         unsigned int const & verbosity,
                                         geosx::dataRepository::RestartFlags const & restart_flag=geosx::dataRepository::RestartFlags::WRITE_AND_READ);


  // TODO THIS ISN'T CORRECT. FIX IT.
  std::string toString() const;

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

     fprintf(outputFile, "%s<%s m_varType=\"%s\"\n%s
         m_varDescription=\"%s\">\n", indent.c_str(),
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


  std::string const & getName() const { return m_name; }

  void setName(std::string className) { m_name = className; }

  std::string const & getStringKey() const { return m_stringKey; }

  int const & getIntKey() const { return m_intKey; }

  void setIntKey( int intKey ) { m_intKey = intKey; }

  std::string const & getDataType() const { return m_dataType; }

  std::string const & getSchemaType() const { return m_schemaType; }

  void setSchemaType( std::string typeName ) { m_schemaType = typeName; }

  std::string const & getShortDescription() const { return m_shortDescription; }

  void setShortDescription( std::string shortDescription ) { m_shortDescription = shortDescription; }

  std::string const & getLongDescription() const { return m_longDescription; }

  std::string const & getDefault() const { return m_default; }

  void setDefault( std::string defVal ) { m_default = defVal; }

  std::string const & getGroups() const { return m_groups; }

  unsigned int const & getLevel() const { return m_level; }

  unsigned int const & getIsInput() const { return m_isInput; }

  int const & getVerbosity() const { return m_verbosity; }

  void setVerbosity( unsigned int verbosity) { m_verbosity = verbosity; }
  
  geosx::dataRepository::RestartFlags getRestartFlags() const { return m_restart_flags; }

  void setRestartFlags( geosx::dataRepository::RestartFlags flags ) { m_restart_flags = flags; } 

  std::map<std::string,DocumentationNode>       & getChildNodes()       { return m_child; }
  std::map<std::string,DocumentationNode> const & getChildNodes() const { return m_child; }

  DocumentationNode       * getChildNode( std::string const & name )       { return &m_child.at(name); }
  DocumentationNode const * getChildNode( std::string const & name ) const { return &m_child.at(name); }


public:

  std::string m_name                                  = "";
  std::string m_stringKey                             = "";
  int         m_intKey                                = -1;
  std::string m_dataType                              = "";
  std::string m_schemaType                            = "";
  std::string m_shortDescription                      = "";
  std::string m_longDescription                       = "";
  std::string m_default                               = "";
  std::string m_groups                                = "";
  int m_managedByParent                               =  0;
  unsigned int m_level                                =  0;
  unsigned int m_isInput                              =  0;
  int m_verbosity                                     =  0;
  unsigned int m_isRegistered                         =  0;
  DocumentationNode * m_parentNode                    = nullptr;
  geosx::dataRepository::RestartFlags m_restart_flags = geosx::dataRepository::RestartFlags::WRITE_AND_READ;
  std::map<std::string,DocumentationNode> m_child     = {};

};


inline bool DocumentationNode::operator==( DocumentationNode const & rhs )
{
  bool rval = false;
  if( m_name==rhs.m_name &&
      m_dataType == rhs.m_dataType &&
      m_intKey == rhs.m_intKey &&
      m_dataType == rhs.m_dataType &&
      m_schemaType == rhs.m_schemaType &&
      m_shortDescription == rhs.m_shortDescription &&
      m_longDescription == rhs.m_longDescription &&
      m_default == rhs.m_default &&
      m_groups == rhs.m_groups &&
      m_level == rhs.m_level &&
      m_isInput == rhs.m_isInput &&
      m_verbosity == rhs.m_verbosity &&
      m_parentNode == rhs.m_parentNode &&
      m_restart_flags == rhs.m_restart_flags )
  {
    rval = true;
  }
  return rval;
}

}
//
//inline bool operator==( cxx_utilities::DocumentationNode const & lhs,
//                        cxx_utilities::DocumentationNode const & rhs )
//{
//  bool rval = false;
//  if( lhs.m_name==rhs.m_name &&
//      lhs.m_dataType == rhs.m_dataType &&
//      lhs.m_intKey == rhs.m_intKey &&
//      lhs.m_dataType == rhs.m_dataType &&
//      lhs.m_schemaType == rhs.m_schemaType &&
//      lhs.m_shortDescription == rhs.m_shortDescription &&
//      lhs.m_longDescription == rhs.m_longDescription &&
//      lhs.m_default == rhs.m_default &&
//      lhs.m_groups == rhs.m_groups &&
//      lhs.m_level == rhs.m_level &&
//      lhs.m_isInput == rhs.m_isInput &&
//      lhs.m_verbosity == rhs.m_verbosity &&
//      lhs.m_parentNode == rhs.m_parentNode )
//  {
//    rval = true;
//  }
//
//  return rval;
//}
//
//inline bool operator!=( cxx_utilities::DocumentationNode const & lhs,
//                        cxx_utilities::DocumentationNode const & rhs )
//{
//  bool rval = true;
//  if( lhs.m_name==rhs.m_name &&
//      lhs.m_dataType == rhs.m_dataType &&
//      lhs.m_intKey == rhs.m_intKey &&
//      lhs.m_dataType == rhs.m_dataType &&
//      lhs.m_schemaType == rhs.m_schemaType &&
//      lhs.m_shortDescription == rhs.m_shortDescription &&
//      lhs.m_longDescription == rhs.m_longDescription &&
//      lhs.m_default == rhs.m_default &&
//      lhs.m_groups == rhs.m_groups &&
//      lhs.m_level == rhs.m_level &&
//      lhs.m_isInput == rhs.m_isInput &&
//      lhs.m_verbosity == rhs.m_verbosity &&
//      lhs.m_parentNode == rhs.m_parentNode )
//  {
//    rval = false;
//  }
//
//  return rval;
//}

#endif /* SRC_COMPONENTS_CXX_UTILITIES_SRC_SRC_DOCUMENTATIONNODE_HPP_ */
