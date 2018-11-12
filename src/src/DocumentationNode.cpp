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


#include "DocumentationNode.hpp"
#include "Logger.hpp"
//#include "dataRepository/RestartFlags.hpp"


namespace cxx_utilities
{


DocumentationNode::DocumentationNode( std::string const & name,
                                      std::string const & stringKey,
                                      int const & intKey,
                                      std::string const & dataType,
                                      std::string const & schemaType,
                                      std::string const & shortDescription,
                                      std::string const & longDescription,
                                      std::string const & default0,
                                      std::string const & groups,
                                      unsigned int const & parentManaged,
                                      unsigned int const & level,
                                      unsigned int const & isInput,
                                      unsigned int const & verbosity,
                                      DocumentationNode * parentNode,
                                      geosx::dataRepository::RestartFlags const & restart_flags ):
  m_name( name ),
  m_stringKey( stringKey ),
  m_intKey( intKey ),
  m_dataType( dataType ),
  m_schemaType( schemaType ),
  m_shortDescription( shortDescription ),
  m_longDescription( longDescription ),
  m_default( default0 ),
  m_groups( groups ),
  m_managedByParent( parentManaged ),
  m_level( level ),
  m_isInput( isInput ),
  m_verbosity( verbosity ),
  m_parentNode( parentNode ),
  m_restart_flags( restart_flags ),
  m_child()
{}

DocumentationNode * DocumentationNode::AllocateChildNode( std::string const & name,
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
                                                          geosx::dataRepository::RestartFlags const & restart_flags )
{
  DocumentationNode newNode( name,
                             stringKey,
                             intKey,
                             dataType,
                             schemaType,
                             shortDescription,
                             longDescription,
                             default0,
                             groups,
                             parentManaged,
                             this->m_level + 1,
                             isInput,
                             verbosity,
                             this,
                             restart_flags );

  // a check to see if nodeName exists. if it does, check to see that members
  // are identical
  std::map<std::string, DocumentationNode>::iterator iter = m_child.find( name );

  // name exists
  if( iter != m_child.end() )
  {
    if( newNode.m_dataType != "" &&
        newNode.m_dataType != iter->second.m_dataType )
    {
      GEOS_LOG_RANK( iter->second.toString());
      GEOS_LOG_RANK( newNode.toString());
      GEOS_ERROR( "documentation node already exists, but has different attributes than specified" );
    }
  }
  else
  {
    m_child.insert( {name, newNode} );
  }
  return &(m_child.at( name ));
}

std::string DocumentationNode::toString() const
{
  std::ostringstream oss;

  if( m_level == 0 )
  {
    oss << "ROOT NODE:"<<m_name<<std::endl;
  }

  std::string indent( m_level*4, ' ' );
  oss << indent << "m_name              = " << m_name << std::endl;
  oss << indent << "m_stringKey         = " << m_stringKey << std::endl;
  oss << indent << "m_intKey            = " << m_intKey << std::endl;
  oss << indent << "m_dataType          = " << m_dataType << std::endl;
  oss << indent << "m_schemaType        = " << m_schemaType << std::endl;
  oss << indent << "m_shortDescription  = " << m_shortDescription << std::endl;
  oss << indent << "m_longDescription   = " << m_longDescription << std::endl;
  oss << indent << "m_default           = " << m_default << std::endl;
  oss << indent << "m_groups            = " << m_groups << std::endl;
  oss << indent << "m_level             = " << m_level << std::endl;
  oss << indent << "m_isInput           = " << m_isInput << std::endl;
  oss << indent << "m_verbosity         = " << m_verbosity << std::endl;

  for( auto const & child : m_child )
  {
    oss << indent << "node " << child.first << std::endl;
    oss << child.second.toString();
  }

  return oss.str();
}

}
