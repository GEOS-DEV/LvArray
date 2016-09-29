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
                     std::string const & default0,
                     std::string const & groups,
                     unsigned int const & level,
                     unsigned int const & isInput,
                     unsigned int const & verbosity,
                     DocumentationNode * m_parentNode );


  bool operator==(DocumentationNode const & rhs);
  bool operator!=(DocumentationNode const & rhs) { return !(*this==rhs); }
//  friend bool operator==( cxx_utilities::DocumentationNode const & lhs,
//                          cxx_utilities::DocumentationNode const & rhs );
//  friend bool operator!=( DocumentationNode const & lhs, DocumentationNode const & rhs );


  DocumentationNode * AllocateChildNode( std::string const & name,
                                       std::string const & stringKey,
                                       int const & intKey,
                                       std::string const & dataType,
                                       std::string const & schemaType,
                                       std::string const & shortDescription,
                                       std::string const & longDescription,
                                       std::string const & default0,
                                       std::string const & groups,
                                       unsigned int const & isInput,
                                       unsigned int const & verbosity );


  // TODO THIS ISN'T CORRECT. FIX IT.
  void Print(  ) const
  {
    std::string indent( m_level*4, ' ');
    std::cout<<indent<<"m_name              = "<<m_name<<std::endl;
    std::cout<<indent<<"m_stringKey         = "<<m_stringKey<<std::endl;
    std::cout<<indent<<"m_intKey            = "<<m_intKey<<std::endl;
    std::cout<<indent<<"m_dataType          = "<<m_dataType<<std::endl;
    std::cout<<indent<<"m_schemaType        = "<<m_schemaType<<std::endl;
    std::cout<<indent<<"m_shortDescription  = "<<m_shortDescription<<std::endl;
    std::cout<<indent<<"m_longDescription   = "<<m_longDescription<<std::endl;
    std::cout<<indent<<"m_default           = "<<m_default<<std::endl;
    std::cout<<indent<<"m_groups            = "<<m_groups<<std::endl;
    std::cout<<indent<<"m_level             = "<<m_level<<std::endl;
    std::cout<<indent<<"m_isInput           = "<<m_isInput<<std::endl;
    std::cout<<indent<<"m_verbosity         = "<<m_verbosity<<std::endl;

    for( auto const & child : m_child )
    {
      std::cout<<indent<<"node "<<child.first<<"\n";
      child.second.Print();
    }
  }



  std::string const & name() const { return m_name; }
  std::string const & stringKey() const { return m_stringKey; }
  int const & intKey() const { return m_intKey; }
  void setIntKey( int intKey ) { m_intKey = intKey; }
  std::string const & dataType() const { return m_dataType; }
  std::string const & schemaType() const { return m_schemaType; }
  std::string const & shortDescription() const { return m_shortDescription; }
  std::string const & longDescription() const { return m_longDescription; }
  std::string const & getDefault() const { return m_default; }
  std::string const & groups() const { return m_groups; }
  unsigned int const & level() const { return m_level; }
  unsigned int const & isInput() const { return m_isInput; }
  unsigned int const & verbosity() const { return m_verbosity; }

  std::map<std::string,DocumentationNode>       & getChildNodes()       { return m_child; }
  std::map<std::string,DocumentationNode> const & getChildNodes() const { return m_child; }

  DocumentationNode       * getChildNode( std::string const & name )       { return &m_child.at(name); }
  DocumentationNode const * getChildNode( std::string const & name ) const { return &m_child.at(name); }


public:

  std::string m_name               = "";
  std::string m_stringKey          = "";
  int         m_intKey             = -1;
  std::string m_dataType           = "";
  std::string m_schemaType         = "";
  std::string m_shortDescription   = "";
  std::string m_longDescription    = "";
  std::string m_default            = "";
  std::string m_groups             = "";
  unsigned int m_level             =  0;
  unsigned int m_isInput           =  0;
  unsigned int m_verbosity         =  0;
  DocumentationNode * m_parentNode = nullptr;
  std::map<std::string,DocumentationNode> m_child = {};

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
      m_parentNode == rhs.m_parentNode )
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
