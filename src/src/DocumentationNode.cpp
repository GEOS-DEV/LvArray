
#include "DocumentationNode.hpp"

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
                                      unsigned int const & level,
                                      unsigned int const & isInput,
                                      unsigned int const & verbosity,
                                      DocumentationNode * parentNode ):
    m_name(name),
    m_stringKey(stringKey),
    m_intKey(intKey),
    m_dataType(dataType),
    m_schemaType(schemaType),
    m_shortDescription(shortDescription),
    m_longDescription(longDescription),
    m_default(default0),
    m_groups(groups),
    m_level(level),
    m_isInput(isInput),
    m_verbosity(verbosity),
    m_parentNode(parentNode),
    m_child()
{
}

DocumentationNode * DocumentationNode::AllocateChildNode( std::string const & name,
                                                          std::string const & stringKey,
                                                          int const & intKey,
                                                          std::string const & dataType,
                                                          std::string const & schemaType,
                                                          std::string const & shortDescription,
                                                          std::string const & longDescription,
                                                          std::string const & default0,
                                                          std::string const & groups,
                                                          unsigned int const & isInput,
                                                          unsigned int const & verbosity )
{
  DocumentationNode newNode(name,
                            stringKey,
                            intKey,
                            dataType,
                            schemaType,
                            shortDescription,
                            longDescription,
                            default0,
                            groups,
                            this->m_level + 1,
                            isInput,
                            verbosity,
                            this );

  // a check to see if nodeName exists. if it does, check to see that members are identical
  std::map<std::string,DocumentationNode>::iterator iter = m_child.find(name);

  // name exists
  if( iter != m_child.end() )
  {
    if( iter->second != newNode )
    {
      std::cout<<"documentation node already exists, but has different attributes than specified"<<std::endl;
      exit(0);
    }
  }
  else
  {
    m_child.insert( {name,newNode} );
  }
  return &(m_child.at(name));
}

}
