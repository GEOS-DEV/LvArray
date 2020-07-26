#pragma once

// Source includes
#include "jittiConfig.hpp"
#include "../StringUtilities.hpp"
#include "../Macros.hpp"

// System includes
#include <iostream>
#include <unordered_map>
#include <typeindex>
#include <string>
#include <cstdlib>

namespace jitti
{

using SymbolTable = std::unordered_map< std::string, std::pair< void *, std::type_index > >;

std::ostream & operator<<( std::ostream & os, SymbolTable const & table );

std::array< int, 6 > getCompileTime();

struct CompilationInfo
{
  std::string compileCommand;
  std::string linker;
  std::string linkArgs;
  std::string header;
  std::string function;
};

class DynamicLibrary
{

public:
  DynamicLibrary( char const * const path );

  DynamicLibrary( std::string const & path ):
    DynamicLibrary( path.data() )
  {}

  DynamicLibrary( DynamicLibrary const & ) = delete;

  DynamicLibrary( DynamicLibrary && src );

  ~DynamicLibrary();

  void * getSymbol( char const * const name ) const;

private:
  void * m_handle;
};

class TypedDynamicLibrary : public DynamicLibrary
{

public:
  TypedDynamicLibrary( DynamicLibrary && dl );

  template< typename T >
  T getSymbol( char const * const name ) const
  {
    auto const iter = m_symbols.find( name );
    LVARRAY_ERROR_IF( iter == m_symbols.end(),
                      "Symbol \"" << name << "\" not found in the exported symbol table.\n" <<
                      "table contains:\n" << m_symbols );

    std::pair< void *, std::type_index > const value = iter->second;

    LVARRAY_ERROR_IF( value.second != std::type_index( typeid( T ) ),
                      "Symbol \"" << name << "\" found but it has type " <<
                      LvArray::demangle( value.second.name() ) << " not " << LvArray::demangleType< T >() );

    return reinterpret_cast< T >( value.first );
  }

private:
  SymbolTable getSymbolTable();

  SymbolTable const m_symbols;
};


class Compiler
{
public:
  Compiler( std::string const & compileCommand, std::string const & linker, std::string const & linkArgs ):
    m_compileCommand( compileCommand ),
    m_linker( linker ),
    m_linkArgs( linkArgs )
  {}

  DynamicLibrary createDynamicLibrary( std::string const & filePath,
                                       std::vector< std::string > const & defines,
                                       std::string const & outputObject,
                                       std::string const & outputLibrary ) const
  {
  #if defined(__CUDACC__)
    std::string compileCommand = m_compileCommand + " -Xcompiler -fPIC -c " + filePath + " -o " + outputObject;
  #else
    std::string compileCommand = m_compileCommand + " -fPIC -c " + filePath + " -o " + outputObject;
  #endif

    for ( std::string const & define : defines )
    { compileCommand += " -D " + define; }

    LVARRAY_LOG( "\nCompiling " << filePath );
    LVARRAY_LOG_VAR( compileCommand );
    LVARRAY_ERROR_IF( std::system( compileCommand.data() ) != 0, compileCommand );

    std::string linkCommand = m_linker + " -shared " + outputObject + " " + m_linkArgs + " -o " + outputLibrary;

    LVARRAY_LOG( "\nLinking " << outputObject );
    LVARRAY_LOG_VAR( linkCommand );
    LVARRAY_ERROR_IF( std::system( linkCommand.data() ) != 0, linkCommand );

    return DynamicLibrary( outputLibrary );
  }

private:
  std::string const m_compileCommand;
  std::string const m_linker;
  std::string const m_linkArgs;
};


class TemplateCompiler: private Compiler
{
public:
  TemplateCompiler( std::string const & compileCommand,
                    std::string const & linker,
                    std::string const & linkArgs ) :
    Compiler( compileCommand, linker, linkArgs )
  {}

  TypedDynamicLibrary instantiateTemplate( std::string const & templateFunction,
                                           std::string const & templateParams,
                                           std::string const & headerFile,
                                           std::string const & outputObject,
                                           std::string const & outputLibrary ) const
  {
    char const * const sourceFile = "/usr/WS2/corbett5/LvArray/src/jitti/templateSource.cpp";

    std::vector< std::string > defines = {
      "JITTI_TEMPLATE_HEADER_FILE='\"" + headerFile + "\"'",
      "JITTI_TEMPLATE_FUNCTION=" + templateFunction,
      "JITTI_TEMPLATE_PARAMS=\"" + templateParams + "\""
    };

    return Compiler::createDynamicLibrary( sourceFile,
                                           defines,
                                           outputObject,
                                           outputLibrary );
  }

};

} // namespace jitti
