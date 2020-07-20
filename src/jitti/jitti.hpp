#pragma once

// Source includes
#include "jittiConfig.hpp"
#include "../StringUtilities.hpp"
#include "../Macros.hpp"

// System includes
#include <iostream>
#include <dlfcn.h>
#include <unordered_map>
#include <typeindex>
#include <string>
#include <cstdlib>

namespace jitti
{

using SymbolTable = std::unordered_map< std::string, std::pair< void *, std::type_index > >;

template< typename T >
std::pair< void *, std::type_index >
createEntry( T * symbol )
{ return { reinterpret_cast< void * >( symbol ), std::type_index( typeid( symbol ) ) }; }


struct CompilationInfo
{
  std::string compilerPath;
  std::string compilerFlags;
  std::string header;
  std::vector< std::string > includeDirs;
  std::vector< std::string > systemIncludeDirs;
  std::vector< std::string > libs;
};

class DynamicLibrary
{

public:
  DynamicLibrary( std::string const & path ):
    m_handle( dlopen( path.data(), RTLD_LAZY ) )
  {}

  DynamicLibrary( DynamicLibrary const & ) = delete;

  DynamicLibrary( DynamicLibrary && src ) :
    m_handle( src.m_handle )
  {
    src.m_handle = nullptr;
  }

  ~DynamicLibrary()
  {
    if ( m_handle != nullptr )
    { LVARRAY_ERROR_IF( dlclose( m_handle ), dlerror() ); }
  }

  void * getSymbol( char const * const name ) const
  {
    LVARRAY_ERROR_IF( m_handle == nullptr, dlerror() );

    void * const symbolAddress = dlsym( m_handle, name );

    char const * const error = dlerror();

    LVARRAY_ERROR_IF( error != nullptr, "Could not find the symbol: " << name << "\n" << error );

    return symbolAddress;
  }

private:
  void * m_handle;
};

class TypedDynamicLibrary : public DynamicLibrary
{

public:
  TypedDynamicLibrary( DynamicLibrary && dl ):
    DynamicLibrary( std::move( dl ) ),
    m_symbols( getSymbolTable() )
  {}

  template< typename T >
  T getSymbol( char const * const name ) const
  {
    auto const iter = m_symbols.find( name );
    LVARRAY_ERROR_IF( iter == m_symbols.end(),
                      "Symbol \"" << name << "\" not found in the exported symbol table.\n" );

    std::pair< void *, std::type_index > const value = iter->second;

    LVARRAY_ERROR_IF( value.second != std::type_index( typeid( T ) ),
                      "Symbol \"" << name << "\" found but it has type " <<
                      LvArray::demangle( value.second.name() ) << " not " << LvArray::demangleType< T >() );

    return reinterpret_cast< T >( value.first );
  }

private:
  SymbolTable getSymbolTable()
  {
    SymbolTable const * const symbols =
      reinterpret_cast< SymbolTable * >( DynamicLibrary::getSymbol( "exportedSymbols") );

    char const * const error = dlerror();

    LVARRAY_ERROR_IF( error != nullptr, "Could not find the symbols table!\n" << error );

    return *symbols;
  }

  SymbolTable const m_symbols;
};


class Compiler
{
public:
  Compiler( std::string const & compilerPath, std::string const & compilerArgs ):
    m_compilerPath( compilerPath ),
    m_compilerArgs( compilerArgs )
  {}

  DynamicLibrary createDynamicLibrary( std::string const & filePath,
                                       std::string const & outputLibrary,
                                       std::vector< std::string > const & includeDirs,
                                       std::vector< std::string > const & systemIncludeDirs,
                                       std::vector< std::string > const & libs,
                                       std::vector< std::string > const & defines ) const
  {
  #if defined(__CUDACC__)
    std::string command = m_compilerPath + " " + m_compilerArgs + " -Xcompiler -fPIC -shared -o " + outputLibrary + " " + filePath;
  #else
    std::string command = m_compilerPath + " " + m_compilerArgs + " -fPIC -shared -o " + outputLibrary + " " + filePath;
  #endif

    for ( std::string const & include : includeDirs )
    { command += " -I " + include; }

    for ( std::string const & include : systemIncludeDirs )
    { command += " -isystem " + include; }

    for ( std::string const & lib : libs )
    { command += " " + lib; }

    for ( std::string const & define : defines )
    { command += " -D " + define; }

    LVARRAY_LOG( "Compiling " << filePath );
    LVARRAY_LOG_VAR( command );
    LVARRAY_ERROR_IF( std::system( command.data() ) != 0, command );

    return DynamicLibrary( outputLibrary );
  }

private:
  std::string const m_compilerPath;
  std::string const m_compilerArgs;
};


class TemplateCompiler: private Compiler
{
public:
  TemplateCompiler( std::string const & compilerPath,
                    std::string const & compilerArgs ) :
    Compiler( compilerPath, compilerArgs )
  {}

  TypedDynamicLibrary instantiateTemplate( std::string const & templateFunction,
                                           std::string const & templateParams,
                                           std::string const & headerFile,
                                           std::string const & outputLibrary,
                                           std::vector< std::string > const & includeDirs,
                                           std::vector< std::string > const & systemIncludeDirs,
                                           std::vector< std::string > const & libs,
                                           std::vector< std::string > defines={} ) const
  {
    char const * const sourceFile = "/usr/WS2/corbett5/LvArray/src/jitti/templateSource.cpp";

    defines.emplace_back( "JITTI_TEMPLATE_HEADER_FILE='\"" + headerFile + "\"'" );
    defines.emplace_back( "JITTI_TEMPLATE_FUNCTION=" + templateFunction );
    defines.emplace_back( "JITTI_TEMPLATE_PARAMS=" + templateParams );
    return Compiler::createDynamicLibrary( sourceFile,
                                           outputLibrary,
                                           includeDirs,
                                           systemIncludeDirs,
                                           libs,
                                           defines );
  }

};

} // namespace jitti
