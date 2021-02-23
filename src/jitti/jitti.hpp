#pragma once

// Source includes
#include "jittiConfig.hpp"
#include "../Macros.hpp"

// System includes
#include <iostream>
#include <unordered_map>
#include <typeindex>
#include <string>
#include <cstdlib>
#include <vector>

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
  TypedDynamicLibrary( char const * const path );

  TypedDynamicLibrary( std::string const & path ):
    TypedDynamicLibrary( path.data() )
  {}

  TypedDynamicLibrary( DynamicLibrary && src);

  TypedDynamicLibrary( TypedDynamicLibrary const & ) = delete;

  TypedDynamicLibrary( TypedDynamicLibrary && src );

  template< typename T >
  T getSymbol( char const * const name ) const
  { return reinterpret_cast< T >( getSymbolCheckType( name, std::type_index( typeid( T ) ) ) ); }

private:

  void * getSymbolCheckType( char const * const name, std::type_index const expectedType ) const;

  SymbolTable m_symbols;
};


class Compiler
{
public:
  Compiler( std::string const & compileCommand,
            bool const cuda,
            std::string const & linker,
            std::string const & linkArgs );

  DynamicLibrary createDynamicLibrary( std::string const & filePath,
                                       std::vector< std::string > const & defines,
                                       std::string const & outputObject,
                                       std::string const & outputLibrary ) const;

private:
  std::string const m_compileCommand;
  bool const m_cuda;
  std::string const m_linker;
  std::string const m_linkArgs;
};


class TemplateCompiler: private Compiler
{
public:
  TemplateCompiler( std::string const & compileCommand,
                    bool const cuda,
                    std::string const & linker,
                    std::string const & linkArgs ) :
    Compiler( compileCommand, cuda, linker, linkArgs )
  {}

  TypedDynamicLibrary instantiateTemplate( std::string const & templateFunction,
                                           std::string const & templateParams,
                                           std::string const & headerFile,
                                           std::string const & outputObject,
                                           std::string const & outputLibrary ) const;

};

} // namespace jitti
