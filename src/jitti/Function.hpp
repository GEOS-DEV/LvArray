#pragma once

// Source includes
#include "types.hpp"
#include "CompilationInfo.hpp"
#include "utils.hpp"
#include "../Macros.hpp"

// System includes
#include <unordered_map>
#include <typeindex>
#include <string>
#include <vector>

namespace jitti
{
namespace internal
{

/**
 * 
 */
class TypedDynamicLibrary
{
public:
  TypedDynamicLibrary( std::string const & path );

  TypedDynamicLibrary( TypedDynamicLibrary const & ) = delete;

  TypedDynamicLibrary( TypedDynamicLibrary && src );

  ~TypedDynamicLibrary();

  template< typename T >
  T getSymbol( std::string const & name ) const
  { return reinterpret_cast< T >( getSymbolCheckType( name, std::type_index( typeid( T ) ) ) ); }

  SymbolTable const & getSymbolTable() const;

private:

  void * getSymbolCheckType( std::string const & name, std::type_index const expectedType ) const;

  void * m_handle;
};

} // namespace internal

/**
 * 
 */
template< typename FUNC_POINTER >
class Function
{
public:
  Function( std::string const & path ):
    m_library( path )
  {
    SymbolTable const & symbols = m_library.getSymbolTable();
    LVARRAY_ERROR_IF_NE( symbols.size(), 1 );

    m_functionName = symbols.begin()->first;
    m_function = m_library.getSymbol< FUNC_POINTER >( m_functionName );
  }

  Function( CompilationInfo const & info ):
    Function( utils::compileTemplate( info.compileCommand,
                                      info.compilerIsNVCC,
                                      info.linker,
                                      info.linkArgs,
                                      info.templateFunction,
                                      info.templateParams,
                                      info.headerFile,
                                      info.outputObject,
                                      info.outputLibrary ) )
  {}

  Function( Function && ) = default;

  Function( Function const & ) = delete;
  Function & operator=( Function const & ) = delete;
  Function & operator=( Function && ) = delete;

  template< typename ... ARGS >
  decltype( auto ) operator()( ARGS && ... args ) const
  {
    LVARRAY_ERROR_IF( m_function == nullptr, "Function not set." );
    return m_function( std::forward< ARGS >( args )... );
  }

  std::string const & getName() const
  { return m_functionName; }

private:
  internal::TypedDynamicLibrary m_library;
  std::string m_functionName = "";
  FUNC_POINTER m_function = nullptr;
};

} // namespace jitti
