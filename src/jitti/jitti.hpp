#pragma once

// Source includes
#include "jittiConfig.hpp"
#include "../Macros.hpp"

// System includes
#include <iostream>
#include <unordered_map>
#include <typeindex>
#include <string>
#include <vector>

namespace jitti
{

using SymbolTable = std::unordered_map< std::string, std::pair< void *, std::type_index > >;

/**
 * 
 */
std::ostream & operator<<( std::ostream & os, SymbolTable const & table );


namespace internal
{

time_t getCompileTime( char const * const data, char const * const time );

} // namespace internal

inline time_t getCompileTime()
{ return internal::getCompileTime( __DATE__, __TIME__ ); }

/**
 * 
 */
struct CompilationInfo
{
  time_t const compilationTime = getCompileTime();
  std::string compileCommand;
  bool compilerIsNVCC;
  std::string linker;
  std::string linkArgs;
  std::string templateFunction;
  std::string templateParams;
  std::string headerFile;
  std::string outputObject;
  std::string outputLibrary;
};

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

/**
 * 
 */
template< typename FUNC_POINTER >
class Function
{
public:
  Function( std::string const & path ):
    Function( TypedDynamicLibrary( path ) )
  {}

  Function( Function && ) = default;

  Function( TypedDynamicLibrary && src ):
    m_library( std::move( src ) )
  {
    SymbolTable const & symbols = m_library.getSymbolTable();
    LVARRAY_ERROR_IF_NE( symbols.size(), 1 );

    m_functionName = symbols.begin()->first;
    m_function = m_library.getSymbol< FUNC_POINTER >( m_functionName );
  }

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
  TypedDynamicLibrary m_library;
  std::string m_functionName = "";
  FUNC_POINTER m_function = nullptr;
};

namespace internal
{

/**
 * 
 */
TypedDynamicLibrary compileTemplate( std::string const & compileCommand,
                                     bool const compilerIsNVCC,
                                     std::string const & linker,
                                     std::string const & linkArgs,
                                     std::string const & templateFunction,
                                     std::string const & templateParams,
                                     std::string const & headerFile,
                                     std::string const & outputObject,
                                     std::string const & outputLibrary );

/**
 * 
 */
void readDirectoryFiles( time_t const time,
                         std::string const & directory,
                         std::unordered_map< std::string, std::string > & libraries );

} // namespace internal

/**
 * 
 */
template< typename FUNC_POINTER >
Function< FUNC_POINTER > compileTemplate( CompilationInfo const & info )
{
  return internal::compileTemplate( info.compileCommand,
                                    info.compilerIsNVCC,
                                    info.linker,
                                    info.linkArgs,
                                    info.templateFunction,
                                    info.templateParams,
                                    info.headerFile,
                                    info.outputObject,
                                    info.outputLibrary );
}

/**
 * 
 */
template< typename FUNC_POINTER >
class TypedCache
{
public:
  TypedCache( time_t const compilationTime,
              std::string const & libraryOutputDir,
              std::vector< std::string > const & libraryDirs={} ):
    m_compilationTime( compilationTime ),
    m_libraryOutputDir( libraryOutputDir ),
    m_libraryDirs( libraryDirs )
  {
    m_libraryDirs.push_back( libraryOutputDir );
    refresh();
  }

  void refresh() const
  {
    for( std::string const & libraryDir : m_libraryDirs )
    {
      internal::readDirectoryFiles( m_compilationTime, libraryDir, m_libraries );
    }
  }

  Function< FUNC_POINTER > const & get( std::string const & functionName ) const
  {
    Function< FUNC_POINTER > const * const function = getImpl( functionName );
    LVARRAY_ERROR_IF( function == nullptr,
                      "Could not find a function called '" << functionName << "' in the cache. " <<
                      "Maybe you need to compile it (use getOrLoadOrCompile)." );

    return *function;
  }

  Function< FUNC_POINTER > const & getOrLoad( std::string const & functionName ) const
  {
    Function< FUNC_POINTER > const * const function = getOrLoadImpl( functionName );
    LVARRAY_ERROR_IF( function == nullptr,
                      "Could not find a function called '" << functionName << "' in the cache. " <<
                      "Maybe you need to compile it (use getOrLoadOrCompile)." );

    return *function;
  }

  Function< FUNC_POINTER > const & getOrLoadOrCompile( CompilationInfo & info )
  {
    std::string const functionName = info.templateFunction + "< " + info.templateParams + " >";

    Function< FUNC_POINTER > const * function = getOrLoadImpl( functionName );
    if( function != nullptr )
    {
      return *function;
    }

    // Otherwise we have to compile it.
    info.outputLibrary = m_libraryOutputDir + "/" + getLibraryName( functionName );
    info.outputObject = m_libraryOutputDir.substr( 0, m_libraryOutputDir.size() - 3 ) + ".o";
    auto const result = m_cache.emplace( std::make_pair( functionName, compileTemplate< FUNC_POINTER >( info ) ) );
    LVARRAY_ERROR_IF( !result.second, "This shouldn't be possible." );
    return result.first->second;
  }

private:

  Function< FUNC_POINTER > const * getImpl( std::string const & functionName )
  {
    auto const iter = m_cache.find( functionName );
    if( iter == m_cache.end() )
    { return nullptr; }

    return &iter->second;
  }

  Function< FUNC_POINTER > const * getOrLoadImpl( std::string const & functionName )
  {
    Function< FUNC_POINTER > const * function = getImpl( functionName );
    if( function != nullptr )
    {
      return function;
    }

    // Then look among those already compiled.
    std::string const libraryName = getLibraryName( functionName );
    auto const iter = m_libraries.find( libraryName );
    if( iter != m_libraries.end() )
    {
      auto const result = m_cache.emplace( std::make_pair( functionName, Function< FUNC_POINTER >( iter->second ) ) );
      LVARRAY_ERROR_IF( !result.second, "This shouldn't be possible." );

      function = &result.first->second;
      LVARRAY_ERROR_IF_NE_MSG( function->getName(), functionName,
                               "Expected a different function at '" << iter->second << "'." );

      return function;
    }

    return nullptr;
  }

  std::string getLibraryName( std::string const & functionName )
  { return "lib" + std::to_string( std::hash< std::string >{}( functionName ) ) + ".so"; }

  time_t const m_compilationTime;

  /// The directory in which to create new libraries.
  std::string const m_libraryOutputDir;

  /// A vector of library directories to search for libraries.
  std::vector< std::string > m_libraryDirs;

  /// A map from library names to library paths.
  std::unordered_map< std::string, std::string > mutable m_libraries;

  /// A map containing all the compiled libraries that have been loaded.
  std::unordered_map< std::string, Function< FUNC_POINTER > > m_cache;
};

} // namespace jitti
