#pragma once

#include "Function.hpp"
#include "utils.hpp"
#include "CompilationInfo.hpp"
#include "../Macros.hpp"

#include <string>
#include <vector>
#include <unordered_map>

namespace jitti
{

/**
 * 
 */
template< typename FUNC_POINTER >
class Cache
{
public:
  Cache( time_t const compilationTime,
         std::string const & libraryOutputDir,
         std::vector< std::string > const & libraryDirs={} ):
    m_compilationTime( compilationTime ),
    m_libraryOutputDir( libraryOutputDir ),
    m_libraryDirs( libraryDirs )
  {
    utils::makeDirsForPath( libraryOutputDir );
    m_libraryDirs.push_back( libraryOutputDir );
    refresh();
  }

  void refresh() const
  {
    for( std::string const & libraryDir : m_libraryDirs )
    {
      utils::readDirectoryFiles( m_compilationTime, libraryDir, m_libraries );
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
    info.outputObject = info.outputLibrary.substr( 0, info.outputLibrary.size() - 3 ) + ".o";
    auto const result = m_cache.emplace( std::make_pair( functionName, Function< FUNC_POINTER >( info ) ) );
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
