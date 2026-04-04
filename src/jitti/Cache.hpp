#pragma once

#include "Function.hpp"
#include "utils.hpp"
#include "CompilationInfo.hpp"
#include "../Macros.hpp"
#include "../typeManipulation.hpp"

#include <string>
#include <vector>
#include <unordered_map>

namespace jitti
{

// template < const char * TYPE, const char * HEADER >
// struct constexpr_jitti_info
// {
//   constexpr static const char * typeName = TYPE;
//   constexpr static const char * headerFile = HEADER;
// };

//
// NOTE: The TYPE value in the following macros needs to be *FULLY SCOPED*!
//
#if JITTI == 1
  //  We can't programmatically get a constexpr expression with the name of the type without fully 
  //   specifying the type e.g. specifying all the template parameters
  //  Since JIITTI is based around delaying that specification until we JIT a function, we can't 
  //   use templates to derive this info, it can only happen via preprocessing
  //
  // C++20 will allow using string literals as template parameters, until then we have to
  //  declare/define a constexpr character array and use a pointer to that instead
  //  but when it is available we can obsolete the JITTI_DECL macro, and jitti::Name struct entirely
  #define JITTI_DECL( VARNAME, TYPE, HEADER ) \
constexpr const char VARNAME##Name[] = STRINGIZE( TYPE ); \
constexpr const char VARNAME##Header[] = HEADER;
  #define JITTI_TPARAM( VARNAME, TYPE ) VARNAME##Name, VARNAME##Header
#else 
  #define JITTI_DECL( VARNAME, TYPE, HEADER )
  #define JITTI_TPARAM( VARNAME, TYPE ) TYPE
#endif

template< typename ARG0, typename ... ARGS >
std::string getInstantiationName( std::string const & templateName,
                                  ARG0 const & firstTemplateParam,
                                  ARGS const & ... remainingTemplateParams )
{
  std::string output = templateName;
  output += "< " + firstTemplateParam;
  LvArray::typeManipulation::forEachArg( [&output] ( auto const & param )
  {
    output += ", " + param;
  }, remainingTemplateParams... );

  output += " >";
  return output;
}

/**
 * 
 */
template< typename FUNC_POINTER >
class Cache
{
public:

  /**
   * 
   */
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

  Cache( Cache const & ) = delete;
  Cache( Cache && ) = delete;

  /**
   * 
   */
  void refresh() const
  {
    for( std::string const & libraryDir : m_libraryDirs )
    {
      utils::readDirectoryFiles( m_compilationTime, libraryDir, m_libraries );
    }
  }

  /**
   * 
   */
  Function< FUNC_POINTER > const * tryGet( std::string const & instantiationName ) const
  {
    auto const iter = m_cache.find( instantiationName );
    if( iter == m_cache.end() )
    { return nullptr; }

    return &iter->second;
  }
  
  /**
   * 
   */
  Function< FUNC_POINTER > const * tryGet( CompilationInfo const & info ) const
  {
    return tryGet( getInstantiationName( info.templateFunction, info.templateParams ) );
  }

  /**
   * 
   */
  Function< FUNC_POINTER > const & get( std::string const & instantiationName ) const
  {
    Function< FUNC_POINTER > const * const function = tryGet( instantiationName );
    LVARRAY_ERROR_IF( function == nullptr,
                      "Could not find an instantiation called '" << instantiationName << "' in the cache. " <<
                      "Maybe you need to compile it (use getOrLoadOrCompile)." );

    return *function;
  }

  /**
   * 
   */
  Function< FUNC_POINTER > const & get( CompilationInfo const & info ) const
  {
    return get( getInstantiationName( info.templateFunction, info.templateParams ) );
  }

  /**
   * 
   */
  Function< FUNC_POINTER > const * tryGetOrLoad( std::string const & instantiationName )
  {
    Function< FUNC_POINTER > const * function = tryGet( instantiationName );
    if( function != nullptr )
    {
      return function;
    }

    // Then look among those already compiled.
    std::string const libraryName = getLibraryName( instantiationName );
    auto const iter = m_libraries.find( libraryName );
    if( iter != m_libraries.end() )
    {
      auto const result = m_cache.emplace( std::make_pair( instantiationName, Function< FUNC_POINTER >( iter->second ) ) );
      LVARRAY_ERROR_IF( !result.second, "This shouldn't be possible." );

      function = &result.first->second;
      LVARRAY_ERROR_IF_NE_MSG( function->getName(), instantiationName,
                               "Expected a different function at '" << iter->second << "'." );

      return function;
    }

    return nullptr;
  }

  /**
   * 
   */
  Function< FUNC_POINTER > const * tryGetOrLoad( CompilationInfo const & info )
  {
    return tryGetOrLoad( getInstantiationName( info.templateFunction, info.templateParams ) );
  }

  /**
   * 
   */
  Function< FUNC_POINTER > const & getOrLoad( std::string const & instantiationName )
  {
    Function< FUNC_POINTER > const * function = tryGetOrLoad( instantiationName );
    LVARRAY_ERROR_IF( function == nullptr,
                      "Could not find an instantiation called '" << instantiationName << "' in the cache. " <<
                      "Maybe you need to compile it (use getOrLoadOrCompile)." );

    return *function;
  }

  /**
   * 
   */
  Function< FUNC_POINTER > const & getOrLoad( CompilationInfo & info )
  {
    return getOrLoad( getInstantiationName( info.templateFunction, info.templateParams ) );
  }

  /**
   * 
   */
  Function< FUNC_POINTER > const & getOrLoadOrCompile( CompilationInfo & info )
  {
    std::string const instantiationName = getInstantiationName( info.templateFunction, info.templateParams );

    Function< FUNC_POINTER > const * function = tryGetOrLoad( instantiationName );
    if( function != nullptr )
    {
      return *function;
    }

    // Otherwise we have to compile it.
    info.outputLibrary = m_libraryOutputDir + "/" + getLibraryName( instantiationName );
    info.outputObject = info.outputLibrary.substr( 0, info.outputLibrary.size() - 3 ) + ".o";
    auto const result = m_cache.emplace( std::make_pair( instantiationName, Function< FUNC_POINTER >( info ) ) );
    LVARRAY_ERROR_IF( !result.second, "This shouldn't be possible." );
    return result.first->second;
  }

private:

  std::string getLibraryName( std::string const & instantiationName ) const
  { return "lib" + std::to_string( std::hash< std::string >{}( instantiationName ) ) + ".so"; }

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
