set(cxx-utilities_headers
    Array.hpp
    ArraySlice.hpp
    ArrayUtilities.hpp
    ArrayView.hpp
    ChaiVector.hpp
    IntegerConversion.hpp
    Logger.hpp
    Macros.hpp
    ObjectCatalog.hpp 
    SetSignalHandling.hpp
    SFINAE_Macros.hpp
    SortedArray.hpp
    StackArrayWrapper.hpp
    stackTrace.hpp
    StringUtilities.hpp )

set(cxx-utilities_sources
    Logger.cpp
    SetSignalHandling.cpp
    stackTrace.cpp
    StringUtilities.cpp )


if( CMAKE_PROJECT_NAME STREQUAL "geosx" )
    list( APPEND cxx-utilities_headers DocumentationNode.hpp )
    list( APPEND cxx-utilities_sources DocumentationNode.cpp )
endif()


set( cxx_dependencies "" )
if( ATK_FOUND OR ENABLE_ATK OR ENABLE_AXOM )
set( cxx_dependencies slic lumberjack )
endif()

if( ${ENABLE_CHAI})
    set( cxx_dependencies ${cxx_dependencies} chai umpire )
endif()

if( ${ENABLE_MPI})
    set( cxx_dependencies ${cxx_dependencies} mpi )
endif()

if( ${ENABLE_CUDA})
    set( cxx_dependencies ${cxx_dependencies} cuda )
endif()

message("adding cxx-utilities library")
blt_add_library( NAME             cxx-utilities
                 SOURCES          ${cxx-utilities_sources}
                 HEADERS          ${cxx-utilities_headers}
                 HEADERS_OUTPUT_SUBDIR cxx-utilities
                 DEPENDS_ON       ${cxx_dependencies}
                 )



target_include_directories( cxx-utilities PUBLIC ${CMAKE_CURRENT_LIST_DIR} )
target_include_directories( cxx-utilities PUBLIC ${CMAKE_BINARY_DIR}/include )

cxxutils_add_code_checks(PREFIX cxx-utilities )

