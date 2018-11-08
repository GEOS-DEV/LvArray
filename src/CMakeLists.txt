set(cxx-utilities_headers
    src/Array.hpp
    src/ArraySlice.hpp
    src/ArrayView.hpp
    src/ChaiVector.hpp
    src/Logger.hpp
    src/Macros.hpp
    src/ObjectCatalog.hpp 
    src/SetSignalHandling.hpp
    src/SFINAE_Macros.hpp
    src/SortedArray.hpp
    src/stackTrace.hpp
    src/StringUtilities.hpp )

set(cxx-utilities_sources
    src/Logger.cpp
    src/SetSignalHandling.cpp
    src/stackTrace.cpp
    src/StringUtilities.cpp )


if( CMAKE_PROJECT_NAME EQUAL "geosx" )
    list( APPEND cxx-utilities_headers src/DocumentationNode.hpp )
    list( APPEND cxx-utilities_sources src/DocumentationNode.cpp )
endif()


add_subdirectory(tests)

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

message("adding cxx-utilities library")
blt_add_library( NAME             cxx-utilities
                 SOURCES          ${cxx-utilities_sources}
                 HEADERS          ${cxx-utilities_headers}
                 HEADERS_OUTPUT_SUBDIR cxx-utilities
                 DEPENDS_ON       ${cxx_dependencies}
                 )

#set(cxx-utilities_headersdest ${HEADER_INCLUDES_DIRECTORY}/cxx-utilities)
#blt_copy_headers_target(NAME        cxx-utilities
#                        HEADERS     ${cxx-utilities_headers}
#                        DESTINATION ${cxx-utilities_headersdest})

target_include_directories( cxx-utilities PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src )
target_include_directories( cxx-utilities PUBLIC ${CMAKE_BINARY_DIR}/include )

