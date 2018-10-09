
set(cxx-utilities_headers
    src/DocumentationNode.hpp
    src/Macros.hpp
    src/MultidimensionalArray.hpp
    src/MultidimensionalArray2.hpp
    src/ManagedArray.hpp
    src/ArrayView.hpp
    src/ObjectCatalog.hpp 
    src/SetSignalHandling.hpp
    src/SFINAE_Macros.hpp
    src/stackTrace.hpp
    src/StringUtilities.hpp
    src/ChaiVector.hpp
    src/SortedArray.hpp
    src/Logger.hpp )

set(cxx-utilities_sources
    src/DocumentationNode.cpp
    src/SetSignalHandling.cpp
    src/stackTrace.cpp
    src/StringUtilities.cpp
    src/Logger.cpp )

add_subdirectory(tests)

set( cxx_dependencies slic lumberjack )

if( ${ENABLE_CHAI})
    set( cxx_dependencies ${cxx_dependencies} chai umpire )
endif()

message("adding cxxutilities library")
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

if( ENABLE_UNCRUSTIFY )
geosx_add_code_checks(PREFIX cxx-utilities)
endif()