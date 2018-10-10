set(cxx-utilities_headers
    src/Array.hpp
    src/ArraySlice.hpp
    src/ArrayView.hpp
    src/Macros.hpp
    src/ObjectCatalog.hpp 
    src/SetSignalHandling.hpp
    src/SFINAE_Macros.hpp
    src/stackTrace.hpp
    src/StringUtilities.hpp
    src/ChaiVector.hpp
    src/SortedArray.hpp
    src/Logger.hpp )

set(cxx-utilities_sources
    src/SetSignalHandling.cpp
    src/stackTrace.cpp
    src/StringUtilities.cpp
    src/Logger.cpp )

add_subdirectory(tests)

set( cxx_dependencies "" )
if( ATK_FOUND OR ENABLE_ATK )
set( cxx_dependencies slic lumberjack )
endif()

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

