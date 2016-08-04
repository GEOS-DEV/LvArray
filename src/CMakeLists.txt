
set(cxx-utilities_headers
    src/MultidimensionalArray.hpp
    src/ObjectCatalog.hpp 
    src/StringUtilities.hpp )

set(cxx-utilities_sources
    src/StringUtilities.cpp )

add_subdirectory(tests)

blt_add_library( NAME             cxx-utilities
                 SOURCES          ${cxx-utilities_sources}
                 HEADERS          ${cxx-utilities_headers}
                 DEPENDS_ON        
                 SYSTEM_DEPENDS_ON gtest
                 )

#set(cxx-utilities_headersdest ${HEADER_INCLUDES_DIRECTORY}/cxx-utilities)
#blt_copy_headers_target(NAME        cxx-utilities
#                        HEADERS     ${cxx-utilities_headers}
#                        DESTINATION ${cxx-utilities_headersdest})

target_include_directories( cxx-utilities PUBLIC ${CMAKE_CURRENT_LIST_DIR} )
target_include_directories( cxx-utilities PUBLIC ${CMAKE_SOURCE_DIR}/src )
