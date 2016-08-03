add_subdirectory(src)

set(cxx-utilities_headers
    PARENT_SCOPE )
    
set(cxx-utilities_sources
    PARENT_SCOPE )

#blt_add_library( NAME             cxx-utilities
#                 SOURCES          ${cxx-utilities_sources}
#                 HEADERS          ${cxx-utilities_headers}
#                 DEPENDS_ON        
#                 SYSTEM_DEPENDS_ON gtest
#                 )

set(cxx-utilities_headersdest ${HEADER_INCLUDES_DIRECTORY}/cxx-utilities)
blt_copy_headers_target(NAME        cxx-utilities
                        HEADERS     ${cxx-utilities_headers}
                        DESTINATION ${cxx-utilities_headersdest})

##target_include_directories( componentName PUBLIC ${CMAKE_CURRENT_LIST_DIR} )
#target_include_directories( componentName PUBLIC ${CMAKE_SOURCE_DIR}/components/core/src )
