
set(cxx-utilities_headers
    src/InputDocumentation.hpp
    src/Macros.hpp
    src/MultidimensionalArray.hpp
    src/MultidimensionalArray2.hpp
    src/ObjectCatalog.hpp 
    src/SetSignalHandling.hpp
    src/SFINAE_Macros.hpp
    src/stackTrace.hpp
    src/StringUtilities.hpp )

set(cxx-utilities_sources
    src/SetSignalHandling.cpp
    src/stackTrace.cpp
    src/StringUtilities.cpp )

add_subdirectory(tests)

message("adding cxxutilities library")
blt_add_library( NAME             cxx-utilities
                 SOURCES          ${cxx-utilities_sources}
                 HEADERS          ${cxx-utilities_headers}
                 HEADERS_OUTPUT_SUBDIR cxx-utilities
                 DEPENDS_ON        
                 SYSTEM_DEPENDS_ON 
                 )

#set(cxx-utilities_headersdest ${HEADER_INCLUDES_DIRECTORY}/cxx-utilities)
#blt_copy_headers_target(NAME        cxx-utilities
#                        HEADERS     ${cxx-utilities_headers}
#                        DESTINATION ${cxx-utilities_headersdest})

target_include_directories( cxx-utilities PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src )
#target_include_directories( cxx-utilities PUBLIC ${CMAKE_SOURCE_DIR}/src )
