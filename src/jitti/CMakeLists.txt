set( jitti_headers
     types.hpp
     CompilationInfo.hpp
     utils.hpp
     Function.hpp
     Cache.hpp
   )

set( jitti_sources
     CompilationInfo.cpp
     Function.cpp
     utils.cpp
    )

blt_add_library( NAME             jitti
                 SOURCES          ${jitti_sources}
                 HEADERS          ${jitti_headers}
                 DEPENDS_ON       lvarray dl
               )

target_include_directories( jitti PUBLIC ${CMAKE_BINARY_DIR}/include )

set( JITTI_OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib/jitti )
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/../../src/jitti/jittiConfig.hpp.in
                ${CMAKE_BINARY_DIR}/include/jittiConfig.hpp )

