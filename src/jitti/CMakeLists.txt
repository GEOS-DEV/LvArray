set( jitti_headers
     jitti.hpp
     json.hpp
   )

set( jitti_sources
     jitti.cpp
    )

blt_add_library( NAME             jitti
                 SOURCES          ${jitti_sources}
                 HEADERS          ${jitti_headers}
                 DEPENDS_ON       dl
               )

target_include_directories( jitti PUBLIC ${CMAKE_BINARY_DIR}/include )

set( JITTI_OUTPUT_DIR ${CMAKE_BINARY_DIR}/lib/jitti )
set( JITTI_RAJA_LIBRARY ${RAJA_LIBRARY} )
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/../../src/jitti/jittiConfig.hpp.in
                ${CMAKE_BINARY_DIR}/include/jittiConfig.hpp )

