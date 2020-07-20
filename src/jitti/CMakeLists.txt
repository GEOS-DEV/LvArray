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

set( JITTI_COMPILE_COMMANDS_JSON ${CMAKE_BINARY_DIR}/compile_commands.json )

set( JITTI_RAJA_LIBRARY ${RAJA_LIBRARY} )

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/../../src/jitti/jittiConfig.hpp.in
                ${CMAKE_BINARY_DIR}/include/jittiConfig.hpp )

