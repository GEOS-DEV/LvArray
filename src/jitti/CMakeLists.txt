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
