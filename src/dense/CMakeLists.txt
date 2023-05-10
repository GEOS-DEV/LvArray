set( lvarraydense_headers
     common.hpp
     backendHelpers.hpp
     BlasLapackInterface.hpp
    )

set( lvarraydense_sources
     common.cpp
     BlasLapackInterface.cpp
    )

set( dependencies lvarray ${lvarray_dependencies} blas lapack )

if( ENABLE_MAGMA )
    set( dependencies ${dependencies} magma )
endif()
    

blt_add_library( NAME             lvarraydense
                 SOURCES          ${lvarraydense_sources}
                 HEADERS          ${lvarraydense_headers}
                 DEPENDS_ON       ${dependencies}
                 SHARED TRUE
                 CLEAR_PREFIX TRUE
                 )

target_include_directories( lvarraydense
                            PUBLIC
                            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
                            $<INSTALL_INTERFACE:include> )

install( TARGETS lvarraydense
         EXPORT lvarraydense
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION lib )

install( EXPORT lvarraydense
         DESTINATION share/lvarray/cmake/ )

lvarray_add_code_checks( PREFIX lvarraydense )
