set( pylvarray_headers
     python.hpp
     pythonForwardDeclarations.hpp
     pythonHelpers.hpp
     numpyHelpers.hpp
     PyArray.hpp
     PySortedArray.hpp
     PyCRSMatrix.hpp
     PyFunc.hpp
     PyArrayOfArrays.hpp
     PyArrayOfSets.hpp
    )

set( pylvarray_sources
     python.cpp
     pythonHelpers.cpp
     numpyHelpers.cpp
     PyArray.cpp
     PySortedArray.cpp
     PyCRSMatrix.cpp
     PyFunc.cpp
     PyArrayOfArrays.cpp
     PyArrayOfSets.cpp
    )

blt_add_library( NAME             pylvarray
                 SOURCES          ${pylvarray_sources}
                 HEADERS          ${pylvarray_headers}
                 DEPENDS_ON       lvarray ${lvarray_dependencies} Python3::Python Python3::NumPy
                 SHARED TRUE
                 CLEAR_PREFIX TRUE
                 )

target_include_directories( pylvarray
                            PUBLIC
                            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
                            $<INSTALL_INTERFACE:include> )

install( TARGETS pylvarray
         EXPORT pylvarray
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION lib )

install( EXPORT pylvarray
         DESTINATION share/lvarray/cmake/ )

lvarray_add_code_checks( PREFIX pylvarray )
