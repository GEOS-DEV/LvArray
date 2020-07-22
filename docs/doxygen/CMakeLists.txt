blt_add_doxygen_target( LvArray_doxygen )

# Required directory for doxygen to generate output to 
add_custom_target( lvarray_build_doxygen_output_dir
                   ALL
                   COMMAND ${CMAKE_COMMAND} -E make_directory
                   ${CMAKE_BINARY_DIR}/docs/sphinx/html/doxygen
                 )
add_dependencies( LvArray_doxygen lvarray_build_doxygen_output_dir )
