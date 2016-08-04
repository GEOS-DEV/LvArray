
set(cxx-utilities_headers
    PARENT_SCOPE )
    
set(cxx-utilities_sources
    PARENT_SCOPE )

file(GLOB children RELATIVE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/*)
foreach(child ${children})
  if(IS_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${child})
    if( NOT ( ${child} STREQUAL "cmake" OR ${child} STREQUAL "docs" ) )
      add_subdirectory(${child})
    endif()
  endif()
endforeach()
