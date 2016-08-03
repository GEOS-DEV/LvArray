file(GLOB children RELATIVE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/*)
message("children = ${children}")

set(cxx-utilities_headers
    PARENT_SCOPE )
    
set(cxx-utilities_sources
    PARENT_SCOPE )

#set(dirlist "")
foreach(child ${children})
  if(IS_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${child})
    if( NOT ( ${child} STREQUAL "cmake" OR ${child} STREQUAL "docs" ) )
      add_subdirectory(${child})
    endif()
  endif()
endforeach()
                   
#if (ENABLE_DOCS)
#  add_subdirectory(docs)
#endif()
