# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "CMakeFiles\\Kurs_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\Kurs_autogen.dir\\ParseCache.txt"
  "Kurs_autogen"
  )
endif()
