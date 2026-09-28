# Compile only the C comparator library used by the Python paper experiments.
set(RUIP_SOURCES
  src/rng_philox.c
  src/mathutil.c
  src/sha256.c
  src/dgp.c
  generated/protocol_generated.c
  generated/quadrature_tables.c
  src/rmap.c
  src/ruip_prior.c
  src/standard_uip.c
  src/power_prior.c
  src/commensurate.c
  src/continuous.c
  src/binary.c
)

add_library(ruip_validation_shared SHARED ${RUIP_SOURCES})
target_include_directories(ruip_validation_shared PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/generated
)
set_target_properties(ruip_validation_shared PROPERTIES
  OUTPUT_NAME ruip_validation
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)
if(NOT MSVC)
  target_link_libraries(ruip_validation_shared PUBLIC m)
endif()
