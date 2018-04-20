
add_executable(loader
                 loader.cpp)
target_link_libraries(loader
                      PUBLIC
                        /Users/rlevenstein/src/build/Halide/genfiles/halide_sgemm_transAB/halide_sgemm_transAB.a
                      PRIVATE
                        Base
                        Importer
                        ExecutionEngine
                        IR
                        Quantization)
