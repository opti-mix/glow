add_library(OpenCL
            OpenCL.cpp
            Transforms.cpp
            LibDNNSupport.cpp)

target_include_directories(OpenCL
                           PUBLIC
                           /Users/rlevenstein/src/libdnn/include
                           /Users/rlevenstein/src/libdnn/build
                           )

target_link_libraries(OpenCL
                      PRIVATE
                      Base
                      Graph
                      CodeGen
                      IR)

target_link_libraries(OpenCL
                      PRIVATE
                      OpenCL::OpenCL
                      CLBlast
                      # Use libdnn.
                      /Users/rlevenstein/src/libdnn/build/src/libgreentea_libdnn.dylib
                      )
