#ifdef GLOW_WITH_OPENCL

BB.newNode("OCLConvolution")
    .addInput("Input")
    .addInput("Filter")
    .addInput("Bias")
    .addMember(MemberType::SizeT, "Kernel")
    .addMember(MemberType::SizeT, "Stride")
    .addMember(MemberType::SizeT, "Pad")
    .addResultFromCtorArg()
    .setDocstring("This is an OpenCL-specific convolution implementation where the "
                  "filter, the bias and the input are in the HCHW format");

BB.newNode("OCLPoolAvg")
    .addInput("Input")
    .addMember(MemberType::SizeT, "Kernel")
    .addMember(MemberType::SizeT, "Stride")
    .addMember(MemberType::SizeT, "Pad")
    .addResultFromCtorArg()
    //.addGradient()
    .setDocstring(
        "This is an OpenCL-specific Average Pool operation on the Input given "
        "provided Kernel, Stride, and Pad. The input and output are in NCHW "
        "format");

BB.newNode("OCLPoolMax")
    .addInput("Input")
    .addMember(MemberType::SizeT, "Kernel")
    .addMember(MemberType::SizeT, "Stride")
    .addMember(MemberType::SizeT, "Pad")
    .addResultFromCtorArg()
    //.addGradient()
    .setDocstring(
        "This is an OpenCL-specific Max Pool operation on the Input given "
        "provided "
        "Kernel, Stride, and Pad. The input and output are in NCHW format");

BB.includeBackendSpecificVerification("OpenCLSpecificNodesVerification.h");

#endif // GLOW_WITH_CPU
