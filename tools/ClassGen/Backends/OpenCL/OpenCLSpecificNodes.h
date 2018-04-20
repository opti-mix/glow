#ifdef GLOW_WITH_CPU

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

BB.includeBackendSpecificVerification("OpenCLSpecificNodesVerification.h");

#endif // GLOW_WITH_CPU
