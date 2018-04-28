#ifdef GLOW_WITH_OPENCL

BB.newBackendSpecificInstr("OCLConvolution")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::SizeT, "Kernel")
    .addMember(MemberType::SizeT, "Stride")
    .addMember(MemberType::SizeT, "Pad")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Filter", "Bias"});

BB.newBackendSpecificInstr("OCLPoolAvg")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::SizeT, "Kernel")
    .addMember(MemberType::SizeT, "Stride")
    .addMember(MemberType::SizeT, "Pad")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"})
    .addGradientInstr({"Dest"}, {"Dest", "Src"});

BB.newBackendSpecificInstr("OCLPoolMax")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::SizeT, "Kernel")
    .addMember(MemberType::SizeT, "Stride")
    .addMember(MemberType::SizeT, "Pad")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.includeBackendSpecificVerification("OpenCLSpecificInstrsVerification.h");

#endif // GLOW_WITH_CPU
