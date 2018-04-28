#ifdef GLOW_WITH_OPENCL

void OCLConvolutionNode::verify() const {
  ShapeNCHW idim(getInput().getType()->dims());
  ShapeNCHW odim(getResult().getType()->dims());
  auto outSz = calculateConvOutputDims(idim.h, idim.w, getKernel(), getStride(),
                                       getPad());
  ShapeNCHW exp(idim.n, getBias().dims()[0], outSz.first, outSz.second);
  (void)exp;
  assert(exp == odim && "Invalid output dimensions");
}

void OCLPoolAvgNode::verify() const {}

void OCLPoolMaxNode::verify() const {}
#endif // GLOW_WITH_OPENCL
