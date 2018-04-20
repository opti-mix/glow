#ifdef GLOW_WITH_CPU

void OCLConvolutionNode::verify() const {
  ShapeNCHW idim(getInput().getType()->dims());
  ShapeNCHW odim(getResult().getType()->dims());
  auto outSz = calculateConvOutputDims(idim.h, idim.w, getKernel(), getStride(),
                                       getPad());
  ShapeNCHW exp(idim.n, getBias().dims()[0], outSz.first, outSz.second);
  (void)exp;
  assert(exp == odim && "Invalid output dimensions");
}

#endif // GLOW_WITH_CPU
