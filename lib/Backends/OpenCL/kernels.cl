
static const char* SHADER_CODE = R"(

/// This type is always 32 bits.
typedef unsigned cl_uint32_t;
/// This type is always 64 bits.
typedef unsigned long cl_uint64_t;

// The types of elements should be always matching the definitions of
// ShapeNHWC in Type.h
typedef struct {
  cl_uint64_t n; // Number of samples
  cl_uint64_t h; // Height
  cl_uint64_t w; // Width
  cl_uint64_t c; // Number of channels
} ShapeNHWC;

typedef struct {
  cl_uint64_t n; // Number of samples
  cl_uint64_t c; // Number of channels
  cl_uint64_t h; // Height
  cl_uint64_t w; // Width
} ShapeNCHW;

/// \returns the index of the element at x,y,z,w.
inline size_t getNHWC(ShapeNHWC s, cl_uint32_t x, cl_uint32_t y, cl_uint32_t z,
               cl_uint32_t w) {
  return (x * s.c * s.w * s.h) + (y * s.c * s.w) + (z * s.c) + w;
}

// w <- c, x <- n, h <- y, w <- z 
inline size_t getNHWC2(ShapeNHWC s, cl_uint32_t n, cl_uint32_t h, cl_uint32_t w,
               cl_uint32_t c) {
  return (n * s.c * s.w * s.h) + (c * s.h * s.w) + (w * s.w) + h;
}

/// \returns the index of the element at x,y,z,w.
inline size_t getNCHW(ShapeNCHW s, cl_uint32_t n, cl_uint32_t c, cl_uint32_t h,
               cl_uint32_t w) {
  return (n * s.c * s.w * s.h) + (c * s.h * s.w) + (h * s.w) + w;
}

__kernel void batchedreduceaddK(__global float *dest, __global float *batch,
                                cl_uint32_t numSlice, cl_uint32_t sliceSize) {
  size_t s = get_global_id(0);
  dest[s] = 0;
  for (size_t n = 0; n < numSlice; n++) {
    dest[s] += batch[n * sliceSize + s];
  }
}

__kernel void batchedreduceaddW(__global void *mem, cl_uint32_t dest,
                                cl_uint32_t batch, size_t numSlice,
                                size_t sliceSize) {
  batchedreduceaddK(&mem[dest], &mem[batch], numSlice, sliceSize);
}

__kernel void batchedaddK(__global float *dest, __global float *batch,
                          __global float *slice, cl_uint32_t numSlice,
                          cl_uint32_t sliceSize) {
  size_t s = get_global_id(0);
  for (size_t n = 0; n < numSlice; n++) {
    dest[n * sliceSize + s] = batch[n * sliceSize + s] + slice[s];
  }
}

__kernel void batchedaddW(__global void *mem, cl_uint32_t dest,
                          cl_uint32_t batch, cl_uint32_t slice,
                          cl_uint32_t numSlice, cl_uint32_t sliceSize) {
  batchedaddK(&mem[dest], &mem[batch], &mem[slice], numSlice, sliceSize);
}

__kernel void matmulK(__global float *dest, __global float *lhs,
                      __global float *rhs, ShapeNHWC ddim, ShapeNHWC ldim,
                      ShapeNHWC rdim) {
  // For each X in the destination matrix.
  size_t x = get_global_id(0);
  // For each Y in the destination matrix.
  size_t y = get_global_id(1);

  // Perform DOT on the row an column.
  float sum = 0;
  for (size_t i = 0; i < ldim.h; i++) {
    sum += lhs[getNHWC(ldim, x, i, 0, 0)] * rhs[getNHWC(rdim, i, y, 0, 0)];
  }

  dest[getNHWC(ddim, x, y, 0, 0)] = sum;
}

__kernel void matmulW(__global void *mem, cl_uint32_t dest, cl_uint32_t lhs,
                      cl_uint32_t rhs, ShapeNHWC ddim, ShapeNHWC ldim,
                      ShapeNHWC rdim) {
  matmulK(&mem[dest], &mem[lhs], &mem[rhs], ddim, ldim, rdim);
}

__kernel void splatK(__global float *dest, float val) {
  size_t i = get_global_id(0);
  dest[i] = val;
}

__kernel void splatW(__global void *mem, cl_uint32_t dest, float val) {
  splatK(&mem[dest], val);
}

__kernel void splatK16(__global float *dest, float val) {
  size_t i = get_global_id(0);
  float16 VAL = (float16)val;
  vstore16(VAL, i, dest);
}

__kernel void splatW16(__global void *mem, cl_uint32_t dest, float val) {
  splatK16(&mem[dest], val);
}

__kernel void sigmoidK(__global float *dest, __global float *src) {
  size_t i = get_global_id(0);
  dest[i] = 1 / (1 + exp(-src[i]));
}

__kernel void sigmoidW(__global void *mem, cl_uint32_t dest, cl_uint32_t src) {
  sigmoidK(&mem[dest], &mem[src]);
}

__kernel void tanhK(__global float *dest, __global float *src) {
  size_t i = get_global_id(0);
  float val = src[i];
  float exp_val = exp(val);
  float exp_neg_val = exp(-val);
  dest[i] = (exp_val - exp_neg_val) / (exp_val + exp_neg_val);
}

__kernel void tanhW(__global void *mem, cl_uint32_t dest, cl_uint32_t src) {
  tanhK(&mem[dest], &mem[src]);
}

#define DP_SLICE_SIZE 16

#define DEFINE_GPU_DATA_PARALLEL_KERNEL(name, type, body)                      \
  __kernel void name##K##16(__global type * dest, __global type * lhs,         \
                            __global type * rhs) {                             \
    size_t j = get_global_id(0);                                               \
    float16 LHS = vload16(j, lhs);                                             \
    float16 RHS = vload16(j, rhs);                                             \
    float16 VAL = body;                                                        \
    vstore16(VAL, j, dest);                                                    \
  }                                                                            \
  __kernel void name##W##16(__global void *mem, cl_uint32_t dest,              \
                            cl_uint32_t LHS, cl_uint32_t RHS) {                \
    name##K##16(&mem[dest], &mem[LHS], &mem[RHS]);                             \
  }                                                                            \
  __kernel void name##K(__global type *dest, __global type *lhs,               \
                        __global type *rhs) {                                  \
    size_t i = get_global_id(0);                                               \
    type RHS = rhs[i];                                                         \
    type LHS = lhs[i];                                                         \
    dest[i] = body;                                                            \
  }                                                                            \
  __kernel void name##W(__global void *mem, cl_uint32_t dest, cl_uint32_t LHS, \
                        cl_uint32_t RHS) {                                     \
    name##K(&mem[dest], &mem[LHS], &mem[RHS]);                                 \
  }

DEFINE_GPU_DATA_PARALLEL_KERNEL(elementadd, float, LHS + RHS)
DEFINE_GPU_DATA_PARALLEL_KERNEL(elementsub, float, LHS - RHS)
DEFINE_GPU_DATA_PARALLEL_KERNEL(elementmul, float, LHS * RHS)
DEFINE_GPU_DATA_PARALLEL_KERNEL(elementdiv, float, LHS / RHS)
DEFINE_GPU_DATA_PARALLEL_KERNEL(elementmax, float, max(LHS, RHS))
DEFINE_GPU_DATA_PARALLEL_KERNEL(elementmin, float, min(LHS, RHS))
//DEFINE_GPU_DATA_PARALLEL_KERNEL(elementcmplte, float, LHS <= RHS)

#if 0
__kernel void elementmaxK16(__global float *dest, __global float *lhs,
                          __global float *rhs) {
  size_t j = get_global_id(0) * 16;
  for (size_t i = j; i < j + 16; i++) {
    float LHS = lhs[i];
    float RHS = rhs[i];
    dest[i] = max(LHS, RHS);
  }
}

__kernel void elementmaxW16(__global void *mem, cl_uint32_t dest, cl_uint32_t LHS,
                          cl_uint32_t RHS) {
  elementmaxK16(&mem[dest], &mem[LHS], &mem[RHS]);
}

__kernel void elementmaxK(__global float *dest, __global float *LHS,
                          __global float *RHS) {
  size_t i = get_global_id(0);
  dest[i] = max(LHS[i], RHS[i]);
}

__kernel void elementmaxW(__global void *mem, cl_uint32_t dest, cl_uint32_t LHS,
                          cl_uint32_t RHS) {
  elementmaxK(&mem[dest], &mem[LHS], &mem[RHS]);
}

__kernel void elementmaxK(__global float *dest, __global float *LHS,
                          __global float *RHS) {
  size_t i = get_global_id(0);
  dest[i] = max(LHS[i], RHS[i]);
}

__kernel void elementmaxW(__global void *mem, cl_uint32_t dest, cl_uint32_t LHS,
                          cl_uint32_t RHS) {
  elementmaxK(&mem[dest], &mem[LHS], &mem[RHS]);
}

__kernel void elementminK(__global float *dest, __global float *LHS,
                          __global float *RHS) {
  size_t i = get_global_id(0);
  dest[i] = min(LHS[i], RHS[i]);
}

__kernel void elementminW(__global void *mem, cl_uint32_t dest, cl_uint32_t LHS,
                          cl_uint32_t RHS) {
  elementminK(&mem[dest], &mem[LHS], &mem[RHS]);
}


#endif

__kernel void elementcmplteK(__global float *dest, __global float *LHS,
                             __global float *RHS) {
  size_t i = get_global_id(0);
  dest[i] = LHS[i] <= RHS[i];
}

__kernel void elementcmplteW(__global void *mem, cl_uint32_t dest,
                             cl_uint32_t LHS, cl_uint32_t RHS) {
  elementcmplteK(&mem[dest], &mem[LHS], &mem[RHS]);
}

__kernel void softmaxK(__global float *dest, __global float *src,
                       __global float *e_cache, cl_uint32_t sliceSize) {
  size_t i = get_global_id(0);
  float max_ = src[i * sliceSize];
  for (size_t j = 0; j < sliceSize; j++) {
    max_ = max(max_, src[i * sliceSize + j]);
  }
  float sum = 0;
  for (size_t j = 0; j < sliceSize; j++) {
    float e = exp(src[i * sliceSize + j] - max_);
    sum += e;
    dest[i * sliceSize + j] = e;
  }
  for (size_t j = 0; j < sliceSize; j++) {
    dest[i * sliceSize + j] /= sum;
    if (e_cache)
      e_cache[i * sliceSize + j] = dest[i * sliceSize + j];
  }
}

__kernel void softmaxW(__global void *mem, cl_uint32_t dest, cl_uint32_t src,
                       cl_uint32_t sliceSize) {
  softmaxK(&mem[dest], &mem[src], (__global float *)0, sliceSize);
}

__kernel void softmaxgradK(__global float *outW, __global float *inG,
                           __global float *selectedW,
                           cl_uint32_t sliceSize) {
  size_t i = get_global_id(0);
  for (size_t j = 0; j < sliceSize; j++) {
      float delta = (selectedW[i * sliceSize + j] == i);
      inG[i*sliceSize + j] = outW[i*sliceSize + j] - delta;
  }
}

__kernel void softmaxgradW(__global void *mem,
                       cl_uint32_t origDest, cl_uint32_t origSrc,
                       cl_uint32_t selected,
                       cl_uint32_t srcGrad,
                       cl_uint32_t sliceSize) {
  softmaxgradK(&mem[srcGrad], &mem[origSrc], &mem[selected], sliceSize);
}

__kernel void convolutionK(__global float *dest, __global float *src,
                           __global float *filter, __global float *bias,
                           cl_uint32_t filterSize, cl_uint32_t stride,
                           cl_uint32_t pad, ShapeNHWC odim, ShapeNHWC idim,
                           ShapeNHWC filterDim) {
  size_t ax = get_global_id(0);
  size_t ay = get_global_id(1);
  size_t d = get_global_id(2);

  typedef int ssize_t;
  // For each convolution 'jump' in the input tensor:
  ssize_t x = -(ssize_t)pad + ax * stride;
  ssize_t y = -(ssize_t)pad + ay * stride;

  // For each input in the batch:
  for (size_t n = 0; n < idim.n; n++) {

    // For each element in the convolution-filter:
    float sum = 0;
    for (size_t fx = 0; fx < filterSize; fx++) {
      for (size_t fy = 0; fy < filterSize; fy++) {
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (ssize_t)idim.h ||
            oy >= (ssize_t)idim.w) {
          continue;
        }

        for (size_t fd = 0; fd < idim.c; fd++) {
          sum += filter[getNHWC(filterDim, d, fx, fy, fd)] *
                 src[getNHWC(idim, n, (size_t)ox, (size_t)oy, fd)];
        }
      }
    }

    sum += bias[d];
    dest[getNHWC(odim, n, ax, ay, d)] = sum;
  } // N
}

/// Optimized version of the convolution kernel that can process
/// 16 floats (channels) at once.
__kernel void convolutionK_float16(__global float *dest,
                           const __global float *src,
                           const __global float *filter,
                           const __global float *bias,
                           cl_uint32_t filterSize, cl_uint32_t stride,
                           cl_uint32_t pad, ShapeNHWC odim, ShapeNHWC idim,
                           ShapeNHWC filterDim) {
  size_t ax = get_global_id(0);
  size_t ay = get_global_id(1);
  size_t d = get_global_id(2);

#if 0
  __local float16 filter[4];
  if (ax == 0 && ay == 0 && d == 0) {
     filter[0] = vload16(0, &filter1[0]);
     filter[1] = vload16(1, &filter1[0]);
     filter[2] = vload16(2, &filter1[0]);
     filter[3] = vload16(3, &filter1[0]);
  }
#endif

  typedef int ssize_t;
  // For each convolution 'jump' in the input tensor:
  ssize_t x = -(ssize_t)pad + ax * stride;
  ssize_t y = -(ssize_t)pad + ay * stride;

  // For each input in the batch:
  for (size_t n = 0; n < idim.n; n++) {

    // For each element in the convolution-filter:
    float16 sum = 0;
    
    for (size_t fx = 0; fx < filterSize; fx++) {
      for (size_t fy = 0; fy < filterSize; fy++) {
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (ssize_t)idim.h ||
            oy >= (ssize_t)idim.w) {
          continue;
        }

        size_t filterBase = getNHWC(filterDim, d, fx, fy, 0);
        size_t srcBase = getNHWC(idim, n, (size_t)ox, (size_t)oy, 0);

        for (size_t fd = 0; fd < idim.c/16; fd++) {
          // Use vload instead of casts, because there is no guarantee that
          // filter or src are aligned.
          sum += vload16(fd, &filter[filterBase]) * vload16(fd, &src[srcBase]); 
          //sum += filter[0] * vload16(fd, &src[srcBase]); 
        }
      }
    }

    dest[getNHWC(odim, n, ax, ay, d)] = bias[d] 
                                      + dot(sum.hi.hi, 1.0) + dot(sum.hi.lo, 1.0)
                                      + dot(sum.lo.hi, 1.0) + dot(sum.lo.lo, 1.0);
  } // N
}

/// Optimized version of the convolution kernel that can process
/// 8 floats (channels) at once.
__kernel void convolutionK_float8(__global float *dest,
                           const __global float *src,
                           const __global float *filter,
                           const __global float *bias,
                           cl_uint32_t filterSize, cl_uint32_t stride,
                           cl_uint32_t pad, ShapeNHWC odim, ShapeNHWC idim,
                           ShapeNHWC filterDim) {
  size_t ax = get_global_id(0);
  size_t ay = get_global_id(1);
  size_t d = get_global_id(2);

  typedef int ssize_t;
  // For each convolution 'jump' in the input tensor:
  ssize_t x = -(ssize_t)pad + ax * stride;
  ssize_t y = -(ssize_t)pad + ay * stride;

  // For each input in the batch:
  for (size_t n = 0; n < idim.n; n++) {

    // For each element in the convolution-filter:
    float8 sum = 0;
    
    for (size_t fx = 0; fx < filterSize; fx++) {
      for (size_t fy = 0; fy < filterSize; fy++) {
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (ssize_t)idim.h ||
            oy >= (ssize_t)idim.w) {
          continue;
        }

        size_t filterBase = getNHWC(filterDim, d, fx, fy, 0);
        size_t srcBase = getNHWC(idim, n, (size_t)ox, (size_t)oy, 0);

        for (size_t fd = 0; fd < idim.c/8; fd++) {
          // Use vload instead of casts, because there is no guarantee that
          // filter or src are aligned.
          sum += vload8(fd, &filter[filterBase]) * vload8(fd, &src[srcBase]); 
        }
      }
    }

    dest[getNHWC(odim, n, ax, ay, d)] = bias[d] 
                                      + dot(sum.hi, 1.0) + dot(sum.lo, 1.0);
  } // N
}

__kernel void convolutionW(__global void *mem, cl_uint32_t dest,
                           cl_uint32_t src, cl_uint32_t filter,
                           cl_uint32_t bias, cl_uint32_t filterSize,
                           cl_uint32_t stride, cl_uint32_t pad, ShapeNHWC odim,
                           ShapeNHWC idim, ShapeNHWC filterDim) {
  
  if (filterDim.c % 16 == 0) {
    convolutionK_float16(&mem[dest], &mem[src], &mem[filter], &mem[bias],
                         filterSize, stride, pad, odim, idim, filterDim);
    return;
  }

  if (filterDim.c % 8 == 0) {
    convolutionK_float8(&mem[dest], &mem[src], &mem[filter], &mem[bias],
                        filterSize, stride, pad, odim, idim, filterDim);
    return;
  }
  
  convolutionK(&mem[dest], &mem[src], &mem[filter], &mem[bias], filterSize,
               stride, pad, odim, idim, filterDim);
}

/// Perform a convolution on the inputs in NCHW format.
__kernel void oclconvolutionK(__global float *dest, __global float *src,
                           __global float *filter, __global float *bias,
                           cl_uint32_t filterSize, cl_uint32_t stride,
                           cl_uint32_t pad, ShapeNCHW odim, ShapeNCHW idim,
                           ShapeNCHW filterDim) {
  size_t ax = get_global_id(0);
  size_t ay = get_global_id(1);
  size_t d = get_global_id(2);

  typedef int ssize_t;
  // For each convolution 'jump' in the input tensor:
  ssize_t x = -(ssize_t)pad + ax * stride;
  ssize_t y = -(ssize_t)pad + ay * stride;

  // For each input in the batch:
  for (size_t n = 0; n < idim.n; n++) {

    // For each element in the convolution-filter:
    float sum = 0;
    for (size_t fx = 0; fx < filterSize; fx++) {
      for (size_t fy = 0; fy < filterSize; fy++) {
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (ssize_t)idim.h ||
            oy >= (ssize_t)idim.w) {
          continue;
        }

        for (size_t fd = 0; fd < idim.c; fd++) {
          sum += filter[getNCHW(filterDim, d, fd, fx, fy)] *
                 src[getNCHW(idim, n, fd, (size_t)ox, (size_t)oy)];
        }
      }
    }

    sum += bias[d];
    dest[getNCHW(odim, n, d, ax, ay)] = sum;
  } // N
}

__kernel void oclconvolutionW(__global void *mem, cl_uint32_t dest,
                           cl_uint32_t src, cl_uint32_t filter,
                           cl_uint32_t bias, cl_uint32_t filterSize,
                           cl_uint32_t stride, cl_uint32_t pad, ShapeNCHW odim,
                           ShapeNCHW idim, ShapeNCHW filterDim) {
  oclconvolutionK(&mem[dest], &mem[src], &mem[filter], &mem[bias], filterSize,
               stride, pad, odim, idim, filterDim);
}

__kernel void poolmaxK(__global float *dest, __global float *src,
                       cl_uint32_t filterSize, cl_uint32_t stride,
                       cl_uint32_t pad, ShapeNHWC odim, ShapeNHWC idim) {
  size_t ax = get_global_id(0);
  size_t ay = get_global_id(1);
  size_t d = get_global_id(2);

  typedef int ssize_t;
  // For each convolution 'jump' in the input tensor:
  ssize_t x = -(ssize_t)pad + ax * stride;
  ssize_t y = -(ssize_t)pad + ay * stride;

  // For each input in the batch:
  for (size_t n = 0; n < idim.n; n++) {
    float maxVal = 0;
    bool first = true;

    // For each element in the convolution-filter:
    for (size_t fx = 0; fx < filterSize; fx++) {
      for (size_t fy = 0; fy < filterSize; fy++) {
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (ssize_t)idim.h ||
            oy >= (ssize_t)idim.w) {
          continue;
        }

        float val = src[getNHWC(idim, n, (size_t)ox, (size_t)oy, d)];

        if (first || (val >= maxVal)) {
          first = false;
          maxVal = val;
        }
      }
    }
    dest[getNHWC(odim, n, ax, ay, d)] = maxVal;
  } // N
}

__kernel void poolmaxW(__global void *mem, cl_uint32_t dest, cl_uint32_t src,
                       cl_uint32_t filterSize, cl_uint32_t stride,
                       cl_uint32_t pad, ShapeNHWC odim, ShapeNHWC idim) {
  poolmaxK(&mem[dest], &mem[src], filterSize, stride, pad, odim, idim);
}

__kernel void poolmaxwithxyK(__global float *dest, __global float *src,
                             __global float *srcXY, cl_uint32_t filterSize,
                             cl_uint32_t stride, cl_uint32_t pad,
                             ShapeNHWC odim, ShapeNHWC idim) {
  size_t ax = get_global_id(0);
  size_t ay = get_global_id(1);
  size_t d = get_global_id(2);

  typedef int ssize_t;
  // For each convolution 'jump' in the input tensor:
  ssize_t x = -(ssize_t)pad + ax * stride;
  ssize_t y = -(ssize_t)pad + ay * stride;

  // For each input in the batch:
  for (size_t n = 0; n < idim.n; n++) {
    float maxVal = 0;
    bool first = true;

    // For each element in the convolution-filter:
    for (size_t fx = 0; fx < filterSize; fx++) {
      for (size_t fy = 0; fy < filterSize; fy++) {
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (ssize_t)idim.h ||
            oy >= (ssize_t)idim.w) {
          continue;
        }

        float val = src[getNHWC(idim, n, (size_t)ox, (size_t)oy, d)];

        if (first || (val >= maxVal)) {
          first = false;
          maxVal = val;
        }
      }
    }
    dest[getNHWC(odim, n, ax, ay, d)] = maxVal;
  } // N
}

__kernel void poolmaxwithxyW(__global void *mem, cl_uint32_t dest,
                             cl_uint32_t src, cl_uint32_t srcXY,
                             cl_uint32_t filterSize, cl_uint32_t stride,
                             cl_uint32_t pad, ShapeNHWC odim, ShapeNHWC idim) {
  poolmaxwithxyK(&mem[dest], &mem[src], &mem[srcXY], filterSize, stride, pad,
                 odim, idim);
}

__kernel void poolavgK(__global float *dest, __global float *src,
                       cl_uint32_t filterSize, cl_uint32_t stride,
                       cl_uint32_t pad, ShapeNHWC odim, ShapeNHWC idim) {
  size_t ax = get_global_id(0);
  size_t ay = get_global_id(1);
  size_t d = get_global_id(2);

  typedef int ssize_t;
  // For each convolution 'jump' in the input tensor:
  ssize_t x = -(ssize_t)pad + ax * stride;
  ssize_t y = -(ssize_t)pad + ay * stride;

  float filterArea = filterSize * filterSize;

  // For each input in the batch:
  for (size_t n = 0; n < idim.n; n++) {
    float sumVal = 0;
    // For each element in the convolution-filter:
    for (size_t fx = 0; fx < filterSize; fx++) {
      for (size_t fy = 0; fy < filterSize; fy++) {
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (ssize_t)idim.h ||
            oy >= (ssize_t)idim.w) {
          continue;
        }

        sumVal += src[getNHWC(idim, n, (size_t)ox, (size_t)oy, d)];
      }
    }
    dest[getNHWC(odim, n, ax, ay, d)] = sumVal / filterArea;
  } // N
}

__kernel void poolavgW(__global void *mem, cl_uint32_t dest, cl_uint32_t src,
                       cl_uint32_t filterSize, cl_uint32_t stride,
                       cl_uint32_t pad, ShapeNHWC odim, ShapeNHWC idim) {
  poolavgK(&mem[dest], &mem[src], filterSize, stride, pad, odim, idim);
}

__kernel void transposeK1(__global float *dest, __global float *src,
                         ShapeNHWC odim, ShapeNHWC idim, ShapeNHWC shuffle) {
  size_t d0 = get_global_id(0);
  size_t res[4];
  res[0] = d0;
  for (size_t d1 = 0; d1 < idim.h; d1++) {
    res[1] = d1;
    for (size_t d2 = 0; d2 < idim.w; d2++) {
      res[2] = d2;
      for (size_t d3 = 0; d3 < idim.c; d3++) {
        res[3] = d3;
        size_t dstIdx = getNHWC(odim, res[shuffle.n], res[shuffle.h],
                                res[shuffle.w], res[shuffle.c]);
        size_t srcIdx = getNHWC(idim, d0, d1, d2, d3);
        dest[dstIdx] = src[srcIdx];
      }
    }
  }
}

__kernel void transposeK(__global float *dest, __global float *src,
                         ShapeNHWC odim, ShapeNHWC idim, ShapeNHWC shuffleMask) {
  size_t d0 = get_global_id(0);
  uint4 res;
  res.s0 = d0;
  uint4 mask;

  mask.s0 = shuffleMask.n;
  mask.s1 = shuffleMask.h;
  mask.s2 = shuffleMask.w;
  mask.s3 = shuffleMask.c;

  for (size_t d1 = 0; d1 < idim.h; d1++) {
    res.s1 = d1;
    for (size_t d2 = 0; d2 < idim.w; d2++) {
      res.s2 = d2;
      for (size_t d3 = 0; d3 < idim.c; d3++) {
        res.s3 = d3;
        uint4 result = shuffle(res, mask);
        size_t dstIdx = getNHWC(odim, result[0], result[1],
                                result[2], result[3]);
        size_t srcIdx = getNHWC(idim, d0, d1, d2, d3);
        dest[dstIdx] = src[srcIdx];
      }
    }
  }
}

__kernel void transposeW(__global void *mem, cl_uint32_t dest, cl_uint32_t src,
                         ShapeNHWC odim, ShapeNHWC idim, ShapeNHWC shuffle) {
  transposeK(&mem[dest], &mem[src], odim, idim, shuffle);
}

__kernel void inserttensorK(__global float *dest, __global float *src,
                            ShapeNHWC odim, ShapeNHWC idim, ShapeNHWC offset) {
  size_t d0 = get_global_id(0);
  for (size_t d1 = 0; d1 < idim.h; d1++) {
    for (size_t d2 = 0; d2 < idim.w; d2++) {
      for (size_t d3 = 0; d3 < idim.c; d3++) {
        size_t r0 = d0 + offset.n;
        size_t r1 = d1 + offset.h;
        size_t r2 = d2 + offset.w;
        size_t r3 = d3 + offset.c;
        size_t srcIdx = getNHWC(idim, d0, d1, d2, d3);
        size_t destIdx = getNHWC(odim, r0, r1, r2, r3);
        dest[destIdx] = src[srcIdx];
      }
    }
  }
}

__kernel void inserttensorW(__global void *mem, cl_uint32_t dest,
                            cl_uint32_t src, ShapeNHWC odim, ShapeNHWC idim,
                            ShapeNHWC offset) {
  inserttensorK(&mem[dest], &mem[src], odim, idim, offset);
}

// CL: grid stride looping
#define CL_KERNEL_LOOP(i, n)                        \
  for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \
      i < (n);                                       \
      i += get_local_size(0) * get_num_groups(0))

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
__kernel void im2col_kernel(const int n, const global float* im_data, int im_offset,
    const int height, const int width, const int ksize_h, const int ksize_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col,
    global float* col_data, int col_offset) {
  global const float *data_im = im_data + im_offset;
  global float *data_col = col_data + col_offset;

  CL_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize_h * ksize_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize_h; ++i) {
      for (int j = 0; j < ksize_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
          data_im[i * width + j] : 0;
        data_col += height_col * width_col;
      }
    }
  }
}

__kernel void col2im_kernel(const int n, global const float* col_data, int col_offset,
    const int height, const int width, const int channels, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    global float* im_data, int im_offset) {
  global float *data_im = im_data + im_offset;
  global const float *data_col = col_data + col_offset;

  CL_KERNEL_LOOP(index, n) {
    float val = 0;
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);
    /*
       for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
       for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
    // the col location: [c * width * height + h_out, w_out]
    int c_col = c * patch_h * patch_w + (h - h_col * stride_h) * ksize + (w - w_col * stride_w);
    val += data_col[(c_col * height_col + h_col) * width_col + w_col];
    }
    }
     */
    // equivalent implementation
    int offset = (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
    int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
    int coeff_w_col = (1 - stride_w * height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

__kernel void testbuffers(global const float *mem, cl_uint32_t in,
                          global const float *inBuf, ShapeNCHW idim) {
  global const float *inputBuf = &mem[in];
  if (inputBuf != inBuf) {
    printf("InputBuf != inBuf: mem=%lu in=%lu inputBuf=&mem[in]=%lu inbuf=%lu\n", mem, in, inputBuf, inBuf);
  }
  int size = idim.n * idim.c * idim.w * idim.h;
  int idx;
  int count = 0;
  for (idx = 0; idx < size && count < 100; ++idx) {
    if (inputBuf[idx] != inBuf[idx]) {
      printf("Elements are different at index %lu: %f (at %lu) != %f (at %lu)\n", idx, inputBuf[idx], &inputBuf[idx],
       inBuf[idx], &inBuf[idx]);
      ++count;
    }
  }
 }

)";
