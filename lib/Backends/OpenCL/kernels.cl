
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

#if defined(cl_khr_int32_base_atomics)
#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable
#define ATOMICS_32_AVAILABLE
#endif
#if defined(cl_khr_global_int32_base_atomics)
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define ATOMICS_32_AVAILABLE
#endif

#ifdef ATOMICS_32_AVAILABLE
#define Dtype float
inline void atomicAdd(volatile __global Dtype* source, const Dtype operand) {
union {
unsigned int intVal;
Dtype floatVal;
} next, expected, current;
current.floatVal = *source;
do {
expected.floatVal = current.floatVal;
next.floatVal = expected.floatVal + operand;
current.intVal = atomic_cmpxchg((volatile __global unsigned int *)source, expected.intVal, next.intVal);
} while (current.intVal != expected.intVal);
}
inline void atomicSub(volatile __global Dtype* source, const Dtype operand) {
union {
unsigned int intVal;
Dtype floatVal;
} next, expected, current;
current.floatVal = *source;
do {
expected.floatVal = current.floatVal;
next.floatVal = expected.floatVal - operand;
current.intVal = atomic_cmpxchg((volatile __global unsigned int *)source, expected.intVal, next.intVal);
} while (current.intVal != expected.intVal);
}
inline void atomicMul(volatile __global Dtype* source, const Dtype operand) {
union {
unsigned int intVal;
Dtype floatVal;
} next, expected, current;
current.floatVal = *source;
do {
expected.floatVal = current.floatVal;
next.floatVal = expected.floatVal * operand;
current.intVal = atomic_cmpxchg((volatile __global unsigned int *)source, expected.intVal, next.intVal);
} while (current.intVal != expected.intVal);
}
inline void atomicDiv(volatile __global Dtype* source, const Dtype operand) {
union {
unsigned int intVal;
Dtype floatVal;
} next, expected, current;
current.floatVal = *source;
do {
expected.floatVal = current.floatVal;
next.floatVal = expected.floatVal / operand;
current.intVal = atomic_cmpxchg((volatile __global unsigned int *)source, expected.intVal, next.intVal);
} while (current.intVal != expected.intVal);
}
#undef Dtype
#endif

/// \returns the index of the element at n, h, w, c.
inline size_t getNHWC(ShapeNHWC s, cl_uint32_t n, cl_uint32_t h, cl_uint32_t w,
               cl_uint32_t c) {
  return (n * s.c * s.w * s.h) + (h * s.c * s.w) + (w * s.c) + c;
}

/// \returns the index of the element at n, c, h, w.
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

__kernel void splatK8(__global float *dest, float val) {
  size_t i = get_global_id(0);
  float8 VAL = (float8)val;
  vstore8(VAL, i, dest);
}

__kernel void splatW8(__global void *mem, cl_uint32_t dest, float val) {
  splatK8(&mem[dest], val);
}

__kernel void splat_uK(__global cl_uint64_t *dest, cl_uint64_t val) {
  size_t i = get_global_id(0);
  dest[i] = val;
}

__kernel void splat_uW(__global void *mem, cl_uint32_t dest, float val) {
  splat_uK(&mem[dest], (cl_uint64_t)val);
}

#define DEFINE_GPU_TERNARY_DATA_PARALLEL_KERNEL(name, type, body)               \
  __kernel void name##K##16(__global type * dest, __global type * cond,  __global type * lhs,         \
                            __global type * rhs) {                             \
    typedef float8 vtype;\
    size_t i = get_global_id(0);                                               \
    {                                                                          \
      float8 COND = vload8(i * 2, cond);                                         \
      float8 LHS = vload8(i * 2, lhs);                                         \
      float8 RHS = vload8(i * 2, rhs);                                         \
      float8 VAL = body;                                                       \
      vstore8(VAL, i * 2, dest);                                               \
    }                                                                          \
    {                                                                          \
      float8 COND = vload8(i * 2, cond);                                         \
      float8 LHS = vload8(i * 2 + 1, lhs);                                     \
      float8 RHS = vload8(i * 2 + 1, rhs);                                     \
      float8 VAL = body;                                                       \
      vstore8(VAL, i * 2 + 1, dest);                                           \
    }                                                                          \
  }                                                                            \
  __kernel void name##W##16(__global void *mem, cl_uint32_t dest,              \
                            cl_uint32_t cond, cl_uint32_t lhs, cl_uint32_t rhs) {                \
    name##K##16(&mem[dest], &mem[cond], &mem[lhs], &mem[rhs]);                             \
  }                                                                            \
  __kernel void name##K##8(__global type * dest, __global type *cond, __global type * lhs,          \
                           __global type * rhs) {                              \
    typedef float8 vtype;\
    size_t i = get_global_id(0);                                               \
    float8 COND = vload8(i, cond);                                         \
    float8 LHS = vload8(i, lhs);                                               \
    float8 RHS = vload8(i, rhs);                                               \
    float8 VAL = body;                                                         \
    vstore8(VAL, i, dest);                                                     \
  }                                                                            \
  __kernel void name##W##8(__global void *mem, cl_uint32_t dest,               \
                           cl_uint32_t cond, cl_uint32_t lhs, cl_uint32_t rhs) {                 \
    name##K##8(&mem[dest], &mem[cond], &mem[lhs], &mem[rhs]);                              \
  }                                                                            \
  __kernel void name##K(__global type *dest, __global type *cond, __global type *lhs,               \
                        __global type *rhs) {                                  \
    typedef float vtype;\
    size_t i = get_global_id(0);                                               \
    type COND = cond[i];                                         \
    type RHS = rhs[i];                                                         \
    type LHS = lhs[i];                                                         \
    dest[i] = body;                                                            \
  }                                                                            \
  __kernel void name##W(__global void *mem, cl_uint32_t dest, cl_uint32_t cond, cl_uint32_t lhs, \
                        cl_uint32_t rhs) {                                     \
    name##K(&mem[dest], &mem[cond], &mem[lhs], &mem[rhs]);                                 \
  }

#define DEFINE_GPU_BINARY_DATA_PARALLEL_KERNEL(name, type, body)               \
  __kernel void name##K##16(__global type * dest, __global type * lhs,         \
                            __global type * rhs) {                             \
    typedef float8 vtype;\
    size_t i = get_global_id(0);                                               \
    {                                                                          \
      float8 LHS = vload8(i * 2, lhs);                                         \
      float8 RHS = vload8(i * 2, rhs);                                         \
      float8 VAL = body;                                                       \
      vstore8(VAL, i * 2, dest);                                               \
    }                                                                          \
    {                                                                          \
      float8 LHS = vload8(i * 2 + 1, lhs);                                     \
      float8 RHS = vload8(i * 2 + 1, rhs);                                     \
      float8 VAL = body;                                                       \
      vstore8(VAL, i * 2 + 1, dest);                                           \
    }                                                                          \
  }                                                                            \
  __kernel void name##W##16(__global void *mem, cl_uint32_t dest,              \
                            cl_uint32_t lhs, cl_uint32_t rhs) {                \
    name##K##16(&mem[dest], &mem[lhs], &mem[rhs]);                             \
  }                                                                            \
  __kernel void name##K##8(__global type * dest, __global type * lhs,          \
                           __global type * rhs) {                              \
    typedef float8 vtype;\
    size_t i = get_global_id(0);                                               \
    float8 LHS = vload8(i, lhs);                                               \
    float8 RHS = vload8(i, rhs);                                               \
    float8 VAL = body;                                                         \
    vstore8(VAL, i, dest);                                                     \
  }                                                                            \
  __kernel void name##W##8(__global void *mem, cl_uint32_t dest,               \
                           cl_uint32_t lhs, cl_uint32_t rhs) {                 \
    name##K##8(&mem[dest], &mem[lhs], &mem[rhs]);                              \
  }                                                                            \
  __kernel void name##K(__global type *dest, __global type *lhs,               \
                        __global type *rhs) {                                  \
    typedef float vtype;\
    size_t i = get_global_id(0);                                               \
    type RHS = rhs[i];                                                         \
    type LHS = lhs[i];                                                         \
    dest[i] = body;                                                            \
  }                                                                            \
  __kernel void name##W(__global void *mem, cl_uint32_t dest, cl_uint32_t lhs, \
                        cl_uint32_t rhs) {                                     \
    name##K(&mem[dest], &mem[lhs], &mem[rhs]);                                 \
  }

#define DEFINE_GPU_UNARY_DATA_PARALLEL_KERNEL(name, type, body)                \
  __kernel void name##K##16(__global type * dest, __global type * src) {       \
    typedef float8 vtype;\
    size_t i = get_global_id(0);                                               \
    {                                                                          \
      float8 SRC = vload8(i * 2, src);                                         \
      float8 VAL = body;                                                       \
      vstore8(VAL, i * 2, dest);                                               \
    }                                                                          \
    {                                                                          \
      float8 SRC = vload8(i * 2 + 1, src);                                     \
      float8 VAL = body;                                                       \
      vstore8(VAL, i * 2 + 1, dest);                                           \
    }                                                                          \
  }                                                                            \
  __kernel void name##W##16(__global void *mem, cl_uint32_t dest,              \
                            cl_uint32_t src) {                                 \
    name##K##16(&mem[dest], &mem[src]);                                        \
  }                                                                            \
  __kernel void name##K##8(__global type * dest, __global type * src) {        \
    typedef float8 vtype;\
    size_t i = get_global_id(0);                                               \
    float8 SRC = vload8(i, src);                                               \
    float8 VAL = body;                                                         \
    vstore8(VAL, i, dest);                                                     \
  }                                                                            \
  __kernel void name##W##8(__global void *mem, cl_uint32_t dest,               \
                           cl_uint32_t src) {                                  \
    name##K##8(&mem[dest], &mem[src]);                                         \
  }                                                                            \
  __kernel void name##K(__global type *dest, __global type *src) {             \
    typedef float vtype;\
    size_t i = get_global_id(0);                                               \
    type SRC = src[i];                                                         \
    dest[i] = body;                                                            \
  }                                                                            \
  __kernel void name##W(__global void *mem, cl_uint32_t dest,                  \
                        cl_uint32_t src) {                                     \
    name##K(&mem[dest], &mem[src]);                                            \
  }

DEFINE_GPU_BINARY_DATA_PARALLEL_KERNEL(elementadd, float, LHS + RHS)
DEFINE_GPU_BINARY_DATA_PARALLEL_KERNEL(elementsub, float, LHS - RHS)
DEFINE_GPU_BINARY_DATA_PARALLEL_KERNEL(elementmul, float, LHS *RHS)
DEFINE_GPU_BINARY_DATA_PARALLEL_KERNEL(elementdiv, float, LHS / RHS)
DEFINE_GPU_BINARY_DATA_PARALLEL_KERNEL(elementmax, float, max(LHS, RHS))
DEFINE_GPU_BINARY_DATA_PARALLEL_KERNEL(elementmin, float, min(LHS, RHS))
//DEFINE_GPU_BINARY_DATA_PARALLEL_KERNEL(elementcmplte, float, convert_float(islessequal(LHS, RHS)))

DEFINE_GPU_UNARY_DATA_PARALLEL_KERNEL(tanh, float, 1 - 2 / (exp(SRC * 2) + 1))
DEFINE_GPU_UNARY_DATA_PARALLEL_KERNEL(sigmoid, float, 1 / (1 + exp(-SRC)))

DEFINE_GPU_TERNARY_DATA_PARALLEL_KERNEL(elementselect, float, (COND != (vtype)0.0) ? LHS : RHS)

#undef DEFINE_GPU_BINARY_DATA_PARALLEL_KERNEL
#undef DEFINE_GPU_UNARY_DATA_PARALLEL_KERNEL

__kernel void elementcmplteK16(__global float *dest, __global float *LHS,
                             __global float *RHS) {
  size_t i = get_global_id(0);
  vstore8(convert_float8(islessequal(vload8(i, LHS), vload8(i, RHS))), i, dest);
  vstore8(convert_float8(islessequal(vload8(i+1, LHS), vload8(i+1, RHS))), i+1, dest);
}

__kernel void elementcmplteW16(__global void *mem, cl_uint32_t dest,
                             cl_uint32_t LHS, cl_uint32_t RHS) {
  elementcmplteK16(&mem[dest], &mem[LHS], &mem[RHS]);
}

__kernel void elementcmplteK8(__global float *dest, __global float *LHS,
                             __global float *RHS) {
  size_t i = get_global_id(0);
  vstore8(convert_float8(islessequal(vload8(i, LHS), vload8(i, RHS))), i, dest);
}

__kernel void elementcmplteW8(__global void *mem, cl_uint32_t dest,
                             cl_uint32_t LHS, cl_uint32_t RHS) {
  elementcmplteK8(&mem[dest], &mem[LHS], &mem[RHS]);
}

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

__kernel void softmaxgradK(__global float *inG, __global float *outW,
                           __global cl_uint64_t *selectedW,
                           cl_uint32_t sliceSize) {
  size_t i = get_global_id(0);
  for (size_t j = 0; j < sliceSize; j++) {
      float delta = (selectedW[i] == j);
      inG[i*sliceSize + j] = outW[i*sliceSize + j] - delta;
  }
}

__kernel void softmaxgradW(__global void *mem,
                       cl_uint32_t origDest, cl_uint32_t origSrc,
                       cl_uint32_t selected,
                       cl_uint32_t srcGrad,
                       cl_uint32_t sliceSize) {
  softmaxgradK(&mem[srcGrad], &mem[origDest], &mem[selected], sliceSize);
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

// builder.CreateCall(F, {srcGradPtr, destGradPtr, srcPtr, filterGradPtr,
//                           biasGradPtr, filterPtr, destGradDims, srcDims,
//                           filterGradDims,
//                           kernel, stride, pad});

//void libjit_convolution_grad_f(float *inG, const float *outG, const float *inW,
//                               float *filterG, float *biasG,
//                               const float *filterW, const size_t *outGdims,
//                               const size_t *inWdims, const size_t *filterGdims,
//                               const size_t kernel, const size_t stride,
//                               const size_t pad)

// inW -> src
// inG -> srcGrad
// filterG -> filterGrad
// outGdims -> destGradDims

#if 1
__kernel void convolutiongradK_whole(
                           const __global float *inW,
                           const __global float *filterW,
                           const __global float *outG,
                           __global float *inG,
                           __global float *filterG,
                           __global float *biasG,
                           cl_uint32_t filterSize,
                           cl_uint32_t stride,
                           cl_uint32_t pad,
                           ShapeNHWC inWdims,
                           ShapeNHWC outGdims,
                           ShapeNHWC filterGdims) {
  typedef int ssize_t;
  // NHWC format is assumed

  // For each input in the batch:
  for (size_t n = 0; n < outGdims.n; n++) {
    for (size_t d = 0; d < outGdims.c; d++) {
      ssize_t x = -(ssize_t)pad;
      for (size_t ax = 0; ax < outGdims.h; ax++, x += stride) {
        ssize_t y = -(ssize_t)pad;
        for (size_t ay = 0; ay < outGdims.w; ay++, y += stride) {
          float grad = outG[getNHWC(outGdims, n, ax, ay, d)];

          for (size_t fx = 0; fx < filterSize; fx++) {
            for (size_t fy = 0; fy < filterSize; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              if (ox < 0 || oy < 0 || ox >= (ssize_t)inWdims.h ||
                  oy >= (ssize_t)inWdims.w) {
                continue;
              }

              for (size_t fd = 0; fd < inWdims.c; fd++) {
                //atomicAdd(&filterG[getNHWC(filterGdims, d, fx, fy, fd)], inW[getNHWC(inWdims, n, (size_t)ox, (size_t)oy, fd)] *
                //    grad);
                //atomicAdd(&inG[getNHWC(inWdims, n, (size_t)ox, (size_t)oy, fd)], filterW[getNHWC(filterGdims, d, fx, fy, fd)] * grad);
#if 1
                filterG[getNHWC(filterGdims, d, fx, fy, fd)] +=
                    inW[getNHWC(inWdims, n, (size_t)ox, (size_t)oy, fd)] *
                    grad;
                inG[getNHWC(inWdims, n, (size_t)ox, (size_t)oy, fd)] +=
                    filterW[getNHWC(filterGdims, d, fx, fy, fd)] * grad;
#endif
              }
            }
          }
          //atomicAdd(&biasG[d], grad);
          biasG[d] += grad;
        } // W
      }   // H
    }     // C
  }       // N
}

__kernel void convolutiongradK(
                           const __global float *inW,
                           const __global float *filterW,
                           const __global float *outG,
                           __global float *inG,
                           __global float *filterG,
                           __global float *biasG,
                           cl_uint32_t filterSize,
                           cl_uint32_t stride,
                           cl_uint32_t pad,
                           ShapeNHWC inWdims,
                           ShapeNHWC outGdims,
                           ShapeNHWC filterGdims) {
  // ax and ay are coordinates in the tensor outG.
  size_t ax = get_global_id(0);
  size_t ay = get_global_id(1);
  size_t d = get_global_id(2);

  typedef int ssize_t;
  // For each convolution 'jump' in the input tensor:
  ssize_t x = -(ssize_t)pad + ax * stride;
  ssize_t y = -(ssize_t)pad + ay * stride;

  // NHWC format is assumed

  // For each input in the batch:
  for (size_t n = 0; n < outGdims.n; n++) {
    //for (size_t d = 0; d < outGdims.c; d++) {
      //ssize_t x = -(ssize_t)pad;
      //for (size_t ax = 0; ax < outGdims.h; ax++, x += stride) {
        //ssize_t y = -(ssize_t)pad;
        //for (size_t ay = 0; ay < outGdims.w; ay++, y += stride) {
          float grad = outG[getNHWC(outGdims, n, ax, ay, d)];

          for (size_t fx = 0; fx < filterSize; fx++) {
            for (size_t fy = 0; fy < filterSize; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              if (ox < 0 || oy < 0 || ox >= (ssize_t)inWdims.h ||
                  oy >= (ssize_t)inWdims.w) {
                continue;
              }

              for (size_t fd = 0; fd < inWdims.c; fd++) {
                atomicAdd(&filterG[getNHWC(filterGdims, d, fx, fy, fd)], inW[getNHWC(inWdims, n, (size_t)ox, (size_t)oy, fd)] *
                    grad);
                atomicAdd(&inG[getNHWC(inWdims, n, (size_t)ox, (size_t)oy, fd)], filterW[getNHWC(filterGdims, d, fx, fy, fd)] * grad);
#if 0
                filterG[getNHWC(filterGdims, d, fx, fy, fd)] +=
                    inW[getNHWC(inWdims, n, (size_t)ox, (size_t)oy, fd)] *
                    grad;
                inG[getNHWC(inWdims, n, (size_t)ox, (size_t)oy, fd)] +=
                    filterW[getNHWC(filterGdims, d, fx, fy, fd)] * grad;
#endif
              }
            }
          }
          atomicAdd(&biasG[d], grad);
          //biasG[d] += grad;
        //} // W
      //}   // H
   // }     // C
  }       // N
}

__kernel void convolutiongradW(__global void *mem, 
                           cl_uint32_t src,
                           cl_uint32_t filter,
                           cl_uint32_t destGrad,
                           cl_uint32_t srcGrad,
                           cl_uint32_t filterGrad,
                           cl_uint32_t biasGrad,
                           cl_uint32_t filterSize,
                           cl_uint32_t stride,
                           cl_uint32_t pad,
                           ShapeNHWC srcDim,
                           ShapeNHWC destGradDim,
                           ShapeNHWC filterGradDim) {
   convolutiongradK(&mem[src], &mem[filter],
                    &mem[destGrad], &mem[srcGrad], &mem[filterGrad],
                    &mem[biasGrad],
                    filterSize, stride, pad, srcDim, destGradDim, filterGradDim); 
}
#endif

/// Perform a convolution on the inputs in NCHW format.
__kernel void oclconvolutionK_std(__global float *dest, __global float *src,
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
    for (size_t fd = 0; fd < idim.c; fd++) {
    for (size_t fx = 0; fx < filterSize; fx++) {
      for (size_t fy = 0; fy < filterSize; fy++) {
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= (ssize_t)idim.h ||
            oy >= (ssize_t)idim.w) {
          continue;
        }
          sum += filter[getNCHW(filterDim, d, fd, fx, fy)] *
                 src[getNCHW(idim, n, fd, (size_t)ox, (size_t)oy)];
        }
      }
    }

    sum += bias[d];
    dest[getNCHW(odim, n, d, ax, ay)] = sum;
  } // N
}

/// Optimized version of the convolution kernel that can process
/// 16 floats (channels) at once.
__kernel void oclconvolutionK_float16(__global float *dest,
                           const __global float *src,
                           const __global float *filter,
                           const __global float *bias,
                           cl_uint32_t filterSize, cl_uint32_t stride,
                           cl_uint32_t pad, ShapeNCHW odim, ShapeNCHW idim,
                           ShapeNCHW filterDim) {
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

        size_t filterBase = getNCHW(filterDim, d, 0, fx, fy);
        size_t srcBase = getNCHW(idim, n, 0, (size_t)ox, (size_t)oy);

        for (size_t fd = 0; fd < idim.c/16; fd++) {
          // Use vload instead of casts, because there is no guarantee that
          // filter or src are aligned.
          sum += vload16(fd, &filter[filterBase]) * vload16(fd, &src[srcBase]); 
          //sum += filter[0] * vload16(fd, &src[srcBase]); 
        }
      }
    }

    dest[getNCHW(odim, n, ax, ay, d)] = bias[d] 
                                      + dot(sum.hi.hi, 1.0) + dot(sum.hi.lo, 1.0)
                                      + dot(sum.lo.hi, 1.0) + dot(sum.lo.lo, 1.0);
  } // N
}

/// Optimized version of the convolution kernel that can process
/// 8 floats (channels) at once.
__kernel void oclconvolutionK_float8(__global float *dest,
                           const __global float *src,
                           const __global float *filter,
                           const __global float *bias,
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

        size_t filterBase = getNCHW(filterDim, d, 0, fx, fy);
        size_t srcBase = getNCHW(idim, n, 0, (size_t)ox, (size_t)oy);

        for (size_t fd = 0; fd < idim.c/8; fd++) {
          // Use vload instead of casts, because there is no guarantee that
          // filter or src are aligned.
          sum += vload8(fd, &filter[filterBase]) * vload8(fd, &src[srcBase]); 
        }
      }
    }

    dest[getNCHW(odim, n, ax, ay, d)] = bias[d] 
                                      + dot(sum.hi, 1.0) + dot(sum.lo, 1.0);
  } // N
}

__kernel void oclconvolutionW(__global void *mem, cl_uint32_t dest,
                           cl_uint32_t src, cl_uint32_t filter,
                           cl_uint32_t bias, cl_uint32_t filterSize,
                           cl_uint32_t stride, cl_uint32_t pad, ShapeNCHW odim,
                           ShapeNCHW idim, ShapeNCHW filterDim) {
#if 0
  if (filterDim.c % 16 == 0) {
    oclconvolutionK_float16(&mem[dest], &mem[src], &mem[filter], &mem[bias],
                         filterSize, stride, pad, odim, idim, filterDim);
    return;
  }

  if (filterDim.c % 8 == 0) {
    oclconvolutionK_float8(&mem[dest], &mem[src], &mem[filter], &mem[bias],
                        filterSize, stride, pad, odim, idim, filterDim);
    return;
  }
#endif
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

__kernel void oclpoolmaxK(__global float *dest, __global float *src,
                       cl_uint32_t filterSize, cl_uint32_t stride,
                       cl_uint32_t pad, ShapeNCHW odim, ShapeNCHW idim) {
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

        float val = src[getNCHW(idim, n, d, (size_t)ox, (size_t)oy)];

        if (first || (val >= maxVal)) {
          first = false;
          maxVal = val;
        }
      }
    }
    dest[getNCHW(odim, n, d, ax, ay)] = maxVal;
  } // N
}

__kernel void oclpoolmaxW(__global void *mem, cl_uint32_t dest, cl_uint32_t src,
                       cl_uint32_t filterSize, cl_uint32_t stride,
                       cl_uint32_t pad, ShapeNCHW odim, ShapeNCHW idim) {
  oclpoolmaxK(&mem[dest], &mem[src], filterSize, stride, pad, odim, idim);
}

__kernel void poolmaxwithxyK(__global float *dest, __global float *src,
                             __global cl_uint64_t *srcXY, cl_uint32_t filterSize,
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
    size_t maxX = x;
    size_t maxY = y;

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
          maxX = (size_t)ox;
          maxY = (size_t)oy;
        }
      }
    }
    dest[getNHWC(odim, n, ax, ay, d)] = maxVal;
    if (srcXY) {
       srcXY[getNHWC(odim, n, ax, ay, d)*2] = maxX;
       srcXY[getNHWC(odim, n, ax, ay, d)*2+1] = maxY;
    }
  } // N
}

__kernel void poolmaxwithxyW(__global void *mem, cl_uint32_t dest,
                             cl_uint32_t src, cl_uint32_t srcXY,
                             cl_uint32_t filterSize, cl_uint32_t stride,
                             cl_uint32_t pad, ShapeNHWC odim, ShapeNHWC idim) {
  poolmaxwithxyK(&mem[dest], &mem[src], &mem[srcXY], filterSize, stride, pad,
                 odim, idim);
}

__kernel void poolmaxwithxygradK(__global float *dest, 
                             __global cl_uint64_t *srcXY,
                             __global float *destGrad,
                             __global float *srcGrad,
                             cl_uint32_t filterSize,
                             cl_uint32_t stride, cl_uint32_t pad,
                             ShapeNHWC srcGradDim,
                             ShapeNHWC destGradDim) {
  size_t n = get_global_id(0);

  // NHWC format is assumed
  //for (size_t n = 0; n < destGradDim.n; n++) {
    for (size_t z = 0; z < destGradDim.c; z++) {
      // Clear inG
      for (size_t x = 0; x < srcGradDim.h; x++) {
        for (size_t y = 0; y < srcGradDim.w; y++) {
          srcGrad[getNHWC(srcGradDim, n, x, y, z)] = 0.0;
        }
      }

      for (size_t ax = 0; ax < destGradDim.h; ax++) {
        for (size_t ay = 0; ay < destGradDim.w; ay++) {
          // For the x and y argmax's, we use a 5-dimensional
          // tensor whose fifth dimension has size 2:
          size_t ix = 2 * getNHWC(destGradDim, n, ax, ay, z);
          size_t maxX = srcXY[ix];
          size_t maxY = srcXY[ix + 1];

          if (maxX > srcGradDim.h || maxY > srcGradDim.w) {
            //printf("Bad maxX=%lu or maxY=%lu coordinates. Max possible values are: X=%lu and Y=%lu\n", maxX, maxY, srcGradDim.h, srcGradDim.w);
          }

          float df = destGrad[getNHWC(destGradDim, n, ax, ay, z)];
          srcGrad[getNHWC(srcGradDim, n, maxX, maxY, z)] += df;
        } // W
      }   // H
    }     // C
  //}       // N
}

__kernel void poolmaxwithxygradW(__global void *mem, cl_uint32_t dest,
                             cl_uint32_t srcXY, cl_uint32_t destGrad, cl_uint32_t srcGrad,
                             cl_uint32_t filterSize, cl_uint32_t stride,
                             cl_uint32_t pad, ShapeNHWC srcGradDim, ShapeNHWC destDim) {
  poolmaxwithxygradK(&mem[dest], &mem[srcXY], &mem[destGrad], &mem[srcGrad],
                     filterSize, stride, pad, srcGradDim, destDim);
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

__kernel void oclpoolavgK(__global float *dest, __global float *src,
                       cl_uint32_t filterSize, cl_uint32_t stride,
                       cl_uint32_t pad, ShapeNCHW odim, ShapeNCHW idim) {
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

        sumVal += src[getNCHW(idim, n, d, (size_t)ox, (size_t)oy)];
      }
    }
    dest[getNCHW(odim, n, d, ax, ay)] = sumVal / filterArea;
  } // N
}

__kernel void oclpoolavgW(__global void *mem, cl_uint32_t dest, cl_uint32_t src,
                       cl_uint32_t filterSize, cl_uint32_t stride,
                       cl_uint32_t pad, ShapeNCHW odim, ShapeNCHW idim) {
  oclpoolavgK(&mem[dest], &mem[src], filterSize, stride, pad, odim, idim);
}

__kernel void transposeK_old(__global float *dest, __global float *src,
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
        //printf("transpose: %lu %lu %lu %lu -> %lu %lu %lu %lu\n", d0, d1, d2, d3, res[shuffle.n], res[shuffle.h], res[shuffle.w], res[shuffle.c]);
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

__kernel void transposeK_u(__global cl_uint64_t *dest, __global cl_uint64_t *src,
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

__kernel void transpose_uW(__global void *mem, cl_uint32_t dest, cl_uint32_t src,
                         ShapeNHWC odim, ShapeNHWC idim, ShapeNHWC shuffle) {
  transposeK_u(&mem[dest], &mem[src], odim, idim, shuffle);
}

__kernel void inserttensorK(__global float *dest, __global float *src,
                            ShapeNHWC odim, ShapeNHWC idim, ShapeNHWC offset) {
  size_t d0 = get_global_id(0);
  size_t offset_w = ((odim.w > 1) ? offset.w : 0);
  size_t offset_c = ((odim.c > 1) ? offset.c : 0);
  for (size_t d1 = 0; d1 < idim.h; d1++) {
    for (size_t d2 = 0; d2 < idim.w; d2++) {
      for (size_t d3 = 0; d3 < idim.c; d3++) {
        size_t r0 = d0 + offset.n;
        size_t r1 = d1 + offset.h;
        size_t r2 = d2 + offset_w;
        size_t r3 = d3 + offset_c;
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

__kernel void inserttensor_uK(__global cl_uint64_t *dest, __global cl_uint64_t *src,
                            ShapeNHWC odim, ShapeNHWC idim, ShapeNHWC offset) {
  size_t d0 = get_global_id(0);
  size_t offset_w = ((odim.w > 1) ? offset.w : 0);
  size_t offset_c = ((odim.c > 1) ? offset.c : 0);
  for (size_t d1 = 0; d1 < idim.h; d1++) {
    for (size_t d2 = 0; d2 < idim.w; d2++) {
      for (size_t d3 = 0; d3 < idim.c; d3++) {
        size_t r0 = d0 + offset.n;
        size_t r1 = d1 + offset.h;
        size_t r2 = d2 + offset_w;
        size_t r3 = d3 + offset_c;
        size_t srcIdx = getNHWC(idim, d0, d1, d2, d3);
        size_t destIdx = getNHWC(odim, r0, r1, r2, r3);
        dest[destIdx] = src[srcIdx];
      }
    }
  }
}

__kernel void inserttensor_uW(__global void *mem, cl_uint32_t dest,
                            cl_uint32_t src, ShapeNHWC odim, ShapeNHWC idim,
                            ShapeNHWC offset) {
  inserttensor_uK(&mem[dest], &mem[src], odim, idim, offset);
}

__kernel void extracttensorK(__global float *dest, __global float *src,
                            ShapeNHWC odim, ShapeNHWC idim, ShapeNHWC offset) {
  size_t d0 = get_global_id(0);
  size_t offset_w = ((odim.w > 1) ? offset.w : 0);
  size_t offset_c = ((odim.c > 1) ? offset.c : 0);
  for (size_t d1 = 0; d1 < odim.h; d1++) {
    for (size_t d2 = 0; d2 < odim.w; d2++) {
      for (size_t d3 = 0; d3 < odim.c; d3++) {
        size_t r0 = d0 + offset.n;
        size_t r1 = d1 + offset.h;
        size_t r2 = d2 + offset_w;
        size_t r3 = d3 + offset_c;
        size_t destIdx = getNHWC(odim, d0, d1, d2, d3);
        size_t srcIdx = getNHWC(idim, r0, r1, r2, r3);
        dest[destIdx] = src[srcIdx];
      }
    }
  }
}

__kernel void extracttensorW(__global void *mem, cl_uint32_t dest,
                            cl_uint32_t src, ShapeNHWC odim, ShapeNHWC idim,
                            ShapeNHWC offset) {
  extracttensorK(&mem[dest], &mem[src], odim, idim, offset);
}

#if 0
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
    //printf("InputBuf != inBuf: mem=%lu in=%lu inputBuf=&mem[in]=%lu inbuf=%lu\n", mem, in, inputBuf, inBuf);
  }
  int size = idim.n * idim.c * idim.w * idim.h;
  int idx;
  int count = 0;
  for (idx = 0; idx < size && count < 100; ++idx) {
    if (inputBuf[idx] != inBuf[idx]) {
      //printf("Elements are different at index %lu: %f (at %lu) != %f (at %lu)\n", idx, inputBuf[idx], &inputBuf[idx],
      // inBuf[idx], &inBuf[idx]);
      ++count;
    }
  }
 }
#endif

#if 0
#define ElemTy float
#define TYPED_NAME(X) X##_f

static size_t TYPED_NAME(get_element_ptr)(const ElemTy *tensor, const size_t *dims,
                              size_t numDims, const size_t *indices,
                              size_t numIndices) {
  size_t index = 0;
  size_t subdimensionSize = 1;
  for (size_t i = numDims; i > 0; i--) {
    size_t curIndicesValue = (i <= numIndices) ? indices[i - 1] : 0;
    index += subdimensionSize * curIndicesValue;
    subdimensionSize *= dims[i - 1];
  }
  return index;
}

#define AT(tensor, dims, numDims, indices, numIndices)                         \
  tensor[TYPED_NAME(get_element_ptr)(tensor, dims, numDims, indices, numIndices)]

__kernel void TYPED_NAME(insert_tensor_impl)(ElemTy *tensor, ElemTy *slice,
                                      cl_uint32_t *offset, cl_uint32_t *sliceCoor,
                                      cl_uint32_t *fusedCoor, cl_uint32_t *tensorDim,
                                      cl_uint32_t *sliceDim, cl_uint32_t numDimsTensor,
                                      cl_uint32_t numDimsSliceCoor,
                                      cl_uint32_t numDimsFusedCoor,
                                      unsigned isInsert, unsigned d) {
  unsigned isDone = (d == numDimsSliceCoor);

  if (isDone) {
    if (isInsert) {
      AT(tensor, tensorDim, numDimsTensor, fusedCoor, numDimsFusedCoor) =
          AT(slice, sliceDim, numDimsSliceCoor, sliceCoor, numDimsSliceCoor);
    } else {
      AT(slice, sliceDim, numDimsSliceCoor, sliceCoor, numDimsSliceCoor) =
          AT(tensor, tensorDim, numDimsTensor, fusedCoor, numDimsFusedCoor);
    }
    return;
  }

  for (size_t i = 0, e = sliceDim[d]; i < e; i++) {
    // Construct the coordinates for the slice and for the joint shape.
    // Add the 'offset' to the dimension that we concat the shapes on.
    sliceCoor[d] = i;
    fusedCoor[d] = i + offset[d];
    TYPED_NAME(insert_tensor_impl)(
        tensor, slice, offset, sliceCoor, fusedCoor, tensorDim, sliceDim,
        numDimsTensor, numDimsSliceCoor, numDimsFusedCoor, isInsert, d + 1);
  }
}

__kernel void TYPED_NAME(insert_tensor)(ElemTy *tensor, ElemTy *slice, cl_uint32_t *offset,
                          cl_uint32_t *tensorDim, cl_uint32_t *sliceDim,
                          cl_uint32_t numDimsTensor, cl_uint32_t numDimsSlice,
                          cl_uint32_t offsetDim) {
  // Reserve statically enough memory to avoid dynamic memory allocation.
  size_t sliceCoor[10];
  size_t fusedCoor[10];
  memcpy(sliceCoor, sliceDim, sizeof(*sliceDim) * numDimsSlice);
  memcpy(fusedCoor, tensorDim, sizeof(*tensorDim) * numDimsTensor);
  TYPED_NAME(insert_tensor_impl)(tensor, slice, offset, sliceCoor, fusedCoor,
                            tensorDim, sliceDim, numDimsTensor, numDimsSlice,
                            offsetDim, 1, 0);
}

__kernel void TYPED_NAME(extract_tensor)(ElemTy *tensor, ElemTy *slice, cl_uint32_t *offset,
                           cl_uint32_t *tensorDim, cl_uint32_t *sliceDim,
                           cl_uint32_t numDimsTensor, cl_uint32_t numDimsSlice,
                           cl_uint32_t offsetDim) {
  // Reserve statically enough memory to avoid dynamic memory allocation.
  size_t sliceCoor[10];
  size_t fusedCoor[10];
  memcpy(sliceCoor, sliceDim, sizeof(*sliceDim) * numDimsSlice);
  memcpy(fusedCoor, tensorDim, sizeof(*tensorDim) * numDimsTensor);
  TYPED_NAME(insert_tensor_impl)(tensor, slice, offset, sliceCoor, fusedCoor,
                            tensorDim, sliceDim, numDimsTensor, numDimsSlice,
                            offsetDim, 0, 0);
}

__kernel void extract_tensor_fW(__global void *mem, cl_uint32_t tensor, cl_uint32_t slice,
                       cl_uint32_t *offset, 
                       cl_uint32_t *tensorDim, cl_uint32_t *sliceDim,
                       cl_uint32_t numDimsTensor, cl_uint32_t numDimsSlice,
                       cl_uint32_t offsetDim) {
  extract_tensor_f(&mem[tensor], &mem[slice], offset, tensorDim,
                   sliceDim, numDimsTensor, numDimsSlice, offsetDim);
}
#endif

void memcpy_float(__global float *dest, const __global float *src, int len) {
    for(int i=0;i<len;i++) {
      dest[i]=src[i];
    }
}

__kernel void gatherK(__global float *dest,
                      __global const float *src,
                      __global cl_uint64_t *indices,
                      cl_uint32_t numIndices,
                      cl_uint32_t sliceSize) {
  int idx = get_global_id(0);
  cl_uint64_t slice = indices[idx];
  memcpy_float(dest + idx * sliceSize, src + slice * sliceSize, sliceSize);
  //memcpy(dest + idx * sliceSize, src + slice * sliceSize, sliceSize * sizeof(float));
}

__kernel void gatherW(__global void *mem,
                      cl_uint32_t dest,
                      cl_uint32_t src,
                      cl_uint32_t indices,
                      cl_uint32_t numIndices,
                      cl_uint32_t sliceSize) {
   gatherK(&mem[dest], &mem[src], &mem[indices], numIndices, sliceSize);
}

)";
