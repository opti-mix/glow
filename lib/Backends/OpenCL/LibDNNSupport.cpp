// Copyright 2017 Facebook Inc.  All Rights Reserved.
#define DEBUG_TYPE "opencl"

#include "OpenCL.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"
#include "glow/Support/Memory.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include "benchmark.hpp"
#include "device.hpp"
#include "libdnn.hpp"

//#define DEBUG(X) X

using namespace glow;
using namespace greentea;

typedef float Dtype;

template <class T>
static void setKernelArg(cl_kernel kernel, unsigned argIdx, T value) {
  cl_int err = clSetKernelArg(kernel, argIdx, sizeof(T), &value);
  GLOW_ASSERT(err == CL_SUCCESS && "Unable to set parameter");
}

void setKernelLocalArg(cl_kernel kernel, unsigned argIdx, size_t size) {
  cl_int err = clSetKernelArg(kernel, argIdx, size, nullptr);
  GLOW_ASSERT(err == CL_SUCCESS && "Unable to set parameter");
}

#if 0
#define DEFINE_GPU_BINARY_DATA_PARALLEL_KERNEL(name, type, body)               \
  __kernel void name##K##16(__global type * dest, __global type * lhs,         \
                            __global type * rhs) {                             \
    size_t j = get_global_id(0);                                               \
    {                                                                          \
      float8 LHS = vload8(j * 2, lhs);                                         \
      float8 RHS = vload8(j * 2, rhs);                                         \
      float8 VAL = body;                                                       \
      vstore8(VAL, j * 2, dest);                                               \
    }                                                                          \
    {                                                                          \
      float8 LHS = vload8(j * 2 + 1, lhs);                                     \
      float8 RHS = vload8(j * 2 + 1, rhs);                                     \
      float8 VAL = body;                                                       \
      vstore8(VAL, j * 2 + 1, dest);                                           \
    }                                                                          \
  }                                                                            \
  __kernel void name##W##16(__global void *mem, cl_uint32_t dest,              \
                            cl_uint32_t lhs, cl_uint32_t rhs) {                \
    name##K##16(&mem[dest], &mem[lhs], &mem[rhs]);                             \
  }                                                                            \
  __kernel void name##K##8(__global type * dest, __global type * lhs,          \
                           __global type * rhs) {                              \
    size_t j = get_global_id(0);                                               \
    float8 LHS = vload8(j, lhs);                                               \
    float8 RHS = vload8(j, rhs);                                               \
    float8 VAL = body;                                                         \
    vstore8(VAL, j, dest);                                                     \
  }                                                                            \
  __kernel void name##W##8(__global void *mem, cl_uint32_t dest,               \
                           cl_uint32_t lhs, cl_uint32_t rhs) {                 \
    name##K##8(&mem[dest], &mem[lhs], &mem[rhs]);                              \
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

#define DEFINE_GPU_UNARY_DATA_PARALLEL_KERNEL(name, type, body)                \
  __kernel void name##K##16(__global type * dest, __global type * src) {       \
    size_t j = get_global_id(0);                                               \
    {                                                                          \
      float8 SRC = vload8(j * 2, src);                                         \
      float8 VAL = body;                                                       \
      vstore8(VAL, j * 2, dest);                                               \
    }                                                                          \
    {                                                                          \
      float8 SRC = vload8(j * 2 + 1, src);                                     \
      float8 VAL = body;                                                       \
      vstore8(VAL, j * 2 + 1, dest);                                           \
    }                                                                          \
  }                                                                            \
  __kernel void name##W##16(__global void *mem, cl_uint32_t dest,              \
                            cl_uint32_t src) {                                 \
    name##K##16(&mem[dest], &mem[src]);                                        \
  }                                                                            \
  __kernel void name##K##8(__global type * dest, __global type * src) {        \
    size_t j = get_global_id(0);                                               \
    float8 SRC = vload8(j, src);                                               \
    float8 VAL = body;                                                         \
    vstore8(VAL, j, dest);                                                     \
  }                                                                            \
  __kernel void name##W##8(__global void *mem, cl_uint32_t dest,               \
                           cl_uint32_t src) {                                  \
    name##K##8(&mem[dest], &mem[src]);                                         \
  }                                                                            \
  __kernel void name##K(__global type *dest, __global type *lhs,               \
                        __global type *rhs) {                                  \
    size_t i = get_global_id(0);                                               \
    type RHS = rhs[i];                                                         \
    type LHS = lhs[i];                                                         \
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
// DEFINE_GPU_BINARY_DATA_PARALLEL_KERNEL(elementcmplte, float, LHS <= RHS)

DEFINE_GPU_UNARY_DATA_PARALLEL_KERNEL(elementmin, float, tanh(SRC))

#undef DEFINE_GPU_BINARY_DATA_PARALLEL_KERNEL
#undef DEFINE_GPU_UNARY_DATA_PARALLEL_KERNEL
#endif

#if 0
void OCLBackend::executeConvolution(ConvolutionInst *CC) {
  GLOW_UNREACHABLE("OpenCL supports only convolutions using NCHW format");
}
#endif

#if 1
#if 0
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
__attribute__((vec_type_hint(Dtype4))) void
conv_forward_mem(__global void *mem, unsigned im_in_offset, unsigned wg_offset,
                 unsigned bias_offset, unsigned im_out_offset) {
  __global const Dtype *im_in = &mem[im_in_offset];
  __global const Dtype *wg = &mem[wg_offset];
  __global const Dtype *bias = &mem[bias_offset];
  __global Dtype *im_out = &mem[im_out_offset];
  // Thread identifiers.
  // Local row ID (max: RTSM=TSM/WPTM).
  const int_tp tidn = get_local_id(0);
  // Local col ID (max: RTSN=TSN/WPTN).
  const int_tp tidm = get_local_id(1);
  // Work-group offset.
  const int_tp offN = TSN * get_group_id(0);
  // Work-group offset.
  const int_tp offM = TSM * get_group_id(1);
  // Local tile memory.
  // Asub for loading weights & shuffling the output.
  volatile __local Dtype Asub[64][8 + v_pad_A];
  // Bsub for loading the input image and shuffling the output image.
  volatile __local Dtype Bsub[8][64 + v_pad_B];
  int_tp batch = get_global_id(2);
  __global const Dtype *Aptr = wg;
  __global const Dtype *Bptr = im_in + v_B_off * batch;
  __global Dtype *Cptr = im_out + v_C_off * batch;
  __global const Dtype *Dptr = bias;
  // Initialize the accumulation registers.
  {
    Dtype4 Creg[WPTM][WPTN / VWN];
// Initialize the accumulation registers.
#pragma unroll
    for (int_tp wm = 0; wm < WPTM; ++wm) {
#pragma unroll
      for (int_tp wn = 0; wn < WPTN / VWN; ++wn) {
        VEC_4_0(Creg[wm][wn]) = 0.0;
        VEC_4_1(Creg[wm][wn]) = 0.0;
        VEC_4_2(Creg[wm][wn]) = 0.0;
        VEC_4_3(Creg[wm][wn]) = 0.0;
      }
    }
    {
// Loop over all tiles.
#pragma unroll 1
      for (int_tp t = 0; t < v_num_tiles; ++t) {
        // Load one tile of A into local memory.
        {
#pragma unroll 4
          for (int_tp la = 0; la < LPTA; ++la) {
            int_tp tid = tidm * RTSN + tidn;
            int_tp id = la * RTSN * RTSM + tid;
            int_tp row = id / TSK;
            int_tp col = id % TSK;
            int_tp tiledIndex = TSK * t + col;
            if ((offM + row) < M && tiledIndex < K) {
              Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
            } else {
              Asub[row][col] = 0.0;
            }
          }
        }
        // Load one tile of B into local memory.
        {
#pragma unroll 4
          for (int_tp lb = 0; lb < LPTB; ++lb) {
            int_tp tid = tidm * RTSN + tidn;
            int_tp id = lb * RTSN * RTSM + tid;
            int_tp col = id % TSN;
            int_tp row = id / TSN;
            int_tp tiledIndex = TSK * t + row;
            if ((offN + col) < N && tiledIndex < K) {
              int_tp d_iter_0;
              int_tp d_temp_0;
              int_tp d_iter_1;
              int_tp d_temp_1;
              int_tp imageIndex = offN + col;
              // Compute d_iter, final tiledIndex becomes input feature map ID.
              // Scale d_iter by the dilation factor.
              d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
              tiledIndex = tiledIndex / v_k_1;
              // Compute d_temp.
              // Scale d_temp by the stride and subtract the padding.
              d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
              imageIndex = imageIndex / v_imso_1;
              // Compute d_iter, final tiledIndex becomes input feature map ID.
              // Scale d_iter by the dilation factor.
              d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
              tiledIndex = tiledIndex / v_k_0;
              // Compute d_temp.
              // Scale d_temp by the stride and subtract the padding.
              d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
              imageIndex = imageIndex / v_imso_0;
              // Recombine final index, compute in-range.
              bool skip_range_check = false;
              // Used only if padding is not 0.
              bool in_range = skip_range_check;
              int_tp d_iter_im;
              // Here, d_temp_ represents the column shift,
              // while d_iter_ is the kernel shift.
              d_iter_im = d_temp_0 + d_iter_0;
              tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
              if (!skip_range_check) {
                in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
              }
              // Here, d_temp_ represents the column shift,
              // while d_iter_ is the kernel shift.
              d_iter_im = d_temp_1 + d_iter_1;
              tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
              if (!skip_range_check) {
                in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
              }
              if (in_range) {
                // tiledIndex now holds the memory offset for the input image.
                Bsub[row][col] = Bptr[tiledIndex];
              } else {
                Bsub[row][col] = 0.0;
              }
            } else {
              Bsub[row][col] = 0.0;
            }
          }
        }
        // Synchronize to make sure the tile is loaded.
        barrier(CLK_LOCAL_MEM_FENCE);
        // Temporary registers for A and B.
        Dtype4 Areg;
        Dtype4 Breg[WPTN / VWN];
// Loop over the values of a single tile.
#pragma unroll 1
        for (int_tp kt = 0; kt < TSK; kt += TSK_UNROLL) {
#pragma unroll 1
          for (int_tp ku = 0; ku < TSK_UNROLL; ++ku) {
            int_tp k = kt + ku;
// Cache the values of Bsub in registers.
#pragma unroll
            for (int_tp wn = 0; wn < WPTN / VWN; ++wn) {
              int_tp col = tidn + wn * VWN * RTSN;
              VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
              VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
              VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
              VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
            }
// Perform the computation.
#pragma unroll
            for (int_tp wm = 0; wm < WPTM / VWM; ++wm) {
              int_tp row = tidm + wm * VWM * RTSM;
              VEC_4_0(Areg) = Asub[row + 0][k];
              VEC_4_1(Areg) = Asub[row + 16][k];
              VEC_4_2(Areg) = Asub[row + 32][k];
              VEC_4_3(Areg) = Asub[row + 48][k];
#pragma unroll
              for (int_tp wn = 0; wn < WPTN / VWN; ++wn) {
                VEC_4_0(Creg[wm * VWM + 0][wn]) +=
                    VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
                VEC_4_0(Creg[wm * VWM + 1][wn]) +=
                    VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
                VEC_4_0(Creg[wm * VWM + 2][wn]) +=
                    VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
                VEC_4_0(Creg[wm * VWM + 3][wn]) +=
                    VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
                VEC_4_1(Creg[wm * VWM + 0][wn]) +=
                    VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
                VEC_4_1(Creg[wm * VWM + 1][wn]) +=
                    VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
                VEC_4_1(Creg[wm * VWM + 2][wn]) +=
                    VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
                VEC_4_1(Creg[wm * VWM + 3][wn]) +=
                    VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
                VEC_4_2(Creg[wm * VWM + 0][wn]) +=
                    VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
                VEC_4_2(Creg[wm * VWM + 1][wn]) +=
                    VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
                VEC_4_2(Creg[wm * VWM + 2][wn]) +=
                    VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
                VEC_4_2(Creg[wm * VWM + 3][wn]) +=
                    VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
                VEC_4_3(Creg[wm * VWM + 0][wn]) +=
                    VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
                VEC_4_3(Creg[wm * VWM + 1][wn]) +=
                    VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
                VEC_4_3(Creg[wm * VWM + 2][wn]) +=
                    VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
                VEC_4_3(Creg[wm * VWM + 3][wn]) +=
                    VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
              }
            }
          }
        }

        // Synchronize before loading the next tile.
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }
// Store the final results in C.
#pragma unroll
    for (int_tp wm = 0; wm < WPTM; ++wm) {
      int_tp globalRow = offM + tidm + wm * RTSM;
      Dtype biasval = Dptr[globalRow];
#pragma unroll
      for (int_tp wn = 0; wn < WPTN; ++wn) {
        int_tp globalCol = offN + tidn + wn * RTSN;
        printf("l0=%d l1=%d g0=%d g1=%d Store out[%d][%d] = ", get_local_id(0),
               get_local_id(1), get_global_id(0), get_global_id(1), globalRow,
               globalCol);
        if (globalRow < M && globalCol < N) {
          printf("%f\n", ((Dtype *)(&(Creg[wm][wn / VWN])))[wn % VWN] +
                             v_bmul * biasval);
          Cptr[globalRow * N + globalCol] =
              ((Dtype *)(&(Creg[wm][wn / VWN])))[wn % VWN] + v_bmul * biasval;
        }
      }
    }
  }
}
#endif

static const char *SimpleConvKernel = R"(
__kernel
void
conv_forward(__global const float *__restrict im_in,
                 __global const float *__restrict wg,
                 __global const float *__restrict bias,
                 __global float *__restrict im_out,
                 volatile __local float *Asub) {
  Asub[0] = 1111;
}

__kernel 
void 
conv_forward_mem(__global void *mem, unsigned im_in, unsigned wg, unsigned bias,
                 unsigned im_out) {
  volatile 
  __local 
  float Asub[1][1];
  Asub[0][0] = 0;
  //printf("Checkpoint 2\n");
  conv_forward(&mem[im_in], &mem[wg], &mem[bias], &mem[im_out], Asub);
}

)";

static const char *ConvKernel = R"(
#if defined(cl_khr_int32_base_atomics)
#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable
#define ATOMICS_32_AVAILABLE
#endif
#if defined(cl_khr_global_int32_base_atomics)
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define ATOMICS_32_AVAILABLE
#endif
#define Dtype float
#define Dtype1 float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16
#define VEC_1_0(X) X
#define VEC_2_0(X) X.x
#define VEC_2_1(X) X.y
#define VEC_4_0(X) X.x
#define VEC_4_1(X) X.y
#define VEC_4_2(X) X.z
#define VEC_4_3(X) X.w
#define VEC_8_0(X) X.s0
#define VEC_8_1(X) X.s1
#define VEC_8_2(X) X.s2
#define VEC_8_3(X) X.s3
#define VEC_8_4(X) X.s4
#define VEC_8_5(X) X.s5
#define VEC_8_6(X) X.s6
#define VEC_8_7(X) X.s7
#define VEC_16_0(X) X.s0
#define VEC_16_1(X) X.s1
#define VEC_16_2(X) X.s2
#define VEC_16_3(X) X.s3
#define VEC_16_4(X) X.s4
#define VEC_16_5(X) X.s5
#define VEC_16_6(X) X.s6
#define VEC_16_7(X) X.s7
#define VEC_16_8(X) X.s8
#define VEC_16_9(X) X.s9
#define VEC_16_10(X) X.sA
#define VEC_16_11(X) X.sB
#define VEC_16_12(X) X.sC
#define VEC_16_13(X) X.sD
#define VEC_16_14(X) X.sE
#define VEC_16_15(X) X.sF
#define int_tp int
#define uint_tp unsigned int
#define int_tpc int
#define uint_tpc unsigned int
#ifdef ATOMICS_32_AVAILABLE
inline void atomicAdd(volatile __global Dtype *source, const Dtype operand) {
  union {
    unsigned int intVal;
    Dtype floatVal;
  } next, expected, current;
  current.floatVal = *source;
  do {
    expected.floatVal = current.floatVal;
    next.floatVal = expected.floatVal + operand;
    current.intVal = atomic_cmpxchg((volatile __global unsigned int *)source,
                                    expected.intVal, next.intVal);
  } while (current.intVal != expected.intVal);
}
inline void atomicSub(volatile __global Dtype *source, const Dtype operand) {
  union {
    unsigned int intVal;
    Dtype floatVal;
  } next, expected, current;
  current.floatVal = *source;
  do {
    expected.floatVal = current.floatVal;
    next.floatVal = expected.floatVal - operand;
    current.intVal = atomic_cmpxchg((volatile __global unsigned int *)source,
                                    expected.intVal, next.intVal);
  } while (current.intVal != expected.intVal);
}
inline void atomicMul(volatile __global Dtype *source, const Dtype operand) {
  union {
    unsigned int intVal;
    Dtype floatVal;
  } next, expected, current;
  current.floatVal = *source;
  do {
    expected.floatVal = current.floatVal;
    next.floatVal = expected.floatVal * operand;
    current.intVal = atomic_cmpxchg((volatile __global unsigned int *)source,
                                    expected.intVal, next.intVal);
  } while (current.intVal != expected.intVal);
}
inline void atomicDiv(volatile __global Dtype *source, const Dtype operand) {
  union {
    unsigned int intVal;
    Dtype floatVal;
  } next, expected, current;
  current.floatVal = *source;
  do {
    expected.floatVal = current.floatVal;
    next.floatVal = expected.floatVal / operand;
    current.intVal = atomic_cmpxchg((volatile __global unsigned int *)source,
                                    expected.intVal, next.intVal);
  } while (current.intVal != expected.intVal);
}
#endif
__kernel void fill_memory(const int_tp n, const Dtype alpha, __global Dtype *x,
                          const int_tp offx) {
  for (int_tp index = get_global_id(0); index < n;
       index += get_global_size(0)) {
    x[index + offx] = alpha;
  }
}
// Number of spacial axes.
#ifdef v_nax
#undef v_nax
#endif
#define v_nax 2
// Number of groups.
#ifdef v_g
#undef v_g
#endif
#define v_g 1
// Input image batch offset.
#ifdef v_B_off
#undef v_B_off
#endif
#define v_B_off 150528
// Output image batch offset.
#ifdef v_C_off
#undef v_C_off
#endif
#define v_C_off 802816
#ifdef v_imsi_0
#undef v_imsi_0
#endif
#define v_imsi_0 224
#ifdef v_imso_0
#undef v_imso_0
#endif
#define v_imso_0 112
#ifdef v_imsi_1
#undef v_imsi_1
#endif
#define v_imsi_1 224
#ifdef v_imso_1
#undef v_imso_1
#endif
#define v_imso_1 112
#ifdef v_imsi
#undef v_imsi
#endif
#define v_imsi 50176
#ifdef v_imso
#undef v_imso
#endif
#define v_imso 12544
// Kernel size in dimension 0.
#ifdef v_k_0
#undef v_k_0
#endif
#define v_k_0 7
// Kernel size in dimension 1.
#ifdef v_k_1
#undef v_k_1
#endif
#define v_k_1 7
// Pad size in dimension 0.
#ifdef v_p_0
#undef v_p_0
#endif
#define v_p_0 3
// Pad size in dimension 1.
#ifdef v_p_1
#undef v_p_1
#endif
#define v_p_1 3
// Stride size in dimension 0.
#ifdef v_s_0
#undef v_s_0
#endif
#define v_s_0 2
// Stride size in dimension 1.
#ifdef v_s_1
#undef v_s_1
#endif
#define v_s_1 2
// Dilation size in dimension 0.
#ifdef v_d_0
#undef v_d_0
#endif
#define v_d_0 0
// Dilation size in dimension 1.
#ifdef v_d_1
#undef v_d_1
#endif
#define v_d_1 0
// Number of input channels.
#ifdef v_fin
#undef v_fin
#endif
#define v_fin 3
// Number of output channels.
#ifdef v_fout
#undef v_fout
#endif
#define v_fout 64
// Bias multiplier.
#ifdef v_bmul
#undef v_bmul
#endif
#define v_bmul (float)1
#ifdef MG
#undef MG
#endif
#define MG 64
#ifdef M
#undef M
#endif
#define M 64
#ifdef N
#undef N
#endif
#define N 12544
#ifdef KG
#undef KG
#endif
#define KG 147
#ifdef K
#undef K
#endif
#define K 147
#ifdef v_pad_A
#undef v_pad_A
#endif
#define v_pad_A 0
#ifdef v_pad_B
#undef v_pad_B
#endif
#define v_pad_B 0
// The tile size in dimension M.
#ifdef TSM
#undef TSM
#endif
#define TSM 4
// The tile size in dimension N.
#ifdef TSN
#undef TSN
#endif
#define TSN 4
// The tile size in dimension K.
#ifdef TSK
#undef TSK
#endif
#define TSK 8
#ifdef TSK_UNROLL
#undef TSK_UNROLL
#endif
#define TSK_UNROLL 1
// Work per thread in dimension M.
#ifdef WPTM
#undef WPTM
#endif
#define WPTM 4
#ifdef VWM
#undef VWM
#endif
#define VWM 4
// Work per thread in dimension N.
#ifdef WPTN
#undef WPTN
#endif
#define WPTN 4
#ifdef VWN
#undef VWN
#endif
#define VWN 4
// The reduced tile size in dimension M.
#ifdef RTSM
#undef RTSM
#endif
#define RTSM 1
// The reduced tile size in dimension N.
#ifdef RTSN
#undef RTSN
#endif
#define RTSN 1
// Loads per thread for A.
#ifdef LPTA
#undef LPTA
#endif
#define LPTA ((TSK * TSM) / (RTSM * RTSN))
// Loads per thread for B.
#ifdef LPTB
#undef LPTB
#endif
#define LPTB ((TSK * TSN) / (RTSM * RTSN))
#ifdef v_num_tiles
#undef v_num_tiles
#endif
#define v_num_tiles (((K - 1) / (TSK * 2) + 1) * 2)

/*
__kernel
    __attribute__((reqd_work_group_size(1, 1, 1)))
    __attribute__((vec_type_hint(Dtype4)))
    void
    conv_forward2(__global const Dtype *__restrict im_in,
                 __global const Dtype *__restrict wg,
                 __global const Dtype *__restrict bias,
                 __global Dtype *__restrict im_out) {
  volatile __local Dtype Asub[2][2];
  printf("Checkpoint 1\n");
  Asub[0][0] = 1111;
  printf("Checkpoint 2\n");
}
*/

/*
__kernel
    __attribute__((reqd_work_group_size(1, 1, 1)))
    __attribute__((vec_type_hint(Dtype4)))
    void
    conv_forward(__global const Dtype *__restrict im_in,
                 __global const Dtype *__restrict wg,
                 __global const Dtype *__restrict bias,
                 __global Dtype *__restrict im_out) {
  // Thread identifiers.
  // Local row ID (max: RTSM=TSM/WPTM).
  const int_tp tidn = get_local_id(0);
  // Local col ID (max: RTSN=TSN/WPTN).
  const int_tp tidm = get_local_id(1);
  // Work-group offset.
  const int_tp offN = TSN * get_group_id(0);
  // Work-group offset.
  const int_tp offM = TSM * get_group_id(1);
  // Local tile memory.
  // Asub for loading weights & shuffling the output.
  volatile 
  __local Dtype Asub[4][8 + v_pad_A];
  // Bsub for loading the input image and shuffling the output image.
  volatile 
  __local Dtype Bsub[8][4 + v_pad_B];
  int_tp batch = get_global_id(2);
  __global const Dtype *Aptr = wg;
  __global const Dtype *Bptr = im_in + v_B_off * batch;
  __global Dtype *Cptr = im_out + v_C_off * batch;
  __global const Dtype *Dptr = bias;
  printf("Checkpoint -2\n");
  Asub[0][0] = 1111;
  printf("Checkpoint -1\n");
  Asub[3][7] = 2222;
  // Initialize the accumulation registers.
  {
#if 1
    Dtype4 Creg[WPTM][WPTN / VWN];
    printf("Checkpoint 0\n");
// Initialize the accumulation registers.
#pragma unroll
    for (int_tp wm = 0; wm < WPTM; ++wm) {
#pragma unroll
      for (int_tp wn = 0; wn < WPTN / VWN; ++wn) {
        VEC_4_0(Creg[wm][wn]) = 0.0;
        VEC_4_1(Creg[wm][wn]) = 0.0;
        VEC_4_2(Creg[wm][wn]) = 0.0;
        VEC_4_3(Creg[wm][wn]) = 0.0;
      }
    }
#endif
    printf("Checkpoint 1\n");
    {
      // Loop over all tiles.
      //#pragma unroll 1
      for (int_tp t = 0; t < v_num_tiles; ++t) {
        // Load one tile of A into local memory.
        {
          //#pragma unroll 4
          for (int_tp la = 0; la < LPTA; ++la) {
#if 1
            int_tp tid = tidm * RTSN + tidn;
            int_tp id = la * RTSN * RTSM + tid;
            int_tp row = id / TSK;
            int_tp col = id % TSK;
            int_tp tiledIndex = TSK * t + col;
            printf("la=%u offM=%u row=%u col=%u tiledIndex=%u t=%u\n", la, offM,
                   row, col, tiledIndex, t);
            if ((offM + row) < M && tiledIndex < K) {
              float val = Aptr[(offM + row) * K + tiledIndex];
              printf("1 Asub[row][col]=%f\n", Asub[row][col]);
              printf("1 Aptr[(offM + row) * K + tiledIndex]=%f\n", val);
              Asub[row][col] = val;
              printf("2 Asub[row][col]=%f\n", Asub[row][col]);
              printf("2 Aptr[(offM + row) * K + tiledIndex]=%f\n",
                     Aptr[(offM + row) * K + tiledIndex]);
            } else {
              Asub[row][col] = 0.0;
            }
#endif
          }
        }
#if 0
        printf("Checkpoint 2\n");
        // Load one tile of B into local memory.
        {
#pragma unroll 4
          for (int_tp lb = 0; lb < LPTB; ++lb) {
            int_tp tid = tidm * RTSN + tidn;
            int_tp id = lb * RTSN * RTSM + tid;
            int_tp col = id % TSN;
            int_tp row = id / TSN;
            int_tp tiledIndex = TSK * t + row;
            if ((offN + col) < N && tiledIndex < K) {
              int_tp d_iter_0;
              int_tp d_temp_0;
              int_tp d_iter_1;
              int_tp d_temp_1;
              int_tp imageIndex = offN + col;
              // Compute d_iter, final tiledIndex becomes input feature map ID.
              // Scale d_iter by the dilation factor.
              d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
              tiledIndex = tiledIndex / v_k_1;
              // Compute d_temp.
              // Scale d_temp by the stride and subtract the padding.
              d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
              imageIndex = imageIndex / v_imso_1;
              // Compute d_iter, final tiledIndex becomes input feature map ID.
              // Scale d_iter by the dilation factor.
              d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
              tiledIndex = tiledIndex / v_k_0;
              // Compute d_temp.
              // Scale d_temp by the stride and subtract the padding.
              d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
              imageIndex = imageIndex / v_imso_0;
              // Recombine final index, compute in-range.
              bool skip_range_check = false;
              // Used only if padding is not 0.
              bool in_range = skip_range_check;
              int_tp d_iter_im;
              // Here, d_temp_ represents the column shift,
              // while d_iter_ is the kernel shift.
              d_iter_im = d_temp_0 + d_iter_0;
              tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
              if (!skip_range_check) {
                in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
              }
              // Here, d_temp_ represents the column shift,
              // while d_iter_ is the kernel shift.
              d_iter_im = d_temp_1 + d_iter_1;
              tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
              if (!skip_range_check) {
                in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
              }
              if (in_range) {
                // tiledIndex now holds the memory offset for the input image.
                Bsub[row][col] = Bptr[tiledIndex];
              } else {
                Bsub[row][col] = 0.0;
              }
            } else {
              Bsub[row][col] = 0.0;
            }
          }
        }
        printf("Checkpoint 3\n");
        // Synchronize to make sure the tile is loaded.
        barrier(CLK_LOCAL_MEM_FENCE);
        // Temporary registers for A and B.
        Dtype4 Areg;
        Dtype4 Breg[WPTN / VWN];
// Loop over the values of a single tile.
#pragma unroll 1
        for (int_tp kt = 0; kt < TSK; kt += TSK_UNROLL) {
#pragma unroll 1
          for (int_tp ku = 0; ku < TSK_UNROLL; ++ku) {
            int_tp k = kt + ku;
// Cache the values of Bsub in registers.
#pragma unroll
            for (int_tp wn = 0; wn < WPTN / VWN; ++wn) {
              int_tp col = tidn + wn * VWN * RTSN;
              VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
              VEC_4_1(Breg[wn]) = Bsub[k][col + 1];
              VEC_4_2(Breg[wn]) = Bsub[k][col + 2];
              VEC_4_3(Breg[wn]) = Bsub[k][col + 3];
            }
            printf("Checkpoint 4\n");
// Perform the computation.
#pragma unroll
            for (int_tp wm = 0; wm < WPTM / VWM; ++wm) {
              int_tp row = tidm + wm * VWM * RTSM;
              VEC_4_0(Areg) = Asub[row + 0][k];
              VEC_4_1(Areg) = Asub[row + 1][k];
              VEC_4_2(Areg) = Asub[row + 2][k];
              VEC_4_3(Areg) = Asub[row + 3][k];
#pragma unroll
              for (int_tp wn = 0; wn < WPTN / VWN; ++wn) {
                VEC_4_0(Creg[wm * VWM + 0][wn]) +=
                    VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
                VEC_4_0(Creg[wm * VWM + 1][wn]) +=
                    VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
                VEC_4_0(Creg[wm * VWM + 2][wn]) +=
                    VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
                VEC_4_0(Creg[wm * VWM + 3][wn]) +=
                    VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
                VEC_4_1(Creg[wm * VWM + 0][wn]) +=
                    VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
                VEC_4_1(Creg[wm * VWM + 1][wn]) +=
                    VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
                VEC_4_1(Creg[wm * VWM + 2][wn]) +=
                    VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
                VEC_4_1(Creg[wm * VWM + 3][wn]) +=
                    VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
                VEC_4_2(Creg[wm * VWM + 0][wn]) +=
                    VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
                VEC_4_2(Creg[wm * VWM + 1][wn]) +=
                    VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
                VEC_4_2(Creg[wm * VWM + 2][wn]) +=
                    VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
                VEC_4_2(Creg[wm * VWM + 3][wn]) +=
                    VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
                VEC_4_3(Creg[wm * VWM + 0][wn]) +=
                    VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
                VEC_4_3(Creg[wm * VWM + 1][wn]) +=
                    VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
                VEC_4_3(Creg[wm * VWM + 2][wn]) +=
                    VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
                VEC_4_3(Creg[wm * VWM + 3][wn]) +=
                    VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
              }
            }
          }
        }
#endif
        printf("Checkpoint 5\n");
        // Synchronize before loading the next tile.
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }
#if 0
// Store the final results in C.
#pragma unroll
    for (int_tp wm = 0; wm < WPTM; ++wm) {
      int_tp globalRow = offM + tidm + wm * RTSM;
      Dtype biasval = Dptr[globalRow];
#pragma unroll
      for (int_tp wn = 0; wn < WPTN; ++wn) {
        int_tp globalCol = offN + tidn + wn * RTSN;
        if (globalRow < M && globalCol < N) {
          Cptr[globalRow * N + globalCol] =
              ((Dtype *)(&(Creg[wm][wn / VWN])))[wn % VWN] + v_bmul * biasval;
        }
      }

      printf("Checkpoint 6\n");
    }
#endif
  }
}

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
__attribute__((vec_type_hint(Dtype4)))
void
conv_forward_mem(__global void *mem, unsigned im_in, unsigned wg, unsigned bias,
                 unsigned im_out) {
  printf("mem=%p in=%u wg=%u bias=%u out=%u\n", mem, im_in, wg, bias, im_out);
  conv_forward(&mem[im_in], &mem[wg], &mem[bias], &mem[im_out]);
}
*/

__kernel
//__attribute__((reqd_work_group_size(16, 16, 1)))
__attribute__((vec_type_hint(Dtype4)))
void conv_forward_mem(__global void* mem, unsigned im_in_offset, unsigned wg_offset, unsigned bias_offset, unsigned im_out_offset) {
__global const Dtype *im_in = &mem[im_in_offset];
__global const Dtype *wg = &mem[wg_offset];
__global const Dtype *bias = &mem[bias_offset];
__global Dtype *im_out = &mem[im_out_offset];
// Thread identifiers.
// Local row ID (max: RTSM=TSM/WPTM).
const int_tp tidn = get_local_id(0);
// Local col ID (max: RTSN=TSN/WPTN).
const int_tp tidm = get_local_id(1);
// Work-group offset.
const int_tp offN = TSN*get_group_id(0);
// Work-group offset.
const int_tp offM = TSM*get_group_id(1);
// Local tile memory.
// Asub for loading weights & shuffling the output.
volatile __local Dtype Asub[64][8 + v_pad_A];
// Bsub for loading the input image and shuffling the output image.
volatile __local Dtype Bsub[8][64 + v_pad_B];
int_tp batch = get_global_id(2);
__global const Dtype* Aptr = wg;
__global const Dtype* Bptr = im_in + v_B_off * batch;
__global Dtype* Cptr = im_out + v_C_off * batch;
__global const Dtype* Dptr = bias;
// Initialize the accumulation registers.
{
Dtype4 Creg[WPTM][WPTN/VWN];
// Initialize the accumulation registers.
#pragma unroll
for (int_tp wm=0; wm<WPTM; ++wm) {
#pragma unroll
for (int_tp wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm][wn]) = 0.0;
VEC_4_1(Creg[wm][wn]) = 0.0;
VEC_4_2(Creg[wm][wn]) = 0.0;
VEC_4_3(Creg[wm][wn]) = 0.0;
}
}
{
// Loop over all tiles.
#pragma unroll 1
for (int_tp t = 0; t < v_num_tiles; ++t) {
// Load one tile of A into local memory.
{
#pragma unroll 4
for (int_tp la = 0; la < LPTA; ++la) {
int_tp tid = tidm * RTSN + tidn;
int_tp id = la * RTSN * RTSM + tid;
int_tp row = id / TSK;
int_tp col = id % TSK;
int_tp tiledIndex = TSK * t + col;
if ((offM + row) < M && tiledIndex < K) {
Asub[row][col] = Aptr[(offM + row) * K + tiledIndex];
} else {
Asub[row][col] = 0.0;
}
}
}
// Load one tile of B into local memory.
{
#pragma unroll 4
for (int_tp lb = 0; lb < LPTB; ++lb) {
int_tp tid = tidm * RTSN + tidn;
int_tp id = lb * RTSN * RTSM + tid;
int_tp col = id % TSN;
int_tp row = id / TSN;
int_tp tiledIndex = TSK * t + row;
if ((offN + col) < N && tiledIndex < K) {
int_tp d_iter_0;
int_tp d_temp_0;
int_tp d_iter_1;
int_tp d_temp_1;
int_tp imageIndex = offN + col;
// Compute d_iter, final tiledIndex becomes input feature map ID.
// Scale d_iter by the dilation factor.
d_iter_1 = (tiledIndex % v_k_1) * v_d_1;
tiledIndex = tiledIndex / v_k_1;
// Compute d_temp.
// Scale d_temp by the stride and subtract the padding.
d_temp_1 = (imageIndex % v_imso_1) * v_s_1 - v_p_1;
imageIndex = imageIndex / v_imso_1;
// Compute d_iter, final tiledIndex becomes input feature map ID.
// Scale d_iter by the dilation factor.
d_iter_0 = (tiledIndex % v_k_0) * v_d_0;
tiledIndex = tiledIndex / v_k_0;
// Compute d_temp.
// Scale d_temp by the stride and subtract the padding.
d_temp_0 = (imageIndex % v_imso_0) * v_s_0 - v_p_0;
imageIndex = imageIndex / v_imso_0;
// Recombine final index, compute in-range.
bool skip_range_check = false;
// Used only if padding is not 0.
bool in_range = skip_range_check;
int_tp d_iter_im;
// Here, d_temp_ represents the column shift,
// while d_iter_ is the kernel shift.
d_iter_im = d_temp_0 + d_iter_0;
tiledIndex = tiledIndex * v_imsi_0 + d_iter_im;
if (!skip_range_check) {
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_0;
}
// Here, d_temp_ represents the column shift,
// while d_iter_ is the kernel shift.
d_iter_im = d_temp_1 + d_iter_1;
tiledIndex = tiledIndex * v_imsi_1 + d_iter_im;
if (!skip_range_check) {
in_range &= d_iter_im >= 0 && d_iter_im < v_imsi_1;
}
if (in_range) {
// tiledIndex now holds the memory offset for the input image.
Bsub[row][col] = Bptr[tiledIndex];
} else {
Bsub[row][col] = 0.0;
}
} else {
Bsub[row][col] = 0.0;
}
}
}
// Synchronize to make sure the tile is loaded.
barrier(CLK_LOCAL_MEM_FENCE);
// Temporary registers for A and B.
Dtype4 Areg;
Dtype4 Breg[WPTN/VWN];
// Loop over the values of a single tile.
#pragma unroll 1
for (int_tp kt=0; kt<TSK; kt+=TSK_UNROLL) {
#pragma unroll 1
for (int_tp ku=0; ku<TSK_UNROLL; ++ku) {
int_tp k = kt + ku;
// Cache the values of Bsub in registers.
#pragma unroll
for (int_tp wn=0; wn<WPTN/VWN; ++wn) {
int_tp col = tidn + wn*VWN*RTSN;
VEC_4_0(Breg[wn]) = Bsub[k][col + 0];
VEC_4_1(Breg[wn]) = Bsub[k][col + 16];
VEC_4_2(Breg[wn]) = Bsub[k][col + 32];
VEC_4_3(Breg[wn]) = Bsub[k][col + 48];
}
// Perform the computation.
#pragma unroll
for (int_tp wm=0; wm<WPTM/VWM; ++wm) {
int_tp row = tidm + wm*VWM*RTSM;
VEC_4_0(Areg) = Asub[row + 0][k];
VEC_4_1(Areg) = Asub[row + 16][k];
VEC_4_2(Areg) = Asub[row + 32][k];
VEC_4_3(Areg) = Asub[row + 48][k];
#pragma unroll
for (int_tp wn=0; wn<WPTN/VWN; ++wn) {
VEC_4_0(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_0(Breg[wn]);
VEC_4_0(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_0(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_1(Breg[wn]);
VEC_4_1(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_1(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_2(Breg[wn]);
VEC_4_2(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_2(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 0][wn]) += VEC_4_0(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 1][wn]) += VEC_4_1(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 2][wn]) += VEC_4_2(Areg) * VEC_4_3(Breg[wn]);
VEC_4_3(Creg[wm * VWM + 3][wn]) += VEC_4_3(Areg) * VEC_4_3(Breg[wn]);
}
}
}
}

// Synchronize before loading the next tile.
barrier(CLK_LOCAL_MEM_FENCE);
}
}
// Store the final results in C.
#pragma unroll
for (int_tp wm=0; wm<WPTM; ++wm) {
int_tp globalRow = offM + tidm + wm * RTSM;
Dtype biasval = Dptr[globalRow];
#pragma unroll
for (int_tp wn=0; wn<WPTN; ++wn) {
int_tp globalCol = offN + tidn + wn * RTSN;
if (globalRow < M && globalCol < N) {
Cptr[globalRow * N + globalCol] = ((Dtype*)(&(Creg[wm][wn/VWN])))[wn%VWN] + v_bmul * biasval;
}
}
}
}
}
__kernel void null_kernel_float(unsigned p) {}
__kernel void null_kernel_double(unsigned p) {}
)";

std::unordered_map<std::string, LibDNNConv<Dtype> *> convolutionKernels;

template <typename Dtype>
std::string getConvIdentifier(LibDNNConvConfig &config) {
  std::stringstream ss;
  ss << "CONV_";
  if (std::is_same<Dtype, double>::value) {
    ss << "double_";
  } else {
    ss << "float_";
  }
  // Device name
  ss << config.dev_ptr->name();
  ss << "_";
  // spatial dims
  ss << 2 << "D_";
  ss << "IN[";
  for (int_tp i = 0; i < config.in_shape.size(); ++i) {
    ss << config.in_shape[i];
    if (i < config.in_shape.size() - 1) {
      ss << ",";
    }
  }
  ss << "]_OUT[";
  for (int_tp i = 0; i < config.out_shape.size(); ++i) {
    ss << config.out_shape[i];
    if (i < config.out_shape.size() - 1) {
      ss << ",";
    }
  }
  ss << "]_K[";
  for (int_tp i = 0; i < config.kernel.size(); ++i) {
    ss << config.kernel[i];
    if (i < config.kernel.size() - 1) {
      ss << ",";
    }
  }
  ss << "]_S[";
  for (int_tp i = 0; i < config.stride.size(); ++i) {
    ss << config.stride[i];
    if (i < config.stride.size() - 1) {
      ss << ",";
    }
  }
  ss << "]_P[";
  for (int_tp i = 0; i < config.pad.size(); ++i) {
    ss << config.pad[i];
    if (i < config.pad.size() - 1) {
      ss << ",";
    }
  }
  ss << "]_D[";
  for (int_tp i = 0; i < config.dilation.size(); ++i) {
    ss << config.dilation[i];
    if (i < config.dilation.size() - 1) {
      ss << ",";
    }
  }
  ss << "]_";
  ss << "FIN[" << config.in_shape[1] << "]_";
  ss << "FOUT[" << config.out_shape[1] << "]_";
  ss << "G[" << config.group << "]";
  return ss.str();
}

#if 1
void OCLBackend::executeConvolution(OCLConvolutionInst *CC) {
  executeConvolutionAlt(CC);
  return;
  auto input = CC->getSrc();
  auto output = CC->getDest();
  auto bias = CC->getBias();
  auto weights = CC->getFilter();
#if 1
  if (!devPtr_) {
    int ctx_id = 0;
    int device = device_;
    greentea::device::setupViennaCLContext(ctx_id, context_, deviceId_,
                                           commands_);
    ocl_devices_.push_back(deviceId_);
    viennacl::ocl::setup_context(ctx_id, ocl_devices_);
    devPtr_ = new greentea::device(ctx_id, device, /* list_id, */
                                   greentea::Backend::BACKEND_OpenCL);
    // Init is necessary to initialize the workgroup sizes properly.
    devPtr_->Init();
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

    std::vector<size_t> temp(3);
    clGetDeviceInfo(ctx.devices()[0].id(), CL_DEVICE_MAX_WORK_ITEM_SIZES,
                    3 * sizeof(size_t), &temp[0], NULL);
    llvm::errs() << "WG sizes: " << temp[0] << ", " << temp[1] << ", "
                 << temp[2] << "\n";
  }
  LibDNNConvConfig config;
  config.dev_ptr = devPtr_;
  // NCHW is the expected format of input and output.
  config.in_shape =
      std::vector<int_tp>(input->dims().begin(), input->dims().end());
  config.out_shape =
      std::vector<int_tp>(output->dims().begin(), output->dims().end());
  // The expected format for kernel, pad and stride is HxW.
  config.kernel = std::vector<int_tp>{static_cast<int_tp>(CC->getKernel()),
                                      static_cast<int_tp>(CC->getKernel())};
  config.pad = std::vector<int_tp>{static_cast<int_tp>(CC->getPad()),
                                   static_cast<int_tp>(CC->getPad())};
  config.stride = std::vector<int_tp>{static_cast<int_tp>(CC->getStride()),
                                      static_cast<int_tp>(CC->getStride())};
  config.dilation = std::vector<int_tp>{1, 1};
  config.group = 1;
  config.bias_term = true;
  config.fast_unsafe_math = true;
  config.weights_backward = true;
  config.bias_backward = true;

  // if (std::is_same<Dtype, float>::value ||
  //  this->device_->CheckCapability("cl_khr_int64_base_atomics")) {
  //  config.wgalgo = LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC;
  //  config.bwalgo = LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC;
  //} else {
  config.wgalgo = LIBDNN_CONVOLUTION_WG_ALGO_DIRECT;
  //  config.bwalgo = LIBDNN_CONVOLUTION_BW_ALGO_IM2COL;
  //}
  //

  auto odim = ShapeNCHW(CC->getDest()->getType()->dims());
  auto idim = ShapeNCHW(CC->getSrc()->getType()->dims());
  auto filterDim = ShapeNCHW(CC->getFilter()->getType()->dims());
  DEBUG(llvm::dbgs() << "\n\nKernel (k, s, p): " << CC->getKernel() << " , "
                     << CC->getStride() << " , " << CC->getPad() << "\n";
        llvm::dbgs() << "Filter dims (n, w, h, c): " << filterDim.n << " , "
                     << filterDim.w << " , " << filterDim.h << " , "
                     << filterDim.c << "\n";
        llvm::dbgs() << "Src dims (n, w, h, c): " << idim.n << " , " << idim.w
                     << " , " << idim.h << " , " << idim.c << "\n";
        llvm::dbgs() << "Dest dims (n, w, h, c): " << odim.n << " , " << odim.w
                     << " , " << odim.h << " , " << odim.c << "\n");
  auto id = getConvIdentifier<Dtype>(config);
  LibDNNConv<Dtype> *libdnn{nullptr};
  if (convolutionKernels.count(id)) {
    DEBUG(llvm::dbgs() << "Reuse existing kernel: " << id << "\n");
    libdnn = convolutionKernels[id];
  } else {
    // Create an optimized convolution kernel.
    libdnn = new LibDNNConv<Dtype>(config);
    // llvm::errs() << libdnn->getKernel() << "\n";
    convolutionKernels[id] = libdnn;
  }
  // llvm::errs() << libdnn->getKernel() << "\n";
  GLOW_ASSERT(tensors_.count(input));
  GLOW_ASSERT(tensors_.count(output));
  GLOW_ASSERT(tensors_.count(bias));
  GLOW_ASSERT(tensors_.count(weights));
  // Run it.
  // libdnn->Forward((float *)inputBuf, (float *)weightsBuf, (float *)biasBuf,
  //                (float *)outputBuf, idim.n);
  libdnn->Forward((float *)deviceBuffer_, tensors_[input], tensors_[weights],
                  tensors_[bias], tensors_[output], idim.n);
#if 0
  // Check only if it is a simple kernel of size 1 and stride 1.
  if (CC->getKernel() > 1) {
    llvm::errs() << libdnn->getKernel() << "\n";
    clFinish(commands_);
    llvm::errs() << "Checking OpenCL convolution for correctness\n";
    float *outNew = (float *)alignedAlloc(output->getType()->getSizeInBytes(),
                                          TensorAlignment);
    float *outOld = (float *)alignedAlloc(output->getType()->getSizeInBytes(),
                                          TensorAlignment);
    // Copy output into a temporary host buffer.
    copyValueFromDevice(output, outNew);
    clFinish(commands_);

    // Run the naive kernel
    executeConvolutionAlt(CC);
    clFinish(commands_);
    // Copy output into a temporary host buffer.
    copyValueFromDevice(output, outOld);
    clFinish(commands_);

    Tensor oldT(outOld, output->getType());
    Tensor newT(outNew, output->getType());

    auto oldH = oldT.getHandle<float>();
    auto newH = newT.getHandle<float>();

    // Compare buffers for equality.
    // for (int idx = 0, e = output->getType()->size(); idx < e; ++idx) {
    newH.dump();
    for (size_t n = 0, ne = 1; n < ne; ++n) {
      for (size_t c = 0, ce = 3; c < ce; ++c) {
        for (size_t h = 0, he = 3; h < he; ++h) {
          for (size_t w = 0, we = 3; w < we; ++w) {
            if (oldH.at({n, c, h, w}) != newH.at({n, c, h, w})) {
              llvm::errs() << "Convolution results differ at index (" << n
                           << ", " << c << ", " << h << ", " << w << ") :"
                           << "new = " << newH.at({n, c, h, w}) << " vs "
                           << "old = " << oldH.at({n, c, h, w}) << "\n";
            }
          }
        }
      }
  }
  alignedFree(outNew);
  alignedFree(outOld);
  }
#endif
#else
  auto prog = createProgram(ConvKernel, {}, commands_);
  auto kernel = createKernel("conv_forward_mem", prog);
  setKernelArg(kernel, 0, deviceBuffer_);
  setKernelArg<cl_uint>(kernel, 1, tensors_[input]);
  setKernelArg<cl_uint>(kernel, 2, tensors_[weights]);
  setKernelArg<cl_uint>(kernel, 3, tensors_[bias]);
  setKernelArg<cl_uint>(kernel, 4, tensors_[output]);
  // setKernelLocalArg(kernel, 5, 4*8*sizeof(float));
  // setKernelLocalArg(kernel, 6, 4*8*sizeof(float));
  enqueueKernel(commands_, kernel, deviceId_, {1, 1, 1}, kernelLaunches_);
#endif
}
#else
#endif
#endif

void OCLBackend::executeConvolutionGrad(ConvolutionGradInst *CC) {
  // executeConvolutionAlt(CC);
  // return;
  auto input = CC->getSrc();
  auto output = CC->getSrcGrad();
  auto bias = CC->getBiasGrad();
  auto weights = CC->getFilterGrad();
#if 1
  if (!devPtr_) {
    int ctx_id = 0;
    int device = device_;
    greentea::device::setupViennaCLContext(ctx_id, context_, deviceId_,
                                           commands_);
    ocl_devices_.push_back(deviceId_);
    viennacl::ocl::setup_context(ctx_id, ocl_devices_);
    devPtr_ = new greentea::device(ctx_id, device, /* list_id, */
                                   greentea::Backend::BACKEND_OpenCL);
    // Init is necessary to initialize the workgroup sizes properly.
    devPtr_->Init();
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(ctx_id);

    std::vector<size_t> temp(3);
    clGetDeviceInfo(ctx.devices()[0].id(), CL_DEVICE_MAX_WORK_ITEM_SIZES,
                    3 * sizeof(size_t), &temp[0], NULL);
    llvm::errs() << "WG sizes: " << temp[0] << ", " << temp[1] << ", "
                 << temp[2] << "\n";
  }
  LibDNNConvConfig config;
  config.dev_ptr = devPtr_;
  // NCHW is the expected format of input and output.
  config.in_shape =
      std::vector<int_tp>(input->dims().begin(), input->dims().end());
  config.out_shape =
      std::vector<int_tp>(output->dims().begin(), output->dims().end());
  // The expected format for kernel, pad and stride is HxW.
  config.kernel = std::vector<int_tp>{static_cast<int_tp>(CC->getKernel()),
                                      static_cast<int_tp>(CC->getKernel())};
  config.pad = std::vector<int_tp>{static_cast<int_tp>(CC->getPad()),
                                   static_cast<int_tp>(CC->getPad())};
  config.stride = std::vector<int_tp>{static_cast<int_tp>(CC->getStride()),
                                      static_cast<int_tp>(CC->getStride())};
  config.dilation = std::vector<int_tp>{1, 1};
  config.group = 1;
  config.bias_term = true;
  config.fast_unsafe_math = true;
  config.weights_backward = true;
  config.bias_backward = true;

  // if (std::is_same<Dtype, float>::value ||
  //  this->device_->CheckCapability("cl_khr_int64_base_atomics")) {
  //  config.wgalgo = LIBDNN_CONVOLUTION_WG_ALGO_ATOMIC;
  //  config.bwalgo = LIBDNN_CONVOLUTION_BW_ALGO_COL2IM_ATOMIC;
  //} else {
  config.wgalgo = LIBDNN_CONVOLUTION_WG_ALGO_DIRECT;
  //  config.bwalgo = LIBDNN_CONVOLUTION_BW_ALGO_IM2COL;
  //}
  //

  auto odim = ShapeNCHW(CC->getDestGrad()->getType()->dims());
  auto idim = ShapeNCHW(CC->getSrc()->getType()->dims());
  auto filterDim = ShapeNCHW(CC->getFilter()->getType()->dims());
  DEBUG(llvm::dbgs() << "\n\nKernel (k, s, p): " << CC->getKernel() << " , "
                     << CC->getStride() << " , " << CC->getPad() << "\n";
        llvm::dbgs() << "Filter dims (n, w, h, c): " << filterDim.n << " , "
                     << filterDim.w << " , " << filterDim.h << " , "
                     << filterDim.c << "\n";
        llvm::dbgs() << "Src dims (n, w, h, c): " << idim.n << " , " << idim.w
                     << " , " << idim.h << " , " << idim.c << "\n";
        llvm::dbgs() << "Dest dims (n, w, h, c): " << odim.n << " , " << odim.w
                     << " , " << odim.h << " , " << odim.c << "\n");
  auto id = getConvIdentifier<Dtype>(config);
  LibDNNConv<Dtype> *libdnn{nullptr};
  if (convolutionKernels.count(id)) {
    DEBUG(llvm::dbgs() << "Reuse existing kernel: " << id << "\n");
    libdnn = convolutionKernels[id];
  } else {
    // Create an optimized convolution kernel.
    libdnn = new LibDNNConv<Dtype>(config);
    // llvm::errs() << libdnn->getKernel() << "\n";
    convolutionKernels[id] = libdnn;
  }
  // llvm::errs() << libdnn->getKernel() << "\n";
  GLOW_ASSERT(tensors_.count(input));
  GLOW_ASSERT(tensors_.count(output));
  GLOW_ASSERT(tensors_.count(bias));
  GLOW_ASSERT(tensors_.count(weights));
  // Run it.
  // libdnn->Forward((float *)inputBuf, (float *)weightsBuf, (float *)biasBuf,
  //                (float *)outputBuf, idim.n);
  // libdnn->Forward((float *)deviceBuffer_, tensors_[input], tensors_[weights],
  //                tensors_[bias], tensors_[output], idim.n);
  libdnn->Backward(/* prop_down_data */ true, /* prop_down_weights */ true,
                   (float *)deviceBuffer_,
                   tensors_[CC->getDestGrad()], tensors_[CC->getDestGrad()],
                   // tensors_[CC->getSrc()], tensors_[CC->getSrcGrad()],
                   tensors_[CC->getFilter()], tensors_[CC->getFilterGrad()],
                   tensors_[CC->getBiasGrad()], tensors_[CC->getBiasGrad()],
                   tensors_[CC->getSrc()], tensors_[CC->getSrcGrad()],
                   // tensors_[CC->getDestGrad()], tensors_[CC->getDestGrad()],
                   idim.n);
#if 0
  // Check only if it is a simple kernel of size 1 and stride 1.
  if (CC->getKernel() == 1) {
    llvm::errs() << libdnn->getKernel() << "\n";
    clFinish(commands_);
    llvm::errs() << "Checking OpenCL convolution for correctness\n";
    float *outNew = (float *)alignedAlloc(output->getType()->getSizeInBytes(),
                                          TensorAlignment);
    float *outOld = (float *)alignedAlloc(output->getType()->getSizeInBytes(),
                                          TensorAlignment);
    // Copy output into a temporary host buffer.
    copyValueFromDevice(output, outNew);
    clFinish(commands_);

    // Run the naive kernel
    executeConvolutionAlt(CC);
    clFinish(commands_);
    // Copy output into a temporary host buffer.
    copyValueFromDevice(output, outOld);
    clFinish(commands_);

    Tensor oldT(outOld, output->getType());
    Tensor newT(outNew, output->getType());

    auto oldH = oldT.getHandle<float>();
    auto newH = newT.getHandle<float>();

    // Compare buffers for equality.
    // for (int idx = 0, e = output->getType()->size(); idx < e; ++idx) {
    newH.dump();
    for (size_t n = 0, ne = 1; n < ne; ++n) {
      for (size_t c = 0, ce = 3; c < ce; ++c) {
        for (size_t h = 0, he = 3; h < he; ++h) {
          for (size_t w = 0, we = 3; w < we; ++w) {
            if (oldH.at({n, c, h, w}) != newH.at({n, c, h, w})) {
              llvm::errs() << "Convolution results differ at index (" << n
                           << ", " << c << ", " << h << ", " << w << ") :"
                           << "new = " << newH.at({n, c, h, w}) << " vs "
                           << "old = " << oldH.at({n, c, h, w}) << "\n";
            }
          }
        }
      }
  }
  alignedFree(outNew);
  alignedFree(outOld);
  }
#endif
#else
  auto prog = createProgram(ConvKernel, {}, commands_);
  auto kernel = createKernel("conv_forward_mem", prog);
  setKernelArg(kernel, 0, deviceBuffer_);
  setKernelArg<cl_uint>(kernel, 1, tensors_[input]);
  setKernelArg<cl_uint>(kernel, 2, tensors_[weights]);
  setKernelArg<cl_uint>(kernel, 3, tensors_[bias]);
  setKernelArg<cl_uint>(kernel, 4, tensors_[output]);
  // setKernelLocalArg(kernel, 5, 4*8*sizeof(float));
  // setKernelLocalArg(kernel, 6, 4*8*sizeof(float));
  enqueueKernel(commands_, kernel, deviceId_, {1, 1, 1}, kernelLaunches_);
#endif
}