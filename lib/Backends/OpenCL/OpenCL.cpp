/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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

// Includes the CLBlast library (C interface)
#include <clblast.h>
#include <clblast_c.h>

//#define DEBUG(X) X

using namespace glow;
using llvm::format;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

typedef uint32_t cl_size_t;

// This defines the string "SHADER_CODE".
#include "kernels.cl"

namespace {
llvm::cl::OptionCategory OpenCLBackendCat("Glow OpenCL Backend Options");

static llvm::cl::opt<unsigned>
    deviceId("device", llvm::cl::desc("OpenCL device to be used"),
             llvm::cl::init(0), llvm::cl::cat(OpenCLBackendCat));
static llvm::cl::opt<bool> doProfile("opencl-profile",
                                     llvm::cl::desc("Profile OpenCL kernels"),
                                     llvm::cl::init(false),
                                     llvm::cl::cat(OpenCLBackendCat));
static llvm::cl::opt<bool>
    useClBlast("opencl-clblast",
               llvm::cl::desc("Use CLBLast optimized routines"),
               llvm::cl::init(false), llvm::cl::cat(OpenCLBackendCat));
} // namespace

Backend *glow::createOCLBackend(IRFunction *F) { return new OCLBackend(F); }

using Kind = Kinded::Kind;
using kernelSrcEnum = struct {
  Kind kind;
  const char *funcName;
};

static void dumpCompileLog(cl_device_id dev, cl_program prog) {
#ifndef NDEBUG
  // Determine the size of the log.
  size_t logSize;
  clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

  // Allocate memory for the log.
  auto *log = (char *)malloc(logSize);

  // Get the log.
  clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr);

  // Print the log.
  llvm::outs() << log << "\n";
  free(log);
#endif
}

OCLBackend::OCLBackend(IRFunction *F) : F_(F), allocator_(0xFFFFFFFF) {
  cl_uint num{0};
  cl_int err = clGetDeviceIDs(nullptr, CL_DEVICE_TYPE_ALL, 0, nullptr, &num);
  GLOW_ASSERT(err == CL_SUCCESS && "clGetDeviceIDs Failed.");
  GLOW_ASSERT(num > deviceId &&
              "Should have at least one GPU for running OpenCL");
  cl_device_id devices[num];
  err = clGetDeviceIDs(nullptr, CL_DEVICE_TYPE_ALL, num, devices, nullptr);
  GLOW_ASSERT(err == CL_SUCCESS && "clGetDeviceIDs Failed.");
  device_ = deviceId;
  deviceId_ = devices[deviceId];
  context_ = clCreateContext(nullptr, 1, &deviceId_, nullptr, nullptr, nullptr);
  GLOW_ASSERT(context_ && "clCreateContext Failed.");
  commands_ = clCreateCommandQueue(
      context_, deviceId_, (doProfile) ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
  GLOW_ASSERT(commands_ && "clCreateCommandQueue Failed.");

  err = CL_SUCCESS;
  /// Create the program from the source.
  createProgram(SHADER_CODE, {}, commands_);
}

OCLBackend::~OCLBackend() {
  for (auto &kv : programsCache_) {
    auto prog = kv.second;
    clReleaseProgram(prog);
  }
  clReleaseCommandQueue(commands_);
  clReleaseContext(context_);
  if (deviceBuffer_) {
    freeDeviceBuffer(deviceBuffer_);
    deviceBuffer_ = nullptr;
  }
  clear();
}

static std::string getKernelName(const char *baseName, ElemKind elemTy) {
  std::string name = baseName;
  switch (elemTy) {
  case ElemKind::FloatTy:
    return name + "W";
  case ElemKind::Int8QTy:
    return name + "_i8W";
  case ElemKind::Int32QTy:
    return name + "_i32W";
  case ElemKind::IndexTy:
    return name + "_uW";
  default:
    GLOW_ASSERT("Unsupported element type");
  }
}

cl_kernel OCLBackend::createKernel(const std::string &name,
                                   cl_program program) {
  cl_int err = CL_SUCCESS;
  cl_kernel kernel = nullptr;
  if (program) {
    cl_kernel kernel = clCreateKernel(program, name.c_str(), &err);
    GLOW_ASSERT((kernel && err == CL_SUCCESS) && "clCreateKernel Failed.");
    return kernel;
  }
  // Inspect all programs.
  for (auto &kv : programsCache_) {
    auto prog = kv.second;
    cl_kernel kernel = clCreateKernel(prog, name.c_str(), &err);
    if (err == CL_SUCCESS) {
      return kernel;
    }
  }
  GLOW_ASSERT(kernel && "clCreateKernel Failed.");
  return kernel;
}

cl_program OCLBackend::createProgram(const std::string &source,
                                     const std::vector<std::string> &options,
                                     cl_command_queue queue) {
  cl_int err = CL_SUCCESS;
  const char *src = source.c_str();
  cl_context ctx;
  err = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx,
                              nullptr);
  GLOW_ASSERT(err == CL_SUCCESS && "clGetCommandQueueInfo Failed.");
  cl_device_id deviceId;
  err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(deviceId),
                              &deviceId, nullptr);
  GLOW_ASSERT(err == CL_SUCCESS && "clGetCommandQueueInfo Failed.");

  // Check if this program was compiled with the same parameters for the
  // provided context and device.
  std::string combinedOptions;
  for (auto &opt : options) {
    combinedOptions.append(opt).append(" ");
  }

  ProgramKey key = std::make_tuple(source, combinedOptions, deviceId);
  cl_program &program = programsCache_[key];
  if (program) {
    return program;
  }
  // Create a new compiled program.
  program = clCreateProgramWithSource(context_, 1, &src, nullptr, &err);
  GLOW_ASSERT(program && "clCreateProgramWithSource Failed.");
  err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
  if (err) {
    dumpCompileLog(deviceId, program);
  }
  GLOW_ASSERT(err == CL_SUCCESS && "clBuildProgram Failed.");
  // Add this program to the program cache.
  return program;
}

template <class T>
static void setKernelArg(cl_kernel kernel, unsigned argIdx, T value) {
  cl_int err = clSetKernelArg(kernel, argIdx, sizeof(T), &value);
  GLOW_ASSERT(err == CL_SUCCESS && "Unable to set parameter");
}

static void setKernelLocalArg(cl_kernel kernel, unsigned argIdx, size_t size) {
  cl_int err = clSetKernelArg(kernel, argIdx, size, nullptr);
  GLOW_ASSERT(err == CL_SUCCESS && "Unable to set parameter");
}

void OCLBackend::fillBuffer(cl_mem buffer, size_t start, size_t len,
                            float value, ElemKind elemKind) {
  auto kernel = createKernel(getKernelName("splat", elemKind));
  setKernelArg(kernel, 0, buffer);
  setKernelArg<cl_uint>(kernel, 1, start);
  setKernelArg(kernel, 2, value);
  enqueueKernel(commands_, kernel, deviceId_, {len}, kernelLaunches_);
}

/// \returns the max local workgroup size for each dimension, under the
/// opencl constraints, with the global workgroup sizes of \p global;
void getMaxLocalWorkgroupSize(cl_kernel kernel, cl_device_id device,
                              llvm::ArrayRef<size_t> global,
                              llvm::MutableArrayRef<size_t> local) {

  // Figure out the max size of the workgroup for a given kernel.
  // This size may be lower than a theoretically possible size for a device,
  // because it may depend on the featues of the kernel (e.g. if it uses memory
  // fence instructions)
  size_t L;
  auto err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                      sizeof(L), &L, nullptr);

  DEBUG(llvm::dbgs() << " max workgroup size: " << L << "\n");

  size_t WIS[3];
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(WIS), &WIS,
                  nullptr);
  DEBUG(llvm::dbgs() << " max work item sizes: " << WIS[0] << ", " << WIS[1]
                     << ", " << WIS[2] << "\n");

  size_t WGS;
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(WGS), &WGS,
                  nullptr);
  DEBUG(llvm::errs() << "WGS: " << WGS << "\n");

  GLOW_ASSERT(err == CL_SUCCESS && "Error in clGetKernelWorkGroupInfo.");
  // The global workgroup size must be a multiple of the local workgroup size,
  // and less than the max size for the specific dimension. Also, the
  // multiplication of all dimensions (size of total local work) needs to be
  // less than WSG. In here we find the highest L that divides the global
  // workgroup size. This is our naive implementation of gcd, with the other
  // constraints:
  size_t totalWorkPrevDims = 1;
  for (int i = 0, e = global.size(); i < e; i++) {
    local[i] = L;

    while (global[i] % local[i] || L % local[i] || local[i] > WIS[i] ||
           local[i] * totalWorkPrevDims > WGS) {
      local[i]--;
    }

    // Remember how much work we are doing in this dimension. Use it to make
    // sure that the next dimenstions don't exceed the total allowed workgroup
    // size.
    totalWorkPrevDims *= local[i];
  }
}

/// Enqueue a \p kernel for execution on the command queue \p commands on a
/// given \p device. The information about the launched kernel will be added to
/// \p kernelLaunches list.
void OCLBackend::enqueueKernel(cl_command_queue commands, cl_kernel kernel,
                               cl_device_id device,
                               llvm::ArrayRef<size_t> global,
                               std::vector<KernelLaunch> &kernelLaunches) {
  llvm::SmallVector<size_t, 4> local(global.size(), 0);
  getMaxLocalWorkgroupSize(kernel, device, global, local);
  char kernelName[128];
  size_t retSize;
  cl_int err = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME,
                               sizeof(kernelName), &kernelName, &retSize);
  GLOW_ASSERT(err == CL_SUCCESS && "Error in clGetKernelInfo.");

  cl_event event{nullptr};

  DEBUG(llvm::errs() << "\nEnqueue kernel: " << kernelName << "\n";
        llvm::errs() << "global.size = " << global.size() << " "
                     << "\n";

        for (unsigned i = 0, e = global.size(); i < e; ++i) {
          llvm::errs() << " global[" << i << "] = " << global[i];
          llvm::errs() << "  local[" << i << "] = " << local[i] << "\n";
        });

  err = clEnqueueNDRangeKernel(commands, kernel, global.size(), nullptr,
                               &global[0], &local[0], 0, nullptr,
                               doProfile ? &event : nullptr);
  GLOW_ASSERT(err == CL_SUCCESS && "Error in clEnqueueNDRangeKernel.");
  kernelLaunches.push_back(KernelLaunch(kernel, kernelName, event));
}

/// Analyze and dump the collected profiling information about the execution of
/// OpenCL kernels.
static void dumpProfileInfo(const std::vector<KernelLaunch> &kernelLaunches) {
  if (!doProfile)
    return;
  cl_ulong total = 0;

  std::unordered_map<std::string, cl_ulong> kernelToDuration;

  for (auto &kl : kernelLaunches) {
    auto &event = kl.event_;
    clWaitForEvents(1, &event);
    auto name = kl.name_;
    assert(!name.empty() && "Kernel name cannot be empty");
    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                            sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end),
                            &time_end, NULL);
    // Duration (in nanoseconds).
    double duration = time_end - time_start;
    kernelToDuration[name] += duration;
    total += duration;
    llvm::outs() << "OpenCl execution time for a launch of kernel " << name
                 << format(" is: %0.3f milliseconds\n", duration / 1000000.0);
  }
  llvm::outs() << format(
      "OpenCl total execution time is: %0.3f milliseconds \n",
      total / 1000000.0);

  // Build a sorted list of kernel durations.
  std::vector<std::pair<cl_ulong, std::string>> sortedKernelDurations;
  sortedKernelDurations.reserve(kernelToDuration.size());
  for (auto kv : kernelToDuration) {
    sortedKernelDurations.push_back(std::make_pair(kv.second, kv.first));
  }
  std::sort(sortedKernelDurations.begin(), sortedKernelDurations.end());

  llvm::outs() << "\n\nSummary information per kernel:\n";
  for (auto k : sortedKernelDurations) {
    llvm::outs() << "OpenCl total execution time for kernel " << k.second
                 << format(" is: %0.3f milliseconds (%lu%%)\n",
                           k.first / 1000000.0,
                           (unsigned long)(k.first * 100 / total));
  }
}

#if 1
void OCLBackend::executeConvolutionAlt(OCLConvolutionInst *CC) {
  std::string kernelName = std::string(CC->getKindName()) + "W";
  cl_kernel kernel = createKernel(kernelName);
#if 1
  setKernelArg(kernel, 0, deviceBuffer_);

  unsigned numArgs = CC->getNumOperands();
  for (unsigned arg = 0; arg < numArgs; arg++) {
    setKernelArg<cl_uint>(kernel, arg + 1, tensors_[CC->getOperand(arg).first]);
  }
#else
  cl_int err;
  cl_uint align = 0;
  clGetDeviceInfo(deviceId_, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint),
                  &align, 0);
  llvm::errs() << "CL_DEVICE_MEM_BASE_ADDR_ALIGN = " << align << "\n";
  // Find the smallest offset from all inputs.
  unsigned numArgs = CC->getNumOperands();
  size_t smallestOffset = tensors_[CC->getOperand(0).first];
  for (unsigned arg = 0; arg < numArgs; arg++) {
    smallestOffset =
        std::min(smallestOffset, tensors_[CC->getOperand(arg).first]);
  }
  cl_buffer_region memRegion{smallestOffset, requiredSpace_ - smallestOffset};
  cl_mem mem =
      clCreateSubBuffer(deviceBuffer_, CL_MEM_READ_WRITE,
                        CL_BUFFER_CREATE_TYPE_REGION, &memRegion, &err);
  size_t offset;
  clGetMemObjectInfo(mem, CL_MEM_OFFSET, sizeof(offset), &offset, nullptr);
  llvm::errs() << "smallest offset: " << smallestOffset << "\n";
  llvm::errs() << "offset: " << offset << "\n";
  assert(err == CL_SUCCESS);
  // assert(smallestOffset % align == 0);
  llvm::errs() << "mem: " << (float *)mem << "\n";
  setKernelArg(kernel, 0, mem);
  for (unsigned arg = 0; arg < numArgs; arg++) {
    assert(tensors_[CC->getOperand(arg).first] >= smallestOffset);
    cl_uint offset = tensors_[CC->getOperand(arg).first] - smallestOffset;
    setKernelArg<cl_uint>(kernel, arg + 1, offset);
    llvm::errs() << "Offset: " << offset << "\n";
  }
  llvm::errs() << "Done\n\n";

#endif

  auto odim = ShapeNCHW(CC->getDest()->getType()->dims());
  auto idim = ShapeNCHW(CC->getSrc()->getType()->dims());

  setKernelArg<cl_uint>(kernel, 5, CC->getKernel());
  setKernelArg<cl_uint>(kernel, 6, CC->getStride());
  setKernelArg<cl_uint>(kernel, 7, CC->getPad());
  setKernelArg(kernel, 8, odim);
  setKernelArg(kernel, 9, idim);
  setKernelArg(kernel, 10, ShapeNCHW(CC->getFilter()->getType()->dims()));

  auto depth = odim.c;

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

  // setKernelLocalArg(kernel, 11,
  // CC->getFilter()->getType()->getSizeInBytes());  setKernelLocalArg(kernel,
  // 11, sizeof(float) * 100);

  // Use a 3D grid where the first dimension is the depth and the second
  // dimension is the slice index in the batch.
  enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, depth},
                kernelLaunches_);
}
#endif

#if 1
void OCLBackend::executeConvolution(ConvolutionInst *CC) {
  std::string kernelName =
      getKernelName(CC->getKindName(), CC->getDest()->getElementType());
  cl_kernel kernel = createKernel(kernelName);
  setKernelArg(kernel, 0, deviceBuffer_);

  unsigned numArgs = CC->getNumOperands();
  for (unsigned arg = 0; arg < numArgs; arg++) {
    setKernelArg<cl_uint>(kernel, arg + 1, tensors_[CC->getOperand(arg).first]);
  }

  auto odim = ShapeNHWC(CC->getDest()->getType()->dims());
  auto idim = ShapeNHWC(CC->getSrc()->getType()->dims());

  setKernelArg<cl_uint>(kernel, 5, CC->getKernel());
  setKernelArg<cl_uint>(kernel, 6, CC->getStride());
  setKernelArg<cl_uint>(kernel, 7, CC->getPad());
  setKernelArg(kernel, 8, odim);
  setKernelArg(kernel, 9, idim);
  setKernelArg(kernel, 10, ShapeNHWC(CC->getFilter()->getType()->dims()));

  auto depth = odim.c;

  auto filterDim = ShapeNHWC(CC->getFilter()->getType()->dims());
  DEBUG(llvm::dbgs() << "\n\nKernel (k, s, p): " << CC->getKernel() << " , "
                     << CC->getStride() << " , " << CC->getPad() << "\n";
        llvm::dbgs() << "Filter dims (n, w, h, c, d): " << filterDim.n << " , "
                     << filterDim.w << " , " << filterDim.h << " , "
                     << filterDim.c << " , " << depth << "\n";
        llvm::dbgs() << "Src dims (n, w, h, c): " << idim.n << " , " << idim.w
                     << " , " << idim.h << " , " << idim.c << "\n";
        llvm::dbgs() << "Dest dims (n, w, h, c): " << odim.n << " , " << odim.w
                     << " , " << odim.h << " , " << odim.c << "\n");

  // setKernelLocalArg(kernel, 11,
  // CC->getFilter()->getType()->getSizeInBytes());  setKernelLocalArg(kernel,
  // 11, sizeof(float) * 100);

  // Use a 3D grid where the first dimension is the depth and the second
  // dimension is the slice index in the batch.
  enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, depth},
                kernelLaunches_);
}
#elif 0
void OCLBackend::executeConvolution(ConvolutionInst *CC) {
  std::string kernelName = std::string(CC->getKindName()) + "W";
  // This is a naive implementation that parallelizes using three dims:
  // the X and the Y in the output filter.
  cl_kernel kernel = createKernel(kernelName);
  setKernelArg(kernel, 0, deviceBuffer_);

  unsigned numArgs = CC->getNumOperands();
  for (unsigned arg = 0; arg < numArgs; arg++) {
    setKernelArg<cl_uint>(kernel, arg + 1, tensors_[CC->getOperand(arg).first]);
  }

  auto odim = ShapeNHWC(CC->getDest()->getType()->dims());
  auto idim = ShapeNHWC(CC->getSrc()->getType()->dims());

  setKernelArg<cl_uint>(kernel, 5, CC->getKernel());
  setKernelArg<cl_uint>(kernel, 6, CC->getStride());
  setKernelArg<cl_uint>(kernel, 7, CC->getPad());
  setKernelArg(kernel, 8, odim);
  setKernelArg(kernel, 9, idim);
  setKernelArg(kernel, 10, ShapeNHWC(CC->getFilter()->getType()->dims()));

  auto depth = odim.c;

  auto filterDim = ShapeNHWC(CC->getFilter()->getType()->dims());
  DEBUG(llvm::dbgs() << "\n\nKernel (k, s, p): " << CC->getKernel() << " , "
                     << CC->getStride() << " , " << CC->getPad() << "\n";
        llvm::dbgs() << "Filter dims (n, w, h, c, d): " << filterDim.n << " , "
                     << filterDim.w << " , " << filterDim.h << " , "
                     << filterDim.c << " , " << depth << "\n";
        llvm::dbgs() << "Src dims (n, w, h, c): " << idim.n << " , " << idim.w
                     << " , " << idim.h << " , " << idim.c << "\n";
        llvm::dbgs() << "Dest dims (n, w, h, c): " << odim.n << " , " << odim.w
                     << " , " << odim.h << " , " << odim.c << "\n");

  // setKernelLocalArg(kernel, 11,
  // CC->getFilter()->getType()->getSizeInBytes());  setKernelLocalArg(kernel,
  // 11, sizeof(float) * 100);

  // Use a 3D grid where the first dimension is the depth and the second
  // dimension is the slice index in the batch.
  enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, depth},
                kernelLaunches_);
}
#elif 0

void OCLBackend::executeConvolution(ConvolutionInst *CC) {
  auto *weight = CC->getFilter();
  auto filterDim = ShapeNHWC(weight->getType()->dims());
  auto odim = ShapeNHWC(CC->getDest()->getType()->dims());
  auto idim = ShapeNHWC(CC->getSrc()->getType()->dims());
  int nInputPlane = filterDim.c;
  int nOutputPlane = filterDim.n;
  auto padW = CC->getPad();
  auto padH = CC->getPad();
  auto kernelW = CC->getKernel();
  auto kernelH = CC->getKernel();
  auto strideW = CC->getStride();
  auto strideH = CC->getStride();
  auto *bias = CC->getBias();
  auto *output = CC->getDest();
  auto *input = CC->getSrc();

  long inputWidth = idim.w;
  long inputHeight = idim.h;
  long outputWidth = (inputWidth + 2 * padW - kernelW) / strideW + 1;
  long outputHeight = (inputHeight + 2 * padH - kernelH) / strideH + 1;

  assert(outputWidth >= 1 && outputHeight >= 1 &&
         "Calculated output size is too small");

  // Batch size + input planes.
  long batchSize = idim.n;

  // Resize output
  // auto output = allocDeviceBuffer(batchSize * nOutputPlane * outputHeight *
  //                                outputWidth * sizeof(float));
  assert(output->getType()->size() ==
             batchSize * nOutputPlane * outputHeight * outputWidth &&
         "Wrong ouput tensor size provided for convolution");

  // Resize temporary columns
  auto columns =
      allocDeviceBuffer((nInputPlane * kernelW * kernelH) *
                        (outputHeight * outputWidth) * sizeof(float));

  // Define a buffer of ones, for bias accumulation.
  // Note: this buffer can be shared with other modules, it only ever gets
  // increased, and always contains ones.
  if (ones->nDimension != 2 ||
      ones->size[0] * ones->size[1] < outputHeight * outputWidth) {
    // Resize plane and fill with ones...
    THClTensor_resize2d(state, ones, outputHeight, outputWidth);
    THClTensor_fill(state, ones, 1);
  }

  // Input and output temporary vectors.
  // input_n and output_n are just views/aliases. They do not own any memory.
  auto input_n = allocDeviceBuffer();
  auto output_n = allocDeviceBuffer();

  // For each element in batch, do:
  for (int elt = 0; elt < batchSize; elt++) {
    // Matrix multiply per output:
    // Select the elt'th element of the input batch.
    THClTensor_select(state, input_n, input, 0, elt);
    // Select the elt'th element of the ouput batch.
    THClTensor_select(state, output_n, output, 0, elt);

    // Do Bias first:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/clblas/#clblas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long n_ = outputHeight * outputWidth;
    long k_ = 1;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major
    // matrices)
    // Bias is m x k, i.e. m x 1
    THClBlas_gemm(state, 't', 'n', n_, m_, k_, 1, ones, k_, bias, k_, 0,
                  output_n, n_);

    // Extract columns:
    im2col(state, input_n, nInputPlane, inputHeight, inputWidth, kernelH,
           kernelW, padH, padW, strideH, strideW, columns);

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/clblas/#clblas-lt-t-gt-gemm)
    long m = nOutputPlane;
    long n = columns->size[1];
    long k = nInputPlane * kH * kW;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major
    // matrices)
    THClBlas_gemm(state, 'n', 'n', n, m, k, 1, columns, n, weight, k,
                  // 1 because output_n contains bias already.
                  1, output_n, n);
  }

  // Free temporaries.
  THClTensor_free(state, input_n);
  THClTensor_free(state, output_n);
}

void run_im2col(THClState *state, THClTensor *im, const int channels,
                const int height, const int width, const int ksize_h,
                const int ksize_w, const int pad_h, const int pad_w,
                const int stride_h, const int stride_w, THClTensor *col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - ksize_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - ksize_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;

  cl_kernel kernel = createKernel(program_, "im2col");

  size_t argNum{0};

  setKernelArg<cl_uint>(kernel, argNum++, num_kernels);
  setKernelArg<cl_uint>(kernel, argNum++, im);
  setKernelArg<cl_uint>(kernel, argNum++, height);
  setKernelArg<cl_uint>(kernel, argNum++, width);
  setKernelArg<cl_uint>(kernel, argNum++, ksize_h);
  setKernelArg<cl_uint>(kernel, argNum++, ksize_w);
  setKernelArg<cl_uint>(kernel, argNum++, pad_h);
  setKernelArg<cl_uint>(kernel, argNum++, pad_w);
  setKernelArg<cl_uint>(kernel, argNum++, stride_h);
  setKernelArg<cl_uint>(kernel, argNum++, stride_w);
  setKernelArg<cl_uint>(kernel, argNum++, height_col);
  setKernelArg<cl_uint>(kernel, argNum++, width_col);
  // This is an out parameter.
  setKernelArg<cl_uint>(kernel, argNum++, col);
  k.run(GET_BLOCKS(state, num_kernels), getNumThreads(state));
}

void run_col2im(THClState *state, THClTensor *col, const int channels,
                const int height, const int width, const int patch_h,
                const int patch_w, const int pad_h, const int pad_w,
                const int stride_h, const int stride_w, THClTensor *im) {
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.

  cl_kernel kernel = createKernel(program_, "col2im");

  size_t argNum{0};
  setKernelArg<cl_uint>(kernel, argNum++, num_kernels);
  setKernelArg<cl_uint>(kernel, argNum++, col);
  setKernelArg<cl_uint>(kernel, argNum++, height);
  setKernelArg<cl_uint>(kernel, argNum++, width);
  setKernelArg<cl_uint>(kernel, argNum++, channels);
  setKernelArg<cl_uint>(kernel, argNum++, channels);
  setKernelArg<cl_uint>(kernel, argNum++, patch_h);
  setKernelArg<cl_uint>(kernel, argNum++, patch_w);
  setKernelArg<cl_uint>(kernel, argNum++, pad_h);
  setKernelArg<cl_uint>(kernel, argNum++, pad_w);
  setKernelArg<cl_uint>(kernel, argNum++, stride_h);
  setKernelArg<cl_uint>(kernel, argNum++, stride_w);
  setKernelArg<cl_uint>(kernel, argNum++, height_col);
  setKernelArg<cl_uint>(kernel, argNum++, width_col);
  // This is an out parameter.
  setKernelArg<cl_uint>(kernel, argNum++, im);

  k.run(GET_BLOCKS(state, num_kernels), getNumThreads(state));
}
#endif

void OCLBackend::doForwardPass() {
  // F_->dumpDAG();
  auto copiedToDeviceBytes = copyMutableWeightsToDevice();
  DEBUG(llvm::dbgs() << "Copied " << copiedToDeviceBytes
                     << " bytes to OpenCL device\n");

  for (auto &I : F_->getInstrs()) {
    // The kernels are named after the name of the instruction, plus the "W"
    // suffix to prevent name colissions for functions like 'tanh' that are also
    // a part of the OpenCL runtime.
    auto elemTy = I->getNumOperands() ? I->getOperand(0).first->getElementType()
                                      : ElemKind::FloatTy;
    std::string kernelName = getKernelName(I->getKindName(), elemTy);

    // Skip memory allocation instructions as they are NOPs.
    if (isa<AllocActivationInst>(I) || isa<DeallocActivationInst>(I) ||
        isa<TensorViewInst>(I)) {
      continue;
    }

    // Element-wise operations, except the copy instruction.
    if (I->isDataParallel() && !isa<CopyInst>(I)) {
      // Figure out how many element-wise elements are there to process:
      size_t global;
      if (I->isDataParallel()) {
        global = I->getOperand(0).first->getType()->size();
#if 0
        if (global % 16 == 0) {
          // Start less kernels and let each kernel do more work using vector
          // instructions.
          global /= 16;
          kernelName += "16";
        } else
            if (global % 8 == 0) {
          // Start less kernels and let each kernel do more work using vector
          // instructions.
          global /= 8;
          kernelName += "8";
        }
#endif
      } else {
        GLOW_UNREACHABLE("Invalid instruction.");
      }

      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);
      unsigned numArgs = I->getNumOperands();

      for (unsigned arg = 0, e = I->getNumOperands(); arg < e; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I->getOperand(arg).first]);
      }

      if (auto *SI = dyn_cast<SplatInst>(I)) {
        // Pass the splat as a parameter.
        setKernelArg(kernel, numArgs + 1, SI->getValue());
      }

      enqueueKernel(commands_, kernel, deviceId_, {global}, kernelLaunches_);
      continue;
    }

    if (auto *SM = dyn_cast<SoftMaxInst>(I)) {
      // Implement Softmax by parallelizing the batch dimension. Each sample in
      // the batch is processed by a different parallel 'thread'.
      cl_kernel kernel = createKernel(kernelName);

      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I->getOperand(arg).first]);
      }

      // This is the number of elements for each slice. There are N slices in
      // our batch.
      auto inputDims = SM->getSrc()->getType()->dims();
      size_t numSlices = inputDims[0];

      // Pass the slice size (size of each sample in the batch) as a parameter.
      setKernelArg<cl_uint>(kernel, numArgs + 1, flattenCdr(inputDims).second);

      enqueueKernel(commands_, kernel, deviceId_, {numSlices}, kernelLaunches_);
      continue;
    }

    if (auto *SM = dyn_cast<SoftMaxGradInst>(I)) {
      // Implement Softmax by parallelizing the batch dimension. Each sample in
      // the batch is processed by a different parallel 'thread'.
      cl_kernel kernel = createKernel(kernelName);

      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I->getOperand(arg).first]);
      }

      // This is the number of elements for each slice. There are N slices in
      // our batch.
      auto inputDims = SM->getSrcGrad()->getType()->dims();
      size_t numSlices = inputDims[0];

      // Pass the slice size (size of each sample in the batch) as a parameter.
      setKernelArg<cl_uint>(kernel, numArgs + 1, flattenCdr(inputDims).second);

      enqueueKernel(commands_, kernel, deviceId_, {numSlices}, kernelLaunches_);
      continue;
    }

    if (auto *IT = dyn_cast<InsertTensorInst>(I)) {
      // assert(IT->getDest()->getElementType() == ElemKind::FloatTy &&
      // "Unexpected element kind");
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I->getOperand(arg).first]);
      }

      // Currently support tensors of 2 and 4 dimensions.
      // TODO: Handle other dimensions.
      const size_t numDimensions = IT->getDest()->getType()->dims().size();
      ShapeNHWC odim = ShapeNHWC::empty();
      ShapeNHWC idim = ShapeNHWC::empty();
      ShapeNHWC offset = ShapeNHWC::empty();

      if (numDimensions == 4) {
        odim = ShapeNHWC(IT->getDest()->getType()->dims());
        idim = ShapeNHWC(IT->getSrc()->getType()->dims());
        offset = ShapeNHWC(IT->getOffsets());
      } else if (numDimensions == 2) {
        odim = ShapeNHWC::fromXY(IT->getDest()->getType()->dims());
        idim = ShapeNHWC::fromXY(IT->getSrc()->getType()->dims());
        offset = ShapeNHWC::fromXY(IT->getOffsets());
      } else if (numDimensions == 3) {
        odim = ShapeNHWC::fromXYZ(IT->getDest()->getType()->dims());
        idim = ShapeNHWC::fromXYZ(IT->getSrc()->getType()->dims());
        offset = ShapeNHWC::fromXYZ(IT->getOffsets());
      } else if (numDimensions == 1) {
        odim = ShapeNHWC::fromX(IT->getDest()->getType()->dims());
        idim = ShapeNHWC::fromX(IT->getSrc()->getType()->dims());
        offset = ShapeNHWC::fromX(IT->getOffsets());
      } else {
        assert(false && "Unsupported tensor dimension");
      }

      setKernelArg(kernel, 3, odim);
      setKernelArg(kernel, 4, idim);
      setKernelArg(kernel, 5, offset);
      // llvm::errs() << "\n\ninserttensor odim (n, h, w, c): " << odim.n << ",
      // " << odim.h << ", " << odim.w << ", " << odim.c << "\n";  llvm::errs()
      // << "inserttensor idim (n, h, w, c): " << idim.n << ", " << idim.h << ",
      // "
      // << idim.w << ", " << idim.c << "\n";  llvm::errs() << "inserttensor
      // offset (n, h, w, c): " << offset.n << ", " << offset.h << ", " <<
      // offset.w << ", " << offset.c << "\n";
      enqueueKernel(commands_, kernel, deviceId_, {idim.n}, kernelLaunches_);
      continue;
    }

    if (auto *ET = dyn_cast<ExtractTensorInst>(I)) {
      // assert(ET->getDest()->getElementType() == ElemKind::FloatTy &&
      // "Unexpected element kind");
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I->getOperand(arg).first]);
      }

      // Currently support tensors of 2 and 4 dimensions.
      // TODO: Handle other dimensions.
      const size_t numDimensions = ET->getDest()->getType()->dims().size();
      ShapeNHWC odim = ShapeNHWC::empty();
      ShapeNHWC idim = ShapeNHWC::empty();
      ShapeNHWC offset = ShapeNHWC::empty();

      if (numDimensions == 4) {
        odim = ShapeNHWC(ET->getDest()->getType()->dims());
        idim = ShapeNHWC(ET->getSrc()->getType()->dims());
        offset = ShapeNHWC(ET->getOffsets());
      } else if (numDimensions == 2) {
        odim = ShapeNHWC::fromXY(ET->getDest()->getType()->dims());
        idim = ShapeNHWC::fromXY(ET->getSrc()->getType()->dims());
        offset = ShapeNHWC::fromXY(ET->getOffsets());
      } else if (numDimensions == 3) {
        odim = ShapeNHWC::fromXYZ(ET->getDest()->getType()->dims());
        idim = ShapeNHWC::fromXYZ(ET->getSrc()->getType()->dims());
        offset = ShapeNHWC::fromXYZ(ET->getOffsets());
      } else {
        assert(false && "Unsupported tensor dimension");
      }

      setKernelArg(kernel, 3, odim);
      setKernelArg(kernel, 4, idim);
      setKernelArg(kernel, 5, offset);
      // llvm::errs() << "\n\nextracttensor odim (n, h, w, c): " << odim.n << ",
      // " << odim.h << ", " << odim.w << ", " << odim.c << "\n";  llvm::errs()
      // << "extracttensor idim (n, h, w, c): " << idim.n << ", " << idim.h <<
      // ", "
      // << idim.w << ", " << idim.c << "\n";  llvm::errs() << "extracttensor
      // offset (n, h, w, c): " << offset.n << ", " << offset.h << ", " <<
      // offset.w << ", " << offset.c << "\n";
      enqueueKernel(commands_, kernel, deviceId_, {odim.n}, kernelLaunches_);
      continue;
    }

    if (auto *BMM = dyn_cast<MatMulInst>(I)) {
#if 0
      // m is number of rows in matrix A.
      const size_t m = BMM->getDest()->dims()[0];
      const size_t n = BMM->getDest()->dims()[1];
      const size_t k = BMM->getRHS()->dims()[0];
#endif
      // opA(A) is of dimension m×k and opB(B) is of dimension k×n,
      const size_t m = BMM->getLHS()->dims()[0];
      const size_t n = BMM->getRHS()->dims()[1];
      const size_t k = BMM->getLHS()->dims()[1];
      const size_t colA = BMM->getLHS()->dims()[1];
      const size_t rowA = BMM->getLHS()->dims()[0];
      const size_t colB = BMM->getRHS()->dims()[1];

      DEBUG(llvm::dbgs() << "About to perform MatMul: "
                         << " m = " << m << " n = " << n << " k = " << k
                         << "\n");
      if (!useClBlast) {
        // This is a naive implementation that parallelizes using three dims:
        // batch, X and Y in the output filter.
        cl_kernel kernel = createKernel(kernelName);
        setKernelArg(kernel, 0, deviceBuffer_);

        unsigned numArgs = I->getNumOperands();
        for (unsigned arg = 0; arg < numArgs; arg++) {
          setKernelArg<cl_uint>(kernel, arg + 1,
                                tensors_[I->getOperand(arg).first]);
        }

        auto ddim = ShapeNHWC::fromXY(BMM->getDest()->getType()->dims());
        auto ldim = ShapeNHWC::fromXY(BMM->getLHS()->getType()->dims());
        auto rdim = ShapeNHWC::fromXY(BMM->getRHS()->getType()->dims());

        setKernelArg(kernel, 4, ddim);
        setKernelArg(kernel, 5, ldim);
        setKernelArg(kernel, 6, rdim);

        // Use a 3D grid where the first dimension is the N and the second and
        // third dimensions are the X and Y in the output buffer.
        enqueueKernel(commands_, kernel, deviceId_, {ddim.n, ddim.h, ddim.w},
                      kernelLaunches_);
      } else {
        // Call the SGEMM routine.
        const float alpha = 1.0;
        const float beta = 0.0;
        cl_event event;
        if (false && (m <= 16 || n <= 16 || k <= 16)) {
          // KWID=2 MDIMAD=8 MDIMCD=8 NDIMBD=8 NDIMCD=8 PADA=1 PADB=1
          // PRECISION=32 VWMD=1 VWND=1 WGD=8
          const char *paramNames[] = {"KWID",   "MDIMAD", "MDIMCD", "NDIMBD",
                                      "NDIMCD", "PADA",   "PADB",   "PRECISION",
                                      "VWMD",   "VWND",   "WGD"};
          size_t paramValues[] = {2, 8, 8, 8, 8, 1, 1, 32, 1, 1, 8};
          CLBlastOverrideParameters(deviceId_, "Xgemm", CLBlastPrecisionSingle,
                                    11, paramNames, paramValues);
          CLBlastOverrideParameters(deviceId_, "XgemmDirect",
                                    CLBlastPrecisionSingle, 11, paramNames,
                                    paramValues);
        }
        DEBUG(llvm::dbgs() << "\n\nCalling CLBlastSgemm\n\n");
        {
          // KWG=32 KWI=2 MDIMA=16 MDIMC=16 MWG=32 NDIMB=8 NDIMC=8 NWG=32
          // PRECISION=32 SA=1 SB=1 STRM=0 STRN=0 VWM=1 VWN=2
          const char *paramNames[] = {"KWG",       "KWI",   "MDIMA", "MDIMC",
                                      "MWG",       "NDIMB", "NDIMC", "NWG",
                                      "PRECISION", "SA",    "SB",    "STRM",
                                      "STRN",      "VWM",   "VWN"};
          size_t paramValues[] = {32, 2, 16, 16, 32, 8, 8, 32,
                                  32, 1, 1,  0,  0,  1, 2};
          CLBlastOverrideParameters(deviceId_, "Xgemm", CLBlastPrecisionSingle,
                                    15, paramNames, paramValues);
        }
#if 0
        CLBlastStatusCode status =
            CLBlastSgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo,
                         CLBlastTransposeNo, m, n, k, alpha, lhs, 0, a_ld, rhs,
                         0, b_ld, beta, dest, 0, c_ld, &commands_, nullptr);
#endif

        CLBlastStatusCode status = CLBlastSgemm(
            CLBlastLayoutColMajor, CLBlastTransposeNo, CLBlastTransposeNo, colB,
            rowA, colA, alpha, deviceBuffer_,
            tensors_[BMM->getRHS()] / sizeof(float), colB, deviceBuffer_,
            tensors_[BMM->getLHS()] / sizeof(float), colA, beta, deviceBuffer_,
            tensors_[BMM->getDest()] / sizeof(float), colB, &commands_, &event);

        GLOW_ASSERT(status == CLBlastSuccess && "Failed CLBlastSgemm");
        kernelLaunches_.push_back(KernelLaunch("matmul_clblast", event));
      }
      continue;
    }

    if (auto *BA = dyn_cast<BatchedAddInst>(I)) {
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I->getOperand(arg).first]);
      }

      auto bdim = flattenCdr(BA->getBatch()->dims());
      setKernelArg<cl_uint>(kernel, 4, bdim.first);
      setKernelArg<cl_uint>(kernel, 5, bdim.second);

      // Parallelize on each element in the slice.
      enqueueKernel(commands_, kernel, deviceId_, {bdim.second},
                    kernelLaunches_);
      continue;
    }

    if (auto *BRA = dyn_cast<BatchedReduceAddInst>(I)) {
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I->getOperand(arg).first]);
      }

      auto bdim = flattenCdr(BRA->getBatch()->dims());
      setKernelArg<cl_uint>(kernel, 3, bdim.first);
      setKernelArg<cl_uint>(kernel, 4, bdim.second);

      // Parallelize on each element in the slice.
      enqueueKernel(commands_, kernel, deviceId_, {bdim.second},
                    kernelLaunches_);
      continue;
    }

    if (auto CC = dyn_cast<ConvolutionInst>(I)) {
      executeConvolution(CC);
      continue;
    }

    if (auto CC = dyn_cast<OCLConvolutionInst>(I)) {
      executeConvolution(CC);
      continue;
    }

    if (auto CG = dyn_cast<ConvolutionGradInst>(I)) {
      executeConvolutionGrad(CG);
      continue;
      auto *src = CG->getSrc();
      auto *filter = CG->getFilter();
      auto *destGrad = CG->getDestGrad();
      auto *srcGrad = CG->getSrcGrad();
      auto *filterGrad = CG->getFilterGrad();
      auto *biasGrad = CG->getBiasGrad();
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = CG->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[CG->getOperand(arg).first]);
      }

      auto srcGradDim = ShapeNHWC(srcGrad->dims());
      auto destGradDim = ShapeNHWC(destGrad->dims());
      auto srcDim = ShapeNHWC(src->dims());
      auto filterGradDim = ShapeNHWC(filterGrad->dims());

      setKernelArg<cl_uint>(kernel, numArgs + 1, CG->getKernel());
      setKernelArg<cl_uint>(kernel, numArgs + 2, CG->getStride());
      setKernelArg<cl_uint>(kernel, numArgs + 3, CG->getPad());
      setKernelArg(kernel, numArgs + 4, srcDim);
      setKernelArg(kernel, numArgs + 5, destGradDim);
      setKernelArg(kernel, numArgs + 6, filterGradDim);

      auto depth = destGradDim.c;

      // setKernelLocalArg(kernel, 11,
      // CC->getFilter()->getType()->getSizeInBytes());
      // setKernelLocalArg(kernel, 11, sizeof(float) * 100);

      // Zero memory.
      fillBuffer(deviceBuffer_, tensors_[srcGrad], srcGrad->size(), 0,
                 srcGrad->getElementType());
      fillBuffer(deviceBuffer_, tensors_[filterGrad], filterGrad->size(), 0,
                 filterGrad->getElementType());
      fillBuffer(deviceBuffer_, tensors_[biasGrad], biasGrad->size(), 0,
                 biasGrad->getElementType());
      // clFinish(commands_);

      assert(filter->dims() == filterGrad->dims() && "Dims should be the same");
      assert(src->dims() == srcGrad->dims() && "Dims should be the same");

      // Use a 3D grid where the first dimension is the depth and the second
      // dimension is the slice index in the batch.
      enqueueKernel(commands_, kernel, deviceId_,
                    {destGradDim.h, destGradDim.w, depth}, kernelLaunches_);
      // enqueueKernel(commands_, kernel, deviceId_,
      //              {1}, kernelLaunches_);
      continue;
    }

    if (auto PA = dyn_cast<OCLPoolAvgInst>(I)) {
      // This is a naive implementation that parallelizes using three dims:
      // the X and the Y in the output filter.
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I->getOperand(arg).first]);
      }

      auto odim = ShapeNCHW(PA->getDest()->getType()->dims());
      auto idim = ShapeNCHW(PA->getSrc()->getType()->dims());

      setKernelArg<cl_uint>(kernel, 3, PA->getKernel());
      setKernelArg<cl_uint>(kernel, 4, PA->getStride());
      setKernelArg<cl_uint>(kernel, 5, PA->getPad());
      setKernelArg(kernel, 6, odim);
      setKernelArg(kernel, 7, idim);

      enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, odim.c},
                    kernelLaunches_);
      continue;
    }

    if (auto *PM = dyn_cast<OCLPoolMaxInst>(I)) {
      // This is a naive implementation that parallelizes using three dims:
      // the X and the Y in the output filter.
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I->getOperand(arg).first]);
      }

      auto odim = ShapeNCHW(PM->getDest()->getType()->dims());
      auto idim = ShapeNCHW(PM->getSrc()->getType()->dims());

      setKernelArg<cl_uint>(kernel, numArgs + 1, PM->getKernel());
      setKernelArg<cl_uint>(kernel, numArgs + 2, PM->getStride());
      setKernelArg<cl_uint>(kernel, numArgs + 3, PM->getPad());
      setKernelArg(kernel, numArgs + 4, odim);
      setKernelArg(kernel, numArgs + 5, idim);

      enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, odim.c},
                    kernelLaunches_);
      continue;
    }

    if (auto *PM = dyn_cast<PoolMaxInst>(I)) {
      // This is a naive implementation that parallelizes using three dims:
      // the X and the Y in the output filter.
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I->getOperand(arg).first]);
      }

      auto odim = ShapeNHWC(PM->getDest()->getType()->dims());
      auto idim = ShapeNHWC(PM->getSrc()->getType()->dims());

      setKernelArg<cl_uint>(kernel, numArgs + 1, PM->getKernel());
      setKernelArg<cl_uint>(kernel, numArgs + 2, PM->getStride());
      setKernelArg<cl_uint>(kernel, numArgs + 3, PM->getPad());
      setKernelArg(kernel, numArgs + 4, odim);
      setKernelArg(kernel, numArgs + 5, idim);

      enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, odim.c},
                    kernelLaunches_);
      continue;
    }

    if (auto *PM = dyn_cast<PoolMaxWithXYInst>(I)) {
      // This is a naive implementation that parallelizes using three dims:
      // the X and the Y in the output filter.
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I->getOperand(arg).first]);
      }

      auto odim = ShapeNHWC(PM->getDest()->getType()->dims());
      auto idim = ShapeNHWC(PM->getSrc()->getType()->dims());

      setKernelArg<size_t>(kernel, numArgs + 1, PM->getKernel());
      setKernelArg<cl_uint>(kernel, numArgs + 2, PM->getStride());
      setKernelArg<cl_uint>(kernel, numArgs + 3, PM->getPad());
      setKernelArg(kernel, numArgs + 4, odim);
      setKernelArg(kernel, numArgs + 5, idim);

      enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, odim.c},
                    kernelLaunches_);
      continue;
    }

    if (auto *PMG = dyn_cast<PoolMaxWithXYGradInst>(I)) {
      // This is a naive implementation that parallelizes using three dims:
      // the X and the Y in the output filter.
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I->getOperand(arg).first]);
      }

      auto destGradDim = ShapeNHWC(PMG->getDestGrad()->dims());
      auto srcGradDim = ShapeNHWC(PMG->getSrcGrad()->dims());

      setKernelArg<size_t>(kernel, numArgs + 1, PMG->getKernel());
      setKernelArg<cl_uint>(kernel, numArgs + 2, PMG->getStride());
      setKernelArg<cl_uint>(kernel, numArgs + 3, PMG->getPad());
      setKernelArg(kernel, numArgs + 4, srcGradDim);
      setKernelArg(kernel, numArgs + 5, destGradDim);

      assert(srcGradDim.n == destGradDim.n && "batch size is wrong");
      assert(srcGradDim.c == destGradDim.c && "depth size is wrong");

      enqueueKernel(commands_, kernel, deviceId_, {srcGradDim.n},
                    kernelLaunches_);
      continue;
    }

    if (auto *PA = dyn_cast<PoolAvgInst>(I)) {
      // This is a naive implementation that parallelizes using three dims:
      // the X and the Y in the output filter.
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I->getOperand(arg).first]);
      }

      auto odim = ShapeNHWC(PA->getDest()->getType()->dims());
      auto idim = ShapeNHWC(PA->getSrc()->getType()->dims());

      setKernelArg<cl_uint>(kernel, 3, PA->getKernel());
      setKernelArg<cl_uint>(kernel, 4, PA->getStride());
      setKernelArg<cl_uint>(kernel, 5, PA->getPad());
      setKernelArg(kernel, 6, odim);
      setKernelArg(kernel, 7, idim);

      enqueueKernel(commands_, kernel, deviceId_, {odim.h, odim.w, odim.c},
                    kernelLaunches_);
      continue;
    }

    if (auto *TR = dyn_cast<TransposeInst>(I)) {
      // assert(TR->getDest()->getElementType() == ElemKind::FloatTy &&
      // "Unexpected element kind");
      // This is a naive implementation that parallelizes using one dimension,
      // the N (batch size).
      GLOW_ASSERT(TR->getShuffle().size() <= 4 &&
                  "This code supports only 4 and lower dimensional transposes");

      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I->getOperand(arg).first]);
      }

      // Temporary hack to support 3-dim transposes.
      // TODO: support any dimensional transposes.
      std::vector<size_t> odim_vec = TR->getDest()->getType()->dims();
      std::vector<size_t> idim_vec = TR->getSrc()->getType()->dims();
      std::vector<unsigned> mask = TR->getShuffle();
      while (mask.size() < 4) {
        odim_vec.push_back(1);
        idim_vec.push_back(1);
        mask.push_back(mask.size());
        continue;
      }
      DEBUG(llvm::dbgs() << "Transpose: "; for (auto m
                                                : mask) {
        llvm::dbgs() << m << " ";
      } llvm::dbgs() << "\n");

      // printf("\n\ntranspose\n");
      // llvm::errs() << "\n\ntranspose odim (n, h, w, c): " << odim_vec[0] <<
      // ", " << odim_vec[1] << ", " << odim_vec[2] << ", " << odim_vec[3] <<
      // "\n";  llvm::errs() << "transpose idim: " << idim_vec[0] << ", " <<
      // idim_vec[1] << ", " << idim_vec[2] << ", " << idim_vec[3] << "\n";
      // llvm::errs() << "transpose mask: " << mask[0] << ", " << mask[1] << ",
      // " << mask[2] << ", " << mask[3] << "\n";

      auto odim = ShapeNHWC(odim_vec);
      auto idim = ShapeNHWC(idim_vec);

      setKernelArg(kernel, 3, odim);
      setKernelArg(kernel, 4, idim);

      ShapeNHWC shuff(mask[0], mask[1], mask[2], mask[3]);
      setKernelArg(kernel, 5, shuff);

      enqueueKernel(commands_, kernel, deviceId_, {idim.n}, kernelLaunches_);
      continue;
    }

    if (auto *TV = dyn_cast<TensorViewInst>(I)) {
      assert(tensors_[TV] == tensors_[TV->getSrc()] &&
             "Memory address for a tensor_view should be the same as the "
             "address of its origin");
      (void)TV;
      continue;
    }

    if (auto *C = dyn_cast<CopyInst>(I)) {
      Value *dest, *src;
      dest = C->getDest();
      src = C->getSrc();
      if (src == dest) {
        continue;
      }
      size_t destOff = tensors_[dest];
      size_t srcOff = tensors_[src];
      size_t sizeInBytes = dest->getSizeInBytes();

      cl_int err =
          clEnqueueCopyBuffer(commands_, deviceBuffer_, deviceBuffer_, srcOff,
                              destOff, sizeInBytes, 0, nullptr, nullptr);
      GLOW_ASSERT(err == CL_SUCCESS && "Error in clEnqueueCopyBuffer.");
      continue;
    }

    if (auto *GI = dyn_cast<GatherInst>(I)) {
      cl_kernel kernel = createKernel(kernelName);
      setKernelArg(kernel, 0, deviceBuffer_);

      unsigned numArgs = I->getNumOperands();
      for (unsigned arg = 0; arg < numArgs; arg++) {
        setKernelArg<cl_uint>(kernel, arg + 1,
                              tensors_[I->getOperand(arg).first]);
      }

      auto *data = GI->getData();
      size_t dataSliceSize =
          data->size() / data->dims()[0] * data->getType()->getElementSize();
      size_t numIndices = GI->getIndices()->size();
      setKernelArg<cl_uint>(kernel, numArgs + 1, numIndices);
      setKernelArg<cl_uint>(kernel, numArgs + 2, dataSliceSize);

      enqueueKernel(commands_, kernel, deviceId_, {numIndices},
                    kernelLaunches_);
      continue;
    }

    if (auto *TK = dyn_cast<TopKInst>(I)) {
      Tensor outW(TK->getValues()->getType());
      Tensor indW(TK->getIndices()->getType());
      Tensor inW(TK->getInput()->getType());
      size_t k = TK->getK();
      auto values = outW.getHandle<float>();
      auto indices = indW.getHandle<size_t>();
      auto in = inW.getHandle<float>();
      size_t n = in.dims().back();

      // Copy current values into host memory.
      clFinish(commands_);
      copyValueFromDevice(TK->getValues(), outW.getUnsafePtr());
      copyValueFromDevice(TK->getIndices(), indW.getUnsafePtr());
      copyValueFromDevice(TK->getInput(), inW.getUnsafePtr());
      clFinish(commands_);

      size_t in_p = 0, out_p = 0;
      size_t tensor_end = in.size();
      using pairType = std::pair<float, size_t>;
      std::vector<pairType> buf(n);

      while (in_p < tensor_end) {
        for (size_t i = 0; i < n; i++) {
          buf[i].first = in.raw(in_p++);
          buf[i].second = i;
        }
        // NOTE: it's possible to do N + KlogK, while this version is NlogN
        std::sort(buf.begin(), buf.end(),
                  [](const pairType &a, const pairType &b) {
                    if (a.first != b.first)
                      return a.first > b.first;
                    return a.second < b.second;
                  });
        for (size_t i = 0; i < k; i++) {
          values.raw(out_p) = buf[i].first;
          indices.raw(out_p) = buf[i].second;
          out_p++;
        }
      }
      // Copy value back to device.
      copyValueToDevice(TK->getValues(), outW.getUnsafePtr());
      copyValueToDevice(TK->getIndices(), indW.getUnsafePtr());
      copyValueToDevice(TK->getInput(), inW.getUnsafePtr());
      clFinish(commands_);
      continue;
    }

    if (auto *DP = dyn_cast<DebugPrintInst>(I)) {
      clFinish(commands_);
      auto *V = DP->getSrc();
      // Allocate a temporary tensor to hold the value.
      // TODO: No need to allocate the tensor for external tensors?
      Tensor T(V->getType());
      // Load the current value of the variable into host memory.
      copyValueFromDevice(V, T.getUnsafePtr());
      clFinish(commands_);
      llvm::outs() << I->getName() << ": ";
      // Dump the content of a value.
      V->dump();
      llvm::outs() << "\n";
      dumpImpl(&T);
      llvm::outs() << "\n";
      llvm::outs().flush();
      continue;
    }

    llvm::errs() << "Cannot select: " << I->getKindName() << "\n";
    GLOW_UNREACHABLE("compilation failed");
  }

  clFinish(commands_);

  // Output profiling information.
  dumpProfileInfo(kernelLaunches_);

  for (auto &kl : kernelLaunches_) {
    clReleaseKernel(kl.kernel_);
  }
  kernelLaunches_.clear();

  auto copiedFromDeviceBytes = copyMutableWeightsFromDevice();
  DEBUG(llvm::dbgs() << "Copied " << copiedFromDeviceBytes
                     << " bytes from OpenCL device\n");
}

size_t OCLBackend::copyValueToDevice(const Value *v, void *buf) {
  size_t copiedBytes = 0;
  auto it = tensors_.find(v);
  assert(it != tensors_.end() && "Unknown value");
  size_t sizeInBytes = v->getType()->getSizeInBytes();
  // Issue a non-blocking command to copy the buffer to the device.
  if (sizeInBytes) {
    if (!buf) {
      Tensor *T = externalTensors_[v];
      assert(T && "Expectded an external tensor");
      buf = T->getUnsafePtr();
    }
    cl_int err =
        clEnqueueWriteBuffer(commands_, deviceBuffer_, CL_FALSE, it->second,
                             sizeInBytes, buf, 0, nullptr, nullptr);
    GLOW_ASSERT(err == CL_SUCCESS && "Unable to copy data to the device");
    copiedBytes += sizeInBytes;
  }
  return copiedBytes;
}

size_t OCLBackend::copyValueFromDevice(const Value *v, void *buf) {
  size_t copiedBytes = 0;
  auto it = tensors_.find(v);
  assert(it != tensors_.end() && "Unknown value");
  size_t sizeInBytes = v->getType()->getSizeInBytes();
  // Issue a non-blocking command to copy the buffer from the device.
  if (sizeInBytes) {
    if (!buf) {
      Tensor *T = externalTensors_[v];
      assert(T && "Expectded an external tensor");
      buf = T->getUnsafePtr();
    }
    cl_int err =
        clEnqueueReadBuffer(commands_, deviceBuffer_, CL_FALSE, it->second,
                            sizeInBytes, buf, 0, nullptr, nullptr);
    GLOW_ASSERT(err == CL_SUCCESS && "Unable to copy from the device");
    DEBUG(llvm::dbgs() << "Copied the value from device: "
                       << it->first->getName() << "\n");
    copiedBytes += sizeInBytes;
  }
  return copiedBytes;
}

size_t OCLBackend::copyMutableWeightsToDevice() {
  size_t copiedBytes = 0;
  for (auto it : tensors_) {
    if (!externalTensors_.count(it.first)) {
      continue;
    }
    if (auto *W = dyn_cast<WeightVar>(it.first)) {
      if (W->getMutability() == WeightVar::MutabilityKind::Constant)
        continue;
    }
    Tensor *T = externalTensors_[it.first];
    copiedBytes += copyValueToDevice(it.first);
  }
  // Do it!
  clFinish(commands_);
  return copiedBytes;
}

size_t OCLBackend::copyConstantWeightsToDevice() {
  size_t copiedBytes = 0;
  for (auto it : tensors_) {
    if (!externalTensors_.count(it.first)) {
      continue;
    }
    if (auto *W = dyn_cast<WeightVar>(it.first)) {
      if (W->getMutability() != WeightVar::MutabilityKind::Constant)
        continue;
    }
    Tensor *T = externalTensors_[it.first];
    copiedBytes += copyValueToDevice(it.first);
  }
  // Do it!
  clFinish(commands_);
  return copiedBytes;
}

size_t OCLBackend::copyMutableWeightsFromDevice() {
  size_t copiedBytes = 0;
  clFinish(commands_);

  for (auto it : tensors_) {
    if (!externalTensors_.count(it.first)) {
      continue;
    }
    if (auto *W = dyn_cast<WeightVar>(it.first)) {
      if (W->getMutability() == WeightVar::MutabilityKind::Constant)
        continue;
    }
    Tensor *T = externalTensors_[it.first];
    copiedBytes += copyValueFromDevice(it.first);
  }
  clFinish(commands_);
  return copiedBytes;
}

void OCLBackend::init() {
  for (auto &v : F_->getGraph()->getParent()->getVars()) {
    auto *w = F_->getWeightForNode(v);
    assert(!externalTensors_.count(w) && "The tensor is already registered");
    externalTensors_[w] = &v->getPayload();
  }

  // Assign device-space addresses to the weights.
  for (auto it : externalTensors_) {
    Tensor *T = it.second;
    size_t sizeInBytes = T->getType().getSizeInBytes();
    size_t addr = allocator_.allocate(sizeInBytes);
    // Associate the new buffer with the weight value.
    tensors_[it.first] = addr;
  }

  // Assign device-space addresses to the activations.
  for (auto &I : F_->getInstrs()) {
    if (auto *A = llvm::dyn_cast<AllocActivationInst>(I)) {
      auto numBytes = I->getSizeInBytes();
      size_t addr = allocator_.allocate(numBytes);
      assert(!tensors_.count(A) && "Allocation already made!");
      tensors_[A] = addr;
      continue;
    }

    if (auto *TV = llvm::dyn_cast<TensorViewInst>(I)) {
      assert(!tensors_.count(TV) && "Allocation already made!");
      tensors_[TV] = tensors_[TV->getSrc()];
      continue;
    }

    if (auto *D = llvm::dyn_cast<DeallocActivationInst>(I)) {
      auto *A = D->getAlloc();
      assert(tensors_.count(A) && "Invalid deallocation!");
      allocator_.deallocate(tensors_[A]);
      continue;
    }
  }

  // Ask the memory allocator how much memory is required. What was the high
  // watermark for this program.
  size_t requiredSpace = allocator_.getMaxMemoryUsage();
  DEBUG(llvm::dbgs() << "Allocated GPU memory block of size: " << requiredSpace
                     << "\n");
  llvm::dbgs() << "Allocated GPU memory block of size: " << requiredSpace
               << "\n";

  // Release the memory from the previous run.
  if (deviceBuffer_) {
    clReleaseMemObject(deviceBuffer_);
    deviceBuffer_ = nullptr;
  }

  deviceBuffer_ = allocDeviceBuffer(requiredSpace);
  requiredSpace_ = requiredSpace;
  // Copy constant weights just once.
  copyConstantWeightsToDevice();
}

void OCLBackend::clear() { externalTensors_.clear(); }

Tensor *OCLBackend::getTensor(const Value *v) const {
  assert(externalTensors_.count(v) && "Unknown value");
  auto ie = externalTensors_.find(v);
  return ie->second;
}

cl_mem OCLBackend::allocDeviceBuffer(size_t size) {
  const size_t alignment = 128;
  // Always allocate buffers properly aligned to hold values of any type.
  size = (size + alignment - 1) & ~(alignment - 1);
  auto buf =
      clCreateBuffer(context_, CL_MEM_READ_WRITE, size, nullptr, nullptr);
  GLOW_ASSERT(buf && "Allocation failed!");
  return buf;
}

void OCLBackend::freeDeviceBuffer(cl_mem buf) { clReleaseMemObject(buf); }