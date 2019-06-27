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

#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

#include "glow/Backend/Backend.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Log.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPasses.h"

#include "llvm/Support/Casting.h"

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

namespace {
/// The name of the temporary function to be used to perform constant folding.
constexpr const char *constEvaluationFunctionName =
    "__constEvaluationFunction__";

/// \returns true if a node \p N is a constant operation, i.e. it is a trivial
/// constant like Constant or Splat or all of its inputs are recursively
/// constant operations, and it has no side-effects and supported by the \p
/// backend.
bool isConstantOperation(const Node *N, const Backend &backend) {
  // An operation with side-effects cannot be computed at compile-time.
  if (N->hasSideEffects()) {
    return false;
  }
  // Constant and splat nodes are trivially constant operations.
  if (llvm::isa<Constant>(N) || llvm::isa<SplatNode>(N)) {
    return true;
  }
  // If the operation is not supported by the backend, it cannot be computed at
  // compile-time.
  if (!backend.shouldLower(N) && !backend.isOpSupported(NodeInfo(*N))) {
    return false;
  }
  if (llvm::isa<Placeholder>(N)) {
    return false;
  }
  for (unsigned idx = 0, e = N->getNumInputs(); idx < e; ++idx) {
    auto input = N->getNthInput(idx);
    if (!isConstantOperation(input.getNode(), backend)) {
      return false;
    }
  }
  return true;
}

/// Compile the function \p F for the provided \p backend using the compilation
/// context \p cctx.
/// \returns compiled function.
std::unique_ptr<CompiledFunction> compile(Backend &backend, Function &F,
                                          CompilationContext &cctx) {
  EXIT_ON_ERR(::glow::optimizeFunction(&F, backend, cctx));
  for (const Node &N : F.getNodes()) {
    CHECK(backend.isOpSupported(N))
        << "Backend must support all nodes after high-level optimizations but "
           "encountered unsupported operator: "
        << N.getDebugDesc();
  }
  auto funcOrErr = backend.compile(&F, cctx.backendOpts);
  EXIT_ON_ERR(funcOrErr.takeError());
  return std::move(*funcOrErr);
}

/// Runs the compiled function \p compiledF on the \p backend using provided \p
/// bindings.
void run(Backend &backend, CompiledFunction &compiledF,
         PlaceholderBindings &bindings) {
  std::unique_ptr<PlaceholderBindings> bindingsPtr(&bindings);
  ExecutionContext context(std::move(bindingsPtr));
  // TODO: Add only constants used by F to the compiled function. This should
  // reduce the amount of data that needs to be copied.
  auto executeErr = compiledF.execute(&context);
  EXIT_ON_ERR(std::move(executeErr));
  // don't delete bindings.
  context.movePlaceholderBindings().release();
}

/// Evaluates a provided constant operation \p C using the provided \p backend
/// and using the compilation context \p cctx.
/// \returns constant results.
std::vector<Constant *>
evaluateConstantOperation(Backend &backend, CompilationContext &cctx, Node *C) {
  PlaceholderBindings bindings;
  assert(isConstantOperation(C, backend) && "Expected a constant expression");
  // Constants and splats do not need to be constant evaluated.
  if (isa<Constant>(C) || isa<SplatNode>(C)) {
    return {};
  }
  Module &mod = *C->getParent()->getParent();
  // Create a temporary function to perform the constant operation.
  Function *constEvaluationF = mod.createFunction(constEvaluationFunctionName);
  // Mapping from existing nodes to the new ones.
  NodeMap currToNew;
  // Clone the constant operation and some of its inputs if necessary.
  auto *clonedC = recursiveClone(constEvaluationF, C, currToNew);
  // Create save nodes for each of the results.
  llvm::SmallVector<SaveNode *, 16> savedResults;
  for (unsigned idx = 0, e = clonedC->getNumResults(); idx < e; ++idx) {
    auto *SN = constEvaluationF->createSave("save", clonedC->getNthResult(idx));
    savedResults.emplace_back(SN);
    bindings.allocate(SN->getPlaceholder());
  }
  // Run the temporary backend to perform this constant operation
  // evaluation.
  EXIT_ON_ERR(executeFunction(backend, *constEvaluationF, bindings, cctx,
                              /* isConstant */ true));
  // Get the results of the constant operation compile-time computation and
  // create new constants from it.
  std::vector<Constant *> constResults;
  constResults.reserve(savedResults.size());
  for (auto *SN : savedResults) {
    Tensor *outputTensor = bindings.get(SN->getPlaceholder());
    auto *constResult =
        mod.createConstant(strFormat("constant"), *outputTensor);
    constResults.emplace_back(constResult);
  }
  // Remove the temporary function.
  mod.eraseFunction(constEvaluationF);
  return constResults;
}

/// Check if function \p F consists of constant operations only.
llvm::Error verifyConstantFunction(Backend &backend, Function &F) {
  for (auto &N : F.getNodes()) {
    // Saving results is fine.
    if (isa<SaveNode>(&N)) {
      continue;
    }
    // Placeholders can be used just to save results.
    if (isa<Placeholder>(&N)) {
      if (N.hasOneUse()) {
        auto SN = dyn_cast<SaveNode>(N.getUsers().begin()->getUser());
        if (SN && SN->getPlaceholder() == &N) {
          continue;
        }
      }
      RETURN_ERR("Expected constant operation");
    }
    RETURN_ERR_IF_NOT(isConstantOperation(&N, backend),
                      "Expected constant operation");
  }
  return llvm::Error::success();
}

} // namespace

llvm::Error glow::executeFunction(Backend &backend, Function &F,
                                  PlaceholderBindings &bindings,
                                  CompilationContext &cctx, bool isConstant) {
  if (isConstant) {
    RETURN_IF_ERR(verifyConstantFunction(backend, F));
  }
  auto compiledF = compile(backend, F, cctx);
  run(backend, *compiledF, bindings);
  return llvm::Error::success();
}

/// Perform constant folding in the function \p F . Any non-trivial node (i.e.
/// not a constant or a splat) that can be computed at compile-time is going to
/// be computed at compile-time. \returns true if any foldings were performed.
bool glow::ConstantFold::run(Function *F) {
  LOG_SCOPE(F->getLogContext(), "glow::constantFold")
  CompilationContext cctx;
  // Graph optimizations may be required to e.g. perform lowering of some nodes
  // that are not supported natively.
  cctx.optimizationOpts.enableGraphOptz = true;
  // Do not recursively call constant folding.
  cctx.optimizationOpts.enableConstantFolding = false;
  cctx.backendOpts.collectConstants = true;
  bool changed = false;
  // Backend to be used for compile-time computations.
  std::unique_ptr<Backend> backend(createBackend("Interpreter"));
  // Set of constant operations to be computed.
  llvm::SmallVector<Node *, 64> worklist;
  // Collect all non-trivial constant operations.
  for (auto &N : F->getNodes()) {
    // Skip trivial nodes/operations that do not require any constant
    // computations.
    if (llvm::isa<Storage>(N) || llvm::isa<Constant>(N) ||
        llvm::isa<SplatNode>(N)) {
      continue;
    }
    if (isConstantOperation(&N, *backend)) {
      worklist.push_back(&N);
    }
  }
  // Compute the result of each collected non-trivial constant operation.
  while (!worklist.empty()) {
    auto *C = worklist.pop_back_val();
    std::vector<Constant *> constResults =
        evaluateConstantOperation(*backend, cctx, C);
    // Replace all results of the original operation by the computed
    // compile-time results of this operation.
    for (unsigned idx = 0, e = constResults.size(); idx < e; ++idx) {
      auto constResult = constResults[idx];
      // Replace the old result by the new constant result.
      C->getNthResult(idx).replaceAllUsesOfWith(constResult);
      // Add all users of constResult to the worklist if they became constant
      // operations.
      for (auto &use : constResult->getUsers()) {
        auto *user = use.getUser();
        if (isConstantOperation(user, *backend)) {
          worklist.push_back(user);
        }
      }
    }
    changed = true;
  }
  return changed;
}

std::vector<Constant *> glow::constantFold(Node *N) {
  LOG_SCOPE(N->getParent()->getLogContext(), "glow::constantFold")

  std::unique_ptr<Backend> backend(createBackend("Interpreter"));
  if (!isConstantOperation(N, *backend)) {
    return {};
  }
  CompilationContext cctx;
  cctx.optimizationOpts.enableGraphOptz = false;
  // Do not recursively call constant folding.
  cctx.optimizationOpts.enableConstantFolding = false;
  cctx.backendOpts.collectConstants = true;
  return evaluateConstantOperation(*backend, cctx, N);
}
