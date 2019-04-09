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

#include "glow/Runtime/HostManager/HostManager.h"
#include "glow/Backends/DeviceManager.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Partitioner/Partitioner.h"
#include "glow/Runtime/Executor/Executor.h"
#include "glow/Runtime/Provisioner/Provisioner.h"
#include "glow/Runtime/RuntimeTypes.h"

#include <future>
#include <queue>

using namespace glow;
using namespace runtime;

HostManager::HostManager(std::vector<std::unique_ptr<DeviceConfig>> configs) {
  // TODO: move all initialization out of constructor.
  TEMP_EXIT_ON_ERR(init(std::move(configs)));
}

llvm::Error
HostManager::init(std::vector<std::unique_ptr<DeviceConfig>> configs) {
  for (auto &config : configs) {
    auto backendKind = config->getBackendKind();
    if (backend_.find(backendKind) == backend_.end()) {
      backend_[backendKind].reset(createBackend(backendKind));
      orderedBackends_.emplace_back(backend_[backendKind].get());
    }
  }

  // Group devices of different kinds into groups.
  // Create provisioners responsible for each group.
  // std::unordered_map<BackendKind, DeviceIDTy> deviceCounts;
  DeviceIDTy deviceCount = 0;
  for (auto &config : configs) {
    auto backendKind = config->getBackendKind();
    // auto &deviceCount = deviceCounts[backendKind];
    if (backend_.find(backendKind) == backend_.end()) {
      backend_[backendKind].reset(createBackend(backendKind));
    }

    if (!config->hasName()) {
      config->setName(backend_[backendKind]->getBackendName() +
                      std::to_string(deviceCount));
    }

    devices_[deviceCount] = std::unique_ptr<DeviceManager>(
        DeviceManager::createDeviceManager(backendKind, std::move(config)));

    RETURN_IF_ERR(devices_[deviceCount]->init());

    deviceCount++;
  }
  for (auto &device : devices_) {
    auto backendKind = device.second->getBackendKind();
    provisioner_[backendKind].reset(new Provisioner(backendKind, devices_));
  }
  executor_.reset(createExecutor(devices_));

  return llvm::Error::success();
}

HostManager::~HostManager() { llvm::toString(clearHost()); }

llvm::Error HostManager::addNetwork(std::unique_ptr<Module> module) {
  std::lock_guard<std::mutex> networkLock(networkLock_);
  auto functions = module->getFunctions();
  for (auto &F : functions) {
    std::string name = F->getName();
    auto it = networks_.find(name);
    if (it != networks_.end()) {
      return MAKE_ERR(GlowErr::ErrorCode::RUNTIME_ERROR,
                      "Failed to add network: already have a function called " +
                          name);
    }
  }
  std::vector<DeviceInfo> deviceInfo;
  for (auto &device : devices_) {
    DeviceInfo info = DeviceInfo();
    info.availableMemory = device.second->getAvailableMemory();
    deviceInfo.push_back(info);
  }
  // First use a backend-based partitioner.
  BackendBasedPartitioner backendBasedPartitioner(module.get(),
                                                  orderedBackends_);
  auto &backendPartitions = backendBasedPartitioner.Partition();
  // Optimize functions before passing to device-based partitioner.
  // Currently hardcoding inference.
  for (auto &partition : backendPartitions) {
    CompilationOptions opts;
    opts.mode = CompilationMode::Infer;
    // auto backendKind = partition.root->backendKind;
    // auto *F = module->getFunction(partition.root->name);
    // if (!F) {
    //   continue;
    // }
    // backend_[backendKind]->optimizeFunction(F, opts);
    for (auto &partNode : partition.nodes) {
      auto *F = module->getFunction(partNode->name);
      assert(F && "No function found for a partition node");
      backend_[partNode->backendKind]->optimizeFunction(F, opts);
    }
  }
#if 0
  // Optimize functions before passing to partitioner.
  // Currently hardcoding inference.
  if (backend_) {
    CompilationOptions opts;
    opts.mode = CompilationMode::Infer;
    for (auto F : module->getFunctions()) {
      backend_->optimizeFunction(F, opts);
    }
  }
#endif
  Partitioner partitioner(module.get(), deviceInfo,
                          orderedBackends_[0]->getBackendKind());
  auto &nodeList = partitioner.Partition();

  for (auto &provisioner : provisioner_) {
    RETURN_IF_ERR(provisioner.second->provision(nodeList, *module));
  }

  // Clear everything but placeholders from the module then put it a shared_ptr
  // to be shared between all of the networks created from each function in the
  // module.
  module->clear(/* clearPlaceholders */ false);
  auto sharedModule = std::shared_ptr<Module>(std::move(module));

  for (auto &node : nodeList) {
    auto &networkData = networks_[(node.root)->name];
    networkData.dag = std::move(node);
    networkData.module = sharedModule;
  }

  return llvm::Error::success();
}

void HostManager::removeNetwork(llvm::StringRef networkName) {
  std::lock_guard<std::mutex> networkLock(networkLock_);
  auto networkIterator = networks_.find(networkName);
  if (networkIterator == networks_.end()) {
    return;
  }
  auto &nodes = networkIterator->second.dag.nodes;
  for (auto &node : nodes) {
    std::promise<void> removeNetwork;
    llvm::Error removeErr = llvm::Error::success();
    auto done = removeNetwork.get_future();
    devices_[node->deviceID]->evictNetwork(
        node->name,
        [&removeNetwork, &removeErr](std::string name, llvm::Error err) {
          removeErr = std::move(err);
          removeNetwork.set_value();
        });
    done.get();
    errToBool(std::move(removeErr));
    // Also remove compiledFunction from Provisioner.
    provisioner_[devices_[node->deviceID]->getBackendKind()]->removeFunction(
        node->name);
  }
  networks_.erase(networkIterator);
}

bool HostManager::networkAdded(llvm::StringRef networkName) {
  std::lock_guard<std::mutex> networkLock(networkLock_);
  return networks_.find(networkName) != networks_.end();
}

llvm::Error HostManager::clearHost() {
  // shutdown the executor, blocking on any current inflight and prevent new
  // requests from being serviced.
  executor_->shutdown();
  assert(activeRequestCount_ == 0 &&
         "All requests should be finished when shutting down HostManager.");

  std::lock_guard<std::mutex> networkLock(networkLock_);
  OneErrOnly errContainer;
  for (auto &it : devices_) {
    errContainer.set(it.second->stop());
  }

  for (auto &network : networks_) {
    for (auto &node : network.second.dag.nodes) {
      devices_[node->deviceID]->evictNetwork(node->name, /*evictCB=*/nullptr);
    }
  }
  networks_.clear();
  return errContainer.get();
}

RunIdentifierTy
HostManager::runNetwork(llvm::StringRef networkName,
                        std::unique_ptr<ExecutionContext> context,
                        ResultCBTy callback) {
  ScopedTraceBlock(context->getTraceContext(),
                   "runFunction_" + networkName.str());
  auto currentRun = totalRequestCount_++;
  std::lock_guard<std::mutex> networkLock(networkLock_);
  if (networks_.find(networkName) == networks_.end()) {
    callback(
        currentRun,
        MAKE_ERR(GlowErr::ErrorCode::RUNTIME_NET_NOT_FOUND,
                 llvm::formatv("Function {0} not found", networkName).str()),
        std::move(context));
    return currentRun;
  }

  size_t activeRequestCount = activeRequestCount_++;
  if (activeRequestCount >= activeRequestLimit_) {
    activeRequestCount_--;
    callback(
        currentRun,
        MAKE_ERR(GlowErr::ErrorCode::RUNTIME_REQUEST_REFUSED,
                 strFormat("The number of allowed requests has been exceeded. "
                           "active requests: %lu allowed requests: %u",
                           activeRequestCount, activeRequestLimit_)),
        std::move(context));
    return currentRun;
  }

  executor_->run(
      networks_[networkName].dag.root.get(), std::move(context), currentRun,
      [&activeRequest = this->activeRequestCount_, callback,
       name = networkName.str()](RunIdentifierTy runID, llvm::Error err,
                                 std::unique_ptr<ExecutionContext> context) {
        --activeRequest;
        TRACE_EVENT_INSTANT(context->getTraceContext(), "finish_" + name);
        callback(runID, std::move(err), std::move(context));
      });
  return currentRun;
}
