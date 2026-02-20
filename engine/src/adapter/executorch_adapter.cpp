#include "executorch_adapter.h"
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/backends/apple/mps/runtime/MPSDelegateHeader.h>
#include <executorch/extension/threadpool/threadpool.h>
#include <vector>
#include <memory>
#include <iostream>
#include <chrono>

using namespace executorch::extension;
using namespace executorch::extension::module;
using namespace executorch::runtime;
using namespace executorch::aten;

struct ExecuTorchModule {
    std::unique_ptr<Module> module;
};

extern "C" {

ExecuTorchModule* executorch_module_load(const char* pte_path) {
    auto start = std::chrono::steady_clock::now();
    fprintf(stderr, "[CPP] Loading module from: %s\n", pte_path);
    fflush(stderr);

    runtime_init();

    // Initialize threadpool for CPU performance. 
    // Usually get_threadpool() creates the singleton if it doesn't exist.
    (void)executorch::extension::threadpool::get_threadpool();

    auto module = std::make_unique<Module>(pte_path, Module::LoadMode::Mmap);
    auto status = module->load();
    if (status != Error::Ok) {
        fprintf(stderr, "[CPP] Failed to load module: %d\n", (int)status);
        fflush(stderr);
        return nullptr;
    }

    auto end = std::chrono::steady_clock::now();
    fprintf(stderr, "[CPP] Module loaded in %.2f ms\n", 
            std::chrono::duration<double, std::milli>(end - start).count());
    fflush(stderr);

    auto* m = new ExecuTorchModule();
    m->module = std::move(module);
    return m;
}

void executorch_module_destroy(ExecuTorchModule* module) {
    if (module) {
        delete module;
    }
}

int32_t executorch_module_forward(
    ExecuTorchModule* module,
    const int64_t* input_tokens,
    size_t input_len,
    float* output_logits,
    size_t* output_vocab_size
) {
    if (!module || !module->module) return -1;

    auto start_all = std::chrono::steady_clock::now();

    // 1. Prepare Input Tensor (1, 128)
    SizesType sizes[] = {1, 128};
    TensorImpl input_impl(ScalarType::Long, 2, sizes, (void*)input_tokens);
    Tensor input_tensor(&input_impl);
    
    std::vector<EValue> inputs = {EValue(input_tensor)};

    // 2. Forward
    auto forward_start = std::chrono::steady_clock::now();
    auto result = module->module->forward(inputs);
    if (!result.ok()) {
        fprintf(stderr, "[CPP] Forward failed with error %d\n", (int)result.error());
        fflush(stderr);
        return -2;
    }
    auto forward_end = std::chrono::steady_clock::now();

    // 3. Process Output
    auto outputs = result.get();
    if (outputs.empty()) return -3;

    EValue& output_evalue = outputs[0];
    if (!output_evalue.isTensor()) return -4;

    Tensor output_tensor = output_evalue.toTensor();
    if (output_tensor.dim() != 3) return -5;
    
    *output_vocab_size = output_tensor.size(2);
    size_t total_elements = output_tensor.numel();
    
    const float* data = output_tensor.const_data_ptr<float>();
    if (!data) return -6;

    memcpy(output_logits, data, total_elements * sizeof(float));

    auto end_all = std::chrono::steady_clock::now();
    fprintf(stderr, "[CPP] Forward total: %.2f ms (exec: %.2f ms)\n", 
            std::chrono::duration<double, std::milli>(end_all - start_all).count(),
            std::chrono::duration<double, std::milli>(forward_end - forward_start).count());
    fflush(stderr);

    return 0; // Success
}
}
