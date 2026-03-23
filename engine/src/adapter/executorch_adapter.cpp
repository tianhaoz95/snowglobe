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
    (void)executorch::extension::threadpool::get_threadpool();

    // Use Mmap for efficiency if possible
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
    const void* input_tokens,
    size_t input_len,
    int32_t use_int32,
    float* output_logits,
    size_t* output_vocab_size
) {
    if (!module || !module->module || !input_tokens || !output_logits || !output_vocab_size) {
        return -1;
    }

    if (input_len != 128) {
        fprintf(stderr, "[CPP] Error: Input length %zu is not 128. Fixed size models require exact input size.\n", input_len);
        fflush(stderr);
        return -8;
    }

    auto start_all = std::chrono::steady_clock::now();

    // 1. Prepare Input Tensor (1, 128)
    TensorImpl::SizesType sizes[] = {1, 128};
    TensorImpl::DimOrderType dim_order[] = {0, 1};
    
    ScalarType input_type = use_int32 ? ScalarType::Int : ScalarType::Long;
    TensorImpl input_impl(input_type, 2, sizes, (void*)input_tokens, dim_order);
    Tensor input_tensor(&input_impl);
    
    std::vector<EValue> inputs = {EValue(input_tensor)};

    // 2. Forward
    auto forward_start = std::chrono::steady_clock::now();
    auto result = module->module->forward(inputs);
    if (!result.ok()) {
        fprintf(stderr, "[CPP] Forward failed with error %d (0x%x)\n", (int)result.error(), (unsigned int)result.error());
        fflush(stderr);
        return -2;
    }
    auto forward_end = std::chrono::steady_clock::now();

    // 3. Process Output
    fprintf(stderr, "[CPP] Processing outputs...\n");
    fflush(stderr);
    auto outputs = result.get();
    if (outputs.empty()) {
        fprintf(stderr, "[CPP] Forward returned empty outputs\n");
        fflush(stderr);
        return -3;
    }

    EValue& output_evalue = outputs[0];
    if (!output_evalue.isTensor()) {
        fprintf(stderr, "[CPP] Output 0 is not a tensor\n");
        fflush(stderr);
        return -4;
    }

    Tensor output_tensor = output_evalue.toTensor();
    if (output_tensor.dim() != 3) {
        fprintf(stderr, "[CPP] Output tensor has wrong dim: %zu (expected 3)\n", (size_t)output_tensor.dim());
        fflush(stderr);
        return -5;
    }
    
    *output_vocab_size = (size_t)output_tensor.size(2);
    size_t total_elements = (size_t)output_tensor.numel();
    
    const float* data = output_tensor.const_data_ptr<float>();
    if (!data) {
        fprintf(stderr, "[CPP] Failed to get output data pointer\n");
        fflush(stderr);
        return -6;
    }

    // Safety Bounds Check to prevent SIGSEGV during memcpy
    // Match the Rust allocation: 128 * 152064
    size_t max_allowed_elements = 128 * 152064;
    if (total_elements > max_allowed_elements) {
        fprintf(stderr, "[CPP] Output size %zu exceeds buffer limit %zu!\n", total_elements, max_allowed_elements);
        fflush(stderr);
        return -7; 
    }

    memcpy(output_logits, data, total_elements * sizeof(float));

    auto end_all = std::chrono::steady_clock::now();
    fprintf(stderr, "[CPP] Forward total: %.2f ms (exec: %.2f ms), output shape: [%zd, %zd, %zd]\n", 
            std::chrono::duration<double, std::milli>(end_all - start_all).count(),
            std::chrono::duration<double, std::milli>(forward_end - forward_start).count(),
            (ssize_t)output_tensor.size(0), (ssize_t)output_tensor.size(1), (ssize_t)output_tensor.size(2));
    fflush(stderr);

    return 0; // Success
}
}
