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

#include <android/log.h>
#define LOG_TAG "SNOWGLOBE_PTE"
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

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
    ALOGE("[CPP] Loading module from: %s", pte_path);

    runtime_init();

    // Initialize threadpool for CPU performance. 
    (void)executorch::extension::threadpool::get_threadpool();

    // Use Mmap for efficiency if possible
    auto module = std::make_unique<Module>(pte_path, Module::LoadMode::Mmap);

    auto status = module->load();
    if (status != Error::Ok) {
        ALOGE("[CPP] Failed to load module: %d", (int)status);
        return nullptr;
    }

    // Log method names to understand what's in the model
    auto num_methods_result = module->num_methods();
    if (num_methods_result.ok()) {
        ALOGE("[CPP] Model has %zu method(s)", *num_methods_result);
    }

    auto end = std::chrono::steady_clock::now();
    ALOGE("[CPP] Module loaded in %.2f ms", 
            std::chrono::duration<double, std::milli>(end - start).count());

    auto* m = new ExecuTorchModule();
    m->module = std::move(module);
    return m;
}

const char* executorch_module_get_name(ExecuTorchModule* module) {
    if (!module || !module->module) {
        return nullptr;
    }
    auto program = module->module->program();
    if (!program) {
        return "forward";
    }
    auto name_result = program->get_method_name(0);
    if (!name_result.ok()) {
        return "forward";
    }
    return *name_result;
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
        ALOGE("[CPP] Error: Input length %zu is not 128. Fixed size models require exact input size.", input_len);
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
    ALOGE("[CPP] Starting forward pass (use_int32=%d, input_len=%zu)...", use_int32, input_len);
    auto forward_start = std::chrono::steady_clock::now();
    auto result = module->module->forward(inputs);
    if (!result.ok()) {
        ALOGE("[CPP] Forward failed with error %d (0x%x)", (int)result.error(), (unsigned int)result.error());
        // Try to get more details about the error
        if (result.error() == Error::OperatorMissing) {
            ALOGE("[CPP] OperatorMissing: A required operator is not registered in the runtime.");
            ALOGE("[CPP] Ensure portable_kernels and xnnpack_backend are linked with whole-archive.");
        }
        return -2;
    }
    auto forward_end = std::chrono::steady_clock::now();
    ALOGE("[CPP] Forward done in %.2f ms", std::chrono::duration<double, std::milli>(forward_end - forward_start).count());

    // 3. Process Output
    ALOGE("[CPP] Processing outputs...");
    auto outputs = result.get();
    if (outputs.empty()) {
        ALOGE("[CPP] Forward returned empty outputs");
        return -3;
    }

    EValue& output_evalue = outputs[0];
    if (!output_evalue.isTensor()) {
        ALOGE("[CPP] Output 0 is not a tensor");
        return -4;
    }

    Tensor output_tensor = output_evalue.toTensor();
    if (output_tensor.dim() != 3) {
        ALOGE("[CPP] Output tensor has wrong dim: %zu (expected 3)", (size_t)output_tensor.dim());
        return -5;
    }
    
    *output_vocab_size = (size_t)output_tensor.size(2);
    size_t total_elements = (size_t)output_tensor.numel();
    
    const float* data = output_tensor.const_data_ptr<float>();
    if (!data) {
        ALOGE("[CPP] Failed to get output data pointer");
        return -6;
    }

    // Safety Bounds Check to prevent SIGSEGV during memcpy
    // Match the Rust allocation: 128 * 152064
    size_t max_allowed_elements = 128 * 152064;
    if (total_elements > max_allowed_elements) {
        ALOGE("[CPP] Output size %zu exceeds buffer limit %zu!", total_elements, max_allowed_elements);
        return -7; 
    }

    memcpy(output_logits, data, total_elements * sizeof(float));

    auto end_all = std::chrono::steady_clock::now();
    ALOGE("[CPP] Forward total: %.2f ms (exec: %.2f ms), output shape: [%zd, %zd, %zd]", 
            std::chrono::duration<double, std::milli>(end_all - start_all).count(),
            std::chrono::duration<double, std::milli>(forward_end - forward_start).count(),
            (ssize_t)output_tensor.size(0), (ssize_t)output_tensor.size(1), (ssize_t)output_tensor.size(2));

    return 0; // Success
}
}
