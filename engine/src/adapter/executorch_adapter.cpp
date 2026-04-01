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

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "SNOWGLOBE_PTE"
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define ALOGE(...) fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n")
#endif

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
    ALOGE("[CPP] executorch_module_load START: %s", pte_path);

    runtime_init();
    ALOGE("[CPP] runtime_init() done");

    // Initialize threadpool for CPU performance. 
    auto* threadpool = executorch::extension::threadpool::get_threadpool();
    if (threadpool) {
        ALOGE("[CPP] Resetting threadpool...");
        // Use 4 threads for better performance on mobile big cores
        threadpool->_unsafe_reset_threadpool(4);
        ALOGE("[CPP] Threadpool initialized with 4 threads");
    }

    // Use File for speed on some Android devices where Mmap might be slow
    ALOGE("[CPP] Creating Module instance (LoadMode::File)...");
    auto module = std::make_unique<Module>(pte_path, Module::LoadMode::File);

    ALOGE("[CPP] Calling module->load()...");
    auto status = module->load();
    if (status != Error::Ok) {
        ALOGE("[CPP] Failed to load module: %d", (int)status);
        return nullptr;
    }
    ALOGE("[CPP] module->load() SUCCESS");

    // Log method names to understand what's in the model
    auto num_methods_result = module->num_methods();
    if (num_methods_result.ok()) {
        ALOGE("[CPP] Model has %zu method(s)", *num_methods_result);
        auto names_result = module->method_names();
        if (names_result.ok()) {
            size_t i = 0;
            for (const auto& name : *names_result) {
                ALOGE("[CPP] Method %zu: %s", i++, name.c_str());
            }
        }
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
        ALOGE("[CPP] executorch_module_get_name: module is null");
        return nullptr;
    }
    auto program = module->module->program();
    if (!program) {
        ALOGE("[CPP] executorch_module_get_name: program is null");
        return "forward";
    }
    auto name_result = program->get_method_name(0);
    if (!name_result.ok()) {
        ALOGE("[CPP] executorch_module_get_name: get_method_name(0) FAILED");
        return "forward";
    }
    ALOGE("[CPP] executorch_module_get_name: found method '%s'", *name_result);
    return *name_result;
}

size_t executorch_module_get_vocab_size(ExecuTorchModule* module) {
    if (!module || !module->module) {
        return 0;
    }
    auto method_name = executorch_module_get_name(module);
    auto meta_result = module->module->method_meta(method_name);
    if (!meta_result.ok()) {
        ALOGE("[CPP] get_vocab_size: method_meta FAILED");
        return 0;
    }
    auto num_outputs = meta_result->num_outputs();
    if (num_outputs == 0) {
        ALOGE("[CPP] get_vocab_size: no outputs");
        return 0;
    }
    auto output_meta = meta_result->output_tensor_meta(0);
    if (!output_meta.ok()) {
        ALOGE("[CPP] get_vocab_size: output_tensor_meta(0) FAILED");
        return 0;
    }
    auto sizes = output_meta->sizes();
    if (sizes.size() < 3) {
        ALOGE("[CPP] get_vocab_size: expected 3 dims, got %zu", sizes.size());
        return 0;
    }
    ALOGE("[CPP] get_vocab_size: %zd", (ssize_t)sizes[2]);
    return (size_t)sizes[2];
}

void executorch_module_destroy(ExecuTorchModule* module) {
    ALOGE("[CPP] executorch_module_destroy START");
    if (module) {
        delete module;
    }
    ALOGE("[CPP] executorch_module_destroy END");
}

int32_t executorch_module_forward(
    ExecuTorchModule* module,
    const void* input_tokens,
    size_t input_len,
    int32_t use_int32,
    float* output_logits,
    size_t* output_vocab_size,
    size_t start_pos,
    size_t num_positions
) {
    if (!module || !module->module || !input_tokens || !output_logits || !output_vocab_size) {
        ALOGE("[CPP] executorch_module_forward: INVALID ARGUMENTS");
        return -1;
    }

    if (input_len != 128) {
        ALOGE("[CPP] Error: Input length %zu is not 128. Fixed size models require exact input size.", input_len);
        return -8;
    }

    ALOGE("[CPP] executorch_module_forward START (input_len=%zu, start_pos=%zu, num_positions=%zu)", 
            input_len, start_pos, num_positions);

    auto start_all = std::chrono::steady_clock::now();

    // 1. Prepare Input Tensor (1, 128)
    TensorImpl::SizesType sizes[] = {1, 128};
    TensorImpl::DimOrderType dim_order[] = {0, 1};
    
    ScalarType input_type = use_int32 ? ScalarType::Int : ScalarType::Long;
    
    TensorImpl input_impl(input_type, 2, sizes, (void*)input_tokens, dim_order);
    Tensor input_tensor(&input_impl);
    
    std::vector<EValue> inputs = {EValue(input_tensor)};

    // 2. Forward
    ALOGE("[CPP] Calling module->forward(inputs)...");
    auto forward_start = std::chrono::steady_clock::now();
    auto result = module->module->forward(inputs);
    if (!result.ok()) {
        ALOGE("[CPP] Forward failed with error %d", (int)result.error());
        return -2;
    }
    auto forward_end = std::chrono::steady_clock::now();
    ALOGE("[CPP] module->forward(inputs) SUCCESS");

    // 3. Process Output
    auto outputs = result.get();
    if (outputs.empty()) {
        ALOGE("[CPP] Forward returned empty outputs");
        return -3;
    }

    EValue& output_evalue = outputs[0];
    Tensor output_tensor = output_evalue.toTensor();
    
    size_t vocab_size = (size_t)output_tensor.size(2);
    *output_vocab_size = vocab_size;
    
    const float* data = output_tensor.const_data_ptr<float>();
    if (!data) {
        ALOGE("[CPP] Could not get output tensor data pointer");
        return -6;
    }

    // Optimization: Only copy the requested positions
    if (num_positions == 0) {
        num_positions = (size_t)output_tensor.size(1); // Default to all
    }
    
    if (start_pos + num_positions > (size_t)output_tensor.size(1)) {
        ALOGE("[CPP] Error: Requested range [%zu, %zu) out of bounds [0, %zd)", 
            start_pos, start_pos + num_positions, (ssize_t)output_tensor.size(1));
        return -9;
    }

    size_t copy_elements = num_positions * vocab_size;
    size_t offset_elements = start_pos * vocab_size;
    
    // Safety Bounds Check
    size_t max_allowed_elements = 128 * 250000; // Allow for larger vocab (Qwen 3.5 is 248k)
    if (copy_elements > max_allowed_elements) {
        ALOGE("[CPP] Error: copy_elements %zu exceeds max_allowed %zu", copy_elements, max_allowed_elements);
        return -7; 
    }

    ALOGE("[CPP] Copying %zu elements (offset=%zu)...", copy_elements, offset_elements);
    memcpy(output_logits, data + offset_elements, copy_elements * sizeof(float));

    auto end_all = std::chrono::steady_clock::now();
    ALOGE("[CPP] Forward total: %.2f ms (exec: %.2f ms), output shape: [%zd, %zd, %zd], copied %zu pos", 
            std::chrono::duration<double, std::milli>(end_all - start_all).count(),
            std::chrono::duration<double, std::milli>(forward_end - forward_start).count(),
            (ssize_t)output_tensor.size(0), (ssize_t)output_tensor.size(1), (ssize_t)output_tensor.size(2),
            num_positions);

    return 0; // Success
}
}
