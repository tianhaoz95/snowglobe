#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ExecuTorchModule ExecuTorchModule;

ExecuTorchModule* executorch_module_load(const char* pte_path);
void executorch_module_destroy(ExecuTorchModule* module);

// For simplicity, we assume a fixed input size of (1, 128) for tokens as per current lib.rs logic
// and output is (1, 128, vocab_size)
int32_t executorch_module_forward(
    ExecuTorchModule* module,
    const int64_t* input_tokens,
    size_t input_len,
    float* output_logits,
    size_t* output_vocab_size
);

#ifdef __cplusplus
}
#endif
